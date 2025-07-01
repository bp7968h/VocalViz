use crate::whisper::{self as m, Config, audio, model::Whisper as Model, timestamps};
use crate::{multilingual, utils};
use anyhow::{Error as E, Result};
use candle_core::{Device, IndexOp, Tensor};
use candle_nn::{VarBuilder, ops::softmax};
use hf_hub::{Repo, RepoType, api::sync::Api};
use rand::SeedableRng;
use rand::distr::Distribution;
use rand::rngs::StdRng;
use rand_distr::weighted::WeightedIndex;
use serde::{Deserialize, Serialize};
use std::process;
use std::{num::NonZeroUsize, path::Path};
use symphonia::core::audio::{AudioBufferRef, Signal};
use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
use symphonia::core::conv::FromSample;
use tokenizers::Tokenizer;

use crate::config::{WhisperLanguage, WhisperModel};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Word {
    pub text: String,
    pub start: f32,
    pub end: f32,
}

impl Word {
    pub fn new(text: String, start: f32, end: f32) -> Self {
        Word { text, start, end }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WordSegment {
    pub text: String,
    pub start: f64,
    pub end: f64,
    pub words: Vec<Word>,
}

impl WordSegment {
    pub fn new(start: f64, end: f64) -> Self {
        WordSegment {
            start,
            end,
            text: String::new(),
            words: Vec::new(),
        }
    }

    pub fn with_text(mut self, text: &str) -> Self {
        self.text.push_str(text);
        self
    }

    pub fn with_words(mut self, words: &[Word]) -> Self {
        self.words.extend_from_slice(words);
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TranscriptSegment {
    pub segments: Vec<WordSegment>,
}

pub struct Transcriber {
    device: Device,
}
impl Transcriber {
    pub fn new() -> Result<Self> {
        let device = utils::device(false)?;
        Ok(Transcriber { device })
    }

    pub fn transcribe<P: AsRef<Path>>(
        &self,
        audio_path: P,
        model: &WhisperModel,
        language: WhisperLanguage,
    ) -> Result<TranscriptSegment> {
        println!("🤖 Loading Whisper model: {:?}", model);

        // Load model files
        let (model_id, revision) = model.model_and_revision(language);
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            model_id.to_string(),
            RepoType::Model,
            revision.to_string(),
        ));

        let config_filename = repo.get("config.json")?;
        let tokenizer_filename = repo.get("tokenizer.json")?;
        let weights_filename = repo.get("model.safetensors")?;

        // Load config and tokenizer
        let mut config: Config = serde_json::from_str(&std::fs::read_to_string(config_filename)?)?;
        let tokenizer = Tokenizer::from_file(tokenizer_filename).map_err(E::msg)?;
        config.use_self_attention_kv_cache = true;
        config.dtw_timestamps = true;

        // Load mel filters
        let mel_bytes = match config.num_mel_bins {
            80 => include_bytes!("../assets/melfilters.bytes").as_slice(),
            128 => include_bytes!("../assets/melfilters128.bytes").as_slice(),
            nmel => anyhow::bail!("unexpected num_mel_bins {nmel}"),
        };
        let mut mel_filters = vec![0f32; mel_bytes.len() / 4];
        <byteorder::LittleEndian as byteorder::ByteOrder>::read_f32_into(
            mel_bytes,
            &mut mel_filters,
        );

        // Process audio
        let (pcm_data, sample_rate) = Self::load_audio(audio_path)?;
        if sample_rate != m::SAMPLE_RATE as u32 {
            anyhow::bail!("input file must have a {} sampling rate", m::SAMPLE_RATE);
        }

        println!("🎵 PCM data loaded: {} samples", pcm_data.len());

        // Convert to mel spectrogram
        let total_frames = (pcm_data.len() as f64 / m::HOP_LENGTH as f64).round() as usize;
        let mel = audio::pcm_to_mel(&config, &pcm_data, &mel_filters);
        let mel_len = mel.len();
        let mel = Tensor::from_vec(
            mel,
            (1, config.num_mel_bins, mel_len / config.num_mel_bins),
            &self.device,
        )?;

        println!("📊 Mel spectrogram: {:?}", mel.dims());

        // Load model
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&[weights_filename], m::DTYPE, &self.device)?
        };
        let mut whisper_model = Model::load(&vb, config)?;

        // Detect language
        let language_token = match language {
            WhisperLanguage::Auto => Some(multilingual::detect_language(
                &mut whisper_model,
                &tokenizer,
                &mel,
            )?),
            WhisperLanguage::English => match utils::token_id(&tokenizer, "<|en|>") {
                Ok(token_id) => Some(token_id),
                Err(_) => None,
            },
        };
        if language_token.is_none() {
            eprintln!("⚠️ Problem detecting language");
            process::exit(1);
        }
        // Run transcription with timestamps
        // let segments = self.run_transcription(model, tokenizer, mel, language_token)?;
        let mut decoder = Decoder::new(
            whisper_model,
            tokenizer,
            &self.device,
            language_token,
            *model,
        )?;
        let segments_with_words = decoder.run(&mel, total_frames, language)?;

        Ok(TranscriptSegment {
            segments: segments_with_words,
        })
    }

    fn load_audio<P: AsRef<Path>>(audio_path: P) -> Result<(Vec<f32>, u32)> {
        // Open the media source.
        let src = std::fs::File::open(audio_path)?;

        // Create the media source stream.
        let mss = symphonia::core::io::MediaSourceStream::new(Box::new(src), Default::default());

        // Create a probe hint using the file's extension. [Optional]
        let hint = symphonia::core::probe::Hint::new();

        // Use the default options for metadata and format readers.
        let meta_opts: symphonia::core::meta::MetadataOptions = Default::default();
        let fmt_opts: symphonia::core::formats::FormatOptions = Default::default();

        // Probe the media source.
        let probed = symphonia::default::get_probe().format(&hint, mss, &fmt_opts, &meta_opts)?;
        // Get the instantiated format reader.
        let mut format = probed.format;

        // Find the first audio track with a known (decodeable) codec.
        let track = format
            .tracks()
            .iter()
            .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
            .expect("no supported audio tracks");

        // Use the default options for the decoder.
        let dec_opts: DecoderOptions = Default::default();

        // Create a decoder for the track.
        let mut decoder = symphonia::default::get_codecs()
            .make(&track.codec_params, &dec_opts)
            .expect("unsupported codec");
        let track_id = track.id;
        let sample_rate = track.codec_params.sample_rate.unwrap_or(0);
        let mut pcm_data = Vec::new();
        // The decode loop.
        while let Ok(packet) = format.next_packet() {
            // Consume any new metadata that has been read since the last packet.
            while !format.metadata().is_latest() {
                format.metadata().pop();
            }

            // If the packet does not belong to the selected track, skip over it.
            if packet.track_id() != track_id {
                continue;
            }
            match decoder.decode(&packet)? {
                AudioBufferRef::F32(buf) => pcm_data.extend(buf.chan(0)),
                AudioBufferRef::U8(data) => Self::conv(&mut pcm_data, data),
                AudioBufferRef::U16(data) => Self::conv(&mut pcm_data, data),
                AudioBufferRef::U24(data) => Self::conv(&mut pcm_data, data),
                AudioBufferRef::U32(data) => Self::conv(&mut pcm_data, data),
                AudioBufferRef::S8(data) => Self::conv(&mut pcm_data, data),
                AudioBufferRef::S16(data) => Self::conv(&mut pcm_data, data),
                AudioBufferRef::S24(data) => Self::conv(&mut pcm_data, data),
                AudioBufferRef::S32(data) => Self::conv(&mut pcm_data, data),
                AudioBufferRef::F64(data) => Self::conv(&mut pcm_data, data),
            }
        }
        if sample_rate != 16000 {
            println!("🔄 Resampling from {}Hz to 16000Hz", sample_rate);
            let resampled = Self::resample_to_16khz(&pcm_data, sample_rate)?;
            Ok((resampled, 16000))
        } else {
            Ok((pcm_data, sample_rate))
        }
    }

    fn resample_to_16khz(samples: &[f32], from_rate: u32) -> Result<Vec<f32>> {
        // Simple linear interpolation resampling
        let ratio = from_rate as f64 / 16000.0;
        let output_len = (samples.len() as f64 / ratio) as usize;
        let mut resampled = Vec::with_capacity(output_len);

        for i in 0..output_len {
            let src_index = (i as f64 * ratio) as usize;
            if src_index < samples.len() {
                resampled.push(samples[src_index]);
            }
        }

        Ok(resampled)
    }

    fn conv<T>(
        samples: &mut Vec<f32>,
        data: std::borrow::Cow<symphonia::core::audio::AudioBuffer<T>>,
    ) where
        T: symphonia::core::sample::Sample,
        f32: symphonia::core::conv::FromSample<T>,
    {
        samples.extend(data.chan(0).iter().map(|v| f32::from_sample(*v)))
    }
}

struct Decoder {
    model: Model,
    rng: StdRng,
    timestamps: bool,
    _verbose: bool,
    pub tokenizer: Tokenizer,
    suppress_tokens: Tensor,
    pub sot_token: u32,
    transcribe_token: u32,
    translate_token: u32,
    pub eot_token: u32,
    no_speech_token: u32,
    pub no_timestamps_token: u32,
    language_token: Option<u32>,
    dtw_timestamps: bool,
    model_type: WhisperModel,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct DecodingResult {
    tokens: Vec<u32>,
    text: String,
    avg_logprob: f64,
    no_speech_prob: f64,
    temperature: f64,
    compression_ratio: f64,
    n_start_tokens: usize,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
struct Segment {
    start: f64,
    duration: f64,
    dr: DecodingResult,
}

impl Decoder {
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        device: &Device,
        language_token: Option<u32>,
        model_type: WhisperModel,
    ) -> Result<Self> {
        let timestamps = true;
        let no_timestamps_token = utils::token_id(&tokenizer, m::NO_TIMESTAMPS_TOKEN)?;
        let suppress_tokens: Vec<f32> = (0..model.config.vocab_size as u32)
            .map(|i| {
                if model.config.suppress_tokens.contains(&i)
                    || timestamps && i == no_timestamps_token
                {
                    f32::NEG_INFINITY
                } else {
                    0f32
                }
            })
            .collect();
        let suppress_tokens = Tensor::new(suppress_tokens.as_slice(), device)?;
        let sot_token = utils::token_id(&tokenizer, m::SOT_TOKEN)?;
        let transcribe_token = utils::token_id(&tokenizer, m::TRANSCRIBE_TOKEN)?;
        let translate_token = utils::token_id(&tokenizer, m::TRANSLATE_TOKEN)?;
        let eot_token = utils::token_id(&tokenizer, m::EOT_TOKEN)?;
        let no_speech_token = m::NO_SPEECH_TOKENS
            .iter()
            .find_map(|token| utils::token_id(&tokenizer, token).ok());
        let no_speech_token = match no_speech_token {
            None => anyhow::bail!("unable to find any non-speech token"),
            Some(n) => n,
        };
        Ok(Self {
            model,
            rng: rand::rngs::StdRng::seed_from_u64(299792458),
            tokenizer,
            timestamps,
            suppress_tokens,
            sot_token,
            transcribe_token,
            translate_token,
            eot_token,
            no_speech_token,
            language_token,
            no_timestamps_token,
            dtw_timestamps: true,
            _verbose: true,
            model_type,
        })
    }

    fn decode_with_fallback(
        &mut self,
        segment: &Tensor,
        n_frames: usize,
    ) -> Result<DecodingResult> {
        for (i, &t) in m::TEMPERATURES.iter().enumerate() {
            let dr: Result<DecodingResult> = self.decode(segment, t, n_frames);
            if i == m::TEMPERATURES.len() - 1 {
                return dr;
            }
            // On errors, we try again with a different temperature.
            match dr {
                Ok(dr) => {
                    let needs_fallback = dr.compression_ratio > m::COMPRESSION_RATIO_THRESHOLD
                        || dr.avg_logprob < m::LOGPROB_THRESHOLD;
                    if !needs_fallback || dr.no_speech_prob > m::NO_SPEECH_THRESHOLD {
                        return Ok(dr);
                    }
                }
                Err(err) => {
                    println!("Error running at {t}: {err}")
                }
            }
        }
        unreachable!()
    }

    fn run(
        &mut self,
        mel: &Tensor,
        total_frames: usize,
        language: WhisperLanguage,
    ) -> Result<Vec<WordSegment>> {
        let (_, _, content_frames) = mel.dims3()?;
        let mut seek = 0;
        let mut segments = vec![];

        while seek < content_frames {
            self.model.reset_kv_cache();
            let time_offset = (seek * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let segment_size = usize::min(content_frames - seek, m::N_FRAMES);
            let n_frames = segment_size.min(
                total_frames
                    .checked_sub(seek)
                    .or_else(|| {
                        seek.checked_sub(m::N_FRAMES)
                            .and_then(|seek| total_frames.checked_sub(seek))
                    })
                    .unwrap_or_default(),
            );
            let mel_segment = mel.narrow(2, seek, segment_size)?;
            let segment_duration = (segment_size * m::HOP_LENGTH) as f64 / m::SAMPLE_RATE as f64;
            let dr = self.decode_with_fallback(&mel_segment, n_frames)?;
            seek += segment_size;

            // Skip if no speech detected
            if dr.no_speech_prob > m::NO_SPEECH_THRESHOLD && dr.avg_logprob < m::LOGPROB_THRESHOLD {
                println!("no speech detected, skipping {seek} {dr:?}");
                continue;
            }

            let segment = Segment {
                start: time_offset,
                duration: segment_duration,
                dr,
            };
            let mut temp_segment =
                WordSegment::new(segment.start, segment.start + segment.duration);

            let mut tokens_to_decode = vec![];
            for &token in segment.dr.tokens.iter() {
                if token == self.sot_token || token == self.eot_token {
                    continue;
                }
                // The no_timestamp_token is the last before the timestamp ones.
                if token > self.no_timestamps_token {
                    if !tokens_to_decode.is_empty() {
                        let text = self
                            .tokenizer
                            .decode(&tokens_to_decode, true)
                            .map_err(E::msg)?;
                        temp_segment = temp_segment.with_text(&text);
                        tokens_to_decode.clear()
                    }
                } else {
                    tokens_to_decode.push(token)
                }
            }
            if !tokens_to_decode.is_empty() {
                let text = self
                    .tokenizer
                    .decode(&tokens_to_decode, true)
                    .map_err(E::msg)?;
                if !text.is_empty() {
                    temp_segment = temp_segment.with_text(&text);
                }
                tokens_to_decode.clear()
            }

            if let Some(timestamps) = self
                .model
                .dtw_timestamps(
                    self.model_type.alignment_heads(language),
                    NonZeroUsize::new(7).unwrap(),
                    n_frames,
                    segment.dr.n_start_tokens,
                )?
                .into_iter()
                .next()
            {
                let words = <Self as timestamps::PostProcessor>::label(
                    self,
                    &timestamps,
                    &segment.dr.tokens,
                )?
                .into_iter()
                .map(|word| word.offset_start(std::time::Duration::from_secs_f64(segment.start)));

                let mut temp_word = Vec::with_capacity(words.len());
                for word in words {
                    temp_word.push(Word::new(word.text, word.start, word.end));
                }
                temp_segment = temp_segment.with_words(&temp_word);
            }
            segments.push(temp_segment);
        }

        Ok(segments)
    }

    fn decode(
        &mut self,
        mel: &Tensor,
        temperature: f64,
        n_frames: usize,
    ) -> Result<DecodingResult> {
        let audio_features = self.model.encoder.forward(mel, true)?;
        let sample_len = self.model.config.max_target_positions / 2;
        let mut sum_logprob = 0f64;
        let mut no_speech_prob = f64::NAN;
        let mut tokens = vec![self.sot_token];
        if let Some(language_token) = self.language_token {
            tokens.push(language_token);
        }
        tokens.push(self.transcribe_token);

        if !self.dtw_timestamps && !self.timestamps {
            tokens.push(self.no_timestamps_token);
        }
        let n_start_tokens = tokens.len();
        for i in 0..sample_len {
            let tokens_t = if tokens.last() == Some(&self.eot_token) {
                let nearest_second = n_frames * m::HOP_LENGTH / m::SAMPLE_RATE;
                let timestamp_token = self
                    .tokenizer
                    .token_to_id(&format!("<|{}.00|>", nearest_second));

                Tensor::new(
                    tokens
                        .iter()
                        .map(|t| {
                            (*t == self.eot_token)
                                .then_some(timestamp_token)
                                .flatten()
                                .unwrap_or(*t)
                        })
                        .collect::<Vec<_>>(),
                    mel.device(),
                )?
            } else if i == 1 && self.dtw_timestamps {
                if let Some(token) = tokens.last_mut() {
                    *token = self.no_timestamps_token;
                }
                Tensor::new(tokens.as_slice(), mel.device())?
            } else {
                Tensor::new(tokens.as_slice(), mel.device())?
            };
            let tokens_t = tokens_t.unsqueeze(0)?;
            let ys = self
                .model
                .decoder
                .forward(&tokens_t, &audio_features, i == 0)?;

            // Extract no speech probability on first iteration
            if i == 0 {
                let logits = self.model.decoder.final_linear(&ys.i(..1)?)?.i(0)?.i(0)?;
                no_speech_prob = softmax(&logits, 0)?
                    .i(self.no_speech_token as usize)?
                    .to_scalar::<f32>()? as f64;
            }

            let (_, seq_len, _) = ys.dims3()?;
            let logits = self
                .model
                .decoder
                .final_linear(&ys.i((..1, seq_len - 1..))?)?
                .i(0)?
                .i(0)?;
            let logits = logits.broadcast_add(&self.suppress_tokens)?;

            let next_token = if temperature > 0f64 {
                let prs = softmax(&(&logits / temperature)?, 0)?;
                let logits_v: Vec<f32> = prs.to_vec1()?;
                let distr = WeightedIndex::new(&logits_v)?;
                distr.sample(&mut self.rng) as u32
            } else {
                let logits_v: Vec<f32> = logits.to_vec1()?;
                logits_v
                    .iter()
                    .enumerate()
                    .max_by(|(_, u), (_, v)| u.total_cmp(v))
                    .map(|(i, _)| i as u32)
                    .unwrap()
            };

            tokens.push(next_token);
            let prob = softmax(&logits, candle_core::D::Minus1)?
                .i(next_token as usize)?
                .to_scalar::<f32>()? as f64;

            if next_token == self.eot_token || tokens.len() > self.model.config.max_target_positions
            {
                break;
            }
            sum_logprob += prob.ln();
        }

        let text = self.tokenizer.decode(&tokens, true).map_err(E::msg)?;
        let avg_logprob = sum_logprob / tokens.len() as f64;

        Ok(DecodingResult {
            tokens,
            text,
            avg_logprob,
            no_speech_prob,
            temperature,
            compression_ratio: f64::NAN,
            n_start_tokens,
        })
    }
}

impl timestamps::PostProcessor for Decoder {
    type Error = candle_core::Error;
    fn decode(
        &mut self,
        tokens: &[u32],
    ) -> candle_core::Result<Vec<crate::whisper::timestamps::Segment>> {
        let full_decode = self
            .tokenizer
            .decode(tokens, true)
            .map_err(candle_core::Error::msg)?;
        let decoded_tokens = tokens
            .iter()
            .filter(|&&n| n < 50_000)
            .copied()
            .map(|n| self.tokenizer.decode(&[n], true))
            .collect::<Result<Vec<_>, _>>()
            .map_err(candle_core::Error::msg)?;

        crate::whisper::timestamps::unicode_segments(full_decode, decoded_tokens)
    }
}
