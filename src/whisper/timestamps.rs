use std::num::NonZeroUsize;

use candle_core::{D, IndexOp, Tensor};
use candle_nn::ops::softmax_last_dim;

/// Raw dtw timestamps
#[derive(Debug)]
pub struct Raw(pub Vec<f32>);

/// Word-level timestamp
#[derive(Debug, Clone)]
pub struct Word {
    pub text: String,
    pub start: f32,
    pub end: f32,
    pub tokens: Vec<u32>,
}

impl Word {
    pub fn offset_start(self, duration: std::time::Duration) -> Self {
        Self {
            start: self.start + duration.as_secs_f32(),
            end: self.end + duration.as_secs_f32(),
            ..self
        }
    }
}

#[derive(Debug, Clone)]
pub struct Segment {
    pub text: String,
    pub token_indices: Vec<usize>,
}

/// Helper trait for processing dtw timestamps.
pub trait PostProcessor {
    type Error: std::error::Error;

    fn decode(&mut self, tokens: &[u32]) -> Result<Vec<Segment>, Self::Error>;
    fn label(&mut self, Raw(timestamps): &Raw, tokens: &[u32]) -> Result<Vec<Word>, Self::Error> {
        const PUNCTUATION: &str = "\"'“¿([{-\"'.。,，!！?？:：”)]}、";
        let segments = self.decode(tokens)?;

        let non_special_tokens = tokens
            .iter()
            .filter(|&&n| n < 50_000)
            .copied()
            .collect::<Vec<_>>();
        let stamped_tokens = timestamps
            .iter()
            .copied()
            .chain(std::iter::repeat(timestamps[timestamps.len() - 1]))
            .zip(
                timestamps
                    .iter()
                    .copied()
                    .skip(1)
                    .chain(std::iter::repeat(timestamps[timestamps.len() - 1])),
            )
            .zip(non_special_tokens.iter().copied())
            .collect::<Vec<_>>();

        let (mut start, mut end) = (0.0, 0.0);
        Ok(segments
            .into_iter()
            .filter(|Segment { text, .. }| !PUNCTUATION.contains(text))
            .map(
                |Segment {
                     text,
                     token_indices,
                 }| {
                    start = token_indices
                        .first()
                        .map(|&i| stamped_tokens[i].0.0)
                        .unwrap_or(end);
                    end = token_indices
                        .last()
                        .map(|&i| stamped_tokens[i].0.1)
                        .unwrap_or(start + 0.2);

                    Word {
                        text: text.trim().to_string(),
                        start,
                        end,
                        tokens: token_indices
                            .into_iter()
                            .map(|i| stamped_tokens[i].1)
                            .collect(),
                    }
                },
            )
            .collect())
    }
}

pub fn unicode_segments(
    full_decode: String,
    decoded_tokens: impl IntoIterator<Item = String>,
) -> candle_core::Result<Vec<Segment>> {
    const RC: char = std::char::REPLACEMENT_CHARACTER;

    let mut buf = vec![0; 3];
    let rc = RC.encode_utf8(&mut buf).as_bytes();
    let bytes = full_decode.bytes().collect::<Vec<_>>();
    let (mut segs, mut seg_tokens, mut current_tokens) = (vec![], vec![], vec![]);
    let mut unicode_offset = 0;
    for (i, decoded) in decoded_tokens.into_iter().enumerate() {
        current_tokens.push(i);

        let rc_idx = unicode_offset + decoded.find(RC).unwrap_or_default();
        if (!decoded.contains(RC)) || &bytes[rc_idx..(rc_idx + rc.len()).min(bytes.len())] == rc {
            unicode_offset += decoded.len();
            segs.push(decoded);
            seg_tokens.push(std::mem::take(&mut current_tokens));
        }
    }

    Ok(segs
        .into_iter()
        .zip(seg_tokens)
        .map(|(text, token_indices)| Segment {
            text,
            token_indices,
        })
        .collect())
}

/// A specific cross-attention head to use for timestamp determination
#[derive(Debug, Clone, Copy)]
pub struct AlignmentHead {
    pub layer: usize,
    pub head: usize,
}

/// The collection of cross-attention heads to use for timestamp determination
#[derive(Debug, Clone)]
pub enum AlignmentHeads {
    /// Uses all heads from the top layers up to the specified maximum
    TopLayerHeads { max_layers: usize },
    /// Uses only the specified alignment heads
    PreDetermined(Vec<AlignmentHead>),
}

impl AlignmentHeads {
    /// Returns the token-level timestamps as a tensor of shape batch x timestamps
    pub(super) fn extract_timestamps(
        &self,
        cross_attentions: &[Tensor],
        filter_width: NonZeroUsize,
        n_frames: usize,
        n_start_tokens: usize,
    ) -> candle_core::Result<Vec<Raw>> {
        // Select relevant cross-attention heads
        let weights = match self {
            Self::TopLayerHeads { max_layers } => {
                let layers = (cross_attentions.len() / 2).min(*max_layers);
                Tensor::cat(
                    &cross_attentions[cross_attentions.len().saturating_sub(layers)..],
                    1,
                )?
            }

            Self::PreDetermined(heads) => Tensor::stack(
                &heads
                    .iter()
                    .copied()
                    .filter_map(|AlignmentHead { layer, head }| {
                        cross_attentions.get(layer)?.i((.., head)).ok()
                    })
                    .collect::<Vec<_>>(),
                0,
            )?
            .permute((1, 0, 2, 3))?,
        }
        .narrow(3, 0, n_frames.min(super::N_FRAMES) / 2)?;

        // Normalize
        let weights = softmax_last_dim(&weights.contiguous()?)?;

        // Smooth
        let weights = &median_filter(
            filter_width,
            weights
                .broadcast_sub(&weights.mean_keepdim(D::Minus2)?)?
                .broadcast_div(&weights.var_keepdim(D::Minus2)?.sqrt()?)?,
        )?;

        // Exclude start tokens
        let cost = weights.mean(1)?.narrow(
            D::Minus2,
            n_start_tokens,
            weights.dim(D::Minus2)? - n_start_tokens - 1,
        )?;

        if cost.dim(D::Minus2)? == 0 {
            // No tokens to be aligned
            return Ok(Default::default());
        }

        // Do the timewarp
        let timestamps = Tensor::stack(
            &((0..weights.dim(0)?).map(|batch_idx| {
                let (text_indices, time_indices) = dynamic_time_warp(
                    cost.neg()?
                        .i(batch_idx)?
                        .to_dtype(candle_core::DType::F32)?
                        .to_vec2::<f32>()?,
                )?;

                let jumps = std::iter::once(1)
                    .chain(
                        text_indices
                            .iter()
                            .skip(1)
                            .copied()
                            .zip(&text_indices)
                            .map(|(a, b)| (a - b) as usize),
                    )
                    .collect::<Vec<_>>();

                let times = jumps
                    .into_iter()
                    .enumerate()
                    .filter(|(_, n)| *n == 1)
                    .map(|(i, _)| {
                        time_indices[i] / (super::SAMPLE_RATE / (super::HOP_LENGTH * 2)) as f32
                    })
                    .collect::<Vec<_>>();

                Tensor::new(times, weights.device())?.to_dtype(weights.dtype())
            }))
            .collect::<candle_core::Result<Vec<_>>>()?,
            0,
        )?;

        Ok(timestamps.to_vec2::<f32>()?.into_iter().map(Raw).collect())
    }
}

/// Computes the lowest cost warping path through the provided cost matrix
fn dynamic_time_warp(matrix: Vec<Vec<f32>>) -> candle_core::Result<(Vec<f32>, Vec<f32>)> {
    let [n, m] = [matrix.len(), matrix[0].len()];
    let (mut cost, mut trace) = (vec![vec![1.; m + 1]; n + 1], vec![vec![1.; m + 1]; n + 1]);

    cost[0][0] = 0.;
    for j in 1..m + 1 {
        for i in 1..n + 1 {
            let (c0, c1, c2) = (cost[i - 1][j - 1], cost[i - 1][j], cost[i][j - 1]);
            let (c, t) = match (c0.lt(&c1), c0.lt(&c2), c1.lt(&c2)) {
                (true, true, _) => (c0, 0.),  // match
                (false, _, true) => (c1, 1.), // insertion
                _ => (c2, 2.),                // deletion
            };

            cost[i][j] = matrix[i - 1][j - 1] + c;
            trace[i][j] = t;
        }
    }

    let (mut i, mut j) = (trace.len() as u32 - 1, trace[0].len() as u32 - 1);
    trace[0] = vec![2.; trace[0].len()];
    for t in &mut trace {
        t[0] = 1.;
    }

    let (mut xs, mut ys) = (vec![], vec![]);
    while i > 0 || j > 0 {
        xs.push(i.saturating_sub(1) as f32);
        ys.push(j.saturating_sub(1) as f32);
        match trace[i as usize][j as usize] as i32 {
            0 => {
                i = i.saturating_sub(1);
                j = j.saturating_sub(1);
            }

            1 => {
                i = i.saturating_sub(1);
            }

            _ => {
                j = j.saturating_sub(1);
            }
        }
    }
    xs.reverse();
    ys.reverse();

    Ok((xs, ys))
}

fn median_filter(filter_width: NonZeroUsize, weights: Tensor) -> candle_core::Result<Tensor> {
    let filter_width = filter_width.get();
    let pad_width = filter_width / 2;
    let (_, _c, _, w) = weights.dims4()?;
    if w <= pad_width {
        return Ok(weights);
    }

    let weights = weights.pad_with_same(3, pad_width, pad_width)?;
    let mut medians = vec![];
    for i in 0..w {
        let weights = weights.narrow(3, i, filter_width)?;
        medians.push(weights.unsqueeze(D::Minus2)?);
    }

    let medians = Tensor::cat(&medians, 3)?
        .contiguous()?
        .to_device(&candle_core::Device::Cpu)? // TODO: investigate why CUDA fails here
        .sort_last_dim(true)?
        .0
        .narrow(4, pad_width, 1)?
        .squeeze(4)?;
    Ok(medians)
}

impl From<[usize; 2]> for AlignmentHead {
    fn from([layer, head]: [usize; 2]) -> Self {
        Self { layer, head }
    }
}

impl<T> FromIterator<T> for AlignmentHeads
where
    T: Into<AlignmentHead>,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        let mut heads = vec![];

        for head in iter {
            heads.push(head.into());
        }

        Self::PreDetermined(heads)
    }
}

impl Default for AlignmentHeads {
    fn default() -> Self {
        Self::TopLayerHeads { max_layers: 3 }
    }
}

/// Creation methods for pre-determined heads
impl AlignmentHeads {
    pub fn tiny() -> Self {
        TINY.iter().copied().collect()
    }

    pub fn tiny_en() -> Self {
        TINY_EN.iter().copied().collect()
    }

    pub fn base() -> Self {
        BASE.iter().copied().collect()
    }

    pub fn base_en() -> Self {
        BASE_EN.iter().copied().collect()
    }

    pub fn small() -> Self {
        SMALL.iter().copied().collect()
    }

    pub fn small_en() -> Self {
        SMALL_EN.iter().copied().collect()
    }

    pub fn medium() -> Self {
        MEDIUM.iter().copied().collect()
    }

    pub fn medium_en() -> Self {
        MEDIUM_EN.iter().copied().collect()
    }

    pub fn large_v1() -> Self {
        LARGE_V1.iter().copied().collect()
    }

    pub fn large_v2() -> Self {
        LARGE_V2_V3.iter().copied().collect()
    }

    pub fn large_v3() -> Self {
        LARGE_V2_V3.iter().copied().collect()
    }

    pub fn large_v3_turbo() -> Self {
        TURBO.iter().copied().collect()
    }

    pub fn distil_small_en() -> Self {
        DISTIL_SMALL_EN.iter().copied().collect()
    }

    pub fn distil_large_v3() -> Self {
        DISTIL_LARGE_V3.iter().copied().collect()
    }
}

const TINY: &[[usize; 2]] = &[[2, 2], [3, 0], [3, 2], [3, 3], [3, 4], [3, 5]];
const TINY_EN: &[[usize; 2]] = &[
    [1, 0],
    [2, 0],
    [2, 5],
    [3, 0],
    [3, 1],
    [3, 2],
    [3, 3],
    [3, 4],
];
const BASE: &[[usize; 2]] = &[
    [3, 1],
    [4, 2],
    [4, 3],
    [4, 7],
    [5, 1],
    [5, 2],
    [5, 4],
    [5, 6],
];
const BASE_EN: &[[usize; 2]] = &[[3, 3], [4, 7], [5, 1], [5, 5], [5, 7]];
const SMALL: &[[usize; 2]] = &[
    [5, 3],
    [5, 9],
    [8, 0],
    [8, 4],
    [8, 7],
    [8, 8],
    [9, 0],
    [9, 7],
    [9, 9],
    [10, 5],
];
const SMALL_EN: &[[usize; 2]] = &[
    [6, 6],
    [7, 0],
    [7, 3],
    [7, 8],
    [8, 2],
    [8, 5],
    [8, 7],
    [9, 0],
    [9, 4],
    [9, 8],
    [9, 10],
    [10, 0],
    [10, 1],
    [10, 2],
    [10, 3],
    [10, 6],
    [10, 11],
    [11, 2],
    [11, 4],
];
const MEDIUM: &[[usize; 2]] = &[[13, 15], [15, 4], [15, 15], [16, 1], [20, 0], [23, 4]];
const MEDIUM_EN: &[[usize; 2]] = &[
    [11, 4],
    [14, 1],
    [14, 12],
    [14, 14],
    [15, 4],
    [16, 0],
    [16, 4],
    [16, 9],
    [17, 12],
    [17, 14],
    [18, 7],
    [18, 10],
    [18, 15],
    [20, 0],
    [20, 3],
    [20, 9],
    [20, 14],
    [21, 12],
];
const LARGE_V1: &[[usize; 2]] = &[
    [9, 19],
    [11, 2],
    [11, 4],
    [11, 17],
    [22, 7],
    [22, 11],
    [22, 17],
    [23, 2],
    [23, 15],
];
const LARGE_V2_V3: &[[usize; 2]] = &[
    [10, 12],
    [13, 17],
    [16, 11],
    [16, 12],
    [16, 13],
    [17, 15],
    [17, 16],
    [18, 4],
    [18, 11],
    [18, 19],
    [19, 11],
    [21, 2],
    [21, 3],
    [22, 3],
    [22, 9],
    [22, 12],
    [23, 5],
    [23, 7],
    [23, 13],
    [25, 5],
    [26, 1],
    [26, 12],
    [27, 15],
];
const TURBO: &[[usize; 2]] = &[[2, 4], [2, 11], [3, 3], [3, 6], [3, 11], [3, 14]];
const DISTIL_SMALL_EN: &[[usize; 2]] = &[
    [6, 6],
    [7, 0],
    [7, 3],
    [7, 8],
    [8, 2],
    [8, 5],
    [8, 7],
    [9, 0],
    [9, 4],
    [9, 8],
    [9, 10],
    [10, 0],
    [10, 1],
    [10, 2],
    [10, 3],
    [10, 6],
    [10, 11],
    [11, 2],
    [11, 4],
];
const DISTIL_LARGE_V3: &[[usize; 2]] = &[
    [1, 0],
    [1, 1],
    [1, 2],
    [1, 3],
    [1, 4],
    [1, 5],
    [1, 6],
    [1, 7],
    [1, 8],
    [1, 9],
    [1, 10],
    [1, 11],
    [1, 12],
    [1, 13],
    [1, 14],
    [1, 15],
    [1, 16],
    [1, 17],
    [1, 18],
    [1, 19],
];
