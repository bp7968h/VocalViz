use crate::config::KarokeTransConfig;
use crate::renderer::TextRenderer;
use crate::transcriber::TranscriptSegment;
use anyhow::Result;
use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

pub struct VideoGenerator {
    config: KarokeTransConfig,
    renderer: TextRenderer,
}

impl VideoGenerator {
    pub fn new(config: KarokeTransConfig) -> Result<Self> {
        let renderer = TextRenderer::new(config.video.clone(), config.text.clone())?;
        Ok(Self { config, renderer })
    }

    pub fn new_with_font_override(
        config: KarokeTransConfig,
        font_path: Option<&std::path::Path>,
    ) -> Result<Self> {
        let renderer = TextRenderer::new_with_font_override(
            config.video.clone(),
            config.text.clone(),
            font_path,
        )?;
        Ok(Self { config, renderer })
    }

    pub fn generate<P: AsRef<Path>>(
        &mut self,
        audio_path: P,
        transcript: &TranscriptSegment,
        output_path: P,
    ) -> Result<()> {
        println!("🎬 Starting karaoke video generation...");
        let output_path_str = output_path
            .as_ref()
            .to_str()
            .ok_or_else(|| anyhow::anyhow!("Invalid output path"))?;

        if !output_path_str.ends_with(".mp4") {
            anyhow::bail!(
                "Output file must have .mp4 extension, got: {}",
                output_path_str
            );
        }

        // Check if FFmpeg is available
        self.check_ffmpeg_available()?;

        // Calculate duration
        let total_duration = transcript.segments.last().map(|seg| seg.end).unwrap_or(0.0);
        let fps = self.config.video.fps as f32;
        let frame_duration = 1.0 / fps;
        let total_frames = (total_duration / frame_duration as f64).ceil() as u64;

        println!(
            "Duration: {:.2}s, FPS: {}, Total frames: {}",
            total_duration, fps, total_frames
        );

        // Use the working streaming approach
        self.generate_streaming(
            &audio_path,
            transcript,
            &output_path,
            fps,
            frame_duration,
            total_frames,
        )?;

        println!("✅ Video generation complete!");
        println!("📁 Output: {}", output_path.as_ref().display());

        Ok(())
    }

    fn check_ffmpeg_available(&self) -> Result<()> {
        let output = Command::new("ffmpeg").args(["-version"]).output();
        match output {
            Ok(_) => Ok(()),
            Err(_) => anyhow::bail!(
                "FFmpeg not found! Please install FFmpeg:\n\
                Fedora: sudo dnf install ffmpeg\n\
                Ubuntu: sudo apt install ffmpeg\n\
                macOS: brew install ffmpeg"
            ),
        }
    }

    fn generate_streaming<P: AsRef<Path>>(
        &mut self,
        audio_path: P,
        transcript: &TranscriptSegment,
        output_path: P,
        fps: f32,
        frame_duration: f32,
        total_frames: u64,
    ) -> Result<()> {
        println!("🎬 Streaming frames directly to FFmpeg...");

        // Get the best available encoder
        let video_encoder = self.get_best_available_encoder()?;

        // Start FFmpeg process to receive raw video frames
        let mut ffmpeg_cmd = Command::new("ffmpeg")
            .args([
                "-y", // Overwrite output
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                &format!("{}x{}", self.config.video.width, self.config.video.height),
                "-r",
                &fps.to_string(),
                "-i",
                "pipe:0", // Read video from stdin
                "-i",
                audio_path.as_ref().to_str().unwrap(), // Audio file
                "-c:v",
                video_encoder.as_str(), // Use available encoder
                "-c:a",
                "aac",
                "-pix_fmt",
                "yuv420p",
                "-shortest", // Match shortest stream duration
                output_path.as_ref().to_str().unwrap(),
            ])
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let stdin = ffmpeg_cmd
            .stdin
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("Failed to open FFmpeg stdin"))?;

        // Generate and stream frames directly to FFmpeg
        println!("🖼️  Generating and streaming frames...");
        for frame_num in 0..total_frames {
            let timestamp = frame_num as f32 * frame_duration;

            // Render frame with enhanced karaoke-style rendering
            let frame_image = self
                .renderer
                .render_frame(timestamp as f64, &transcript.segments)?;

            // Convert to raw RGB bytes and write directly to FFmpeg
            let raw_bytes = frame_image.into_raw();
            if let Err(e) = stdin.write_all(&raw_bytes) {
                // Check if this is just a broken pipe (FFmpeg finished early)
                if e.kind() == std::io::ErrorKind::BrokenPipe {
                    println!("\n📺 FFmpeg finished processing (this is normal)");
                    break; // Exit the loop, FFmpeg is done
                } else {
                    // Some other error occurred
                    let _ = ffmpeg_cmd.kill();
                    let output = ffmpeg_cmd.wait_with_output()?;
                    anyhow::bail!(
                        "FFmpeg write failed: {}\nFFmpeg error: {}",
                        e,
                        String::from_utf8_lossy(&output.stderr)
                    );
                }
            }

            // Progress update
            if frame_num % (fps as u64 / 3) == 0 || frame_num == total_frames - 1 {
                let progress = (frame_num as f32 / total_frames as f32) * 100.0;
                let bar_width = 40;
                let filled = (progress / 100.0 * bar_width as f32) as usize;
                let empty = bar_width - filled;

                print!(
                    "\r🎬 Progress: |{}{}| {:.1}% ({}/{} frames)",
                    "█".repeat(filled),
                    "░".repeat(empty),
                    progress,
                    frame_num + 1,
                    total_frames
                );
                std::io::stdout().flush().unwrap();
            }
        }

        // Close stdin to signal end of video stream
        drop(ffmpeg_cmd.stdin.take());

        // Wait for FFmpeg to finish
        println!("\n🔄 Finalizing video...");
        let output = ffmpeg_cmd.wait_with_output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("FFmpeg failed: {}", stderr);
        }

        Ok(())
    }

    fn get_best_available_encoder(&self) -> Result<String> {
        let output = Command::new("ffmpeg").args(["-encoders"]).output()?;

        let encoders_output = String::from_utf8_lossy(&output.stdout);

        // Try encoders in order of preference
        let preferred_encoders = ["libx264", "mpeg4", "mpeg2video"];

        for encoder in &preferred_encoders {
            if encoders_output.contains(encoder) {
                println!("✅ Using video encoder: {}", encoder);
                return Ok(encoder.to_string());
            }
        }

        anyhow::bail!("No suitable video encoder found")
    }
}
