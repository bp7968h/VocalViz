use std::collections::HashMap;

use crate::config::{BackgroundType, Scaling, TextConfig, VideoConfig};
use crate::transcriber::{Word, WordSegment};
use anyhow::Result;
use fontdue::{Font, FontSettings};
use image::{DynamicImage, ImageBuffer, Rgb, RgbImage};

pub struct TextRenderer {
    width: u32,
    height: u32,
    video_config: VideoConfig,
    text_config: TextConfig,
    font: Font,
    image_cache: HashMap<String, DynamicImage>,
    video_frame_cache: HashMap<String, Vec<RgbImage>>,
}

#[derive(Debug, Clone)]
struct WordGroup {
    words: Vec<(Word, bool)>, // word and whether it's highlighted
    start_time: f64,
    end_time: f64,
}

impl TextRenderer {
    pub fn new(video_config: VideoConfig, text_config: TextConfig) -> Result<Self> {
        let font = Self::load_font(&text_config.font_path)?;

        Ok(Self {
            width: video_config.width as u32,
            height: video_config.height as u32,
            video_config,
            text_config,
            font,
            image_cache: HashMap::new(),
            video_frame_cache: HashMap::new(),
        })
    }

    pub fn new_with_font_override(
        video_config: VideoConfig,
        text_config: TextConfig,
        font_override: Option<&std::path::Path>,
    ) -> Result<Self> {
        // CLI font override takes precedence over config font
        let font_path = if let Some(override_path) = font_override {
            Some(override_path.to_string_lossy().to_string())
        } else {
            text_config.font_path.clone()
        };

        let font = Self::load_font(&font_path)?;

        Ok(Self {
            width: video_config.width as u32,
            height: video_config.height as u32,
            video_config,
            text_config,
            font,
            image_cache: HashMap::new(),
            video_frame_cache: HashMap::new(),
        })
    }

    pub fn render_frame(&mut self, timestamp: f64, segments: &[WordSegment]) -> Result<RgbImage> {
        let mut img = self.create_background(timestamp)?; // Now takes timestamp for video backgrounds

        let word_groups = self.create_word_groups(timestamp, segments);

        self.render_word_groups(&mut img, &word_groups, timestamp)?;
        Ok(img)
    }

    fn create_background(&mut self, timestamp: f64) -> Result<RgbImage> {
        let mut img = ImageBuffer::new(self.width, self.height);

        match &self.video_config.background.clone() {
            BackgroundType::Solid { color } => {
                let rgb = self.parse_hex_color(color)?;
                for pixel in img.pixels_mut() {
                    *pixel = Rgb(rgb);
                }
            }
            BackgroundType::Gradient {
                start_color,
                end_color,
                direction,
            } => {
                self.render_gradient(&mut img, start_color, end_color, direction)?;
            }
            BackgroundType::Image {
                path,
                scaling,
                opacity,
            } => {
                self.render_image_background(&mut img, path, scaling, *opacity)?;
            }
            BackgroundType::Video {
                path,
                start_time,
                scaling,
                opacity,
            } => {
                self.render_video_background(
                    &mut img,
                    path,
                    *start_time,
                    timestamp,
                    scaling,
                    *opacity,
                )?;
            }
        }

        Ok(img)
    }

    fn render_image_background(
        &mut self,
        img: &mut RgbImage,
        image_path: &str,
        scaling: &Scaling,
        opacity: f32,
    ) -> Result<()> {
        // Load image (with caching)
        let background_image = if let Some(cached) = self.image_cache.get(image_path) {
            cached.clone()
        } else {
            let loaded = image::open(image_path).map_err(|e| {
                anyhow::anyhow!("Failed to load background image '{}': {}", image_path, e)
            })?;
            self.image_cache
                .insert(image_path.to_string(), loaded.clone());
            loaded
        };

        // Scale the image according to the scaling mode
        let scaled_image = self.scale_image(background_image, scaling)?;
        let scaled_rgb = scaled_image.to_rgb8();

        // Apply the image to the background
        self.blend_image_onto_background(img, &scaled_rgb, opacity)?;

        Ok(())
    }

    fn render_video_background(
        &mut self,
        img: &mut RgbImage,
        video_path: &str,
        start_time: f32,
        current_time: f64,
        scaling: &Scaling,
        opacity: f32,
    ) -> Result<()> {
        // Calculate which frame we need from the video
        let video_timestamp = start_time + current_time as f32;

        // For this example, I'll show how you could extract frames using FFmpeg
        // In a real implementation, you'd want to pre-extract frames or use ffmpeg-next

        // Simple approach: Pre-extract key frames and cache them
        if !self.video_frame_cache.contains_key(video_path) {
            self.extract_video_frames(video_path)?;
        }

        if let Some(frames) = self.video_frame_cache.get(video_path) {
            // Calculate which frame to use (assuming 30 FPS)
            let frame_index = (video_timestamp * 30.0) as usize % frames.len();
            let frame = &frames[frame_index];

            // Scale the frame
            let dynamic_frame = DynamicImage::ImageRgb8(frame.clone());
            let scaled_frame = self.scale_image(dynamic_frame, scaling)?;
            let scaled_rgb = scaled_frame.to_rgb8();

            // Blend onto background
            self.blend_image_onto_background(img, &scaled_rgb, opacity)?;
        } else {
            // Fallback to solid color if video loading fails
            let rgb = [32, 32, 32];
            for pixel in img.pixels_mut() {
                *pixel = Rgb(rgb);
            }
        }

        Ok(())
    }

    fn scale_image(&self, image: DynamicImage, scaling: &Scaling) -> Result<DynamicImage> {
        let (img_width, img_height) = (image.width(), image.height());
        let (target_width, target_height) = (self.width, self.height);

        let scaled = match scaling {
            Scaling::Stretch => {
                // Stretch to fill exactly (may distort aspect ratio)
                image.resize_exact(
                    target_width,
                    target_height,
                    image::imageops::FilterType::Lanczos3,
                )
            }
            Scaling::Fit => {
                // Fit inside while maintaining aspect ratio
                image.resize(
                    target_width,
                    target_height,
                    image::imageops::FilterType::Lanczos3,
                )
            }
            Scaling::Fill => {
                // Fill entire area while maintaining aspect ratio (may crop)
                let scale_x = target_width as f32 / img_width as f32;
                let scale_y = target_height as f32 / img_height as f32;
                let scale = scale_x.max(scale_y);

                let new_width = (img_width as f32 * scale) as u32;
                let new_height = (img_height as f32 * scale) as u32;

                let resized = image.resize_exact(
                    new_width,
                    new_height,
                    image::imageops::FilterType::Lanczos3,
                );

                // Crop to center
                let crop_x = (new_width.saturating_sub(target_width)) / 2;
                let crop_y = (new_height.saturating_sub(target_height)) / 2;

                resized.crop_imm(crop_x, crop_y, target_width, target_height)
            }
            Scaling::Center => {
                // Center the image without scaling
                image
            }
            Scaling::Loop => {
                // Tile the image (useful for patterns)
                let mut tiled = DynamicImage::new_rgb8(target_width, target_height);
                for y in (0..target_height).step_by(img_height as usize) {
                    for x in (0..target_width).step_by(img_width as usize) {
                        image::imageops::overlay(&mut tiled, &image, x as i64, y as i64);
                    }
                }
                tiled
            }
        };

        Ok(scaled)
    }

    fn blend_image_onto_background(
        &self,
        background: &mut RgbImage,
        foreground: &RgbImage,
        opacity: f32,
    ) -> Result<()> {
        let opacity = opacity.clamp(0.0, 1.0);

        for (x, y, bg_pixel) in background.enumerate_pixels_mut() {
            if let Some(fg_pixel) = foreground.get_pixel_checked(x, y) {
                // Alpha blend the foreground onto the background
                for i in 0..3 {
                    bg_pixel.0[i] = ((1.0 - opacity) * bg_pixel.0[i] as f32
                        + opacity * fg_pixel.0[i] as f32) as u8;
                }
            }
        }

        Ok(())
    }

    fn extract_video_frames(&mut self, video_path: &str) -> Result<()> {
        // This is a simplified example - in practice you'd want more sophisticated frame extraction
        use std::process::Command;

        // Extract frames using FFmpeg command (you could also use ffmpeg-next for this)
        let temp_dir = std::env::temp_dir().join("karaoke_frames");
        std::fs::create_dir_all(&temp_dir)?;

        // Extract one frame per second for simplicity
        let output = Command::new("ffmpeg")
            .args([
                "-i",
                video_path,
                "-vf",
                "fps=1", // Extract 1 frame per second
                "-y",    // Overwrite
                &format!("{}/frame_%03d.png", temp_dir.display()),
            ])
            .output()?;

        if !output.status.success() {
            anyhow::bail!(
                "Failed to extract video frames: {}",
                String::from_utf8_lossy(&output.stderr)
            );
        }

        // Load the extracted frames
        let mut frames = Vec::new();
        for i in 1..=3600 {
            // Up to 1 hour of video
            let frame_path = temp_dir.join(format!("frame_{:03}.png", i));
            if frame_path.exists() {
                if let Ok(frame) = image::open(&frame_path) {
                    frames.push(frame.to_rgb8());
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        self.video_frame_cache
            .insert(video_path.to_string(), frames);

        // Clean up temporary files
        std::fs::remove_dir_all(&temp_dir).ok();

        Ok(())
    }

    fn create_word_groups(&self, timestamp: f64, segments: &[WordSegment]) -> Vec<WordGroup> {
        let mut word_groups = Vec::new();
        let look_ahead_time = 8.0; // Show words 8 seconds ahead
        let words_per_group = 4; // Group 4 words together

        for segment in segments {
            if timestamp >= segment.start - look_ahead_time && timestamp <= segment.end + 2.0 {
                // Group words into chunks of 4
                for chunk in segment.words.chunks(words_per_group) {
                    let group_start = chunk
                        .first()
                        .map(|w| w.start as f64)
                        .unwrap_or(segment.start);
                    let group_end = chunk.last().map(|w| w.end as f64).unwrap_or(segment.end);

                    // Only show groups that are current or upcoming
                    if group_end >= timestamp - 1.0 {
                        let words_with_highlight: Vec<(Word, bool)> = chunk
                            .iter()
                            .map(|word| {
                                let is_highlighted =
                                    timestamp >= word.start as f64 && timestamp <= word.end as f64;
                                (word.clone(), is_highlighted)
                            })
                            .collect();

                        word_groups.push(WordGroup {
                            words: words_with_highlight,
                            start_time: group_start,
                            end_time: group_end,
                        });
                    }
                }
            }
        }

        // Sort by start time and limit to 3 groups maximum for screen space
        word_groups.sort_by(|a, b| a.start_time.partial_cmp(&b.start_time).unwrap());
        word_groups.truncate(3);

        word_groups
    }

    fn render_word_groups(
        &self,
        img: &mut RgbImage,
        word_groups: &[WordGroup],
        timestamp: f64,
    ) -> Result<()> {
        if word_groups.is_empty() {
            return Ok(());
        }

        let font_size = self.text_config.size as f32;
        let line_height = font_size * 2.0; // More spacing between groups
        let max_line_width = self.width as f32 - 100.0;

        // Calculate starting Y position (bottom third of screen)
        let total_height = word_groups.len() as f32 * line_height;
        let start_y = self.height as f32 - total_height - 100.0;

        // Render each word group
        for (group_idx, group) in word_groups.iter().enumerate() {
            let y = start_y + group_idx as f32 * line_height;

            // Determine group opacity based on timing
            let opacity = self.calculate_group_opacity(group, timestamp);

            // Calculate line width for centering
            let line_width = self.calculate_group_width(&group.words, font_size)?;
            let start_x = (self.width as f32 - line_width) / 2.0;

            // Render words in this group
            let mut current_x = start_x;
            for (word, is_highlighted) in &group.words {
                let color = if *is_highlighted {
                    self.apply_opacity(
                        &self.parse_hex_color(&self.text_config.highlight_color)?,
                        opacity,
                    )
                } else {
                    self.apply_opacity(
                        &self.parse_hex_color(&self.text_config.color)?,
                        opacity * 0.7, // Unhighlighted words are dimmer
                    )
                };

                let word_width =
                    self.render_word(img, &word.text, current_x, y, font_size, color)?;
                current_x += word_width + 20.0; // Space between words
            }
        }

        Ok(())
    }

    fn calculate_group_opacity(&self, group: &WordGroup, timestamp: f64) -> f32 {
        let fade_duration = 0.5; // Fade in/out duration

        if timestamp < group.start_time - fade_duration {
            // Future group - fade in as it approaches
            let fade_progress =
                (timestamp - (group.start_time - fade_duration * 2.0)) / fade_duration;
            fade_progress.max(0.0).min(1.0) as f32 * 0.3 // Preview opacity
        } else if timestamp >= group.start_time - fade_duration
            && timestamp <= group.end_time + fade_duration
        {
            // Current group - full opacity
            1.0
        } else {
            // Past group - fade out
            let fade_progress = (group.end_time + fade_duration * 2.0 - timestamp) / fade_duration;
            fade_progress.max(0.0).min(1.0) as f32 * 0.5 // Past opacity
        }
    }

    fn apply_opacity(&self, color: &[u8; 3], opacity: f32) -> [u8; 3] {
        [
            (color[0] as f32 * opacity) as u8,
            (color[1] as f32 * opacity) as u8,
            (color[2] as f32 * opacity) as u8,
        ]
    }

    fn calculate_group_width(&self, words: &[(Word, bool)], font_size: f32) -> Result<f32> {
        let mut total_width = 0.0;

        for (word, _) in words {
            total_width += self.estimate_word_width(&word.text, font_size)?;
            total_width += 20.0; // Space between words
        }

        // Remove the last space
        if !words.is_empty() {
            total_width -= 20.0;
        }

        Ok(total_width)
    }

    fn render_word(
        &self,
        img: &mut RgbImage,
        text: &str,
        x: f32,
        y: f32,
        size: f32,
        color: [u8; 3],
    ) -> Result<f32> {
        let mut current_x = x;

        // Add text shadow/outline for better readability
        let shadow_color = [0, 0, 0]; // Black shadow
        let shadow_offset = 2.0;

        // Render shadow first
        self.render_text_at_position(
            img,
            text,
            current_x + shadow_offset,
            y + shadow_offset,
            size,
            shadow_color,
        )?;

        // Render main text
        let width = self.render_text_at_position(img, text, current_x, y, size, color)?;

        Ok(width)
    }

    fn render_text_at_position(
        &self,
        img: &mut RgbImage,
        text: &str,
        x: f32,
        y: f32,
        size: f32,
        color: [u8; 3],
    ) -> Result<f32> {
        let mut current_x = x;

        for ch in text.chars() {
            let (metrics, bitmap) = self.font.rasterize(ch, size);

            let start_x = current_x as i32;
            let start_y = y as i32;

            for (i, &alpha) in bitmap.iter().enumerate() {
                if alpha > 0 {
                    let px = i % metrics.width;
                    let py = i / metrics.width;

                    let img_x = start_x + px as i32;
                    let img_y = start_y + py as i32;

                    if img_x >= 0
                        && img_x < self.width as i32
                        && img_y >= 0
                        && img_y < self.height as i32
                    {
                        let pixel = img.get_pixel_mut(img_x as u32, img_y as u32);

                        let alpha_f = alpha as f32 / 255.0;
                        for (i, &color) in color.iter().enumerate() {
                            pixel.0[i] = ((1.0 - alpha_f) * pixel.0[i] as f32
                                + alpha_f * color as f32)
                                as u8;
                        }
                    }
                }
            }

            current_x += metrics.advance_width;
        }

        Ok(current_x - x)
    }

    fn parse_hex_color(&self, hex: &str) -> Result<[u8; 3]> {
        let hex = hex.trim_start_matches('#');
        if hex.len() != 6 {
            anyhow::bail!("Invalid hex color: {}", hex);
        }

        Ok([
            u8::from_str_radix(&hex[0..2], 16)?,
            u8::from_str_radix(&hex[2..4], 16)?,
            u8::from_str_radix(&hex[4..6], 16)?,
        ])
    }

    fn render_gradient(
        &self,
        img: &mut RgbImage,
        start_color: &str,
        end_color: &str,
        _direction: &crate::config::GradientDirection,
    ) -> Result<()> {
        let start_rgb = self.parse_hex_color(start_color)?;
        let end_rgb = self.parse_hex_color(end_color)?;

        for (_x, y, pixel) in img.enumerate_pixels_mut() {
            let ratio = y as f32 / self.height as f32;
            let r = (start_rgb[0] as f32 * (1.0 - ratio) + end_rgb[0] as f32 * ratio) as u8;
            let g = (start_rgb[1] as f32 * (1.0 - ratio) + end_rgb[1] as f32 * ratio) as u8;
            let b = (start_rgb[2] as f32 * (1.0 - ratio) + end_rgb[2] as f32 * ratio) as u8;

            *pixel = Rgb([r, g, b]);
        }

        Ok(())
    }

    fn estimate_word_width(&self, word: &str, font_size: f32) -> Result<f32> {
        let avg_char_width = font_size * 0.6;
        Ok(word.len() as f32 * avg_char_width)
    }

    fn load_font(font_path: &Option<String>) -> Result<Font> {
        match font_path {
            Some(path) => {
                println!("📝 Loading custom font: {}", path);

                // Validate font file exists
                if !std::path::Path::new(path).exists() {
                    anyhow::bail!("Font file not found: {}", path);
                }

                // Load custom font
                let font_data = std::fs::read(path)
                    .map_err(|e| anyhow::anyhow!("Failed to read font file '{}': {}", path, e))?;

                Font::from_bytes(font_data.as_slice(), FontSettings::default())
                    .map_err(|e| anyhow::anyhow!("Failed to load font '{}': {}", path, e))
            }
            None => {
                println!("📝 Using default embedded font");

                // Load default embedded font
                let font_data = include_bytes!("../assets/fonts/LoveDays.ttf");
                Font::from_bytes(font_data as &[u8], FontSettings::default())
                    .map_err(|e| anyhow::anyhow!("Failed to load default font: {}", e))
            }
        }
    }
}
