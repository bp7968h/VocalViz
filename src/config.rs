use std::{fs, path::Path};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

use crate::whisper::timestamps::AlignmentHeads;

#[derive(Debug, Default, Deserialize, Serialize, Clone)]
pub struct KarokeTransConfig {
    pub(crate) video: VideoConfig,
    pub text: TextConfig,
    pub whisper: WhisperConfig,
}

impl KarokeTransConfig {
    pub fn load_or_default(config_path: Option<&Path>) -> Result<Self> {
        match config_path {
            Some(path) => {
                if !path.exists() {
                    anyhow::bail!("Config file '{}' does not exist", path.display());
                }
                Self::from_file(path)
            }
            None => Ok(KarokeTransConfig::default()),
        }
    }

    fn from_file(path: &Path) -> Result<Self> {
        let file_content = fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file '{}'", path.display()))?;
        let config: KarokeTransConfig = toml::from_str(&file_content).with_context(|| {
            format!(
                "Failed to parse config file '{}' - check TOML syntax",
                path.display()
            )
        })?;

        Ok(config)
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub(crate) struct VideoConfig {
    pub background: BackgroundType,
    pub height: u16,
    pub width: u16,
    pub fps: u16,
    pub codec: String,
    pub bitrate: String,
    pub quality: String,
}

impl Default for VideoConfig {
    fn default() -> Self {
        VideoConfig {
            background: BackgroundType::default(),
            height: 1080,
            width: 1920,
            fps: 30,
            codec: "libx264".to_string(),
            bitrate: "2M".to_string(),
            quality: "medium".to_string(),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum BackgroundType {
    Solid {
        color: String,
    },
    Image {
        path: String,
        scaling: Scaling,
        opacity: f32,
    },
    Video {
        path: String,
        start_time: f32,
        scaling: Scaling,
        opacity: f32,
    },
    Gradient {
        start_color: String,
        end_color: String,
        direction: GradientDirection,
    },
}

impl Default for BackgroundType {
    fn default() -> Self {
        BackgroundType::Solid {
            color: "#000000".to_string(),
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum Scaling {
    Stretch,
    Fit,
    Fill,
    Loop,
    Center,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub enum GradientDirection {
    Horizontal,
    Vertical,
    Diagonal,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct TextConfig {
    pub size: u16,
    pub color: String,
    pub highlight_color: String,
    pub background_color: Option<String>,
    pub font_path: Option<String>,
}

impl Default for TextConfig {
    fn default() -> Self {
        TextConfig {
            size: 32,
            color: "#FFFFFF".to_string(),
            highlight_color: "#FFD700".to_string(),
            background_color: None,
            font_path: None,
        }
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct WhisperConfig {
    pub model: WhisperModel,
    pub language: String,
    pub timestamps: bool,
}

impl Default for WhisperConfig {
    fn default() -> Self {
        WhisperConfig {
            model: WhisperModel::Base,
            language: "auto".to_string(),
            timestamps: true,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WhisperModel {
    #[serde(rename = "tiny")]
    Tiny,
    #[serde(rename = "base")]
    Base,
    #[serde(rename = "small")]
    Small,
    #[serde(rename = "medium")]
    Medium,
}

impl WhisperModel {
    pub fn model_and_revision(&self, lang: WhisperLanguage) -> (&'static str, &'static str) {
        match lang {
            WhisperLanguage::Auto => match self {
                Self::Tiny => ("openai/whisper-tiny", "main"),
                Self::Base => ("openai/whisper-base", "refs/pr/22"),
                Self::Small => ("openai/whisper-small", "main"),
                Self::Medium => ("openai/whisper-medium", "main"),
            },
            WhisperLanguage::English => match self {
                Self::Tiny => ("openai/whisper-tiny.en", "refs/pr/15"),
                Self::Base => ("openai/whisper-base.en", "refs/pr/13"),
                Self::Small => ("openai/whisper-small.en", "refs/pr/10"),
                Self::Medium => ("openai/whisper-medium.en", "main"),
            },
        }
    }

    pub fn alignment_heads(&self, lang: WhisperLanguage) -> AlignmentHeads {
        match lang {
            WhisperLanguage::Auto => match self {
                Self::Tiny => AlignmentHeads::tiny(),
                Self::Base => AlignmentHeads::base(),
                Self::Small => AlignmentHeads::small(),
                Self::Medium => AlignmentHeads::medium(),
            },
            WhisperLanguage::English => match self {
                Self::Tiny => AlignmentHeads::tiny_en(),
                Self::Base => AlignmentHeads::base_en(),
                Self::Small => AlignmentHeads::small_en(),
                Self::Medium => AlignmentHeads::medium_en(),
            },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WhisperLanguage {
    #[serde(rename = "auto")]
    Auto,
    #[serde(rename = "english")]
    English,
}

impl From<&str> for WhisperLanguage {
    fn from(value: &str) -> Self {
        match value {
            "auto" => WhisperLanguage::Auto,
            "english" => WhisperLanguage::English,
            _ => unreachable!(),
        }
    }
}
