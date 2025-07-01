use std::path::{Path, PathBuf};

use anyhow::Result;
use clap::Parser;
use vocalviz::{Cli, KarokeTransConfig, Transcriber, VideoGenerator, WhisperLanguage};

fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = match KarokeTransConfig::load_or_default(cli.config.as_deref()) {
        Ok(conf) => conf,
        Err(e) => {
            eprintln!("Config Error: {}", e);
            std::process::exit(1);
        }
    };
    println!("✅ Configuration loaded successfully!");
    // println!("Loaded config: {:#?}", config);

    let audio_path = cli.input;
    println!("\n🎵 Processing audio: {}", audio_path.display());

    let transcriber = match Transcriber::new() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("Failed to initialize Whisper: {}", e);
            std::process::exit(1);
        }
    };

    println!("🚀 Starting transcription...");
    // Now using await since transcribe is async
    let transcript = match transcriber.transcribe(
        &audio_path,
        &config.whisper.model,
        WhisperLanguage::from(config.whisper.language.as_str()),
    ) {
        // <-- Added .await here
        Ok(transcript) => transcript,
        Err(e) => {
            eprintln!("Transcription failed: {}", e);
            std::process::exit(1);
        }
    };

    // println!("{:#?}", transcript);

    println!("\n📝 Transcription complete!");

    let output_path = match cli.output {
        Some(path) => path,
        None => generate_output_path(&audio_path)?,
    };

    println!("\n🎬 Generating karaoke video...");
    let mut video_generator = match VideoGenerator::new(config) {
        Ok(generator) => generator,
        Err(e) => {
            eprintln!("Failed to initialize video generator: {}", e);
            std::process::exit(1);
        }
    };

    match video_generator.generate(&audio_path, &transcript, &output_path) {
        Ok(()) => {
            println!("\n🎉 Karaoke video generation complete!");
            println!("📁 Output saved to: {}", output_path.display());
        }
        Err(e) => {
            eprintln!("Video generation failed: {}", e);
            std::process::exit(1);
        }
    }

    println!(
        "\n🎉 Processing complete! This transcript could now be used for karaoke video generation."
    );

    Ok(())
}

fn generate_output_path(audio_path: &Path) -> Result<PathBuf> {
    // Get the file stem (filename without extension)
    let file_stem = audio_path
        .file_stem()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow::anyhow!("Invalid audio file path"))?;

    // Get the parent directory (or use current directory if no parent)
    let parent_dir = audio_path
        .parent()
        .unwrap_or_else(|| std::path::Path::new("."));

    // Create new filename with _karaoke suffix and .mp4 extension and combine parent dir
    let output_filename = format!("{}_karaoke.mp4", file_stem);
    let output_path = parent_dir.join(output_filename);

    Ok(output_path)
}
