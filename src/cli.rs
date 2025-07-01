use std::path::PathBuf;

use clap::Parser;

#[derive(Parser)]
#[command(version, about, long_about = None)]
pub struct Cli {
    /// Sets a custom config file
    #[arg(long, value_name = "CONFIG FILE")]
    pub config: Option<PathBuf>,

    /// Input audio file (MP3, WAV, etc.)
    #[arg(short, long, value_name = "AUDIO FILE")]
    pub input: PathBuf,

    /// Output video file
    #[arg(short, long, value_name = "OUTPUT PATH")]
    pub output: Option<PathBuf>,

    /// Custom font file (TTF, OTF)
    #[arg(
        short,
        long,
        value_name = "FONT FILE",
        help = "Custom font file to use for text rendering"
    )]
    pub font: Option<PathBuf>,
}
