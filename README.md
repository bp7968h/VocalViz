# VocalViz

Transform any audio into karaoke videos with automatic word-level highlighting.

> [!WARNING]  
> This is still work in progress, currently, you can generate video by supplying audio file, however the video will have default background.

## ✨ Features

- **AI-Powered Transcription** - Uses Whisper with Candle ML framework for accurate speech-to-text
- **Word-Level Highlighting** - Precise timing for karaoke-style text animation
- **Advanced Timestamping** - Enhanced word-level timing using DTW (Dynamic Time Warping)
- **Custom Backgrounds** - Solid colors, gradients, images, and videos
- **Custom Fonts** - Use any TTF/OTF font file
- **Fast Generation** - Built with Rust and Candle for optimal performance
- **Professional Output** - High-quality MP4 videos

## 📦 Installation (Source)

```bash
git clone https://github.com/yourusername/vocalviz.git
cd vocalviz
cargo build --release
```

## 🚀 Quick Start

```bash
# Basic usage
vocalviz -i audio.mp3 -o karaoke_video.mp4
```

### System Requirements
- **FFmpeg** - For video encoding
- **Rust** - For building from source

## 🧠 Technical Details

### Whisper Integration
VocalViz uses a custom Whisper implementation built on Candle ML framework with enhanced word-level timestamping capabilities. The implementation includes:

- **DTW-based alignment** for precise word timing
- **Cross-attention head analysis** for improved accuracy
- **Custom timestamp processing** with unmerged Candle improvements
- **Optimized inference** for faster processing

### Architecture
```
Audio Input → Whisper/Candle → Word Timestamps → Text Renderer → FFmpeg → MP4 Output
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## ⭐ Acknowledgments

- [Candle ML](https://github.com/huggingface/candle) - Rust ML framework
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition model
- [FFmpeg](https://ffmpeg.org/) - Video processing
- [Rust](https://rust-lang.org/) - Systems programming language