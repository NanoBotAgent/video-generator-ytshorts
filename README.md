# Automated CPU-Only Video Generation Pipeline

A fully automated, open-source short-form vertical video generator designed to run entirely on **free GitHub Actions CPU runners** (Ubuntu 22.04/24.04, 4 vCPUs, 16 GB RAM). Zero GPU costs, zero API fees.

## Features

- **TTS**: espeak-ng (CLI-invoked directly) - deterministic, CPU-native voiceover generation
- **BGM**: Facebook MusicGen-Small (300M) for ambient background music, matched to voiceover duration
- **Captions**: UsefulSensors Moonshine Base (61M) for word-level transcription → CapCut/TikTok-style `.ass` subtitles
- **Visuals**: 1080x1920 @ 60fps animated gradient background rendered with FFmpeg lavfi filter graph (no browser/Chromium needed)
- **Assembly**: FFmpeg merge with sidechain auto-ducking (BGM → 15% during speech) + burned subtitles

> **Why not Step-Audio-EditX / ACE-Step 1.5 XL?** Those were the original targets for TTS and BGM, but both require a GPU with 12GB+ VRAM (Step-Audio-EditX also needs a separate CosyVoice vocoder pipeline that is not bundled with the model weights). Neither runs meaningfully on a free CPU-only GitHub Actions runner, so this pipeline uses the CPU-viable substitutes above. If you want the original models, you would need to point this pipeline at a GPU-backed runner (e.g. a self-hosted runner, or offload just those two steps to a service like an HF Space with ZeroGPU).

## Quick Start

### Local Development

```bash
# Clone and enter
git clone https://github.com/yourusername/video-generator-ytshorts.git
cd video-generator-ytshorts

# Install Python dependencies (CPU-only PyTorch)
pip install -r requirements.txt

# Initialize models (downloads ~2GB weights: MusicGen-Small + Moonshine Base)
python scripts/init.py

# Configure your script
# Edit config.json with your script_text and bgm_prompt

# Run full pipeline
python main.py
```

### GitHub Actions (Zero-Cost Cloud)

1. Fork this repository
2. Go to **Actions** → **Generate Video** → **Run workflow**
3. Optionally override `script_text` and `bgm_prompt`
4. Download `final_video.mp4` from workflow artifacts (available for 7 days)

## Configuration

Edit `config.json`:

```json
{
  "script_text": "Your voiceover script. [sigh] Paralinguistic tags are stripped before TTS. [laugh]",
  "bgm_prompt": "Ambient electronic, subtle tech atmosphere, minimal synth pads, slow tempo",
  "video_width": 1080,
  "video_height": 1920,
  "fps": 60,
  "bgm_volume_duck": 0.15
}
```

Note: `[sigh]`, `[laugh]`, etc. in `script_text` are stripped (not voiced) since espeak-ng has no equivalent for them - they are just kept as writing hints for now.

## Pipeline Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  config.json│────▶│  01_tts.py  │────▶│ voiceover.wav│
└─────────────┘     └─────────────┘     └──────┐──────┘
                                               │
┌─────────────┐     ┌─────────────┐            │
│  02_bgm.py  │────▶│   bgm.wav   │◀─────────┘ (duration match)
└─────────────┘     └─────────────┘
                                               │
┌─────────────┐     ┌─────────────┐            │
│03_captions.py│────▶│ captions.ass│◀─────────┘ (transcription)
└─────────────┘     └─────────────┘
                                               │
┌─────────────┐     ┌─────────────┐            │
│04_visuals.py│────▶│ visuals.mp4 │◀─────────┘ (duration match)
└─────────────┘     └─────────────┘
                           │
                           ▼
                  ┌───────────────┐
                  │ 05_assembly.py  │
                  │   (FFmpeg)      │
                  └────────┄────────┘
                           │
                           ▼
                  ┌───────────────┐
                  │ final_video.mp4 │
                  └───────────────┘
```

## Output Structure

```
output/
├── voiceover.wav    # 44.1kHz mono, normalized
├── bgm.wav          # 44.1kHz stereo, duration = voiceover + 2s
├── captions.ass     # ASS format with word-level highlighting
├── visuals.mp4      # 1080x1920 @ 60fps, H.264
└── final_video.mp4  # Merged output with burned captions
```

## Requirements

- **Python**: 3.10+ (tested on 3.10, 3.11)
- **System**: ffmpeg, libass-dev, fontconfig, espeak-ng
- **RAM**: <8 GB peak (fits comfortably in GHA 16 GB runner)
- **Disk**: ~3 GB for model weights (cached across runs)

## Model Weights (Auto-Downloaded)

| Component | Model | Size | Notes |
|-----------|-------|------|-------|
| TTS | espeak-ng | N/A | No model download - system TTS engine |
| BGM | MusicGen-Small | ~2 GB | FP32 (300M params) |
| Captions | Moonshine Base | ~120 MB | FP32 (61M params) |

Models are cached in `~/.cache/huggingface/` and persist across GitHub Actions runs via cache action.

## Customization

### Visual Style
Visuals are procedural animated gradients generated directly by an FFmpeg `lavfi` filter graph in `scripts/04_visuals.py` (colors, blend modes, speed). Edit the `filter_complex` string there to change the look.

> The `templates/index.html` and `templates/styles.css` files are leftover from an earlier Chromium/timecut-based renderer and are no longer used by the pipeline. Safe to ignore or delete.

### Audio Processing
Modify `scripts/05_assembly.py` filter graph for different ducking ratios, compression settings, or audio effects.

### Caption Styling
Adjust ASS style definitions in `scripts/03_captions.py` for fonts, colors, positions, animations.

## GitHub Actions Workflow

The workflow (`.github/workflows/generate_video.yml`):
- Triggers on `push` to main, `workflow_dispatch`, and daily schedule
- Uses `ubuntu-latest` (4 vCPU, 16 GB RAM)
- Caches pip and Hugging Face directories
- Installs FFmpeg, libass, fonts, espeak-ng
- Runs full pipeline with 60-minute timeout
- Uploads `final_video.mp4` as downloadable artifact

## Memory Optimization

- PyTorch CPU threads limited to 4 (OMP/MKL)
- Models loaded with `low_cpu_mem_usage=True` and `torch.float32`
- `torch.inference_mode()` for all forward passes
- Sequential execution (no parallel model loading)

## Troubleshooting

### OOM Errors
- Reduce `fps` in config (30fps halves visual render memory)
- Lower `ffmpeg_crf` to 28 for faster encoding
- Ensure no other heavy processes running

### Model Download Failures
- Check disk space (`df -h`)
- Verify network access to huggingface.co
- Re-run `scripts/init.py` manually

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [MusicGen](https://huggingface.co/facebook/musicgen-small) by Meta/Facebook
- [Moonshine](https://github.com/usefulsensors/moonshine) by UsefulSensors
- [espeak-ng](https://github.com/espeak-ng/espeak-ng) for offline TTS
