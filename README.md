# Automated CPU-Only Video Generation Pipeline

A fully automated, open-source short-form vertical video generator designed to run entirely on **free GitHub Actions CPU runners** (Ubuntu 22.04/24.04, 4 vCPUs, 16 GB RAM). Zero GPU costs, zero API fees.

## Features

- **TTS**: Step Audio EditX (3B Q8 quantized) with zero-shot voice cloning and paralinguistic tags (`[sigh]`, `[laugh]`, `[chuckle]`)
- **BGM**: ACE-Step 1.5 4B XL-Turbo (Q8, 8-step turbo distillation) for ambient background music
- **Captions**: UsefulSensors Moonshine Base (61M) for millisecond-exact word timestamps → CapCut/TikTok-style `.ass` subtitles
- **Visuals**: 1080x1920 @ 60fps dark-mode tech UI rendered via `timecut` (Chromium frame capture)
- **Assembly**: FFmpeg merge with sidechain auto-ducking (BGM → 15% during speech) + burned subtitles

## Quick Start

### Local Development

```bash
# Clone and enter
git clone https://github.com/yourusername/video-generator-ytshorts.git
cd video-generator-ytshorts

# Install Python dependencies (CPU-only PyTorch)
pip install -r requirements.txt

# Install Node dependencies (timecut, playwright)
npm ci
npx playwright install chromium --with-deps

# Initialize models (downloads ~12GB weights)
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
  "script_text": "Your voiceover script. [sigh] Add paralinguistic tags. [laugh]",
  "bgm_prompt": "Ambient electronic, subtle tech atmosphere, minimal synth pads, slow tempo",
  "video_width": 1080,
  "video_height": 1920,
  "fps": 60,
  "bgm_volume_duck": 0.15
}
```

## Pipeline Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  config.json│────▶│  01_tts.py  │────▶│ voiceover.wav│
└─────────────┘     └─────────────┘     └──────┬──────┘
                                               │
┌─────────────┐     ┌─────────────┐            │
│  02_bgm.py  │────▶│   bgm.wav   │◀───────────┘ (duration match)
└─────────────┘     └─────────────┘
                                               │
┌─────────────┐     ┌─────────────┐            │
│03_captions.py│────▶│ captions.ass│◀───────────┘ (transcription)
└─────────────┘     └─────────────┘
                                               │
┌─────────────┐     ┌─────────────┐            │
│04_visuals.py│────▶│ visuals.mp4 │◀───────────┘ (duration match)
└─────────────┘     └─────────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ 05_assembly.py  │
                  │   (FFmpeg)      │
                  └────────┬────────┘
                           │
                           ▼
                  ┌─────────────────┐
                  │ final_video.mp4 │
                  └─────────────────┘
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
- **Node.js**: 20+ (for timecut/playwright)
- **System**: ffmpeg, libass-dev, fontconfig
- **RAM**: <12 GB peak (fits in GHA 16 GB runner)
- **Disk**: ~15 GB for model weights (cached across runs)

## Model Weights (Auto-Downloaded)

| Component | Model | Size | Quantization |
|-----------|-------|------|--------------|
| TTS | Step Audio EditX | ~6 GB | Q8 (3B params) |
| BGM | ACE-Step XL-Turbo | ~8 GB | Q8 (4B params) |
| Captions | Moonshine Base | ~120 MB | FP16 (61M params) |

Models are cached in `~/.cache/huggingface/` and persist across GitHub Actions runs via cache action.

## Customization

### Visual Template
Edit `templates/index.html` and `templates/styles.css` for custom branding, colors, animations.

### Audio Processing
Modify `scripts/05_assembly.py` filter graph for different ducking ratios, compression settings, or audio effects.

### Caption Styling
Adjust ASS style definitions in `scripts/03_captions.py` for fonts, colors, positions, animations.

## GitHub Actions Workflow

The workflow (`.github/workflows/generate_video.yml`):
- Triggers on `push` to main, `workflow_dispatch`, and daily schedule
- Uses `ubuntu-latest` (4 vCPU, 16 GB RAM)
- Caches pip and Hugging Face directories
- Installs FFmpeg, libass, fonts, Chromium
- Runs full pipeline with 60-minute timeout
- Uploads `final_video.mp4` as downloadable artifact

## Memory Optimization

- PyTorch CPU threads limited to 4 (OMP/MKL)
- Models loaded with `low_cpu_mem_usage=True` and `torch.float16`
- `torch.inference_mode()` for all forward passes
- Explicit `del` and `torch.cuda.empty_cache()` (no-op on CPU) between steps
- Sequential execution (no parallel model loading)

## Troubleshooting

### OOM Errors
- Reduce `fps` in config (30fps halves visual render memory)
- Lower `ffmpeg_crf` to 28 for faster encoding
- Ensure no other heavy processes running

### Timecut/Chromium Issues
- Verify `npx playwright install chromium --with-deps` ran
- Check `--viewport` matches config dimensions
- Increase timeout in `04_visuals.py` if render hangs

### Model Download Failures
- Check disk space (`df -h`)
- Verify network access to huggingface.co
- Re-run `scripts/init.py` manually

## License

MIT License - See LICENSE file for details.

## Acknowledgments

- [Step Audio EditX](https://huggingface.co/stepfun-ai/Step-Audio-EditX) by StepFun
- [ACE-Step](https://huggingface.co/ACE-Step) by ACE-Step Team
- [Moonshine](https://github.com/usefulsensors/moonshine) by UsefulSensors
- [timecut](https://github.com/Timecut/timecut) for Chromium frame capture