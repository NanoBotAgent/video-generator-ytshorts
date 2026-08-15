#!/usr/bin/env python3
"""
Visuals Module - Renders procedural gradient video using ffmpeg lavfi.
Generates 1080x1920 @ 60fps visuals.mp4 matching voiceover duration.
Uses ffmpeg color/geq filter - no Chromium/timecut needed (works on CPU CI).
"""

import os
import sys
import json
import logging
import time
import subprocess
import wave
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class VisualsRenderer:
    """Renders procedural gradient video using ffmpeg."""

    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.width = config.get("video_width", 1080)
        self.height = config.get("video_height", 1920)
        self.fps = config.get("fps", 60)

    def get_audio_duration(self, audio_path: Path) -> float:
        """Get duration of audio file in seconds."""
        try:
            with wave.open(str(audio_path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                return frames / float(rate)
        except Exception as e:
            logger.warning(f"Could not read audio duration: {e}, using default 30s")
            return 30.0

    def render(self, duration: float) -> Optional[Path]:
        """Render procedural gradient video using ffmpeg lavfi."""
        try:
            logger.info(f"Rendering visuals: {self.width}x{self.height} @ {duration:.1f}s @ {self.fps}fps")
            start_time = time.time()

            output_path = self.output_dir / "visuals.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Animated gradient via geq: each channel is a smooth sine function of
            # position and time, with per-channel phase offsets (120 degrees apart,
            # like a hue rotation) so the colors drift between navy/purple/pink.
            # This replaced an earlier version built from two independently-seeded
            # `gradients` source filters combined with stacked `overlay` blends -
            # that combo visibly flickered, since each `gradients` layer picks its
            # own random walk with no shared seed, and overlay blend mode amplifies
            # small per-frame differences between them. geq has no randomness at
            # all: the color at a given pixel is a continuous function of time, so
            # frame-to-frame change is always tiny and smooth - no flicker possible.
            filter_complex = (
                f"color=c=black:size={self.width}x{self.height}:rate={self.fps}:duration={duration:.3f},"
                "geq="
                "r='clip(50+40*sin(2*PI*(X/W*0.6+Y/H*0.3+T*0.04)),0,255)':"
                "g='clip(45+35*sin(2*PI*(X/W*0.5+Y/H*0.4+T*0.035)+2.094),0,255)':"
                "b='clip(90+60*sin(2*PI*(X/W*0.4+Y/H*0.5+T*0.03)+4.188),0,255)'"
                "[vout]"
            )

            cmd = [
                "ffmpeg", "-y",
                "-filter_complex", filter_complex,
                "-map", "[vout]",
                "-c:v", "libx264",
                "-preset", self.config.get("ffmpeg_preset", "fast"),
                "-crf", str(self.config.get("ffmpeg_crf", 22)),
                "-pix_fmt", "yuv420p",
                "-r", str(self.fps),
                "-shortest",
                str(output_path),
            ]

            logger.info(f"Executing: {' '.join(cmd[:8])}... [filter_complex omitted]")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(Path.cwd()),
            )

            elapsed = time.time() - start_time

            if result.returncode != 0:
                logger.error(f"ffmpeg failed (exit {result.returncode}): {result.stderr[-2000:]}")
                # Fallback: simple solid color with text
                return self._render_fallback(duration)

            if not output_path.exists() or output_path.stat().st_size == 0:
                logger.error("Output file not created or empty")
                return self._render_fallback(duration)

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"Visuals rendered in {elapsed:.1f}s ({file_size_mb:.1f} MB)")
            return output_path

        except subprocess.TimeoutExpired:
            logger.error("ffmpeg timed out after 10 minutes")
            return self._render_fallback(duration)
        except Exception as e:
            logger.error(f"Visuals rendering failed: {e}")
            return self._render_fallback(duration)

    def _render_fallback(self, duration: float) -> Optional[Path]:
        """Fallback: simple color with drawtext."""
        try:
            logger.info("Rendering fallback visuals...")
            output_path = self.output_dir / "visuals.mp4"

            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"color=c=0x1a1a2e:size={self.width}x{self.height}:rate={self.fps}:duration={duration:.3f}",
                "-vf", (
                    f"drawtext=text='VIDEO-GEN PIPELINE':fontcolor=white:fontsize=48:"
                    f"x=(w-text_w)/2:y=(h-text_h)/2:fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
                ),
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "22",
                "-pix_fmt", "yuv420p",
                "-r", str(self.fps),
                str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if result.returncode == 0 and output_path.exists():
                logger.info(f"Fallback visuals rendered: {output_path}")
                return output_path
            return None
        except Exception as e:
            logger.error(f"Fallback rendering failed: {e}")
            return None


def main() -> int:
    config_path = Path("config.json")
    if not config_path.exists():
        logger.error("config.json not found")
        return 1

    with open(config_path) as f:
        config = json.load(f)

    output_dir = Path(config.get("output_dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    voiceover_path = output_dir / "voiceover.wav"
    if not voiceover_path.exists():
        logger.error("voiceover.wav not found. Run 01_tts.py first.")
        return 1

    renderer = VisualsRenderer(config, output_dir)
    duration = renderer.get_audio_duration(voiceover_path)
    result = renderer.render(duration)

    if result and result.exists():
        logger.info(f"SUCCESS: Visuals saved to {result}")
        return 0
    else:
        logger.error("Visuals rendering failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
