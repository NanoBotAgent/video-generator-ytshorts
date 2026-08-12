#!/usr/bin/env python3
"""
Visuals Module - Renders procedural gradient video using ffmpeg lavfi.
Generates 1080x1920 @ 60fps visuals.mp4 matching voiceover duration.
Uses ffmpeg color/gradients filter - no Chromium/timecut needed (works on CPU CI).
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

            # Build ffmpeg lavfi filter graph for animated gradients
            # Multiple gradient layers with animation
            filter_complex = (
                f"color=c=0x1a1a2e:size={self.width}x{self.height}:rate={self.fps}:duration={duration:.3f}[bg];"
                f"gradients=size={self.width}x{self.height}:rate={self.fps}:duration={duration:.3f}:"
                f"c0=0x16213e:c1=0x0f3460:c2=0xe94560:c3=0x533483:speed=0.1[grad1];"
                f"gradients=size={self.width}x{self.height}:rate={self.fps}:duration={duration:.3f}:"
                f"c0=0x0f0f23:c1=0x1a1a2e:c2=0x16213e:c3=0x0f3460:speed=0.05[grad2];"
                f"[bg][grad1]blend=all_mode=overlay:all_opacity=0.3[tmp];"
                f"[tmp][grad2]blend=all_mode=overlay:all_opacity=0.2[vout]"
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