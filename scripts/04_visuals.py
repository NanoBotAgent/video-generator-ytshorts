#!/usr/bin/env python3
"""
Visuals Module - Generates MP4 video using ffmpeg (fallback for CI).
Generates 1080x1920 @ 60fps visuals.mp4 matching voiceover duration.
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
    """Generates video using ffmpeg (reliable in CI)."""

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
        """Generate video using ffmpeg with animated gradient."""
        try:
            logger.info(f"Generating visuals with ffmpeg: {self.width}x{self.height} @ {duration:.1f}s @ {self.fps}fps")
            start_time = time.time()

            output_path = self.output_dir / "visuals.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Generate a video with animated gradient using ffmpeg
            # Using lavfi (libavfilter) for programmatic video generation
            total_frames = int(duration * self.fps)

            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"color=c=0x0a0a0f:size={self.width}x{self.height}:rate={self.fps}:duration={duration:.3f}",
                "-f", "lavfi",
                "-i", f"gradients=s={self.width}x{self.height}:c0=0x0064ff:c1=0x00ff88:c2=0xff3296:c3=0x12121a:duration={duration:.3f}:rate={self.fps}",
                "-filter_complex",
                "[1:v]format=rgba,loop=loop=-1:size=1,setpts=N/(FRAME_RATE*TB)[grad];"
                "[0:v][grad]blend=all_mode='overlay':all_opacity=0.3[v]",
                "-map", "[v]",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-r", str(self.fps),
                "-t", f"{duration:.3f}",
                str(output_path),
            ]

            logger.info(f"Executing: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=str(Path.cwd()),
            )

            elapsed = time.time() - start_time

            if result.returncode != 0:
                logger.warning(f"ffmpeg gradient failed (exit {result.returncode}): {result.stderr}")
                # Fallback to simple solid color
                return self._render_simple_fallback(duration, output_path, start_time)

            if not output_path.exists() or output_path.stat().st_size == 0:
                logger.error("Output file not created or empty")
                return None

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"Visuals generated in {time.time() - start_time:.1f}s ({file_size_mb:.1f} MB)")
            return output_path

        except subprocess.TimeoutExpired:
            logger.error("ffmpeg timed out after 10 minutes")
            return None
        except Exception as e:
            logger.error(f"Visuals generation failed: {e}")
            return self._render_simple_fallback(duration, output_path, start_time)

    def _render_simple_fallback(self, duration: float, output_path: Path, start_time: float) -> Optional[Path]:
        """Simple fallback: solid color video with timecode."""
        try:
            logger.info("Generating simple fallback visuals...")
            cmd = [
                "ffmpeg", "-y",
                "-f", "lavfi",
                "-i", f"color=c=0x12121a:size={self.width}x{self.height}:rate={self.fps}:duration={duration:.3f}",
                "-vf", f"drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf:fontcolor=white:fontsize=48:text='VIDEO-GEN PIPELINE':x=(w-text_w)/2:y=(h-text_h)/2,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:fontcolor=0x00ff88:fontsize=24:text='CPU-Only | Zero GPU | GitHub Actions':x=(w-text_w)/2:y=(h-text_h)/2+80",
                "-c:v", "libx264",
                "-preset", "fast",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                "-r", str(self.fps),
                "-t", f"{duration:.3f}",
                str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                logger.error(f"Fallback ffmpeg failed: {result.stderr}")
                return None

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"Fallback visuals generated in {time.time() - start_time:.1f}s ({file_size_mb:.1f} MB)")
            return output_path
        except Exception as e:
            logger.error(f"Fallback visuals failed: {e}")
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
        logger.error("Visuals generation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())