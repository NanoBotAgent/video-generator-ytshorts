#!/usr/bin/env python3
"""
Visuals Module - Renders HTML template to MP4 using timecut (Chromium frame capture).
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
    """Renders HTML template to video using timecut/puppeteer."""

    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.template_path = Path("templates/index.html")
        self.width = config.get("video_width", 1080)
        self.height = config.get("video_height", 1920)
        self.fps = config.get("fps", 60)
        self.viewport = config.get("timecut_viewport", f"{self.width}x{self.height}")

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

    def check_timecut_available(self) -> bool:
        """Verify timecut is installed and accessible."""
        try:
            result = subprocess.run(
                ["npx", "timecut", "--version"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                logger.info(f"timecut version: {result.stdout.strip()}")
                return True
            else:
                logger.error(f"timecut check failed: {result.stderr}")
                return False
        except FileNotFoundError:
            logger.error("npx not found. Node.js/npm not installed.")
            return False
        except Exception as e:
            logger.error(f"timecut check error: {e}")
            return False

    def render(self, duration: float) -> Optional[Path]:
        """Render HTML template to MP4 video."""
        if not self.template_path.exists():
            logger.error(f"Template not found: {self.template_path}")
            return None

        if not self.check_timecut_available():
            logger.error("timecut not available. Run 'npm install' first.")
            return None

        try:
            logger.info(f"Rendering visuals: {self.width}x{self.height} @ {duration:.1f}s @ {self.fps}fps")
            start_time = time.time()

            output_path = self.output_dir / "visuals.mp4"
            output_path.parent.mkdir(parents=True, exist_ok=True)

            abs_template = self.template_path.resolve()
            abs_output = output_path.resolve()

            cmd = [
                "npx", "timecut",
                str(abs_template),
                f"--viewport={self.viewport}",
                f"--fps={self.fps}",
                f"--duration={duration:.3f}",
                f"--output={abs_output}",
                "--selector=body",
                "--left=0",
                "--top=0",
                "--round=1",
                "--parallel=1",
                "--no-sandbox",  # Required for GitHub Actions / CI environments
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
                logger.error(f"timecut failed (exit {result.returncode}): {result.stderr}")
                if "Cannot find module" in result.stderr or "ENOENT" in result.stderr:
                    logger.error("Try running: npm install")
                return None

            if not output_path.exists() or output_path.stat().st_size == 0:
                logger.error("Output file not created or empty")
                return None

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"Visuals rendered in {elapsed:.1f}s ({file_size_mb:.1f} MB)")
            return output_path

        except subprocess.TimeoutExpired:
            logger.error("timecut timed out after 10 minutes")
            return None
        except Exception as e:
            logger.error(f"Visuals rendering failed: {e}")
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