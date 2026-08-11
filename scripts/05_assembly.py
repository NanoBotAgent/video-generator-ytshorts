#!/usr/bin/env python3
"""
Assembly Module - FFmpeg merge with sidechain ducking and subtitle burning.
Combines visuals.mp4, voiceover.wav, bgm.wav, and captions.ass into final_video.mp4.
"""

import os
import sys
import json
import logging
import time
import subprocess
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class VideoAssembler:
    """Assembles final video using FFmpeg with sidechain compression and subtitle burning."""

    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.width = config.get("video_width", 1080)
        self.height = config.get("video_height", 1920)
        self.fps = config.get("fps", 60)
        self.bgm_duck_level = config.get("bgm_volume_duck", 0.15)
        self.ffmpeg_preset = config.get("ffmpeg_preset", "fast")
        self.ffmpeg_crf = config.get("ffmpeg_crf", 22)

    def check_ffmpeg(self) -> bool:
        """Verify FFmpeg is available with required features."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                logger.info(f"FFmpeg: {version_line}")
                return True
            return False
        except FileNotFoundError:
            logger.error("FFmpeg not found. Install with: apt-get install ffmpeg")
            return False
        except Exception as e:
            logger.error(f"FFmpeg check failed: {e}")
            return False

    def check_ass_support(self) -> bool:
        """Verify FFmpeg has libass support for subtitle burning."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-filters"],
                capture_output=True,
                text=True,
                timeout=10
            )
            return "ass" in result.stdout
        except Exception:
            return False

    def assemble(self) -> Optional[Path]:
        """Run FFmpeg to assemble final video."""
        if not self.check_ffmpeg():
            return None

        if not self.check_ass_support():
            logger.warning("libass support not detected in FFmpeg, subtitles may not render")

        required_files = {
            "visuals": self.output_dir / "visuals.mp4",
            "voiceover": self.output_dir / "voiceover.wav",
            "bgm": self.output_dir / "bgm.wav",
            "captions": self.output_dir / "captions.ass",
        }

        for name, path in required_files.items():
            if not path.exists():
                logger.error(f"Required file missing: {name} ({path})")
                return None

        try:
            logger.info("Assembling final video with FFmpeg...")
            start_time = time.time()

            output_path = self.output_dir / "final_video.mp4"
            duck = self.bgm_duck_level
            captions_path = str(required_files["captions"]).replace("'", "\\'")

            # Simplified filter graph:
            # Input 0: visuals.mp4 (video only)
            # Input 1: voiceover.wav (audio only)  
            # Input 2: bgm.wav (audio only)
            #
            # We use explicit stream mapping:
            # - 0:v for video from visuals
            # - 1:a for audio from voiceover
            # - 2:a for audio from bgm
            filter_graph = (
                f"[1:a]volume=1.0[voice];"
                f"[2:a]volume=1.0[bgm];"
                f"[bgm][voice]sidechaincompress=threshold=0.003:ratio=20:attack=5:release=100:makeup=1[ducked];"
                f"[ducked]volume={duck}[bgm_final];"
                f"[voice][bgm_final]amix=inputs=2:duration=first:dropout_transition=0[audio_out];"
                f"[0:v]ass='{captions_path}'[vout]"
            )

            cmd = [
                "ffmpeg", "-y",
                "-i", str(required_files["visuals"]),
                "-i", str(required_files["voiceover"]),
                "-i", str(required_files["bgm"]),
                "-filter_complex", filter_graph,
                "-map", "[vout]",
                "-map", "[audio_out]",
                "-c:v", "libx264",
                "-preset", self.ffmpeg_preset,
                "-crf", str(self.ffmpeg_crf),
                "-pix_fmt", "yuv420p",
                "-r", str(self.fps),
                "-c:a", "aac",
                "-b:a", "128k",
                "-ar", "44100",
                "-movflags", "+faststart",
                "-shortest",
                str(output_path),
            ]

            logger.info(f"FFmpeg command: {' '.join(cmd[:10])}... [filter_complex omitted]")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )

            elapsed = time.time() - start_time

            if result.returncode != 0:
                logger.error(f"FFmpeg failed (exit {result.returncode}): {result.stderr[-2000:]}")
                # Try fallback without sidechain
                return self._assemble_fallback(required_files, output_path, captions_path, start_time)

            if not output_path.exists() or output_path.stat().st_size == 0:
                logger.error("Output file not created or empty")
                return None

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"Assembly complete in {elapsed:.1f}s ({file_size_mb:.1f} MB)")
            return output_path

        except subprocess.TimeoutExpired:
            logger.error("FFmpeg timed out after 10 minutes")
            return None
        except Exception as e:
            logger.error(f"Assembly failed: {e}")
            return self._assemble_fallback(required_files, output_path, captions_path, start_time)

    def _assemble_fallback(self, required_files: dict, output_path: Path, captions_path: str, start_time: float) -> Optional[Path]:
        """Fallback assembly without sidechain compression."""
        try:
            logger.info("Trying fallback assembly without sidechain...")
            filter_graph = (
                f"[1:a]volume=1.0[voice];"
                f"[2:a]volume={self.bgm_duck_level}[bgm];"
                f"[voice][bgm]amix=inputs=2:duration=first:dropout_transition=0[audio_out];"
                f"[0:v]ass='{captions_path}'[vout]"
            )

            cmd = [
                "ffmpeg", "-y",
                "-i", str(required_files["visuals"]),
                "-i", str(required_files["voiceover"]),
                "-i", str(required_files["bgm"]),
                "-filter_complex", filter_graph,
                "-map", "[vout]",
                "-map", "[audio_out]",
                "-c:v", "libx264",
                "-preset", self.ffmpeg_preset,
                "-crf", str(self.ffmpeg_crf),
                "-pix_fmt", "yuv420p",
                "-r", str(self.fps),
                "-c:a", "aac",
                "-b:a", "128k",
                "-ar", "44100",
                "-movflags", "+faststart",
                "-shortest",
                str(output_path),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)

            if result.returncode != 0:
                logger.error(f"Fallback FFmpeg failed: {result.stderr[-2000:]}")
                return None

            if not output_path.exists() or output_path.stat().st_size == 0:
                logger.error("Output file not created or empty")
                return None

            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            logger.info(f"Fallback assembly complete in {time.time() - start_time:.1f}s ({file_size_mb:.1f} MB)")
            return output_path

        except Exception as e:
            logger.error(f"Fallback assembly failed: {e}")
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

    assembler = VideoAssembler(config, output_dir)
    result = assembler.assemble()

    if result and result.exists():
        logger.info(f"SUCCESS: Final video saved to {result}")
        return 0
    else:
        logger.error("Video assembly failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())