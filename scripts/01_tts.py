#!/usr/bin/env python3
"""
TTS Module - CPU-only voiceover generation via the espeak-ng CLI.

NOTE: The original plan called for stepfun-ai/Step-Audio-EditX (3B) for zero-shot voice
cloning with paralinguistic tags. That model requires a GPU with 12GB+ VRAM and a separate
CosyVoice vocoder pipeline (it outputs semantic tokens, not audio) - neither is available
on free GitHub Actions CPU runners. This module previously used the pyttsx3 library, but
its espeak driver relies on a ctypes callback loop that is known to silently fail (no
error, no output file) in headless CI/Docker environments - see e.g.
github.com/nateshmbhat/pyttsx3/issues/151. Calling the espeak-ng binary directly avoids
that whole class of failure: it's a synchronous subprocess with a real exit code.
Paralinguistic tags like [sigh] are stripped since espeak-ng has no equivalent for them.
"""

import os
import sys
import json
import logging
import re
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

PARALINGUISTIC_TAG_PATTERN = re.compile(
    r"\[(sigh|laugh|chuckle|cough|breath|inhale|exhale|clears throat|snort|giggle)\]",
    re.IGNORECASE,
)


class TTSGenerator:
    """espeak-ng CLI-based CPU voiceover generator."""

    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.sample_rate = config.get("voiceover_sample_rate", 44100)

    def strip_paralinguistic_tags(self, text: str) -> str:
        """Remove paralinguistic tags (e.g. [sigh]) - espeak-ng has no equivalent for them."""
        cleaned = PARALINGUISTIC_TAG_PATTERN.sub("", text)
        return re.sub(r"\s{2,}", " ", cleaned).strip()

    def generate(self, text: str) -> Optional[Path]:
        """Generate voiceover audio from text using the espeak-ng CLI."""
        try:
            logger.info("Generating voiceover with espeak-ng...")
            start_time = time.time()

            clean_text = self.strip_paralinguistic_tags(text)
            logger.info(f"Text (tags stripped): {clean_text[:100]}...")

            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = (self.output_dir / "voiceover.wav").resolve()

            # Write text to a temp file and use -f so arbitrary punctuation/quotes/unicode
            # never has to survive shell/argv escaping.
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as tf:
                tf.write(clean_text)
                text_file = tf.name

            try:
                result = subprocess.run(
                    [
                        "espeak-ng",
                        "-s", "170",       # words per minute
                        "-v", "en-us",
                        "-f", text_file,
                        "-w", str(output_path),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
            finally:
                os.unlink(text_file)

            if result.returncode != 0:
                raise RuntimeError(f"espeak-ng exited {result.returncode}: {result.stderr.strip()}")

            if not output_path.exists() or output_path.stat().st_size == 0:
                raise RuntimeError(f"espeak-ng did not produce a valid file at {output_path}")

            import soundfile as sf
            audio, sr = sf.read(str(output_path))
            if sr != self.sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                sf.write(str(output_path), audio, self.sample_rate, subtype='PCM_16')

            duration = len(audio) / self.sample_rate
            logger.info(f"Voiceover generated in {time.time() - start_time:.1f}s ({duration:.2f}s audio)")
            return output_path

        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return None


def get_audio_duration(wav_path: Path) -> float:
    """Get duration of WAV file in seconds."""
    try:
        with wave.open(str(wav_path), "rb") as wf:
            return wf.getnframes() / float(wf.getframerate())
    except Exception:
        return 0.0


def main() -> int:
    config_path = Path("config.json")
    if not config_path.exists():
        logger.error("config.json not found")
        return 1

    with open(config_path) as f:
        config = json.load(f)

    output_dir = Path(config.get("output_dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    script_text = config.get("script_text", "")
    if not script_text:
        logger.error("No script_text in config")
        return 1

    generator = TTSGenerator(config, output_dir)
    result = generator.generate(script_text)

    if result and result.exists():
        duration = get_audio_duration(result)
        logger.info(f"SUCCESS: Voiceover saved to {result} ({duration:.2f}s)")
        return 0
    else:
        logger.error("Voiceover generation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
