#!/usr/bin/env python3
"""
TTS Module - CPU-only voiceover generation via pyttsx3.

NOTE: The original plan called for stepfun-ai/Step-Audio-EditX (3B) for zero-shot voice
cloning with paralinguistic tags. That model requires a GPU with 12GB+ VRAM and a separate
CosyVoice vocoder pipeline (it outputs semantic tokens, not audio) - neither is available
on free GitHub Actions CPU runners. Loading the 3B model just to discard its output and
fall back anyway was wasting ~6GB of downloads and CPU time on every run, so this module
goes straight to pyttsx3: a lightweight, deterministic, CPU-native TTS engine.
Paralinguistic tags like [sigh] are stripped since pyttsx3 has no equivalent for them.
"""

import os
import sys
import json
import logging
import re
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
    """pyttsx3-based CPU voiceover generator."""

    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.sample_rate = config.get("voiceover_sample_rate", 44100)

    def strip_paralinguistic_tags(self, text: str) -> str:
        """Remove paralinguistic tags (e.g. [sigh]) - pyttsx3 has no equivalent for them."""
        cleaned = PARALINGUISTIC_TAG_PATTERN.sub("", text)
        return re.sub(r"\s{2,}", " ", cleaned).strip()

    def generate(self, text: str) -> Optional[Path]:
        """Generate voiceover audio from text using pyttsx3."""
        try:
            logger.info("Generating voiceover with pyttsx3...")
            start_time = time.time()

            clean_text = self.strip_paralinguistic_tags(text)
            logger.info(f"Text (tags stripped): {clean_text[:100]}...")

            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 180)
            engine.setProperty('volume', 1.0)

            # pyttsx3's espeak driver on Linux fails with a relative path
            # ("Error opening '...': System error.") - it must be absolute.
            self.output_dir.mkdir(parents=True, exist_ok=True)
            output_path = (self.output_dir / "voiceover.wav").resolve()
            engine.save_to_file(clean_text, str(output_path))
            engine.runAndWait()

            if not output_path.exists() or output_path.stat().st_size == 0:
                raise RuntimeError(f"pyttsx3 did not produce a valid file at {output_path}")

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
