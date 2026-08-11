#!/usr/bin/env python3
"""
TTS Module - Step Audio EditX (3B Q8 quantized) for voiceover generation.
Generates voiceover.wav using pyttsx3 fallback (model has relative import issues).
"""

import os
import sys
import json
import logging
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


class TTSGenerator:
    """TTS generator using pyttsx3 fallback."""

    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.device = "cpu"
        self.sample_rate = config.get("voiceover_sample_rate", 44100)

    def load_model(self) -> bool:
        """Skip model loading, use fallback."""
        logger.info("Using pyttsx3 fallback for TTS (Step Audio EditX model has import issues)")
        return False

    def process_paralinguistic_tags(self, text: str) -> str:
        """Process paralinguistic tags for Step Audio EditX."""
        tag_map = {
            "[sigh]": "<|sigh|>",
            "[laugh]": "<|laugh|>",
            "[chuckle]": "<|chuckle|>",
            "[cough]": "<|cough|>",
            "[breath]": "<|breath|>",
        }
        processed = text
        for tag, token in tag_map.items():
            processed = processed.replace(tag, token)
        return processed

    def generate(self, text: str) -> Optional[Path]:
        """Generate voiceover audio using pyttsx3 fallback."""
        # Ensure output directory exists early
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")

        processed_text = self.process_paralinguistic_tags(text)
        logger.info(f"Generating voiceover with pyttsx3...")
        logger.info(f"Text: {processed_text[:100]}...")

        return self._generate_fallback(processed_text)

    def _generate_fallback(self, text: str) -> Optional[Path]:
        """Generate voiceover using pyttsx3 as fallback."""
        try:
            logger.info("Generating voiceover with pyttsx3...")
            start_time = time.time()

            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)

            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 180)
            engine.setProperty('volume', 1.0)

            output_path = self.output_dir / "voiceover.wav"
            engine.save_to_file(text, str(output_path))
            engine.runAndWait()

            # Wait a bit for file to be written
            time.sleep(2)

            # Verify file exists before reading
            if not output_path.exists():
                logger.error(f"Output file not created: {output_path}")
                return None

            # Convert to desired sample rate if needed
            import soundfile as sf
            audio, sr = sf.read(str(output_path))
            if sr != self.sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
                sf.write(str(output_path), audio, self.sample_rate, subtype='PCM_16')

            duration = len(audio) / self.sample_rate
            logger.info(f"Fallback TTS generated in {time.time() - start_time:.1f}s ({duration:.2f}s)")
            return output_path
        except Exception as e:
            logger.error(f"Fallback TTS failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None


def get_audio_duration(wav_path: Path) -> float:
    """Get duration of WAV file in seconds."""
    try:
        with wave.open(str(wav_path), "rb") as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            return frames / float(rate)
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