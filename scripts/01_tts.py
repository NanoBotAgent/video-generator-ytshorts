#!/usr/bin/env python3
"""
TTS Module - Fish Audio S2.1 Pro (free tier) via the official hosted API.

Inference runs on Fish Audio's servers, not the runner, so this is genuinely a real
neural TTS model on a free CPU-only GitHub Actions runner - the runner just makes one
HTTP call. Falls back to the local espeak-ng CLI if FISH_API_KEY isn't set or the API
call fails for any reason, so the pipeline still produces audio rather than hard-failing.

NOTE ON MODEL HISTORY:
- Step-Audio-EditX (3B) was rejected earlier for local CPU inference: GPU-only, needs a
  separate CosyVoice vocoder not bundled with the weights.
- Running Fish Audio S2 Pro locally via s2.cpp/GGUF was also considered, but s2.cpp is an
  early-stage/experimental GGML engine and S2 Pro is a ~4.56B-param Dual-AR model - every
  available benchmark for it is GPU VRAM-based (7-11GB+), with no established CPU numbers.
  That's larger than Step-Audio-EditX, so it has the same problem, worse.
- Fish Audio separately offers the *same* S2.1 Pro model for free via a hosted API
  (model="s2.1-pro-free"), specifically for evaluation/prototyping/small-scale use with no
  hard character cap as of writing. That sidesteps the local-compute problem entirely,
  which is what's used below. Omitting the `model` param would silently default to the
  PAID "s2.1-pro" - it must always be passed explicitly.
- Unlike espeak-ng, S2.1 Pro natively understands paralinguistic/emotion tags like [sigh],
  [laugh], [excited] - so script_text is passed through unmodified on this path (only the
  espeak-ng fallback strips them, since espeak-ng has no equivalent).

Requires a FISH_API_KEY secret (free account + API key at https://fish.audio). If it's not
set, or the API call fails for any reason (network, rate limit, auth, quota), this logs a
clear warning and falls back to the local espeak-ng CLI.
"""

import os
import sys
import json
import logging
import time
import re
import subprocess
import tempfile
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

# The FREE tier model string. Must be passed explicitly on every request - if the
# `model` param is omitted or misspelled, Fish Audio's API silently falls back to the
# PAID "s2.1-pro" instead of erroring, so a typo here would incur real cost rather than
# fail loudly. Double-check this against https://docs.fish.audio if requests start
# getting billed.
FISH_MODEL = "s2.1-pro-free"


class TTSGenerator:
    """Fish Audio S2.1 Pro (free API) with a local espeak-ng fallback."""

    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.sample_rate = config.get("voiceover_sample_rate", 44100)
        self.engine_used = "unknown"

    def generate(self, text: str) -> Optional[Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = (self.output_dir / "voiceover.wav").resolve()

        api_key = os.environ.get("FISH_API_KEY", "").strip()
        if not api_key:
            logger.warning(
                "FISH_API_KEY is not set - skipping Fish Audio S2.1 Pro and falling "
                "back to local espeak-ng. Add a FISH_API_KEY repo secret (free account "
                "at https://fish.audio) to use the neural voice."
            )
            return self._generate_espeak_fallback(text, output_path)

        result = self._generate_fish_audio(text, output_path, api_key)
        if result is not None:
            return result

        logger.warning(
            "Fish Audio API call failed (see error above) - falling back to local "
            "espeak-ng so the pipeline can still complete."
        )
        return self._generate_espeak_fallback(text, output_path)

    def _generate_fish_audio(self, text: str, output_path: Path, api_key: str) -> Optional[Path]:
        """Generate voiceover via Fish Audio S2.1 Pro (free tier, hosted API)."""
        try:
            from fishaudio import FishAudio
            from fishaudio.types import TTSConfig

            logger.info(f"Requesting TTS from Fish Audio (model={FISH_MODEL})...")
            logger.info(f"Text sent as-is, tags included (S2.1 Pro understands them natively): {text[:100]}...")
            start_time = time.time()

            client = FishAudio(api_key=api_key)
            audio_bytes = client.tts.convert(
                text=text,
                model=FISH_MODEL,
                config=TTSConfig(
                    format="wav",
                    sample_rate=self.sample_rate,
                    normalize=True,
                ),
            )
            elapsed = time.time() - start_time

            if not audio_bytes:
                raise RuntimeError("Fish Audio returned an empty response body")

            output_path.write_bytes(audio_bytes)

            if not output_path.exists() or output_path.stat().st_size == 0:
                raise RuntimeError(f"Wrote 0 bytes to {output_path}")

            size_kb = output_path.stat().st_size / 1024
            duration = get_audio_duration(output_path)
            logger.info(
                f"Fish Audio S2.1 Pro response received in {elapsed:.1f}s "
                f"({size_kb:.0f} KB, {duration:.2f}s audio, model={FISH_MODEL})"
            )
            self.engine_used = f"fish-audio ({FISH_MODEL})"
            return output_path

        except Exception as e:
            logger.error(f"Fish Audio TTS request failed: {type(e).__name__}: {e}")
            return None

    def _generate_espeak_fallback(self, text: str, output_path: Path) -> Optional[Path]:
        """Fallback: local espeak-ng CLI. Robotic-sounding but needs no API key/network."""
        try:
            logger.info("Generating voiceover with local espeak-ng (fallback)...")
            start_time = time.time()

            clean_text = self._strip_paralinguistic_tags(text)
            logger.info(f"Text (tags stripped for espeak-ng): {clean_text[:100]}...")

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", delete=False, encoding="utf-8"
            ) as tf:
                tf.write(clean_text)
                text_file = tf.name

            try:
                result = subprocess.run(
                    [
                        "espeak-ng",
                        "-s", "170",
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
            logger.info(f"espeak-ng fallback generated in {time.time() - start_time:.1f}s ({duration:.2f}s audio)")
            self.engine_used = "espeak-ng (fallback, robotic)"
            return output_path

        except Exception as e:
            logger.error(f"espeak-ng fallback also failed: {e}")
            return None

    def _strip_paralinguistic_tags(self, text: str) -> str:
        cleaned = PARALINGUISTIC_TAG_PATTERN.sub("", text)
        return re.sub(r"\s{2,}", " ", cleaned).strip()


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

    # Write a small marker file so the workflow's debug/summary steps can report which
    # engine actually produced the audio without parsing full logs.
    (output_dir / "tts_engine.txt").write_text(generator.engine_used, encoding="utf-8")

    if result and result.exists():
        duration = get_audio_duration(result)
        logger.info(f"SUCCESS: Voiceover saved to {result} ({duration:.2f}s, engine={generator.engine_used})")
        return 0
    else:
        logger.error("Voiceover generation failed (both Fish Audio and espeak-ng fallback failed)")
        return 1


if __name__ == "__main__":
    sys.exit(main())
