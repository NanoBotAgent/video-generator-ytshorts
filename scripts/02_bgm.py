#!/usr/bin/env python3
"""
BGM Module - Facebook MusicGen-Small (300M) for background music generation.
ACE-Step requires custom pipeline; MusicGen is natively supported in transformers.
Generates bgm.wav via text-to-audio model on CPU, matching voiceover duration.
"""

import os
import sys
import json
import logging
import time
import wave
from pathlib import Path
from typing import Optional

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class BGMGenerator:
    """MusicGen-Small BGM generator for CPU."""

    MODEL_ID = "facebook/musicgen-small"

    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device("cpu")
        self.model = None
        self.processor = None
        self.target_sr = 44100

    def load_model(self) -> bool:
        """Load MusicGen-Small model."""
        try:
            logger.info(f"Loading {self.MODEL_ID} on CPU...")
            start_time = time.time()

            from transformers import AutoProcessor, MusicgenForConditionalGeneration

            self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
            self.model = MusicgenForConditionalGeneration.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.float32,
            ).to(self.device)
            self.model.eval()

            logger.info(f"BGM model loaded in {time.time() - start_time:.1f}s")
            return True
        except Exception as e:
            logger.error(f"Failed to load BGM model: {e}")
            return False

    def get_voiceover_duration(self) -> float:
        """Get duration of generated voiceover."""
        voiceover_path = self.output_dir / "voiceover.wav"
        if not voiceover_path.exists():
            return 30.0
        try:
            with wave.open(str(voiceover_path), "rb") as wf:
                return wf.getnframes() / float(wf.getframerate())
        except Exception:
            return 30.0

    def generate(self, prompt: str, duration: float) -> Optional[Path]:
        """Generate background music matching target duration."""
        if self.model is None or self.processor is None:
            if not self.load_model():
                logger.error("Model failed to load")
                return None

        try:
            logger.info(f"Generating BGM for ~{duration:.1f}s...")
            start_time = time.time()

            buffer_seconds = self.config.get("bgm_duration_buffer_seconds", 2)
            target_duration = duration + buffer_seconds

            # MusicGen generates ~50 tokens/second (32kHz, 4 codebooks)
            # Max new tokens = duration * 50, cap to avoid timeout
            max_tokens = min(int(target_duration * 50), 1500)
            logger.info(f"Using max_new_tokens={max_tokens}")

            inputs = self.processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
            ).to(self.device)

            torch.manual_seed(42)
            with torch.inference_mode():
                audio_values = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=True,
                    guidance_scale=3.0,
                )

            audio = audio_values[0, 0].cpu().float().numpy()
            model_sr = self.model.config.audio_encoder.sampling_rate
            actual_duration = len(audio) / model_sr
            logger.info(f"Raw audio: {len(audio)} samples @ {model_sr}Hz "
                        f"({actual_duration:.2f}s)")

            if model_sr != self.target_sr:
                import librosa
                audio = librosa.resample(
                    audio.astype(np.float32),
                    orig_sr=model_sr,
                    target_sr=self.target_sr,
                )

            # Stereo pad/trim
            if len(audio.shape) == 1:
                stereo = np.stack([audio, audio], axis=0)
            else:
                stereo = audio
            target_samples = int(target_duration * self.target_sr)
            if stereo.shape[-1] < target_samples:
                pad = target_samples - stereo.shape[-1]
                stereo = np.pad(stereo, ((0, 0), (0, pad)), mode="wrap")
            else:
                stereo = stereo[:, :target_samples]

            stereo = stereo / (np.abs(stereo).max() + 1e-8) * 0.7

            import soundfile as sf
            output_path = self.output_dir / "bgm.wav"
            sf.write(str(output_path), stereo.T, self.target_sr, subtype="PCM_16")

            logger.info(f"BGM generated in {time.time() - start_time:.1f}s "
                        f"({stereo.shape[-1]/self.target_sr:.2f}s, {self.target_sr}Hz)")
            return output_path

        except Exception as e:
            logger.error(f"BGM generation failed: {e}")
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

    bgm_prompt = config.get("bgm_prompt", "")
    if not bgm_prompt:
        logger.error("No bgm_prompt in config")
        return 1

    voiceover_path = output_dir / "voiceover.wav"
    if not voiceover_path.exists():
        logger.error("voiceover.wav not found. Run 01_tts.py first.")
        return 1

    with wave.open(str(voiceover_path), "rb") as wf:
        voiceover_duration = wf.getnframes() / float(wf.getframerate())

    generator = BGMGenerator(config, output_dir)
    result = generator.generate(bgm_prompt, voiceover_duration)

    if result and result.exists():
        logger.info(f"SUCCESS: BGM saved to {result}")
        return 0
    else:
        logger.error("BGM generation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())