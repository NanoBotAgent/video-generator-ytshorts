#!/usr/bin/env python3
"""
BGM Module - Background music generation.
Generates bgm.wav using a simple procedural ambient fallback.
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
    """Procedural ambient BGM generator."""

    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device("cpu")
        self.sample_rate = 44100

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
        return self._generate_fallback(duration)

    def _generate_fallback(self, duration: float) -> Optional[Path]:
        """Generate a simple ambient fallback."""
        try:
            logger.info("Generating fallback ambient audio...")
            start_time = time.time()
            buffer_seconds = self.config.get("bgm_duration_buffer_seconds", 2)
            target_duration = duration + buffer_seconds
            target_samples = int(target_duration * self.sample_rate)

            t = torch.linspace(0, target_duration, target_samples)
            base_freq = 55.0
            audio = torch.zeros(2, target_samples)

            for i, freq_mult in enumerate([1.0, 1.5, 2.0, 3.0]):
                freq = base_freq * freq_mult
                amplitude = 0.02 / freq_mult
                audio[0] += amplitude * torch.sin(2 * np.pi * freq * t + np.random.random() * 2 * np.pi)
                audio[1] += amplitude * torch.sin(2 * np.pi * freq * t + np.random.random() * 2 * np.pi)

            noise = torch.randn(2, target_samples) * 0.005
            audio += noise

            envelope = torch.exp(-t * 0.1) * 0.5 + 0.5
            audio *= envelope

            audio = audio / (audio.abs().max() + 1e-8)

            output_path = self.output_dir / "bgm.wav"
            import soundfile as sf
            sf.write(str(output_path), audio.T.numpy(), self.sample_rate, subtype='PCM_16')

            actual_duration = audio.shape[-1] / self.sample_rate
            logger.info(f"BGM generated in {time.time() - start_time:.1f}s "
                       f"({actual_duration:.2f}s, {self.sample_rate}Hz)")
            return output_path
        except Exception as e:
            logger.error(f"Fallback generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
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