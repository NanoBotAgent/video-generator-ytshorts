#!/usr/bin/env python3
"""
BGM Module - Background music generation using procedural synthesis.
Generates bgm.wav using procedural ambient synthesis (ACE-Step model not available on HF).
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
        return self._generate_procedural(prompt, duration)

    def _generate_procedural(self, prompt: str, duration: float) -> Optional[Path]:
        """Generate a procedural ambient background track."""
        try:
            logger.info(f"Generating procedural BGM for {duration:.1f}s...")
            start_time = time.time()

            buffer_seconds = self.config.get("bgm_duration_buffer_seconds", 2)
            target_duration = duration + buffer_seconds
            target_samples = int(target_duration * self.sample_rate)

            t = torch.linspace(0, target_duration, target_samples)
            
            # Create a more musical ambient track based on prompt keywords
            audio = torch.zeros(2, target_samples)
            
            # Base frequencies for different moods
            base_freqs = [55.0, 82.41, 110.0, 164.81]  # A1, E2, A2, E3
            
            # Add harmonic layers
            for i, freq in enumerate(base_freqs):
                amplitude = 0.03 / (i + 1)
                # Slowly evolving phase
                phase = np.random.random() * 2 * np.pi
                audio[0] += amplitude * torch.sin(2 * np.pi * freq * t + phase)
                audio[1] += amplitude * torch.sin(2 * np.pi * freq * t + phase + 0.5)
            
            # Add subtle pad with slow modulation
            for freq_mult in [1.5, 2.0, 3.0]:
                freq = 110.0 * freq_mult
                amplitude = 0.015 / freq_mult
                phase = np.random.random() * 2 * np.pi
                audio[0] += amplitude * torch.sin(2 * np.pi * freq * t + phase)
                audio[1] += amplitude * torch.sin(2 * np.pi * freq * t + phase + 1.0)
            
            # Add very subtle noise for texture
            noise = torch.randn(2, target_samples) * 0.003
            audio += noise
            
            # Apply slow envelope for natural fade in/out
            envelope = torch.exp(-t * 0.05) * 0.6 + 0.4
            envelope[target_samples//2:] = torch.linspace(0.7, 0.3, target_samples//2)
            audio *= envelope
            
            # Apply gentle low-pass filter effect by smoothing (per channel)
            kernel_size = 5
            kernel = torch.ones(kernel_size) / kernel_size
            # Process each channel separately
            audio_filtered = torch.zeros_like(audio)
            for ch in range(2):
                audio_filtered[ch] = torch.nn.functional.conv1d(
                    audio[ch].unsqueeze(0).unsqueeze(0), 
                    kernel.unsqueeze(0).unsqueeze(0), 
                    padding=kernel_size//2
                ).squeeze(0).squeeze(0)
            audio = audio_filtered
            
            audio = audio / (audio.abs().max() + 1e-8)

            output_path = self.output_dir / "bgm.wav"
            import soundfile as sf
            sf.write(str(output_path), audio.T.numpy(), self.sample_rate, subtype='PCM_16')

            actual_duration = audio.shape[-1] / self.sample_rate
            logger.info(f"BGM generated in {time.time() - start_time:.1f}s "
                       f"({actual_duration:.2f}s, {self.sample_rate}Hz)")
            return output_path
        except Exception as e:
            logger.error(f"Procedural BGM generation failed: {e}")
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