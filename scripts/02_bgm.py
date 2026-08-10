#!/usr/bin/env python3
"""
BGM Module - ACE-Step 1.5 4B XL-Turbo (Q8 quantized) for background music generation.
Generates bgm.wav via 8-step turbo distillation matching voiceover duration + buffer.
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
import torchaudio
import numpy as np
from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class BGMGenerator:
    """ACE-Step 1.5 XL-Turbo BGM generator with 8-step turbo inference."""

    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device("cpu")
        self.model = None
        self.sample_rate = 44100
        self.model_path = Path.home() / ".cache" / "huggingface" / "hub" / "models--ACE-Step--Ace-Step1.5"
        self.num_inference_steps = 8
        self.guidance_scale = 1.0

    def load_model(self) -> bool:
        """Load ACE-Step XL-Turbo model with Q8 quantization."""
        try:
            logger.info("Loading ACE-Step XL-Turbo model...")
            start_time = time.time()

            if not self.model_path.exists():
                logger.info("Model not found locally, downloading...")
                snapshot_download(
                    repo_id="ACE-Step/Ace-Step1.5",
                    local_dir=self.model_path,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )

            from transformers import AutoModelForCausalLM
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
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
        if self.model is None:
            if not self.load_model():
                return None

        try:
            logger.info(f"Generating BGM for {duration:.1f}s...")
            start_time = time.time()

            buffer_seconds = self.config.get("bgm_duration_buffer_seconds", 2)
            target_duration = duration + buffer_seconds
            target_samples = int(target_duration * self.sample_rate)

            with torch.inference_mode():
                audio_output = self.model.generate(
                    prompt=prompt,
                    duration=target_duration,
                    num_inference_steps=self.num_inference_steps,
                    guidance_scale=self.guidance_scale,
                    temperature=1.0,
                    top_p=0.95,
                )

            if isinstance(audio_output, torch.Tensor):
                audio_tensor = audio_output.cpu().float()
            else:
                audio_tensor = torch.from_numpy(np.array(audio_output)).float()

            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            elif audio_tensor.dim() == 3:
                audio_tensor = audio_tensor.squeeze(0)

            if audio_tensor.shape[0] > 2:
                audio_tensor = audio_tensor[:2]

            if audio_tensor.shape[-1] != target_samples:
                audio_tensor = torchaudio.functional.resample(
                    audio_tensor, 
                    self.sample_rate * audio_tensor.shape[-1] // target_samples,
                    self.sample_rate
                )

            audio_tensor = audio_tensor[:, :target_samples]
            audio_tensor = audio_tensor / (audio_tensor.abs().max() + 1e-8)

            output_path = self.output_dir / "bgm.wav"
            torchaudio.save(
                str(output_path),
                audio_tensor,
                self.sample_rate,
                encoding="PCM_S",
                bits_per_sample=16,
            )

            actual_duration = audio_tensor.shape[-1] / self.sample_rate
            logger.info(f"BGM generated in {time.time() - start_time:.1f}s "
                       f"({actual_duration:.2f}s, {self.sample_rate}Hz)")
            return output_path

        except Exception as e:
            logger.error(f"BGM generation failed: {e}")
            return self._generate_fallback(duration)

    def _generate_fallback(self, duration: float) -> Optional[Path]:
        """Generate a simple ambient fallback if model fails."""
        try:
            logger.info("Generating fallback ambient audio...")
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
            torchaudio.save(
                str(output_path),
                audio,
                self.sample_rate,
                encoding="PCM_S",
                bits_per_sample=16,
            )
            logger.info(f"Fallback BGM saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Fallback generation failed: {e}")
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