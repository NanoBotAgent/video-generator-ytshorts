#!/usr/bin/env python3
"""
BGM Module - ACE-Step 1.5 for background music generation.
Generates bgm.wav using ACE-Step model with CPU support (slow but works).
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
from transformers import AutoModelForCausalLM, AutoConfig
from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class BGMGenerator:
    """ACE-Step 1.5 BGM generator with CPU support (slow but works)."""

    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device("cpu")
        self.model = None
        self.sample_rate = 44100
        self.model_path = Path.home() / ".cache" / "huggingface" / "hub" / "models--ACE-Step--Ace-Step1.5"
        self.num_inference_steps = 8
        self.guidance_scale = 1.0

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

    def load_model(self) -> bool:
        """Load ACE-Step XL-Turbo model."""
        try:
            logger.info("Loading ACE-Step 1.5 model (this may take several minutes on CPU)...")
            start_time = time.time()

            if not self.model_path.exists():
                logger.info("Model not found locally, downloading...")
                snapshot_download(
                    repo_id="ACE-Step/Ace-Step1.5",
                    local_dir=self.model_path,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )

            config = AutoConfig.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
            logger.info(f"Config class: {config.__class__.__name__}")

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                config=config,
            ).to(self.device)

            self.model.eval()
            logger.info(f"BGM model loaded in {time.time() - start_time:.1f}s")
            return True
        except Exception as e:
            logger.error(f"Failed to load BGM model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def generate(self, prompt: str, duration: float) -> Optional[Path]:
        """Generate background music matching target duration."""
        if self.model is None:
            if not self.load_model():
                logger.error("BGM model failed to load")
                return None

        try:
            logger.info(f"Generating BGM for {duration:.1f}s with ACE-Step (CPU mode - this may take several minutes)...")
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
                import torchaudio
                orig_sr = self.sample_rate * audio_tensor.shape[-1] // target_samples
                audio_tensor = torchaudio.functional.resample(
                    audio_tensor, orig_sr, self.sample_rate
                )

            audio_tensor = audio_tensor[:, :target_samples]
            audio_tensor = audio_tensor / (audio_tensor.abs().max() + 1e-8)

            output_path = self.output_dir / "bgm.wav"
            import torchaudio
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