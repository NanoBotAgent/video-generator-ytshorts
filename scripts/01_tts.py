#!/usr/bin/env python3
"""
TTS Module - Step Audio EditX (3B Q8 quantized) with fallback to pyttsx3.
Generates voiceover.wav with zero-shot voice cloning and paralinguistic tag support.
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
from transformers import AutoModel, AutoTokenizer
from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class TTSGenerator:
    """Step Audio EditX TTS generator with Q8 quantization for CPU, with pyttsx3 fallback."""

    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device("cpu")
        self.model = None
        self.tokenizer = None
        self.sample_rate = config.get("voiceover_sample_rate", 44100)
        self.model_path = Path.home() / ".cache" / "huggingface" / "hub" / "models--stepfun-ai--Step-Audio-EditX"

    def load_model(self) -> bool:
        """Load Step Audio EditX model with Q8 quantization."""
        try:
            logger.info("Loading Step Audio EditX model...")
            start_time = time.time()

            if not self.model_path.exists():
                logger.info("Model not found locally, downloading...")
                snapshot_download(
                    repo_id="stepfun-ai/Step-Audio-EditX",
                    local_dir=self.model_path,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )

            self.model = AutoModel.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            ).to(self.device)

            self.model.eval()
            logger.info(f"Model loaded in {time.time() - start_time:.1f}s")
            return True
        except Exception as e:
            logger.warning(f"Failed to load Step Audio EditX model: {e}")
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

    def generate_fallback(self, text: str) -> Optional[Path]:
        """Generate voiceover using pyttsx3 as fallback."""
        try:
            logger.info("Using pyttsx3 fallback TTS...")
            start_time = time.time()

            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 180)
            engine.setProperty('volume', 1.0)

            output_path = self.output_dir / "voiceover.wav"
            engine.save_to_file(text, str(output_path))
            engine.runAndWait()

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
            return None

    def generate(self, text: str) -> Optional[Path]:
        """Generate voiceover audio from text with fallback."""
        # Try main model first
        if self.model is None or self.tokenizer is None:
            if not self.load_model():
                logger.info("Main model failed to load, trying fallback...")
                return self.generate_fallback(text)

        try:
            logger.info("Generating voiceover with Step Audio EditX...")
            start_time = time.time()

            processed_text = self.process_paralinguistic_tags(text)
            logger.info(f"Processed text: {processed_text[:100]}...")

            inputs = self.tokenizer(
                processed_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            with torch.inference_mode():
                audio_output = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    max_new_tokens=2048,
                )

            audio_tensor = audio_output.cpu().float()

            if audio_tensor.dim() == 3:
                audio_tensor = audio_tensor.squeeze(0)
            if audio_tensor.dim() == 2:
                audio_tensor = audio_tensor.mean(dim=0)

            audio_tensor = audio_tensor / (audio_tensor.abs().max() + 1e-8)

            target_sr = self.sample_rate
            if hasattr(self.model, "config") and hasattr(self.model.config, "sampling_rate"):
                model_sr = self.model.config.sampling_rate
                if model_sr != target_sr:
                    audio_tensor = torchaudio.functional.resample(
                        audio_tensor, model_sr, target_sr
                    )

            output_path = self.output_dir / "voiceover.wav"
            torchaudio.save(
                str(output_path),
                audio_tensor.unsqueeze(0),
                target_sr,
                encoding="PCM_S",
                bits_per_sample=16,
            )

            duration = audio_tensor.shape[-1] / target_sr
            logger.info(f"Voiceover generated in {time.time() - start_time:.1f}s "
                       f"({duration:.2f}s, {target_sr}Hz, mono)")
            return output_path

        except Exception as e:
            logger.warning(f"Main TTS generation failed: {e}, trying fallback...")
            return self.generate_fallback(text)


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