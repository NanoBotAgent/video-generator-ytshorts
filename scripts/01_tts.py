#!/usr/bin/env python3
"""
TTS Module - Step Audio EditX (3B Q8 quantized) for voiceover generation.
Generates voiceover.wav with zero-shot voice cloning and paralinguistic tag support.
Falls back to pyttsx3 if tokenizer/model fails.
"""

import os
import sys
import json
import logging
import time
import wave
from pathlib import Path
from typing import Optional, Tuple

import torch
import numpy as np
from transformers import AutoTokenizer, AutoConfig
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
                    local_local_dir_use_symlinks=False,
                    resume_download=True,
                )

            # Add model path to sys.path to import custom model class
            transformers_modules_path = self.model_path / "transformers_modules"
            if transformers_modules_path.exists():
                sys.path.insert(0, str(transformers_modules_path))

            # Load tokenizer with proper handling for SentencePiece tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=False,  # Disable fast tokenizer to avoid tiktoken issues
            )

            # Load config to get model class name
            config = AutoConfig.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )

            # Import and load the custom model class directly
            try:
                # The model uses a custom modeling file - import it
                from models.step1.modeling_step1 import Step1ForCausalLM
                self.model = Step1ForCausalLM.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    config=config,
                ).to(self.device)
            except ImportError as e:
                logger.warning(f"Could not import custom model class: {e}")
                # Try alternative import path
                try:
                    from modeling_step1 import Step1ForCausalLM
                    self.model = Step1ForCausalLM.from_pretrained(
                        self.model_path,
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                        config=config,
                    ).to(self.device)
                except ImportError as e2:
                    logger.error(f"Failed to import custom model class: {e2}")
                    return False

            self.model.eval()
            logger.info(f"Model loaded in {time.time() - start_time:.1f}s")
            return True
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
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
        """Generate voiceover audio from text with fallback to pyttsx3."""
        # Ensure output directory exists early
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Try main model first
        if self.model is None or self.tokenizer is None:
            if not self.load_model():
                logger.info("Main model failed to load, using pyttsx3 fallback")
                return self._generate_fallback(text)

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
                    import torchaudio
                    audio_tensor = torchaudio.functional.resample(
                        audio_tensor, model_sr, target_sr
                    )

            output_path = self.output_dir / "voiceover.wav"
            import torchaudio
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
            logger.error(f"TTS generation failed: {e}")
            return self._generate_fallback(text)

    def _generate_fallback(self, text: str) -> Optional[Path]:
        """Generate voiceover using pyttsx3 as fallback."""
        try:
            logger.info("Generating voiceover with pyttsx3 fallback...")
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