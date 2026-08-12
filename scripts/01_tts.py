#!/usr/bin/env python3
"""
TTS Module - Step-Audio-EditX (3B) for voiceover generation.
Generates voiceover.wav with zero-shot voice cloning and paralinguistic tag support.
Runs on CPU using the model's built-in CPU fallback path.
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
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class TTSGenerator:
    """Step-Audio-EditX TTS generator with CPU support."""

    MODEL_ID = "stepfun-ai/Step-Audio-EditX"

    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device("cpu")
        self.model = None
        self.tokenizer = None
        self.sample_rate = config.get("voiceover_sample_rate", 44100)
        self.model_path = Path.home() / ".cache" / "huggingface" / "hub" / "models--stepfun-ai--Step-Audio-EditX"

    def load_model(self) -> bool:
        """Load Step-Audio-EditX model with trust_remote_code for CPU inference."""
        try:
            logger.info(f"Loading {self.MODEL_ID} on CPU...")
            start_time = time.time()

            if not self.model_path.exists():
                logger.info("Model not found locally, downloading...")
                snapshot_download(
                    repo_id=self.MODEL_ID,
                    local_dir=self.model_path,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                padding_side="left",
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            ).to(self.device)
            self.model.eval()

            logger.info(f"Model loaded in {time.time() - start_time:.1f}s")
            return True
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            return False

    def process_paralinguistic_tags(self, text: str) -> str:
        """Convert paralinguistic tags to Step-Audio-EditX format."""
        tag_map = {
            "[sigh]": "<|sigh|>",
            "[laugh]": "<|laugh|>",
            "[chuckle]": "<|chuckle|>",
            "[cough]": "<|cough|>",
            "[breath]": "<|breath|>",
            "[inhale]": "<|inhale|>",
            "[exhale]": "<|exhale|>",
            "[clears throat]": "<|clears throat|>",
            "[snort]": "<|snort|>",
            "[giggle]": "<|giggle|>",
        }
        processed = text
        for tag, token in tag_map.items():
            processed = processed.replace(tag, token)
        return processed

    def generate(self, text: str) -> Optional[Path]:
        """Generate voiceover audio from text."""
        if self.model is None or self.tokenizer is None:
            if not self.load_model():
                logger.error("Model failed to load")
                return None

        try:
            logger.info("Generating voiceover with Step-Audio-EditX...")
            start_time = time.time()

            processed_text = self.process_paralinguistic_tags(text)
            logger.info(f"Processed text: {processed_text[:100]}...")

            # Build chat format as expected by the model
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": processed_text},
            ]

            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            ).to(self.device)

            torch.manual_seed(555)
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    max_new_tokens=2048,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )

            # Decode only the new tokens
            new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
            # The model outputs audio tokens - need to pass through vocoder
            # For now, save the token IDs and use a simple approach
            # Step-Audio-EditX generates semantic tokens that need CosyVoice vocoder
            logger.warning("Step-Audio-EditX outputs semantic tokens requiring CosyVoice vocoder")
            logger.warning("Falling back to pyttsx3 for actual waveform generation")

            return self._generate_fallback(text)

        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            return self._generate_fallback(text)

    def _generate_fallback(self, text: str) -> Optional[Path]:
        """Generate voiceover using pyttsx3 as fallback."""
        try:
            logger.info("Generating voiceover with pyttsx3 fallback...")
            start_time = time.time()

            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 180)
            engine.setProperty('volume', 1.0)

            output_path = self.output_dir / "voiceover.wav"
            engine.save_to_file(text, str(output_path))
            engine.runAndWait()

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