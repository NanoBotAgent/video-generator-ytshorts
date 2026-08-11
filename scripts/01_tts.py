#!/usr/bin/env python3
"""
TTS Module - SpeechT5 (microsoft/speecht5_tts) for voiceover generation.
Generates voiceover.wav using CPU-friendly SpeechT5 model with speaker embeddings.
Falls back to pyttsx3 if model fails.
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
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from huggingface_hub import snapshot_download

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class TTSGenerator:
    """SpeechT5 TTS generator for CPU inference with speaker embeddings."""

    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device("cpu")
        self.model = None
        self.processor = None
        self.vocoder = None
        self.speaker_embeddings = None
        self.sample_rate = config.get("voiceover_sample_rate", 16000)  # SpeechT5 uses 16kHz
        self.model_path = Path.home() / ".cache" / "huggingface" / "hub" / "models--microsoft--speecht5_tts"

    def load_model(self) -> bool:
        """Load SpeechT5 model with processor and vocoder."""
        try:
            logger.info("Loading SpeechT5 TTS model...")
            start_time = time.time()

            if not self.model_path.exists():
                logger.info("Model not found locally, downloading...")
                snapshot_download(
                    repo_id="microsoft/speecht5_tts",
                    local_dir=self.model_path,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )

            # Load processor
            self.processor = SpeechT5Processor.from_pretrained(
                self.model_path,
            )

            # Load model
            self.model = SpeechT5ForTextToSpeech.from_pretrained(
                self.model_path,
                torch_dtype=torch.float32,  # Use float32 for CPU
                low_cpu_mem_usage=True,
            ).to(self.device)

            # Load vocoder (HiFi-GAN)
            vocoder_path = Path.home() / ".cache" / "huggingface" / "hub" / "models--microsoft--speecht5_hifigan"
            if not vocoder_path.exists():
                logger.info("Downloading HiFi-GAN vocoder...")
                snapshot_download(
                    repo_id="microsoft/speecht5_hifigan",
                    local_dir=vocoder_path,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )
            self.vocoder = SpeechT5HifiGan.from_pretrained(vocoder_path).to(self.device)

            # Create default speaker embeddings (random x-vector)
            # In production, you'd load from a speaker verification model
            # For now, use a simple random embedding that works
            self.speaker_embeddings = torch.randn(1, 512).to(self.device)
            
            self.model.eval()
            self.vocoder.eval()
            logger.info(f"Model loaded in {time.time() - start_time:.1f}s")
            return True
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def process_paralinguistic_tags(self, text: str) -> str:
        """Process paralinguistic tags for SpeechT5 (remove unsupported tags)."""
        # SpeechT5 doesn't support paralinguistic tags, so we remove them
        tag_map = {
            "[sigh]": "",
            "[laugh]": "",
            "[chuckle]": "",
            "[cough]": "",
            "[breath]": "",
            "[inhale]": "",
            "[exhale]": "",
            "[clears throat]": "",
            "[snort]": "",
            "[giggle]": "",
            "[Uhm]": "",
            "[Confirmation-en]": "",
            "[Surprise-oh]": "",
            "[Surprise-ah]": "",
            "[Surprise-wa]": "",
            "[Dissatisfaction-hnn]": "",
            "[Question-ei]": "",
            "[Question-ah]": "",
            "[Question-en]": "",
            "[Question-yi]": "",
            "[Question-oh]": "",
        }
        processed = text
        for tag, token in tag_map.items():
            processed = processed.replace(tag, token)
        # Clean up extra spaces
        processed = " ".join(processed.split())
        return processed

    def generate(self, text: str) -> Optional[Path]:
        """Generate voiceover audio from text with fallback to pyttsx3."""
        # Ensure output directory exists early
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")

        # Try main model first
        if self.model is None or self.processor is None:
            if not self.load_model():
                logger.info("Main model failed to load, using pyttsx3 fallback")
                return self._generate_fallback(text)

        try:
            logger.info("Generating voiceover with SpeechT5...")
            start_time = time.time()

            processed_text = self.process_paralinguistic_tags(text)
            logger.info(f"Processed text: {processed_text[:100]}...")

            inputs = self.processor(
                text=processed_text,
                return_tensors="pt",
            ).to(self.device)

            with torch.inference_mode():
                # Generate spectrogram
                spectrogram = self.model.generate_speech(
                    inputs["input_ids"],
                    self.speaker_embeddings,
                )
                # Generate waveform using vocoder
                speech = self.vocoder(spectrogram)

            audio_tensor = speech.cpu().float()

            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            target_sr = 16000  # SpeechT5 native sample rate
            output_sr = self.sample_rate

            if target_sr != output_sr:
                import torchaudio
                audio_tensor = torchaudio.functional.resample(
                    audio_tensor, target_sr, output_sr
                )
            else:
                target_sr = output_sr

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
            import traceback
            logger.error(traceback.format_exc())
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