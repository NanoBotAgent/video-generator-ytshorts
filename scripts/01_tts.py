#!/usr/bin/env python3
"""
TTS Module - Step Audio EditX (3B) for voiceover generation.
Generates voiceover.wav with zero-shot voice cloning and paralinguistic tag support.
Creates proper package structure to handle relative imports.
"""

import os
import sys
import json
import logging
import time
import wave
import tempfile
import shutil
from pathlib import Path
from typing import Optional

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
    """Step Audio EditX TTS generator with CPU support (slow but works)."""

    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device("cpu")
        self.model = None
        self.tokenizer = None
        self.sample_rate = config.get("voiceover_sample_rate", 44100)
        self.model_path = Path.home() / ".cache" / "huggingface" / "hub" / "models--stepfun-ai--Step-Audio-EditX"
        self._temp_package_dir = None

    def _create_package_structure(self) -> Path:
        """Create a temporary package structure to handle relative imports."""
        # Create a temporary directory for our package
        temp_dir = Path(tempfile.mkdtemp(prefix="step_audio_editx_"))
        self._temp_package_dir = temp_dir
        
        # Create the package structure: step_audio_editx/
        pkg_dir = temp_dir / "step_audio_editx"
        pkg_dir.mkdir(parents=True)
        
        # Create __init__.py
        (pkg_dir / "__init__.py").write_text("")
        
        # Copy modeling files to the package
        model_files = {
            "modeling_step1.py": self.model_path / "modeling_step1.py",
            "configuration_step1.py": self.model_path / "configuration_step1.py",
        }
        
        for name, src in model_files.items():
            if src.exists():
                dst = pkg_dir / name
                shutil.copy2(src, dst)
                logger.info(f"Copied {name} to package")
            else:
                logger.warning(f"File not found: {src}")
        
        # Fix relative imports in modeling_step1.py
        modeling_path = pkg_dir / "modeling_step1.py"
        if modeling_path.exists():
            content = modeling_path.read_text()
            # Replace relative imports with absolute imports
            content = content.replace(
                "from .configuration_step1 import Step1Config",
                "from step_audio_editx.configuration_step1 import Step1Config"
            )
            content = content.replace(
                "from .configuration_step1 import Step1Config",
                "from step_audio_editx.configuration_step1 import Step1Config"
            )
            modeling_path.write_text(content)
            logger.info("Fixed relative imports in modeling_step1.py")
        
        return temp_dir

    def load_model(self) -> bool:
        """Load Step Audio EditX model by creating proper package structure."""
        try:
            logger.info("Loading Step Audio EditX model (this may take several minutes on CPU)...")
            start_time = time.time()

            if not self.model_path.exists():
                logger.info("Model not found locally, downloading...")
                snapshot_download(
                    repo_id="stepfun-ai/Step-Audio-EditX",
                    local_dir=self.model_path,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                )

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                use_fast=False,
            )

            # Load config first
            config = AutoConfig.from_pretrained(
                self.model_path,
                trust_remote_code=True,
            )
            logger.info(f"Config class: {config.__class__.__name__}")
            logger.info(f"Config model_type: {getattr(config, 'model_type', 'unknown')}")

            # Create package structure to handle relative imports
            pkg_dir = self._create_package_structure()
            
            # Add package directory to sys.path
            sys.path.insert(0, str(pkg_dir))
            
            # Import the model class
            try:
                import importlib
                modeling_module = importlib.import_module("step_audio_editx.modeling_step1")
                
                # Find the model class
                model_class = None
                for attr_name in ["Step1ForCausalLM", "StepAudioEditXForCausalLM", "StepAudioForCausalLM"]:
                    model_class = getattr(modeling_module, attr_name, None)
                    if model_class:
                        logger.info(f"Found model class: {attr_name}")
                        break
                
                if model_class is None:
                    logger.error("Could not find model class in modeling_step1")
                    return False
                
                # Load model with the custom class
                self.model = model_class.from_pretrained(
                    self.model_path,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,  # Use float32 for CPU
                    low_cpu_mem_usage=True,
                    config=config,
                ).to(self.device)
                
            except Exception as e:
                logger.error(f"Failed to import model class: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return False

            self.model.eval()
            logger.info(f"Model loaded in {time.time() - start_time:.1f}s")
            return True
        except Exception as e:
            logger.error(f"Failed to load TTS model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
        finally:
            # Clean up temp directory
            if self._temp_package_dir and self._temp_package_dir.exists():
                try:
                    shutil.rmtree(self._temp_package_dir)
                except Exception:
                    pass

    def process_paralinguistic_tags(self, text: str) -> str:
        """Process paralinguistic tags for Step Audio EditX."""
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
            "[Uhm]": "<|Uhm|>",
            "[Confirmation-en]": "<|Confirmation-en|>",
            "[Surprise-oh]": "<|Surprise-oh|>",
            "[Surprise-ah]": "<|Surprise-ah|>",
            "[Surprise-wa]": "<|Surprise-wa|>",
            "[Dissatisfaction-hnn]": "<|Dissatisfaction-hnn|>",
            "[Question-ei]": "<|Question-ei|>",
            "[Question-ah]": "<|Question-ah|>",
            "[Question-en]": "<|Question-en|>",
            "[Question-yi]": "<|Question-yi|>",
            "[Question-oh]": "<|Question-oh|>",
        }
        processed = text
        for tag, token in tag_map.items():
            processed = processed.replace(tag, token)
        return processed

    def generate(self, text: str) -> Optional[Path]:
        """Generate voiceover audio from text."""
        # Ensure output directory exists early
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.output_dir}")

        # Try main model first
        if self.model is None or self.tokenizer is None:
            if not self.load_model():
                logger.error("Main model failed to load")
                return None

        try:
            logger.info("Generating voiceover with Step Audio EditX (CPU mode - this may take several minutes)...")
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