#!/usr/bin/env python3
"""
Initialization script for video generation pipeline.
Downloads and caches all required models to avoid repeated downloads in CI.
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor, AutoModelForSpeechSeq2Seq, MusicgenForConditionalGeneration

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

MODEL_CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODELS_TO_DOWNLOAD = {
    "tts": {
        "repo_id": "stepfun-ai/Step-Audio-EditX",
        "revision": "main",
        "local_dir": MODEL_CACHE_DIR / "models--stepfun-ai--Step-Audio-EditX",
    },
    "bgm": {
        "repo_id": "facebook/musicgen-small",
        "revision": "main",
        "local_dir": MODEL_CACHE_DIR / "models--facebook--musicgen-small",
    },
    "captions": {
        "repo_id": "UsefulSensors/moonshine-base",
        "revision": "main",
        "local_dir": MODEL_CACHE_DIR / "models--UsefulSensors--moonshine-base",
    },
}


def check_disk_space(path: Path, required_gb: float = 20.0) -> bool:
    """Check if there's enough disk space for model downloads."""
    try:
        import shutil
        total, used, free = shutil.disk_usage(path)
        free_gb = free / (1024 ** 3)
        logger.info(f"Disk space: {free_gb:.1f} GB free of {total / (1024**3):.1f} GB total")
        return free_gb >= required_gb
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return True


def download_model(repo_id: str, revision: str, local_dir: Path) -> bool:
    """Download a model from Hugging Face Hub with progress tracking."""
    try:
        logger.info(f"Downloading {repo_id}...")
        snapshot_download(
            repo_id=repo_id,
            revision=revision,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4,
        )
        logger.info(f"Successfully downloaded {repo_id} to {local_dir}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {repo_id}: {e}")
        return False


def verify_model(local_dir: Path) -> bool:
    """Verify model files exist."""
    try:
        if not local_dir.exists():
            return False
        files = list(local_dir.rglob("*.bin")) + list(local_dir.rglob("*.safetensors")) + list(local_dir.rglob("config.json"))
        if not files:
            logger.warning(f"No model weights found in {local_dir}")
            return False
        logger.info(f"Verified {local_dir}: {len(files)} weight/config files found")
        return True
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return False


def setup_torch_environment() -> None:
    """Configure PyTorch for optimal CPU performance."""
    torch.set_num_threads(min(4, os.cpu_count() or 4))
    torch.set_num_interop_threads(2)
    os.environ["OMP_NUM_THREADS"] = str(min(4, os.cpu_count() or 4))
    os.environ["MKL_NUM_THREADS"] = str(min(4, os.cpu_count() or 4))
    logger.info(f"PyTorch threads: {torch.get_num_threads()}, Interop: {torch.get_num_interop_threads()}")


def main() -> int:
    logger.info("=" * 60)
    logger.info("Video Generation Pipeline - Model Initialization")
    logger.info("=" * 60)

    setup_torch_environment()

    if not check_disk_space(MODEL_CACHE_DIR.parent):
        logger.error("Insufficient disk space for model downloads")
        return 1

    success_count = 0
    for name, config in MODELS_TO_DOWNLOAD.items():
        local_dir = Path(config["local_dir"])
        if verify_model(local_dir):
            logger.info(f"Model {name} already cached, skipping download")
            success_count += 1
            continue

        if download_model(config["repo_id"], config["revision"], local_dir):
            if verify_model(local_dir):
                success_count += 1
            else:
                logger.error(f"Model {name} downloaded but verification failed")
        else:
            logger.error(f"Model {name} download failed")

    logger.info("=" * 60)
    logger.info(f"Initialization complete: {success_count}/{len(MODELS_TO_DOWNLOAD)} models ready")
    logger.info("=" * 60)

    return 0 if success_count == len(MODELS_TO_DOWNLOAD) else 1


if __name__ == "__main__":
    sys.exit(main())