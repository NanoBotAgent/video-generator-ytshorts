#!/usr/bin/env python3
"""
Main Orchestrator - Master pipeline controller.
Executes all 5 steps sequentially with detailed timing metrics.
"""

import os
import sys
import json
import logging
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

PIPELINE_STEPS = [
    ("01_tts.py", "Voiceover Generation (TTS)"),
    ("02_bgm.py", "Background Music Generation"),
    ("03_captions.py", "Caption/Subtitle Generation"),
    ("04_visuals.py", "Visual Rendering"),
    ("05_assembly.py", "Final Video Assembly"),
]

SCRIPTS_DIR = Path("scripts")


def load_config() -> Dict:
    """Load configuration from config.json."""
    config_path = Path("config.json")
    if not config_path.exists():
        logger.error("config.json not found")
        sys.exit(1)
    with open(config_path) as f:
        return json.load(f)


def ensure_output_dir(config: Dict) -> Path:
    """Ensure output directory exists."""
    output_dir = Path(config.get("output_dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def run_step(script_name: str, description: str) -> Tuple[bool, float]:
    """Run a single pipeline step and return success status and duration."""
    script_path = SCRIPTS_DIR / script_name
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return False, 0.0

    logger.info(f"[STEP] Starting: {description}")
    start_time = time.time()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=1800,
            cwd=str(Path.cwd()),
        )

        elapsed = time.time() - start_time

        if result.returncode == 0:
            logger.info(f"[STEP] COMPLETE: {description} in {elapsed:.1f}s")
            # NOTE: this used to be logger.debug(), which is invisible because the
            # root logger is configured at INFO level - meaning every script's stdout
            # (which model ran, timings, warnings) was silently dropped on success and
            # only showed up if the step failed. Always surface it now.
            if result.stdout:
                logger.info("  --- step output ---")
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        logger.info(f"  {line}")
                logger.info("  --------------------")
            return True, elapsed
        else:
            logger.error(f"[STEP] FAILED: {description} after {elapsed:.1f}s")
            if result.stderr:
                for line in result.stderr.strip().split('\n'):
                    if line.strip():
                        logger.error(f"  {line}")
            if result.stdout:
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        logger.error(f"  {line}")
            return False, elapsed

    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        logger.error(f"[STEP] TIMEOUT: {description} after {elapsed:.1f}s")
        return False, elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"[STEP] ERROR: {description} after {elapsed:.1f}s: {e}")
        return False, elapsed


def verify_outputs(output_dir: Path) -> Dict[str, bool]:
    """Verify all expected output files exist."""
    expected = {
        "voiceover.wav": output_dir / "voiceover.wav",
        "bgm.wav": output_dir / "bgm.wav",
        "captions.ass": output_dir / "captions.ass",
        "visuals.mp4": output_dir / "visuals.mp4",
        "final_video.mp4": output_dir / "final_video.mp4",
    }

    results = {}
    for name, path in expected.items():
        exists = path.exists() and path.stat().st_size > 0
        results[name] = exists
        status = "OK" if exists else "MISSING"
        logger.info(f"  {name}: {status} ({path})")

    return results


def print_summary(step_times: List[Tuple[str, float]], success: bool) -> None:
    """Print pipeline execution summary."""
    total_time = sum(t for _, t in step_times)
    logger.info("=" * 60)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 60)
    for i, (desc, elapsed) in enumerate(step_times, 1):
        status = "✓" if elapsed > 0 else "✗"
        logger.info(f"  Step {i}: {desc:<40} {elapsed:>8.1f}s {status}")
    logger.info("-" * 60)
    logger.info(f"  TOTAL TIME: {total_time:.1f}s ({total_time/60:.1f} min)")
    logger.info(f"  STATUS: {'SUCCESS' if success else 'FAILED'}")
    logger.info("=" * 60)


def main() -> int:
    logger.info("=" * 60)
    logger.info("VIDEO GENERATION PIPELINE - Starting")
    logger.info("=" * 60)

    pipeline_start = time.time()

    config = load_config()
    output_dir = ensure_output_dir(config)

    logger.info(f"Config loaded: {config.get('script_text', '')[:50]}...")
    logger.info(f"Output directory: {output_dir.absolute()}")

    step_times = []
    all_success = True

    for i, (script_name, description) in enumerate(PIPELINE_STEPS, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"[STEP {i}/5] {description}")
        logger.info(f"{'='*60}")

        success, elapsed = run_step(script_name, description)
        step_times.append((description, elapsed))

        if not success:
            all_success = False
            logger.error(f"Pipeline halted at step {i}")
            break

    if all_success:
        logger.info("\nVerifying outputs...")
        verify_outputs(output_dir)

    total_elapsed = time.time() - pipeline_start
    step_times.append(("TOTAL", total_elapsed))

    print_summary(step_times[:-1], all_success)

    if all_success:
        final_video = output_dir / "final_video.mp4"
        if final_video.exists():
            size_mb = final_video.stat().st_size / (1024 * 1024)
            logger.info(f"\nFinal video: {final_video} ({size_mb:.1f} MB)")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())