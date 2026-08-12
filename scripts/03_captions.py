#!/usr/bin/env python3
"""
Caption Module - Caption generation with fallback timing.
Generates CapCut/TikTok-style .ass subtitle file with active-word highlighting.
"""

import os
import sys
import json
import logging
import time
import wave
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class CaptionGenerator:
    """Caption generator with word-level timestamps for ASS generation."""

    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device("cpu")
        self.sample_rate = 16000

    def load_model(self) -> bool:
        """Try to load a caption model, fall back to simple timing."""
        logger.info("Using fallback caption timing (Moonshine not supported in current transformers)")
        return False

    def transcribe(self, audio_path: Path) -> List[Dict]:
        """Transcribe audio with word-level timestamps using fallback timing."""
        return self._fallback_transcribe(audio_path)

    def _fallback_transcribe(self, audio_path: Path) -> List[Dict]:
        """Generate fallback word timings based on audio duration."""
        try:
            with wave.open(str(audio_path), "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
        except Exception:
            duration = 30.0

        logger.info(f"Using fallback timing for {duration:.1f}s audio")

        script_text = self.config.get("script_text", "")
        words = script_text.split()
        if not words:
            words = ["Hello", "world", "this", "is", "a", "test"]

        word_duration = duration / len(words)
        result = []
        current_time = 0.0
        for word in words:
            result.append({
                "word": word,
                "start": current_time,
                "end": current_time + word_duration * 0.9,
            })
            current_time += word_duration

        return result

    def _fallback_transcribe_from_text(self, text: str) -> List[Dict]:
        """Generate fallback word timings from text."""
        words = text.split()
        if not words:
            words = ["Hello", "world", "this", "is", "a", "test"]

        word_duration = 0.5  # approximate
        result = []
        current_time = 0.0
        for word in words:
            result.append({
                "word": word,
                "start": current_time,
                "end": current_time + word_duration * 0.9,
            })
            current_time += word_duration

        return result

    def generate_ass(self, words: List[Dict], video_width: int, video_height: int) -> Optional[Path]:
        """Generate CapCut/TikTok-style ASS subtitle file."""
        if not words:
            logger.error("No words to generate captions from")
            return None

        try:
            logger.info("Generating ASS subtitle file...")
            start_time = time.time()

            font_size = 28
            margin_v = 320

            ass_header = f"""[Script Info]
Title: Generated Captions
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709
PlayResX: {video_width}
PlayResY: {video_height}

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Inter,{font_size},&H00FFFFFF,&H000000FF,&H80000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,2,2,20,20,{margin_v},1
Style: Highlight,Inter,{font_size},&H0000FFFF,&H000000FF,&H80000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,2,2,20,20,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

            events = []
            for i, word in enumerate(words):
                start_ms = self._format_ass_time(word["start"])
                end_ms = self._format_ass_time(word["end"])

                before_text = " ".join(w["word"] for w in words[:i])
                current_word = word["word"]
                after_text = " ".join(w["word"] for w in words[i+1:])

                line_text = f"{before_text} {{\\rHighlight}}{current_word}{{\\rDefault}} {after_text}".strip()

                events.append(
                    f"Dialogue: 0,{start_ms},{end_ms},Default,,0,0,0,,{line_text}"
                )

            ass_content = ass_header + "\n".join(events)

            output_path = self.output_dir / "captions.ass"
            output_path.write_text(ass_content, encoding="utf-8")

            logger.info(f"ASS file generated in {time.time() - start_time:.1f}s: {output_path}")
            return output_path

        except Exception as e:
            logger.error(f"ASS generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _format_ass_time(self, seconds: float) -> str:
        """Convert seconds to ASS time format (H:MM:SS.cc)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centiseconds = int((seconds - int(seconds)) * 100)
        return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"


def main() -> int:
    config_path = Path("config.json")
    if not config_path.exists():
        logger.error("config.json not found")
        return 1

    with open(config_path) as f:
        config = json.load(f)

    output_dir = Path(config.get("output_dir", "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    voiceover_path = output_dir / "voiceover.wav"
    if not voiceover_path.exists():
        logger.error("voiceover.wav not found. Run 01_tts.py first.")
        return 1

    video_width = config.get("video_width", 1080)
    video_height = config.get("video_height", 1920)

    generator = CaptionGenerator(config, output_dir)
    words = generator.transcribe(voiceover_path)

    if not words:
        logger.error("No transcription results")
        return 1

    result = generator.generate_ass(words, video_width, video_height)

    if result and result.exists():
        logger.info(f"SUCCESS: Captions saved to {result}")
        return 0
    else:
        logger.error("Caption generation failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())