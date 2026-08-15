#!/usr/bin/env python3
"""
Caption Module - Moonshine Base (61M) for word-level transcription.
Generates CapCut/TikTok-style .ass subtitle file with active-word highlighting.
Uses AutoModelForSpeechSeq2Seq (requires transformers >= 4.49).
"""

import os
import sys
import json
import logging
import time
import wave
from pathlib import Path
from typing import List, Dict, Optional

import torch
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)


class CaptionGenerator:
    """Moonshine Base transcription with word-level timestamps for ASS generation."""

    MODEL_ID = "UsefulSensors/moonshine-base"

    def __init__(self, config: dict, output_dir: Path):
        self.config = config
        self.output_dir = output_dir
        self.device = torch.device("cpu")
        self.model = None
        self.processor = None
        self.sample_rate = 16000

    def load_model(self) -> bool:
        """Load Moonshine Base model via AutoModelForSpeechSeq2Seq."""
        try:
            logger.info(f"Loading {self.MODEL_ID} on CPU...")
            start_time = time.time()

            from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

            self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
            self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.MODEL_ID,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            ).to(self.device)
            self.model.eval()

            logger.info(f"Caption model loaded in {time.time() - start_time:.1f}s")
            return True
        except Exception as e:
            logger.error(f"Failed to load caption model: {e}")
            return False

    def transcribe(self, audio_path: Path) -> List[Dict]:
        """Transcribe audio with estimated word-level timestamps."""
        if self.model is None or self.processor is None:
            if not self.load_model():
                return []

        try:
            logger.info("Transcribing audio for captions...")
            start_time = time.time()

            import soundfile as sf
            audio, sr = sf.read(str(audio_path))
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if sr != self.sample_rate:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)

            # Prepare inputs
            inputs = self.processor(
                audio,
                sampling_rate=self.sample_rate,
                return_tensors="pt",
            ).to(self.device)

            # Generate transcription
            torch.manual_seed(42)
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=448,  # ~6.5 tokens/sec * max audio length
                )

            transcription = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()

            logger.info(f"Transcription: {transcription[:100]}...")

            # Moonshine doesn't provide word timestamps directly.
            # We estimate them by aligning the transcript to the audio duration.
            words = transcription.split()
            if not words:
                return []

            # Get audio duration
            audio_duration = len(audio) / self.sample_rate

            # Estimate word timings proportionally
            total_chars = sum(len(w) for w in words) + len(words) - 1  # +spaces
            words_with_times = []
            current_time = 0.0

            for i, word in enumerate(words):
                word_duration = (len(word) / total_chars) * audio_duration * 0.9  # 90% for words, 10% gaps
                gap = audio_duration * 0.1 / max(len(words), 1)

                words_with_times.append({
                    "word": word,
                    "start": max(0.0, current_time),
                    "end": min(audio_duration, current_time + word_duration),
                })
                current_time += word_duration + gap

            logger.info(f"Transcription complete in {time.time() - start_time:.1f}s "
                        f"({len(words_with_times)} words, {audio_duration:.2f}s audio)")
            return words_with_times

        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return []

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
