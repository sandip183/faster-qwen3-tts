"""
Cog predictor for faster-qwen3-tts on Replicate.

Supports three generation modes depending on the inputs supplied:

  Voice Clone    — provide `ref_audio` + `ref_text`.  The model will mimic the
                   speaker in the reference audio while synthesising `text`.
                   Works with any *-Base model variant.

  Custom Voice   — provide `speaker` (e.g. "aiden").  Requires a
                   Qwen3-TTS-*-CustomVoice model variant (set MODEL_ID).

  Voice Design   — provide `voice_instruction` (e.g. "Warm British narrator").
                   Requires a Qwen3-TTS-*-VoiceDesign model variant (set MODEL_ID).

The model variant is chosen at *build time* via the MODEL_ID environment variable
(default: Qwen/Qwen3-TTS-12Hz-0.6B-Base).  Change it by editing cog.yaml or by
passing a different model_id when using the deploy-to-replicate workflow.

Environment variables (optional, set as Replicate deployment secrets):
  HF_TOKEN — Hugging Face token if the model weights are gated.
"""

from __future__ import annotations

import io
import os
import tempfile
from typing import Optional

import numpy as np
import soundfile as sf
from cog import BasePredictor, Input, Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"

# Speaker IDs available in the CustomVoice variants
CUSTOM_VOICE_SPEAKERS = [
    "aiden", "amber", "ashley", "bella", "brandon", "brianna",
    "carter", "claire", "cody", "crystal", "danielle", "derek",
    "dylan", "emily", "ethan", "eva", "felix", "fiona", "grace",
    "hunter", "iris", "jack", "jake", "jasmine", "jessica", "jordan",
    "julia", "justin", "kayla", "kevin", "kim", "kyle", "laura",
    "liam", "lily", "madison", "mason", "mia", "michael", "mila",
    "monica", "nathan", "nicole", "noah", "olivia", "paige", "parker",
    "ryan", "samantha", "sara", "savannah", "sean", "skylar", "sophia",
    "taylor", "tiffany", "tyler", "victoria", "zoe",
]

SUPPORTED_LANGUAGES = [
    "English", "Chinese", "Japanese", "Korean",
    "French", "German", "Spanish", "Italian",
    "Portuguese", "Arabic", "Russian", "Dutch",
    "Polish", "Swedish", "Turkish", "Hindi",
]


class Predictor(BasePredictor):
    """Cog predictor wrapping FasterQwen3TTS."""

    # ── Setup (called once when the container starts) ───────────────────────
    def setup(self) -> None:
        """Load model weights into GPU memory for fast repeated inference."""
        from faster_qwen3_tts import FasterQwen3TTS

        model_id = os.environ.get("MODEL_ID", DEFAULT_MODEL_ID)
        print(f"Loading model: {model_id}")

        hf_token = os.environ.get("HF_TOKEN") or None
        self.model = FasterQwen3TTS.from_pretrained(model_id, token=hf_token)
        self.model_id = model_id
        self._is_custom_voice = "CustomVoice" in model_id
        self._is_voice_design = "VoiceDesign" in model_id

        print(f"Model loaded successfully: {model_id}")

    # ── Predict ─────────────────────────────────────────────────────────────
    def predict(
        self,
        # ── Core inputs ──────────────────────────────────────────────────────
        text: str = Input(
            description="Text to synthesise into speech.",
            default="Hello! This is faster Qwen3 TTS running on Replicate.",
        ),
        language: str = Input(
            description="Language of the input text.",
            default="English",
            choices=SUPPORTED_LANGUAGES,
        ),
        # ── Voice Clone mode (requires a *-Base model) ───────────────────────
        ref_audio: Optional[Path] = Input(
            description=(
                "[Voice Clone mode] Reference WAV/MP3 audio file whose speaker "
                "identity the model will imitate.  5–30 seconds works best. "
                "Required together with `ref_text`."
            ),
            default=None,
        ),
        ref_text: Optional[str] = Input(
            description=(
                "[Voice Clone mode] Exact transcript of `ref_audio`. "
                "Accuracy here directly affects cloning quality."
            ),
            default=None,
        ),
        # ── Custom Voice mode (requires a *-CustomVoice model) ───────────────
        speaker: Optional[str] = Input(
            description=(
                "[Custom Voice mode] Speaker ID to use.  Only valid when the "
                "model was built with a *-CustomVoice variant.  "
                f"Available: {', '.join(CUSTOM_VOICE_SPEAKERS[:10])} …"
            ),
            default=None,
        ),
        # ── Voice Design mode (requires a *-VoiceDesign model) ───────────────
        voice_instruction: Optional[str] = Input(
            description=(
                "[Voice Design mode] Free-form instruction that describes the "
                "desired voice style, e.g. 'Warm, confident narrator with a "
                "slight British accent'.  Only valid when the model was built "
                "with a *-VoiceDesign variant."
            ),
            default=None,
        ),
        # ── Generation hyper-parameters ──────────────────────────────────────
        seed: int = Input(
            description="Random seed for reproducible outputs (-1 = random).",
            default=-1,
            ge=-1,
        ),
    ) -> Path:
        """
        Generate speech and return a WAV file.

        Mode is determined automatically from the supplied inputs:
          • ref_audio + ref_text  → Voice Clone
          • speaker               → Custom Voice  (requires CustomVoice model)
          • voice_instruction     → Voice Design   (requires VoiceDesign model)
          • none of the above     → raises ValueError with a clear message
        """
        import torch

        # ── Seed ─────────────────────────────────────────────────────────────
        if seed != -1:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        # ── Dispatch to the correct generation method ─────────────────────────
        if ref_audio is not None and ref_text:
            audio_chunks, sr = self._voice_clone(text, language, ref_audio, ref_text)

        elif speaker is not None:
            if not self._is_custom_voice:
                raise ValueError(
                    f"Speaker IDs require a CustomVoice model variant, but the "
                    f"loaded model is '{self.model_id}'.  Re-deploy with "
                    f"MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice."
                )
            audio_chunks, sr = self._custom_voice(text, language, speaker)

        elif voice_instruction is not None:
            if not self._is_voice_design:
                raise ValueError(
                    f"Voice-design instructions require a VoiceDesign model "
                    f"variant, but the loaded model is '{self.model_id}'.  "
                    f"Re-deploy with MODEL_ID=Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign."
                )
            audio_chunks, sr = self._voice_design(text, language, voice_instruction)

        else:
            raise ValueError(
                "You must supply at least one of: "
                "(1) ref_audio + ref_text for voice cloning, "
                "(2) speaker for custom-voice mode, or "
                "(3) voice_instruction for voice-design mode."
            )

        # ── Concatenate chunks and write WAV ──────────────────────────────────
        audio = np.concatenate(
            [c if isinstance(c, np.ndarray) else c.numpy() for c in audio_chunks],
            axis=-1,
        )
        if audio.ndim > 1:
            # (channels, samples) → (samples,) mono
            audio = audio.mean(axis=0)

        fd, tmp_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        output_path = Path(tmp_path)
        sf.write(str(output_path), audio, sr, subtype="PCM_16")
        return output_path

    # ── Private helpers ──────────────────────────────────────────────────────

    def _voice_clone(
        self,
        text: str,
        language: str,
        ref_audio: Path,
        ref_text: str,
    ):
        """Voice cloning — mimic the speaker in ref_audio."""
        audio_list, sr = self.model.generate_voice_clone(
            text=text,
            language=language,
            ref_audio=str(ref_audio),
            ref_text=ref_text,
        )
        return audio_list, sr

    def _custom_voice(self, text: str, language: str, speaker: str):
        """Generate speech with a predefined speaker ID (CustomVoice variants)."""
        audio_list, sr = self.model.generate_custom_voice(
            text=text,
            language=language,
            speaker=speaker,
        )
        return audio_list, sr

    def _voice_design(self, text: str, language: str, instruction: str):
        """Generate speech styled by a free-form instruction (VoiceDesign variants)."""
        audio_list, sr = self.model.generate_voice_design(
            text=text,
            language=language,
            instruct=instruction,
        )
        return audio_list, sr
