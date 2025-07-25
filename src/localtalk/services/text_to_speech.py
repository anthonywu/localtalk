"""Text-to-speech service using ChatterBox."""

import warnings
from pathlib import Path

import nltk
import numpy as np
import torch
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS
from rich.console import Console

from localtalk.models.config import ChatterBoxConfig

warnings.filterwarnings(
    "ignore",
    message="torch.nn.utils.weight_norm is deprecated in favor of torch.nn.utils.parametrizations.weight_norm.",
)


class TextToSpeechService:
    """Service for converting text to speech using ChatterBox."""

    def __init__(self, config: ChatterBoxConfig, console: Console | None = None):
        self.config = config
        self.console = console or Console()
        self.device = self._get_device()
        self._patch_torch_load()
        self.model = self._load_model()
        self.sample_rate = self.model.sr

        # Ensure NLTK data is available
        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)

    def _get_device(self) -> str:
        """Determine the device to use."""
        if self.config.device:
            return self.config.device

        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _patch_torch_load(self):
        """Patch torch.load for device compatibility."""
        map_location = torch.device(self.device)

        if not hasattr(torch, "_original_load"):
            torch._original_load = torch.load

        def patched_torch_load(*args, **kwargs):
            if "map_location" not in kwargs:
                kwargs["map_location"] = map_location
            return torch._original_load(*args, **kwargs)

        torch.load = patched_torch_load

    def _load_model(self) -> ChatterboxTTS:
        """Load the ChatterBox model."""
        self.console.print(f"[cyan]Loading ChatterBox model on {self.device}")

        # Patch the model to use eager attention to avoid warnings
        self._patch_attention_implementation()

        # Suppress tqdm progress bars
        self._suppress_tqdm()

        return ChatterboxTTS.from_pretrained(device=self.device)

    def _patch_attention_implementation(self):
        """Patch transformers to use eager attention implementation."""
        import transformers

        # Store the original LlamaModel
        if not hasattr(transformers, '_original_LlamaModel'):
            transformers._original_LlamaModel = transformers.LlamaModel

            # Create a patched version
            class PatchedLlamaModel(transformers._original_LlamaModel):
                def __init__(self, config):
                    # Set attention implementation to eager
                    config._attn_implementation = "eager"
                    super().__init__(config)

            # Replace the original with the patched version
            transformers.LlamaModel = PatchedLlamaModel

    def _suppress_tqdm(self):
        """Suppress tqdm progress bars by redirecting to null."""
        import os

        from tqdm import tqdm

        # Disable tqdm globally
        tqdm.disable = True

        # Alternative: redirect tqdm to null
        os.environ['TQDM_DISABLE'] = '1'

    def synthesize(
        self,
        text: str,
        audio_prompt_path: Path | None = None,
        exaggeration: float | None = None,
        cfg_weight: float | None = None,
    ) -> tuple[int, np.ndarray]:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize
            audio_prompt_path: Optional path to voice sample for cloning
            exaggeration: Optional emotion exaggeration override
            cfg_weight: Optional CFG weight override

        Returns:
            Tuple of (sample_rate, audio_array)
        """
        audio_prompt_path = audio_prompt_path or self.config.voice_sample_path
        exaggeration = exaggeration if exaggeration is not None else self.config.exaggeration
        cfg_weight = cfg_weight if cfg_weight is not None else self.config.cfg_weight

        with self.console.status("Generating speech...", spinner="dots"):
            wav = self.model.generate(
                text,
                audio_prompt_path=str(audio_prompt_path) if audio_prompt_path else None,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )

        audio_array = wav.squeeze().cpu().numpy()
        return self.sample_rate, audio_array

    def synthesize_long_form(
        self,
        text: str,
        audio_prompt_path: Path | None = None,
        exaggeration: float | None = None,
        cfg_weight: float | None = None,
    ) -> tuple[int, np.ndarray]:
        """Synthesize long-form text by splitting into sentences.

        Args:
            text: Long text to synthesize
            audio_prompt_path: Optional path to voice sample for cloning
            exaggeration: Optional emotion exaggeration override
            cfg_weight: Optional CFG weight override

        Returns:
            Tuple of (sample_rate, audio_array)
        """
        pieces = []
        sentences = nltk.sent_tokenize(text)
        silence = np.zeros(int(0.25 * self.sample_rate))

        for sent in sentences:
            _, audio_array = self.synthesize(
                sent,
                audio_prompt_path=audio_prompt_path,
                exaggeration=exaggeration,
                cfg_weight=cfg_weight,
            )
            pieces.extend([audio_array, silence.copy()])

        return self.sample_rate, np.concatenate(pieces)

    def save_voice_sample(
        self,
        text: str,
        output_path: Path,
        audio_prompt_path: Path | None = None,
    ):
        """Save a voice sample to file.

        Args:
            text: Text to synthesize
            output_path: Path to save the audio file
            audio_prompt_path: Optional path to voice sample for cloning
        """
        audio_prompt_path = audio_prompt_path or self.config.voice_sample_path

        wav = self.model.generate(
            text,
            audio_prompt_path=str(audio_prompt_path) if audio_prompt_path else None,
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        ta.save(str(output_path), wav, self.sample_rate)
        self.console.print(f"[dim]Voice sample saved to: {output_path}[/dim]")
