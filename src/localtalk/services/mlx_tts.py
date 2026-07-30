"""Text-to-speech service using mlx-audio."""

import contextlib
import io
import sys

import numpy as np
from rich.console import Console

from localtalk.models.config import ChatterBoxConfig

# chatterbox_turbo submodules that render tqdm progress bars during generation.
_CHATTERBOX_NOISY_MODULES = (
    "mlx_audio.tts.models.chatterbox_turbo.models.t3.t3",
    "mlx_audio.tts.models.chatterbox_turbo.models.s3gen.flow_matching",
)


def _quiet_tqdm(iterable, *args, **kwargs):
    """Drop-in tqdm replacement that yields the iterable without rendering a bar."""
    return iterable


def _silence_chatterbox_output():
    """Monkey-patch tqdm out of chatterbox_turbo's generation loops.

    mlx-audio's chatterbox_turbo renders tqdm progress bars to stderr while
    generating speech tokens and running the S3 mel decoder. Both modules bind
    ``tqdm`` at import time, so replacing the module attribute silences the
    bars without affecting tqdm anywhere else. Only patches modules that are
    already imported (guaranteed right after model load); never imports them.
    """
    for module_name in _CHATTERBOX_NOISY_MODULES:
        module = sys.modules.get(module_name)
        if module is not None:
            module.tqdm = _quiet_tqdm


class MLXTextToSpeechService:
    """Service for converting text to speech using mlx-audio TTS models."""

    def __init__(self, config: ChatterBoxConfig, console: Console | None = None):
        self.config = config
        self.console = console or Console()
        self.model_id = config.model_id
        self.model = self._load_model()
        self.sample_rate = self.model.sample_rate

    def _load_model(self):
        """Load the mlx-audio TTS model."""
        from mlx_audio.tts.utils import load_model

        self.console.print(f"[cyan]Loading TTS model: {self.model_id}[/cyan]")
        model = load_model(model_path=self.model_id)
        # The chatterbox modules are imported as a side effect of load_model.
        _silence_chatterbox_output()
        return model

    @staticmethod
    def _to_numpy_audio(audio) -> np.ndarray:
        """Convert MLX or other audio array to float32 numpy."""
        if hasattr(audio, "tolist"):
            return np.array(audio.tolist(), dtype=np.float32)
        if not isinstance(audio, np.ndarray):
            return np.array(audio, dtype=np.float32)
        return audio.astype(np.float32)

    def synthesize(self, text: str) -> tuple[int, np.ndarray]:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize

        Returns:
            Tuple of (sample_rate, audio_array)

        """
        # chatterbox prints chatter like "S3 Token -> Mel Inference..." to
        # stdout during generation; keep the console clean.
        with contextlib.redirect_stdout(io.StringIO()):
            results = list(self.model.generate(text=text, verbose=False))
        if not results:
            return self.sample_rate, np.array([], dtype=np.float32)

        audio = self._to_numpy_audio(results[0].audio)

        return self.sample_rate, audio

    def synthesize_long_form(self, text: str) -> tuple[int, np.ndarray]:
        """Synthesize long-form text.

        The model handles sentence splitting internally via its generate() method.

        Args:
            text: Long text to synthesize

        Returns:
            Tuple of (sample_rate, audio_array)

        """
        pieces = []
        silence = np.zeros(
            int(self.config.silence_between_pieces_ms / 1000 * self.sample_rate),
            dtype=np.float32,
        )

        # See synthesize(): suppress chatterbox's stray prints during generation.
        with contextlib.redirect_stdout(io.StringIO()):
            for result in self.model.generate(text=text, verbose=False):
                audio = self._to_numpy_audio(result.audio)
                pieces.extend([audio, silence.copy()])

        if not pieces:
            return self.sample_rate, np.array([], dtype=np.float32)

        return self.sample_rate, np.concatenate(pieces)
