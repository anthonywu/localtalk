"""Local Talk App - A voice assistant that runs entirely offline."""

import os
import warnings


def _migrate_deprecated_hf_transfer_env() -> None:
    """Translate Hugging Face's retired transfer flag before it is imported.

    ``huggingface_hub`` 1.x no longer uses ``hf_transfer`` and emits a
    FutureWarning when the old flag is inherited from a shell profile. Xet is
    its supported replacement for high-performance transfers.
    """
    legacy_value = os.environ.pop("HF_HUB_ENABLE_HF_TRANSFER", None)
    if legacy_value and legacy_value.casefold() in {"1", "on", "true", "yes"}:
        os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")


_migrate_deprecated_hf_transfer_env()

# Suppress the pkg_resources deprecation warning from perth module
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

# Suppress torch.backends.cuda.sdp_kernel deprecation warning
warnings.filterwarnings("ignore", message="torch.backends.cuda.sdp_kernel\\(\\) is deprecated", category=FutureWarning)

# Note: mx.metal.device_info deprecation is handled in mlx_compat.py by
# redirecting to mx.device_info, since MLX prints it at the C++ level
# (bypassing Python's warnings module).

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Disable Hugging Face telemetry for offline privacy
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ.setdefault("TQDM_DISABLE", "1")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

from importlib.metadata import PackageNotFoundError, version  # noqa: E402

try:
    __version__ = version("localtalk")
except PackageNotFoundError:
    __version__ = "0.0.0-dev"


def __getattr__(name: str):  # noqa: ANN001
    """Lazy-load heavy imports to avoid pulling in the full ML stack on every import."""
    if name == "VoiceAssistant":
        from localtalk.core.assistant import VoiceAssistant

        return VoiceAssistant
    if name == "AppConfig":
        from localtalk.models.config import AppConfig

        return AppConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["AppConfig", "VoiceAssistant", "__version__"]
