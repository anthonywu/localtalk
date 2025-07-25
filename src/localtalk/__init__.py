"""Local Talk App - A voice assistant that runs entirely offline."""

import os
import warnings

# Suppress the pkg_resources deprecation warning from perth module
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

# Suppress torch.backends.cuda.sdp_kernel deprecation warning
warnings.filterwarnings("ignore", message="torch.backends.cuda.sdp_kernel\\(\\) is deprecated", category=FutureWarning)

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Apply MLX compatibility patches early
try:
    import localtalk.utils.mlx_compat
except ImportError:
    pass

__version__ = "0.1.0"

from localtalk.core.assistant import VoiceAssistant
from localtalk.models.config import AppConfig

__all__ = ["VoiceAssistant", "AppConfig"]
