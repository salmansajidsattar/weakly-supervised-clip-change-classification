# Lightweight package init. seed.py pulls in torch and numpy; logging/config
# are stdlib + pyyaml. Lazy-import seed_everything so label-building scripts
# work without a torch install.

from .config import load_config
from .logging import setup_logger

__all__ = ["load_config", "setup_logger", "seed_everything"]


def __getattr__(name):
    if name == "seed_everything":
        from .seed import seed_everything
        return seed_everything
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
