from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_file: Path | None = None, mode: str = "a") -> logging.Logger:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.insert(0, logging.FileHandler(log_file, encoding="utf-8", mode=mode))

    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=handlers)
    root.setLevel(logging.INFO)
    return root


def quiet_third_party_loggers() -> None:
    for name in ("httpx", "httpcore", "huggingface_hub", "transformers"):
        logging.getLogger(name).setLevel(logging.WARNING)

    try:
        from transformers.utils import logging as transformers_logging

        transformers_logging.set_verbosity_error()
    except Exception:
        pass
