"""FastAPI server configuration."""

import dataclasses
import logging
import logging.config
import os
from pathlib import Path

import dotenv
from singleton import Singleton

dotenv.load_dotenv()
base_dir = Path(__file__).resolve().parent.parent


@dataclasses.dataclass
class Settings(metaclass=Singleton):
    """Server config settings."""

    root_url: str = os.getenv("DOMAIN", default="http://localhost:8000")

    testing: bool = os.getenv("TESTING", default=False)

    llm_model: str = os.getenv("LLM_MODEL", default="phi-2")
    phi_tensorrt_path: str = os.getenv("PHI_TENSORRT_PATH")
    phi_tokenizer_path: str = Path(phi_tensorrt_path) / "tokenizer"
    whisper_tensorrt_path: str = os.getenv("WHISPER_TENSORRT_PATH")

    log_config = {
        "version": 1,
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "INFO",
                "filename": base_dir / "logs" / "info.log",
                "formatter": "standard",
            },
        },
        "formatters": {
            "standard": {
                "format": "[{levelname} : {filename}:{lineno} : {asctime} -> {funcName:10}] {message}",
                # "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                "style": "{",
            }
        },
        "loggers": {
            "": {
                "handlers": [
                    "console",
                    "file",
                ],
                "level": "INFO",
                "propagate": True,
            }
        },
    }

    def config_logger(self):
        if not (base_dir / "logs").exists():
            (base_dir / "logs").mkdir()

        logging.config.dictConfig(self.log_config)
