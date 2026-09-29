"""Runtime configuration for ChemX."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def get_settings() -> "Settings":
    return Settings()


class Settings:
    def __init__(self) -> None:
        self.root = _ROOT
        self.env = os.getenv("CHEMX_ENV", "development").lower()
        self.log_level = os.getenv("CHEMX_LOG_LEVEL", "INFO").upper()

        self.groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        self.groq_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip()

        self.api_host = os.getenv("CHEMX_API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("PORT", os.getenv("CHEMX_API_PORT", "8000")))

        origins = os.getenv("CHEMX_CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501")
        self.cors_origins = [x.strip() for x in origins.split(",") if x.strip()]

        self.data_dir = _ROOT / os.getenv("CHEMX_DATA_DIR", "data")
        self.model_dir = _ROOT / os.getenv("CHEMX_MODEL_DIR", "models")
        self.reports_dir = _ROOT / os.getenv("CHEMX_REPORTS_DIR", "reports")
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        self.pls_components = int(os.getenv("CHEMX_PLS_COMPONENTS", "10"))
        self.pca_components = int(os.getenv("CHEMX_PCA_COMPONENTS", "5"))
        self.test_size = float(os.getenv("CHEMX_TEST_SIZE", "0.25"))
        self.random_seed = int(os.getenv("CHEMX_RANDOM_SEED", "42"))
        self.allow_runtime_training = os.getenv("CHEMX_ALLOW_RUNTIME_TRAINING", "false").lower() == "true"

        if self.is_production and "*" in self.cors_origins:
            raise ValueError("CHEMX_CORS_ORIGINS cannot contain '*' in production.")

    @property
    def is_production(self) -> bool:
        return self.env in {"production", "prod"}

    @property
    def groq_configured(self) -> bool:
        return bool(self.groq_api_key) and not self.groq_api_key.startswith("gsk_your")
