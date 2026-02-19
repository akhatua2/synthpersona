"""Configuration via pydantic-settings, loaded from .env."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Vertex AI
    vertexai_project: str = "soe-gemini-llm-agents"
    vertexai_location: str = "global"

    # LLM model names
    fast_model: str = "vertex_ai/gemini-3-flash-preview"
    smart_model: str = "vertex_ai/gemini-3-pro-preview"

    # Persona generation
    population_size: int = 25

    # Evolution
    num_islands: int = 10
    evolution_iterations: int = 500
    extinction_interval: int = 100
    extinction_bottom_pct: float = 0.30
    extinction_top_pct: float = 0.30

    # Metrics
    mc_samples: int = 10_000
    calibration_trials: int = 1_000
    coverage_target: float = 0.99

    # LLM concurrency
    max_concurrency: int = 10
    simulation_temperature: float = 0.0


def get_settings() -> Settings:
    return Settings()
