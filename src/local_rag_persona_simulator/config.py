"""Local RAG Persona Simulator - Configuration Management."""

from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with pydantic validation."""

    ollama_base_url: str = Field(
        default="http://localhost:11434", description="Base URL for Ollama API"
    )
    ollama_model: str = Field(default="llama3.2", description="Ollama model to use for chat")
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2", description="Sentence-transformers model for embeddings"
    )
    chroma_persist_directory: str = Field(
        default="data/chroma_db", description="Directory for ChromaDB persistence"
    )
    chunk_size: int = Field(default=1000, description="Chunk size for text splitting")
    chunk_overlap: int = Field(default=200, description="Overlap between chunks")
    retrieval_top_k: int = Field(
        default=5, description="Number of documents to retrieve for context"
    )
    persona_directory: str = Field(
        default="data/personas", description="Directory to store persona data"
    )
    transcript_directory: str = Field(
        default="data/transcripts", description="Directory to store fetched transcripts"
    )

    model_config = SettingsConfigDict(
        env_prefix="RAG_PERSONA_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    def get_chroma_path(self) -> Path:
        """Get the ChromaDB persistence path."""
        return Path(self.chroma_persist_directory)

    def get_persona_path(self) -> Path:
        """Get the persona directory path."""
        return Path(self.persona_directory)

    def get_transcript_path(self) -> Path:
        """Get the transcript directory path."""
        return Path(self.transcript_directory)

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        self.get_chroma_path().mkdir(parents=True, exist_ok=True)
        self.get_persona_path().mkdir(parents=True, exist_ok=True)
        self.get_transcript_path().mkdir(parents=True, exist_ok=True)


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get cached settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
        _settings.ensure_directories()
    return _settings


def reload_settings() -> Settings:
    """Reload settings (useful for testing)."""
    global _settings
    _settings = Settings()
    _settings.ensure_directories()
    return _settings
