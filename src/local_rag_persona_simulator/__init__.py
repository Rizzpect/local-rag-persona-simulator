"""Local RAG Persona Simulator package."""

__version__ = "0.1.0"

from .config import get_settings, reload_settings, Settings
from .core.chatbot import PersonaChatbot, Persona, create_persona_interactive
from .core.rag import PersonaKnowledgeBase, RAGPipeline
from .core.transcript import TranscriptFetcher
from .utils.text_utils import (
    clean_text,
    truncate_text,
    extract_sentences,
    count_words,
    sanitize_for_filename,
)
