"""Test suite for Local RAG Persona Simulator."""

import pytest
from pathlib import Path


class TestConfig:
    """Tests for configuration module."""

    def test_settings_defaults(self):
        """Test default settings values."""
        from local_rag_persona_simulator.config import Settings

        settings = Settings()
        assert settings.ollama_base_url == "http://localhost:11434"
        assert settings.ollama_model == "llama3.2"
        assert settings.embedding_model == "all-MiniLM-L6-v2"
        assert settings.chunk_size == 1000
        assert settings.chunk_overlap == 200

    def test_settings_paths(self):
        """Test path generation."""
        from local_rag_persona_simulator.config import Settings

        settings = Settings()
        assert settings.get_chroma_path() == Path("data/chroma_db")
        assert settings.get_persona_path() == Path("data/personas")
        assert settings.get_transcript_path() == Path("data/transcripts")


class TestTextUtils:
    """Tests for text utilities."""

    def test_clean_text(self):
        """Test text cleaning."""
        from local_rag_persona_simulator.utils import clean_text

        assert clean_text("  hello   world  ") == "hello world"
        assert clean_text("\n\ttab\n") == "tab"

    def test_truncate_text(self):
        """Test text truncation."""
        from local_rag_persona_simulator.utils import truncate_text

        assert truncate_text("hello world", 8) == "hello..."
        assert truncate_text("hi", 10) == "hi"

    def test_count_words(self):
        """Test word counting."""
        from local_rag_persona_simulator.utils import count_words

        assert count_words("hello world") == 2
        assert count_words("") == 0

    def test_sanitize_for_filename(self):
        """Test filename sanitization."""
        from local_rag_persona_simulator.utils import sanitize_for_filename

        assert sanitize_for_filename("hello<world>") == "hello_world"
        assert sanitize_for_filename("test:file") == "test_file"


class TestRAGPipeline:
    """Tests for RAG pipeline."""

    def test_rag_pipeline_init(self):
        """Test RAG pipeline initialization."""
        from local_rag_persona_simulator.core import RAGPipeline

        rag = RAGPipeline(persona_name="test_persona")
        assert rag.persona_name == "test_persona"
        assert rag.collection_name == "persona_test_persona"


class TestPersonaKnowledgeBase:
    """Tests for knowledge base."""

    def test_list_personas_empty(self):
        """Test listing personas when none exist."""
        from local_rag_persona_simulator.core import PersonaKnowledgeBase

        kb = PersonaKnowledgeBase()
        personas = kb.list_personas()
        assert isinstance(personas, list)


class TestTranscriptFetcher:
    """Tests for transcript fetcher."""

    def test_transcript_fetcher_init(self):
        """Test transcript fetcher initialization."""
        from local_rag_persona_simulator.core import TranscriptFetcher

        fetcher = TranscriptFetcher()
        assert fetcher.transcript_dir.exists()


class TestChatbot:
    """Tests for chatbot."""

    def test_chatbot_init(self):
        """Test chatbot initialization."""
        from local_rag_persona_simulator.core import PersonaChatbot

        chatbot = PersonaChatbot(persona_name="test")
        assert chatbot.persona_name == "test"
        assert chatbot.chat_history == []
