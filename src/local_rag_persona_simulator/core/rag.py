"""RAG Pipeline with local embeddings and ChromaDB."""

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

from ..config import get_settings


class RAGPipeline:
    """Retrieval Augmented Generation pipeline using local embeddings."""

    def __init__(
        self,
        persona_name: str,
        collection_name: Optional[str] = None,
    ) -> None:
        """
        Initialize the RAG pipeline.

        Args:
            persona_name: Name of the persona
            collection_name: Optional ChromaDB collection name
        """
        self.settings = get_settings()
        self.persona_name = persona_name
        self.collection_name = (
            collection_name or f"persona_{persona_name.replace(' ', '_').lower()}"
        )
        self._vectorstore: Optional[Chroma] = None
        self._embeddings: Optional[SentenceTransformerEmbeddings] = None

    @property
    def embeddings(self) -> SentenceTransformerEmbeddings:
        """Get or create embeddings model."""
        if self._embeddings is None:
            self._embeddings = SentenceTransformerEmbeddings(
                model_name=self.settings.embedding_model
            )
        return self._embeddings

    @property
    def vectorstore(self) -> Chroma:
        """Get or create vectorstore."""
        if self._vectorstore is None:
            self._vectorstore = self._create_vectorstore()
        return self._vectorstore

    def _create_vectorstore(self) -> Chroma:
        """Create a new ChromaDB vectorstore."""
        persist_dir = str(self.settings.get_chroma_path() / self.collection_name)

        return Chroma(
            client=chromadb.PersistentClient(
                path=persist_dir,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            ),
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
        )

    def add_transcript(
        self,
        transcript_path: str | Path,
        source_name: Optional[str] = None,
    ) -> int:
        """
        Add a transcript to the knowledge base.

        Args:
            transcript_path: Path to the transcript file
            source_name: Optional name for the source

        Returns:
            Number of chunks added
        """
        transcript_path = Path(transcript_path)
        if not transcript_path.exists():
            raise FileNotFoundError(f"Transcript file not found: {transcript_path}")

        text = transcript_path.read_text(encoding="utf-8")
        source = source_name or transcript_path.stem

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            length_function=len,
        )

        chunks = text_splitter.split_text(text)

        metadatas = [
            {"source": source, "chunk_id": i, "persona": self.persona_name}
            for i in range(len(chunks))
        ]

        self.vectorstore.add_texts(texts=chunks, metadatas=metadatas)

        return len(chunks)

    def add_text(
        self,
        text: str,
        source: str = "manual",
    ) -> int:
        """
        Add raw text to the knowledge base.

        Args:
            text: Text to add
            source: Source identifier

        Returns:
            Number of chunks added
        """
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            length_function=len,
        )

        chunks = text_splitter.split_text(text)

        metadatas = [
            {"source": source, "chunk_id": i, "persona": self.persona_name}
            for i in range(len(chunks))
        ]

        self.vectorstore.add_texts(texts=chunks, metadatas=metadatas)

        return len(chunks)

    def similarity_search(
        self,
        query: str,
        k: Optional[int] = None,
    ) -> list[dict]:
        """
        Perform similarity search.

        Args:
            query: Query string
            k: Number of results to return

        Returns:
            List of documents with metadata
        """
        k = k or self.settings.retrieval_top_k

        docs = self.vectorstore.similarity_search_with_score(
            query=query,
            k=k,
        )

        results = []
        for doc, score in docs:
            results.append(
                {
                    "content": doc.page_content,
                    "metadata": doc.metadata,
                    "score": score,
                }
            )

        return results

    def get_relevant_context(self, query: str, k: Optional[int] = None) -> str:
        """
        Get relevant context for a query.

        Args:
            query: Query string
            k: Number of chunks to retrieve

        Returns:
            Combined context string
        """
        results = self.similarity_search(query, k)

        if not results:
            return "No relevant context found."

        context_parts = [
            f"[Source: {r['metadata'].get('source', 'unknown')}]\n{r['content']}" for r in results
        ]

        return "\n\n---\n\n".join(context_parts)

    def get_collection_stats(self) -> dict:
        """Get statistics about the collection."""
        try:
            count = self.vectorstore._collection.count()
            return {
                "persona": self.persona_name,
                "collection_name": self.collection_name,
                "document_count": count,
            }
        except Exception:
            return {
                "persona": self.persona_name,
                "collection_name": self.collection_name,
                "document_count": 0,
            }

    def reset(self) -> None:
        """Reset the vectorstore."""
        if self._vectorstore is not None:
            self._vectorstore._client.delete_collection(self.collection_name)
            self._vectorstore = None


class PersonaKnowledgeBase:
    """Manages multiple persona knowledge bases."""

    def __init__(self) -> None:
        """Initialize the knowledge base manager."""
        self.settings = get_settings()

    def list_personas(self) -> list[str]:
        """List all available personas."""
        chroma_path = self.settings.get_chroma_path()
        if not chroma_path.exists():
            return []

        personas = []
        for item in chroma_path.iterdir():
            if item.is_dir() and item.name.startswith("persona_"):
                persona_name = item.name.replace("persona_", "").replace("_", " ")
                personas.append(persona_name)

        return personas

    def delete_persona(self, persona_name: str) -> bool:
        """Delete a persona's knowledge base."""
        collection_name = f"persona_{persona_name.replace(' ', '_').lower()}"

        try:
            client = chromadb.PersistentClient(
                path=str(self.settings.get_chroma_path()),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            client.delete_collection(collection_name)

            persona_dir = (
                self.settings.get_persona_path() / f"{persona_name.replace(' ', '_')}.json"
            )
            if persona_dir.exists():
                persona_dir.unlink()

            return True
        except Exception:
            return False

    def get_persona_pipeline(self, persona_name: str) -> RAGPipeline:
        """Get a RAG pipeline for a persona."""
        return RAGPipeline(persona_name=persona_name)

    def persona_exists(self, persona_name: str) -> bool:
        """Check if a persona exists."""
        collection_name = f"persona_{persona_name.replace(' ', '_').lower()}"

        try:
            client = chromadb.PersistentClient(
                path=str(self.settings.get_chroma_path()),
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            collections = client.list_collections()
            return collection_name in [c.name for c in collections]
        except Exception:
            return False
