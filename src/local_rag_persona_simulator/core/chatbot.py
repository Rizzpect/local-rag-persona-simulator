"""Persona Chatbot with Ollama integration."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import ollama
import requests

from ..config import get_settings
from .rag import PersonaKnowledgeBase, RAGPipeline


@dataclass
class ChatMessage:
    """Represents a chat message."""

    role: str
    content: str
    timestamp: float = field(default_factory=lambda: __import__("time").time())


@dataclass
class Persona:
    """Represents a persona configuration."""

    name: str
    description: str
    system_prompt: str
    sources: list[str]
    created_at: float
    last_interaction: float


class PersonaChatbot:
    """Chatbot that simulates a persona using RAG and Ollama."""

    def __init__(
        self,
        persona_name: str,
        model: Optional[str] = None,
    ) -> None:
        """
        Initialize the persona chatbot.

        Args:
            persona_name: Name of the persona
            model: Optional Ollama model override
        """
        self.settings = get_settings()
        self.persona_name = persona_name
        self.model = model or self.settings.ollama_model
        self.rag = RAGPipeline(persona_name=persona_name)
        self.knowledge_base = PersonaKnowledgeBase()
        self.persona = self._load_persona()
        self.chat_history: list[ChatMessage] = []

    def _get_persona_path(self) -> Path:
        """Get the path to the persona file."""
        safe_name = self.persona_name.replace(" ", "_")
        return self.settings.get_persona_path() / f"{safe_name}.json"

    def _load_persona(self) -> Optional[Persona]:
        """Load persona configuration."""
        persona_path = self._get_persona_path()
        if not persona_path.exists():
            return None

        try:
            data = json.loads(persona_path.read_text(encoding="utf-8"))
            return Persona(
                name=data["name"],
                description=data.get("description", ""),
                system_prompt=data["system_prompt"],
                sources=data.get("sources", []),
                created_at=data.get("created_at", 0),
                last_interaction=data.get("last_interaction", 0),
            )
        except Exception:
            return None

    def _save_persona(self) -> None:
        """Save persona configuration."""
        if self.persona is None:
            return

        persona_path = self._get_persona_path()
        persona_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "name": self.persona.name,
            "description": self.persona.description,
            "system_prompt": self.persona.system_prompt,
            "sources": self.persona.sources,
            "created_at": self.persona.created_at,
            "last_interaction": self.persona.last_interaction,
        }

        persona_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _build_system_prompt(self) -> str:
        """Build the system prompt for the persona."""
        if self.persona is None:
            return (
                f"You are a helpful AI assistant. Your name is {self.persona_name}. "
                "Answer questions based on your knowledge base."
            )

        base_prompt = f"""You are {self.persona.name}.

{self.persona.description}

---

IMPORTANT INSTRUCTIONS:
1. You must respond AS {self.persona.name}, using their knowledge, perspective, and communication style.
2. Only use information from the provided context to answer questions about {self.persona.name}'s expertise.
3. If you're asked about something not in the context, be honest that you don't have that information.
4. Stay in character as {self.persona.name} at all times.
5. Your responses should reflect {self.persona.name}'s knowledge and personality based on the transcripts provided.

---

Context from transcripts:
{self.rag.get_relevant_context(" ".join([m.content for m in self.chat_history[-3:]]))}

Now, remember: You are {self.persona.name}. Respond accordingly."""

        return base_prompt

    def _build_prompt(self, user_message: str) -> str:
        """Build the full prompt with context and history."""
        context = self.rag.get_relevant_context(user_message)

        history_text = ""
        if self.chat_history:
            history_messages = self.chat_history[-10:]
            history_text = "\n".join(
                f"{msg.role.capitalize()}: {msg.content}" for msg in history_messages
            )

        prompt = f"""Based on the following context from {self.persona.name}'s transcripts, answer the user's question.

Context:
{context}

{"Previous conversation:\n" + history_text if history_text else ""}

User: {user_message}

Remember: You are {self.persona.name}. Answer in their voice and style using the provided context."""

        return prompt

    def check_ollama_connection(self) -> tuple[bool, str]:
        """
        Check if Ollama is running and accessible.

        Returns:
            Tuple of (is_connected, message)
        """
        try:
            response = requests.get(
                f"{self.settings.ollama_base_url}/api/tags",
                timeout=5,
            )
            if response.status_code == 200:
                return True, "Connected to Ollama"
            return False, f"Ollama returned status {response.status_code}"
        except requests.exceptions.ConnectionError:
            return False, "Cannot connect to Ollama. Is it running?"
        except requests.exceptions.Timeout:
            return False, "Connection to Ollama timed out"
        except Exception as e:
            return False, f"Error connecting to Ollama: {str(e)}"

    def list_available_models(self) -> list[dict[str, Any]]:
        """
        List available Ollama models.

        Returns:
            List of model information dictionaries
        """
        try:
            response = requests.get(
                f"{self.settings.ollama_base_url}/api/tags",
                timeout=10,
            )
            if response.status_code == 200:
                data = response.json()
                return data.get("models", [])
            return []
        except Exception:
            return []

    def generate_response(
        self,
        user_message: str,
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a response from the persona.

        Args:
            user_message: The user's message
            temperature: Sampling temperature (0.0 to 1.0)

        Returns:
            Generated response text
        """
        import time

        self.chat_history.append(ChatMessage(role="user", content=user_message))

        prompt = self._build_prompt(user_message)

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "num_ctx": 4096,
                    "top_p": 0.9,
                },
                stream=False,
            )

            response_text = response.get("response", "")

            self.chat_history.append(ChatMessage(role="assistant", content=response_text))

            if self.persona:
                self.persona.last_interaction = time.time()
                self._save_persona()

            return response_text

        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            return error_msg

    def stream_response(
        self,
        user_message: str,
        temperature: float = 0.7,
    ) -> str:
        """
        Stream a response from the persona.

        Args:
            user_message: The user's message
            temperature: Sampling temperature

        Yields:
            Response text chunks
        """
        import time

        self.chat_history.append(ChatMessage(role="user", content=user_message))

        prompt = self._build_prompt(user_message)

        full_response = []

        try:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": temperature,
                    "num_ctx": 4096,
                    "top_p": 0.9,
                },
                stream=True,
            )

            for chunk in response:
                text = chunk.get("response", "")
                if text:
                    full_response.append(text)
                    yield text

            response_text = "".join(full_response)

            self.chat_history.append(ChatMessage(role="assistant", content=response_text))

            if self.persona:
                self.persona.last_interaction = time.time()
                self._save_persona()

        except Exception as e:
            yield f"Error generating response: {str(e)}"

    def clear_history(self) -> None:
        """Clear the chat history."""
        self.chat_history = []

    def get_history(self) -> list[dict[str, str]]:
        """Get chat history as dictionaries."""
        return [{"role": msg.role, "content": msg.content} for msg in self.chat_history]

    @staticmethod
    def create_persona(
        name: str,
        description: str,
        transcript_paths: list[str],
    ) -> Persona:
        """
        Create a new persona from transcripts.

        Args:
            name: Persona name
            description: Persona description
            transcript_paths: List of transcript file paths

        Returns:
            Created Persona object
        """
        import time

        settings = get_settings()
        rag = RAGPipeline(persona_name=name)

        for path in transcript_paths:
            rag.add_transcript(path)

        all_text = ""
        for path in transcript_paths:
            transcript_file = Path(path)
            if transcript_file.exists():
                all_text += transcript_file.read_text(encoding="utf-8") + "\n\n"

        sample_text = all_text[:5000] if all_text else ""

        system_prompt = f"""You are {name}.

{description}

You have knowledge from the following transcripts and sources. Use this knowledge to answer questions as {name} would.

Sample of your knowledge:
{sample_text}

Remember to respond in {name}'s voice and style."""

        persona = Persona(
            name=name,
            description=description,
            system_prompt=system_prompt,
            sources=transcript_paths,
            created_at=time.time(),
            last_interaction=time.time(),
        )

        safe_name = name.replace(" ", "_")
        persona_path = settings.get_persona_path() / f"{safe_name}.json"
        persona_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "name": persona.name,
            "description": persona.description,
            "system_prompt": persona.system_prompt,
            "sources": persona.sources,
            "created_at": persona.created_at,
            "last_interaction": persona.last_interaction,
        }

        persona_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

        return persona


def create_persona_interactive(
    name: str,
    description: str,
    transcript_paths: list[str],
) -> Persona:
    """Create a persona interactively with progress output."""
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn

    console = Console()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task(f"Creating persona '{name}'...", total=None)

        persona = PersonaChatbot.create_persona(
            name=name,
            description=description,
            transcript_paths=transcript_paths,
        )

        progress.update(task, completed=True)

    return persona
