"""Command-line interface for Local RAG Persona Simulator."""

import os
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table

from .config import get_settings
from .core.chatbot import PersonaChatbot, create_persona_interactive
from .core.rag import PersonaKnowledgeBase, RAGPipeline
from .core.transcript import TranscriptFetcher

console = Console()


@click.group()
@click.version_option(version="0.1.0")
def cli() -> None:
    """Local RAG Persona Simulator - Chat with personas from YouTube transcripts."""
    pass


@cli.command("fetch-transcript")
@click.argument("url")
@click.option("-o", "--output", type=click.Path(path_type=Path), help="Output file path")
def fetch_transcript(url: str, output: Path | None) -> None:
    """Fetch transcript from a YouTube video or playlist."""
    console.print(f"[bold blue]Fetching transcript from:[/bold blue] {url}")

    try:
        fetcher = TranscriptFetcher()
        transcript_path = fetcher.fetch_transcript(url, output)

        info = fetcher.get_video_info(url)

        console.print(
            Panel(
                f"[green]✓ Transcript saved successfully![/green]\n\n"
                f"Title: {info.get('title', 'N/A')}\n"
                f"Duration: {info.get('duration', 'N/A')} seconds\n"
                f"Subtitles available: {info.get('has_subtitles', False)}\n"
                f"File: [cyan]{transcript_path}[/cyan]",
                title="Success",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(
            Panel(f"[red]Error:[/red] {str(e)}", title="Fetch Failed", border_style="red")
        )
        sys.exit(1)


@cli.command("add-transcript")
@click.argument("persona_name")
@click.argument("transcript_path", type=click.Path(exists=True, path_type=Path))
@click.option("--source-name", help="Name for the source")
def add_transcript(
    persona_name: str,
    transcript_path: Path,
    source_name: str | None,
) -> None:
    """Add a transcript to a persona's knowledge base."""
    console.print(f"[bold blue]Adding transcript to persona '{persona_name}'...[/bold blue]")

    try:
        rag = RAGPipeline(persona_name=persona_name)
        chunks_added = rag.add_transcript(transcript_path, source_name)

        console.print(
            Panel(
                f"[green]✓ Successfully added {chunks_added} chunks to knowledge base![/green]\n\n"
                f"Persona: {persona_name}\n"
                f"Source: {source_name or transcript_path.stem}\n"
                f"Chunks: {chunks_added}",
                title="Success",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(Panel(f"[red]Error:[/red] {str(e)}", title="Add Failed", border_style="red"))
        sys.exit(1)


@cli.command("create-persona")
@click.argument("name")
@click.option(
    "--description", "-d", prompt="Enter a description for this persona", help="Persona description"
)
@click.option(
    "--transcripts",
    "-t",
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="Transcript files to include",
)
def create_persona(
    name: str,
    description: str,
    transcripts: tuple[Path, ...],
) -> None:
    """Create a new persona from transcripts."""
    if not transcripts:
        console.print("[yellow]No transcripts provided. Creating empty persona...[/yellow]")
        transcripts = []

    console.print(f"[bold blue]Creating persona '{name}'...[/bold blue]")

    try:
        transcript_paths = [str(p) for p in transcripts]
        persona = create_persona_interactive(
            name=name,
            description=description,
            transcript_paths=transcript_paths,
        )

        console.print(
            Panel(
                f"[green]✓ Persona created successfully![/green]\n\n"
                f"Name: {persona.name}\n"
                f"Sources: {len(persona.sources)}\n"
                f"System prompt length: {len(persona.system_prompt)} chars",
                title="Success",
                border_style="green",
            )
        )

    except Exception as e:
        console.print(
            Panel(f"[red]Error:[/red] {str(e)}", title="Creation Failed", border_style="red")
        )
        sys.exit(1)


@cli.command("chat")
@click.argument("persona_name")
@click.option("--model", help="Ollama model to use (default: from config)")
@click.option("--clear-history", is_flag=True, help="Clear chat history before starting")
def chat(
    persona_name: str,
    model: str | None,
    clear_history: bool,
) -> None:
    """Start an interactive chat with a persona."""
    console.print(f"[bold blue]Initializing chat with '{persona_name}'...[/bold blue]")

    try:
        chatbot = PersonaChatbot(persona_name=persona_name, model=model)

        connected, message = chatbot.check_ollama_connection()
        if not connected:
            console.print(
                Panel(
                    f"[red]Ollama not connected:[/red] {message}\n\n"
                    "Please ensure Ollama is running and accessible.",
                    title="Connection Error",
                    border_style="red",
                )
            )
            sys.exit(1)

        console.print(f"[green]✓ {message}[/green]")

        if not chatbot.knowledge_base.persona_exists(persona_name):
            console.print(
                Panel(
                    f"[yellow]Warning:[/yellow] Persona '{persona_name}' does not exist. "
                    "You can create it with 'create-persona' command.",
                    title="Unknown Persona",
                    border_style="yellow",
                )
            )

        if clear_history:
            chatbot.clear_history()
            console.print("[dim]Chat history cleared.[/dim]")

        if chatbot.persona:
            console.print(
                Panel(
                    f"[bold cyan]{chatbot.persona.name}[/bold cyan]\n\n"
                    f"{chatbot.persona.description}",
                    title="Persona Info",
                    border_style="cyan",
                )
            )
        else:
            console.print(
                Panel(
                    f"[yellow]No persona configuration found. Chatting with RAG context only.[/yellow]",
                    title="Warning",
                    border_style="yellow",
                )
            )

        console.print("\n[dim]Type 'quit' or 'exit' to end the chat.[/dim]\n")

        while True:
            try:
                user_input = Prompt.ask(
                    "[bold green]You[/bold green]",
                    default="",
                    show_default=False,
                )

                if not user_input.strip():
                    continue

                if user_input.lower() in ("quit", "exit", "q"):
                    console.print("[dim]Goodbye![/dim]")
                    break

                if user_input.lower() == "clear":
                    chatbot.clear_history()
                    console.print("[dim]Chat history cleared.[/dim]\n")
                    continue

                if user_input.lower() == "history":
                    history = chatbot.get_history()
                    if history:
                        console.print("\n[bold]Chat History:[/bold]")
                        for msg in history:
                            console.print(
                                f"  {msg['role'].capitalize()}: {msg['content'][:100]}..."
                            )
                    else:
                        console.print("[dim]No chat history.[/dim]")
                    console.print()
                    continue

                console.print()
                console.print(f"[bold cyan]{chatbot.persona_name}[/bold cyan]: ", end="")

                response = chatbot.generate_response(user_input)
                console.print(response)
                console.print()

            except KeyboardInterrupt:
                console.print("\n[dim]Use 'quit' to exit.[/dim]\n")

    except Exception as e:
        console.print(Panel(f"[red]Error:[/red] {str(e)}", title="Chat Error", border_style="red"))
        sys.exit(1)


@cli.command("list-personas")
def list_personas() -> None:
    """List all available personas."""
    knowledge_base = PersonaKnowledgeBase()
    personas = knowledge_base.list_personas()

    if not personas:
        console.print(
            Panel(
                "No personas found. Create one with 'create-persona' command.",
                title="Personas",
                border_style="yellow",
            )
        )
        return

    table = Table(title="Available Personas")
    table.add_column("Name", style="cyan")
    table.add_column("Status", style="green")

    for name in personas:
        has_kb = knowledge_base.persona_exists(name)
        status = "[green]Active[/green]" if has_kb else "[red]Missing KB[/red]"
        table.add_row(name, status)

    console.print(table)


@cli.command("delete-persona")
@click.argument("persona_name")
@click.option("--force", is_flag=True, help="Skip confirmation prompt")
def delete_persona(persona_name: str, force: bool) -> None:
    """Delete a persona and its knowledge base."""
    if not force:
        if not Confirm.ask(
            f"Are you sure you want to delete persona '{persona_name}'?",
            default=False,
        ):
            console.print("[dim]Cancelled.[/dim]")
            return

    knowledge_base = PersonaKnowledgeBase()

    try:
        success = knowledge_base.delete_persona(persona_name)

        if success:
            console.print(
                Panel(
                    f"[green]✓ Persona '{persona_name}' deleted successfully![/green]",
                    title="Success",
                    border_style="green",
                )
            )
        else:
            console.print(
                Panel(
                    f"[yellow]Persona '{persona_name}' not found or could not be deleted.[/yellow]",
                    title="Warning",
                    border_style="yellow",
                )
            )

    except Exception as e:
        console.print(
            Panel(f"[red]Error:[/red] {str(e)}", title="Delete Failed", border_style="red")
        )
        sys.exit(1)


@cli.command("info")
@click.argument("persona_name")
def info(persona_name: str) -> None:
    """Show detailed information about a persona."""
    knowledge_base = PersonaKnowledgeBase()

    if not knowledge_base.persona_exists(persona_name):
        console.print(
            Panel(
                f"Persona '{persona_name}' does not exist.", title="Not Found", border_style="red"
            )
        )
        sys.exit(1)

    try:
        chatbot = PersonaChatbot(persona_name=persona_name)
        stats = chatbot.rag.get_collection_stats()

        info_text = f"""
[bold]Name:[/bold] {persona_name}
[bold]Collection:[/bold] {stats["collection_name"]}
[bold]Documents:[/bold] {stats["document_count"]}
"""

        if chatbot.persona:
            info_text += f"""
[bold]Description:[/bold]
{chatbot.persona.description}

[bold]Sources:[/bold]
{", ".join(chatbot.persona.sources) if chatbot.persona.sources else "None"}
"""

        console.print(
            Panel(info_text.strip(), title=f"Persona: {persona_name}", border_style="cyan")
        )

    except Exception as e:
        console.print(Panel(f"[red]Error:[/red] {str(e)}", title="Error", border_style="red"))
        sys.exit(1)


@cli.command("check-ollama")
def check_ollama() -> None:
    """Check Ollama connection and list available models."""
    chatbot = PersonaChatbot(persona_name="_temp")

    connected, message = chatbot.check_ollama_connection()

    if connected:
        console.print(
            Panel(f"[green]✓ {message}[/green]", title="Ollama Status", border_style="green")
        )

        models = chatbot.list_available_models()
        if models:
            table = Table(title="Available Models")
            table.add_column("Name", style="cyan")
            table.add_column("Size", style="dim")

            for model in models:
                name = model.get("name", "Unknown")
                size = model.get("size", 0)
                size_gb = size / (1024**3) if size else 0
                table.add_row(name, f"{size_gb:.1f} GB")

            console.print(table)
        else:
            console.print(
                "[yellow]No models available. Pull a model with: ollama pull <model>[/yellow]"
            )
    else:
        console.print(
            Panel(
                f"[red]✗ {message}[/red]\n\n"
                "To use Ollama:\n"
                "1. Install Ollama from https://ollama.ai\n"
                "2. Run 'ollama serve' in a terminal\n"
                "3. Pull a model: 'ollama pull llama3.2'",
                title="Ollama Status",
                border_style="red",
            )
        )
        sys.exit(1)


def main() -> None:
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
