# Local RAG Persona Simulator

> Chat with AI personas created from YouTube transcripts using local LLMs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Powered by Ollama](https://img.shields.io/badge/Powered%20by-Ollama-purple.svg)](https://ollama.ai/)

Local RAG Persona Simulator is a CLI tool that lets you create AI personas from YouTube video transcripts and chat with them using local Large Language Models. No API keys required — everything runs on your machine.

## Features

- **YouTube Transcript Fetching** — Extract transcripts from any YouTube video or playlist using `yt-dlp`
- **Local LLM Inference** — Powered by [Ollama](https://ollama.ai/) for completely offline AI interactions
- **Vector Storage** — ChromaDB provides persistent vector storage for persona knowledge bases
- **Semantic Search** — Sentence-transformers embeddings enable relevant context retrieval
- **CLI Interface** — Intuitive command-line interface built with Click and Rich
- **Multiple Personas** — Create and manage multiple different personas
- **Chat History** — Maintains conversation context within sessions

## Why This Project?

### No API Keys Needed

Most RAG and chatbot solutions require paid API access to OpenAI, Anthropic, or other cloud LLM providers. This project runs entirely locally using:

- **[Ollama](https://ollama.ai/)** — Download and run LLMs locally (Llama 3.2, Mistral, Phi, etc.)
- **[ChromaDB](https://www.trychroma.com/)** — Local vector database for embeddings
- **[sentence-transformers](https://sbert.net/)** — Local embedding models

### Privacy First

All data stays on your machine:

- Transcripts are downloaded and stored locally
- Embeddings are computed on your CPU/GPU
- LLM inference happens locally — no data leaves your computer

### Cost Effective

Zero API costs after initial setup. Once you have Ollama and the required models installed, there are no per-token charges.

## Prerequisites

### Python 3.10+

```bash
# Check your Python version
python --version
```

### Ollama

Ollama runs local LLM inference. Install it from [ollama.ai](https://ollama.ai/) or use one of these methods:

```bash
# macOS
brew install ollama

# Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Windows
# Download and install from https://ollama.ai/download/windows
```

After installation, pull a model:

```bash
# Pull the default model (Llama 3.2)
ollama pull llama3.2

# Or use a smaller model for faster inference
ollama pull mistral
ollama pull phi3

# List available models
ollama list
```

Start the Ollama service:

```bash
# Run in background (required for the app to work)
ollama serve
```

## Installation

### 1. Clone or Navigate to the Project

```bash
cd /path/to/local-rag-persona-simulator
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate on macOS/Linux
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install the package and dependencies
pip install -e .

# Or install from requirements.txt
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
# Check Ollama connection
rag-persona check-ollama
```

Expected output:
```
┌─────────────────────────────────────────────────────┐
│  ✓ Connected to Ollama                             │
├─────────────────────────────────────────────────────┤
│  Name          │ Size                               │
│  llama3.2      │ 3.8 GB                             │
│  mistral       │ 4.1 GB                             │
└─────────────────────────────────────────────────────┘
```

## Quick Start

The complete workflow has three steps:

### Step 1: Fetch a YouTube Transcript

```bash
# Fetch transcript from a YouTube video
rag-persona fetch-transcript "https://www.youtube.com/watch?v=dQw4w9WgXcQ"

# Or from a playlist (all videos)
rag-persona fetch-transcript "https://www.youtube.com/playlist?list=PLxxxxx"
```

### Step 2: Create a Persona

```bash
# Create a persona with one or more transcript files
rag-persona create-persona "Rick Astley" \
    --description "An enthusiastic presenter known for his engaging talks" \
    --transcripts ./data/transcripts/rick_roll.txt
```

### Step 3: Chat with the Persona

```bash
# Start an interactive chat
rag-persona chat "Rick Astley"
```

That's it! You can now chat with the AI persona.

## Usage Guide

### Command Overview

| Command | Description |
|---------|-------------|
| `fetch-transcript` | Download transcripts from YouTube videos/playlists |
| `add-transcript` | Add a transcript to an existing persona's knowledge base |
| `create-persona` | Create a new persona from transcripts |
| `chat` | Start an interactive chat session with a persona |
| `list-personas` | List all available personas |
| `info` | Show detailed information about a persona |
| `delete-persona` | Delete a persona and its knowledge base |
| `check-ollama` | Check Ollama connection and list available models |

### Detailed Command Reference

#### fetch-transcript

Fetch transcripts from YouTube videos or playlists.

```bash
# Basic usage - saves to default directory
rag-persona fetch-transcript "https://www.youtube.com/watch?v=VIDEO_ID"

# Save to specific location
rag-persona fetch-transcript "https://www.youtube.com/watch?v=VIDEO_ID" \
    --output ./my-transcripts/lecture.txt
```

#### add-transcript

Add more transcripts to an existing persona's knowledge base.

```bash
rag-persona add-transcript "Rick Astley" ./data/transcripts/another_lecture.txt
```

#### create-persona

Create a new persona with a name, description, and optional transcripts.

```bash
# Interactive mode (will prompt for description)
rag-persona create-persona "Professor Smith"

# With all options
rag-persona create-persona "Professor Smith" \
    --description "A computer science professor specializing in AI" \
    --transcripts ./transcripts/intro_to_ai.txt \
    --transcripts ./transcripts/machine_learning.txt
```

#### chat

Start an interactive chat session with a persona.

```bash
# Basic chat
rag-persona chat "Rick Astley"

# Use a specific Ollama model
rag-persona chat "Rick Astley" --model mistral

# Clear chat history before starting
rag-persona chat "Rick Astley" --clear-history
```

In chat mode:

- Type your message and press Enter to send
- Type `quit` or `exit` to end the session
- Type `clear` to clear chat history
- Type `history` to view conversation history

#### list-personas

View all available personas.

```bash
rag-persona list-personas
```

Output:
```
┌────────────────────────┬────────────────────┐
│ Name                  │ Status             │
│ Rick Astley           │ Active             │
│ Professor Smith       │ Active             │
│ Historical Figure     │ Missing KB         │
└────────────────────────┴────────────────────┘
```

#### info

Get detailed information about a persona.

```bash
rag-persona info "Rick Astley"
```

#### delete-persona

Delete a persona and all its knowledge base data.

```bash
# With confirmation prompt
rag-persona delete-persona "Rick Astley"

# Skip confirmation
rag-persona delete-persona "Rick Astley" --force
```

#### check-ollama

Verify Ollama is running and see available models.

```bash
rag-persona check-ollama
```

## Configuration

### Environment Variables

The application can be configured using environment variables. All settings use the prefix `RAG_PERSONA_`:

| Variable | Default | Description |
|----------|---------|-------------|
| `RAG_PERSONA_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama API base URL |
| `RAG_PERSONA_OLLAMA_MODEL` | `llama3.2` | Default Ollama model for chat |
| `RAG_PERSONA_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model for embeddings |
| `RAG_PERSONA_CHROMA_PERSIST_DIRECTORY` | `data/chroma_db` | ChromaDB storage directory |
| `RAG_PERSONA_CHUNK_SIZE` | `1000` | Text chunk size for RAG |
| `RAG_PERSONA_CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RAG_PERSONA_RETRIEVAL_TOP_K` | `5` | Number of documents to retrieve |
| `RAG_PERSONA_PERSONA_DIRECTORY` | `data/personas` | Persona configuration storage |
| `RAG_PERSONA_TRANSCRIPT_DIRECTORY` | `data/transcripts` | Transcript file storage |

### Using .env Files

Create a `.env` file in the project root to set custom values:

```bash
# .env
RAG_PERSONA_OLLAMA_MODEL=mistral
RAG_PERSONA_CHUNK_SIZE=1500
RAG_PERSONA_EMBEDDING_MODEL=all-mpnet-base-v2
```

### Configuration Precedence

Settings are loaded in this order (later overrides earlier):

1. Default values in `config.py`
2. `.env` file (if present)
3. Environment variables

## How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        User (CLI)                               │
│                  rag-persona <command>                         │
└────────────────────────────┬────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌──────────────┐   ┌──────────────┐    ┌──────────────┐
│ Transcript   │   │     RAG      │    │   Chatbot    │
│   Fetcher    │   │   Pipeline   │    │              │
│              │   │              │    │              │
│  yt-dlp      │   │ ChromaDB     │    │   Ollama     │
│              │   │ sentence-    │    │              │
│              │   │ transformers │    │              │
└──────────────┘   └──────────────┘    └──────────────┘
       │                  │                   │
       │                  │                   │
       ▼                  ▼                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                        YouTube                                  │
│                   (Transcript Source)                          │
└─────────────────────────────────────────────────────────────────┘
```

### Component Details

#### 1. Transcript Fetcher (`transcript.py`)

Uses `yt-dlp` to download and parse YouTube subtitles:

- Supports single videos and playlists
- Prefers English subtitles, falls back to available languages
- Parses JSON3 and SRT subtitle formats
- Saves transcripts as plain text files

#### 2. RAG Pipeline (`rag.py`)

Retrieval Augmented Generation pipeline:

- **Text Splitting** — Uses `RecursiveCharacterTextSplitter` to chunk transcripts
- **Embeddings** — Computes semantic embeddings using `sentence-transformers`
- **Vector Store** — Stores embeddings in ChromaDB for fast similarity search
- **Retrieval** — Finds relevant context based on user queries

#### 3. Chatbot (`chatbot.py`)

Ollama-powered chat interface:

- Builds context-aware prompts with retrieved knowledge
- Maintains chat history for conversational context
- Streams responses for real-time feedback
- Handles persona configuration and persistence

### Data Flow

```
YouTube Video
     │
     ▼
┌─────────────┐
│  yt-dlp     │ ──► Transcript Text
└─────────────┘
     │
     ▼
┌─────────────────┐
│ Text Splitter   │ ──► Chunks
└─────────────────┘
     │
     ▼
┌─────────────────────┐
│ Embedding Model     │ ──► Vectors
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ ChromaDB            │ ──► Stored
└─────────────────────┘
     │
     │ (at query time)
     ▼
┌─────────────────────┐
│ Similarity Search  │ ──► Relevant Context
└─────────────────────┘
     │
     ▼
┌─────────────────────┐
│ LLM (Ollama)        │ ──► Response
└─────────────────────┘
```

## Troubleshooting

### Ollama Connection Issues

**Error**: `Cannot connect to Ollama. Is it running?`

**Solution**:

1. Start Ollama service: `ollama serve`
2. Verify it's running: `rag-persona check-ollama`
3. Check the URL in config matches your setup

### No Transcripts Available

**Error**: `No transcripts available for this video`

**Solution**:

- The video must have subtitles/captions enabled
- Try a different video with closed captions
- Some videos only have auto-generated captions (yt-dlp attempts to fetch these)

### Model Not Found

**Error**: `model with name not found`

**Solution**:

```bash
# List available models
ollama list

# Pull a model if needed
ollama pull llama3.2
```

### Out of Memory

**Solution**:

- Use a smaller embedding model: `RAG_PERSONA_EMBEDDING_MODEL=all-MiniLM-L6-v2`
- Use a smaller LLM: `ollama pull phi3` or `ollama pull mistral`
- Reduce chunk size: `RAG_PERSONA_CHUNK_SIZE=500`

### ChromaDB Errors

**Error**: Collection or database errors

**Solution**:

```bash
# Delete the ChromaDB directory to reset
rm -rf data/chroma_db

# Recreate persona if needed
```

### Slow Embeddings

**Solution**:

- The first run downloads the embedding model (~90MB for all-MiniLM-L6-v2)
- Subsequent runs use cached models
- Consider using a smaller model: `all-MiniLM-L6-v2` is already quite fast

## Project Structure

```
local-rag-persona-simulator/
├── src/
│   └── local_rag_persona_simulator/
│       ├── __init__.py
│       ├── cli.py              # Click CLI commands
│       ├── config.py           # Configuration management
│       ├── core/
│       │   ├── chatbot.py      # Ollama integration
│       │   ├── rag.py          # RAG pipeline
│       │   └── transcript.py   # YouTube transcript fetching
│       └── utils/
│           └── text_utils.py
├── data/                       # Generated data directory
│   ├── chroma_db/             # Vector database
│   ├── personas/              # Persona configurations
│   └── transcripts/           # Downloaded transcripts
├── tests/                     # Test suite
├── requirements.txt           # Python dependencies
└── pyproject.toml            # Project configuration
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- [yt-dlp](https://github.com/yt-dlp/yt-dlp) — YouTube downloader
- [Ollama](https://ollama.ai/) — Local LLM inference
- [ChromaDB](https://www.trychroma.com/) — Vector database
- [LangChain](https://langchain.com/) — LLM application framework
- [sentence-transformers](https://sbert.net/) — Sentence embeddings
