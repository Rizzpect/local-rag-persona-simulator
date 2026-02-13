# Local RAG Persona Simulator

A local RAG (Retrieval Augmented Generation) persona simulator that lets you create chatbots from YouTube video transcripts using local LLMs.

## Features

- **YouTube Transcript Fetching**: Extract subtitles/transcripts from YouTube videos and playlists using yt-dlp
- **Local Embeddings**: Use sentence-transformers for generating embeddings (no API keys needed)
- **Local LLM Inference**: Use Ollama for running LLMs locally
- **RAG Pipeline**: ChromaDB for vector storage and similarity search
- **Persona Creation**: Create custom personas from any YouTube content
- **Interactive Chat**: Chat with your personas in an interactive CLI

## Requirements

- Python 3.10+
- Ollama installed and running locally
- No API keys required (fully local!)

## Installation

### 1. Clone and install dependencies

```bash
cd local-rag-persona-simulator
pip install -r requirements.txt
```

### 2. Install and set up Ollama

1. Download Ollama from https://ollama.ai
2. Run `ollama serve` to start the Ollama server
3. Pull a model:
   ```bash
   ollama pull llama3.2
   # or
   ollama pull mistral
   ```

### 3. (Optional) Create a .env file

```bash
cp .env.example .env
```

Configuration options (defaults work fine):
```
RAG_PERSONA_OLLAMA_BASE_URL=http://localhost:11434
RAG_PERSONA_OLLAMA_MODEL=llama3.2
RAG_PERSONA_EMBEDDING_MODEL=all-MiniLM-L6-v2
RAG_PERSONA_CHUNK_SIZE=1000
RAG_PERSONA_RETRIEVAL_TOP_K=5
```

## Usage

### Check Ollama status

```bash
rag-persona check-ollama
```

### Fetch a transcript

```bash
# Single video
rag-persona fetch-transcript "https://www.youtube.com/watch?v=VIDEO_ID"

# Playlist
rag-persona fetch-transcript "https://www.youtube.com/playlist?list=PLAYLIST_ID"

# Save to custom location
rag-persona fetch-transcript "URL" -o ./my-transcript.txt
```

### Create a persona

```bash
rag-persona create-persona "Persona Name" \
  --description "A helpful AI assistant based on..." \
  --transcripts ./transcript1.txt \
  --transcripts ./transcript2.txt
```

### Add more transcripts to a persona

```bash
rag-persona add-transcript "Persona Name" ./new-transcript.txt --source-name "Video Title"
```

### List all personas

```bash
rag-persona list-personas
```

### Chat with a persona

```bash
rag-persona chat "Persona Name"
```

Chat commands:
- `quit` or `exit` - End the chat
- `clear` - Clear chat history
- `history` - Show chat history

### Delete a persona

```bash
rag-persona delete-persona "Persona Name"
# With confirmation skip
rag-persona delete-persona "Persona Name" --force
```

### Get persona info

```bash
rag-persona info "Persona Name"
```

## Configuration

All settings can be configured via:

1. **Environment variables** (prefix with `RAG_PERSONA_`):
   - `OLLAMA_BASE_URL` - Ollama server URL (default: http://localhost:11434)
   - `OLLAMA_MODEL` - Model to use (default: llama3.2)
   - `EMBEDDING_MODEL` - Embedding model (default: all-MiniLM-L6-v2)
   - `CHUNK_SIZE` - Text chunk size (default: 1000)
   - `CHUNK_OVERLAP` - Chunk overlap (default: 200)
   - `RETRIEVAL_TOP_K` - Documents to retrieve (default: 5)
   - `CHROMA_PERSIST_DIRECTORY` - ChromaDB storage path
   - `PERSONA_DIRECTORY` - Persona config storage path
   - `TRANSCRIPT_DIRECTORY` - Transcript storage path

2. **.env file**: Create a `.env` file in the project root

## Project Structure

```
local-rag-persona-simulator/
├── src/local_rag_persona_simulator/
│   ├── __init__.py
│   ├── config.py          # Configuration management
│   ├── core/
│   │   ├── transcript.py  # YouTube transcript fetcher
│   │   ├── rag.py         # RAG pipeline
│   │   └── chatbot.py     # Persona chatbot
│   ├── utils/
│   │   └── text_utils.py  # Text utilities
│   └── cli.py             # CLI interface
├── tests/
├── pyproject.toml
└── requirements.txt
```

## Troubleshooting

### Ollama not connecting

1. Ensure Ollama is running: `ollama serve`
2. Check the port: default is 11434
3. Verify models are installed: `ollama list`

### No transcripts available

- The video must have either manual subtitles or auto-generated captions
- Some videos don't have subtitles available
- Try a different video or playlist

### ChromaDB errors

- Ensure write permissions to the data directory
- Try deleting the `data/chroma_db` folder if corrupted

### Model download issues

- Pull models manually: `ollama pull llama3.2`
- Check available models: `ollama list`

## License

MIT License.
