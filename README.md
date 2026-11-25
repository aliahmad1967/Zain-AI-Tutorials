# Zain AI Tutorials

## Setup

This project uses [uv](https://docs.astral.sh/uv/) and Python 3.12.

```powershell
uv sync
Copy-Item .env.example .env
uv run python main.py
```

The default `.env` configuration expects Ollama at `http://localhost:11434`. Start Ollama and install the configured model:

```powershell
ollama pull gemma4:e4b
```

The assistant can call `get_current_time` to retrieve the current time in the
Asia/Baghdad timezone, and `get_weather` to retrieve current conditions for a
city or named location using the free Open-Meteo API. It can also use
`read_document` to read UTF-8 text files from the `documents/` folder; provide
the file path relative to that folder.

## Local RAG

Before each chat request, the assistant automatically retrieves relevant
passages from `.txt` and text-based `.pdf` files in `knowledge/` and includes
them as reference context. The retriever ranks overlapping text chunks with
BM25 keyword matching and returns source filenames for citations; it does not
require an embeddings model. Scanned PDFs require OCR and are not supported.
The assistant can also call `search_knowledge` for another search or
`read_knowledge_document` when you name a specific knowledge file.