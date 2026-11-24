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
city or named location using the free Open-Meteo API.