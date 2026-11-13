# LangGraph-OpenAI

A small Python project demonstrating a LangGraph workflow powered by OpenAI.

## Requirements

- Python 3.10+
- An OpenAI API key

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`, then run:

```bash
langgraph-openai "Explain LangGraph in one sentence"
```

You can also run the module directly:

```bash
python -m langgraph_openai "What is a state graph?"
```

## Project structure

- `src/langgraph_openai/graph.py` defines the LangGraph workflow.
- `src/langgraph_openai/main.py` provides the command-line entry point.
- `.env.example` documents the required environment variables.

## License

This sample is provided for demonstration purposes.
