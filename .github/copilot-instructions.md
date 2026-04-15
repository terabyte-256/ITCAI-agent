# Copilot instructions for this repository

## Scope to prioritize
- Primary project for implementation work: `BroncoBook-ITC-MISSA/Application`.
- Use `BroncoBook-ITC-MISSA/campus-knowledge-agent 3/campus-knowledge-agent 3` mainly as reference context (it is where the existing pytest suite lives).

## Build, run, test, and lint commands
Run commands from repository root unless noted.

| Task | Command |
|---|---|
| Create venv + install deps (primary project) | `cd "BroncoBook-ITC-MISSA/Application" && python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt` |
| Run provider-switching FastAPI app (`LLM_PROVIDER` controls backend) | `cd "BroncoBook-ITC-MISSA/Application" && uvicorn app.main.main:app --reload --host 127.0.0.1 --port 8000` |
| Run OpenAI-only FastAPI app | `cd "BroncoBook-ITC-MISSA/Application" && uvicorn app.openAI.main:app --reload --host 127.0.0.1 --port 8060` |
| Run Ollama-only FastAPI app | `cd "BroncoBook-ITC-MISSA/Application" && uvicorn app.ollama.main:app --reload --host 127.0.0.1 --port 8000` |
| Run full existing tests (sibling module) | `cd "BroncoBook-ITC-MISSA/campus-knowledge-agent 3/campus-knowledge-agent 3" && python -m pytest` |
| Run a single existing test | `cd "BroncoBook-ITC-MISSA/campus-knowledge-agent 3/campus-knowledge-agent 3" && python -m pytest tests/test_retriever.py::test_get_page` |

No lint command/config is currently defined in this repository.

## High-level architecture
- `BroncoBook-ITC-MISSA/Application/app/main/main.py` is the main server entrypoint. It serves the static UI and exposes `POST /chat`, selecting OpenAI vs Ollama via `LLM_PROVIDER`.
- `BroncoBook-ITC-MISSA/Application/app/openAI/main.py` and `.../app/ollama/main.py` are provider-specific FastAPI variants. They also mount the same static directory and expose separate chat routes.
- Frontend lives in `BroncoBook-ITC-MISSA/Application/app/static/`:
  - `index.html` provides the shell and chat form.
  - `app.js` calls `/health` and `/chat`, and renders provider/model metadata from the backend response.
  - `styles.css` defines the dark themed UI styling.
- `BroncoBook-ITC-MISSA/DESIGN.md` documents the intended visual system (Digital Curator, tonal layering, no-line rule) and should guide major UI restyling decisions.

## Key conventions in this codebase
- Backend/Frontend response contract for the primary app (`app/main/main.py` + `app/static/app.js`): `/chat` responses are expected to include `response`, with optional `provider` and `model`; preserve this shape when changing the API.
- The primary app depends on environment-driven provider switching:
  - `LLM_PROVIDER` defaults to `ollama`.
  - `OPENAI_API_KEY` is optional unless using the OpenAI path.
  - `OLLAMA_HOST` defaults differ between modules (`app/main/main.py` defaults to `http://localhost:8000`, while provider-specific modules default to `http://localhost:8050`), so keep host/port choices explicit when editing or running.
- Static asset serving pattern is consistent across backend entrypoints: compute `BASE_DIR`, mount `app/static` at `/static`, and serve `/` via `FileResponse(index.html)`.
- There is an extra `app/static/app.js.txt` file containing alternate chat logic; the active client script loaded by `index.html` is `app/static/app.js`.
