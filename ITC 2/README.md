# BroncoBook AI Assistant

BroncoBook AI Assistant is a university-focused AI assistant designed to help students quickly find reliable campus information from an indexed university knowledge base. Instead of answering from general internet knowledge, the assistant is grounded in a curated campus corpus and only responds using information that has been indexed into the system.

This makes it useful for student support, admissions questions, campus services discovery, and hackathon demos where trustworthy, source-backed answers matter.

## What It Does

BroncoBook helps students and campus users:

- Ask natural-language questions about university information
- Get answers grounded in indexed campus documents
- View source-backed responses with linked citations
- Explore topics such as admissions, financial aid, deadlines, campus services, dining, and student resources
- Continue multi-turn conversations with saved conversation history
- Receive a safe fallback response when the answer is not found in the indexed corpus

## Core Features

### 1. Grounded University Q&A
The assistant answers questions using only the campus content that has been indexed into the application. It does not rely on open-ended background knowledge for university facts.

### 2. Source-Cited Responses
Each answer can include source cards with:

- page title
- section or heading path
- supporting snippet
- original source URL

This helps users verify where the answer came from.

### 3. Hybrid Retrieval with SQLite
The project uses a SQLite-backed retrieval system that combines:

- keyword search with SQLite FTS5
- semantic search with embeddings
- hybrid ranking to improve relevance

This allows the assistant to find both exact matches and semantically similar campus information.

### 4. Conversation Memory
The assistant stores conversations and messages so users can ask follow-up questions in the same session.

### 5. Multi-Provider AI Support
The system supports:

- OpenAI for tool-calling and grounded answer generation
- Ollama as a local model fallback option

If OpenAI is unavailable, the app can fall back to Ollama-based workflows.

### 6. Retrieval Safety and Hallucination Control
The assistant is designed to reduce misinformation by:

- retrieving evidence before answering factual questions
- refusing to invent citations or policies
- returning a clear fallback message when evidence is too weak

Fallback response:

```text
I could not find that information in the indexed corpus.
```

### 7. Analytics and Debug Visibility
The application tracks useful demo metrics such as:

- total queries
- unanswered queries
- tool calls
- average sources per answer

It also supports retrieval debug output so developers can inspect how results were found and ranked.

### 8. Admin Indexing and Embedding Tools
The backend includes endpoints and scripts for:

- indexing the campus corpus
- rebuilding embeddings
- searching the indexed corpus directly
- checking application health and retrieval status

## Example Questions

Users can ask questions like:

- What are the freshman admission requirements?
- How do I change my major?
- Where is Student Health Services located?
- What financial aid resources are available?
- What dining options are on campus?
- When is the application deadline?

## How It Works

1. Campus webpages are converted into markdown files and mapped through `data/corpus/index.json`.
2. The app chunks and indexes those documents into SQLite.
3. Embeddings are generated for semantic search.
4. When a user asks a question, the retriever finds the most relevant chunks.
5. The assistant generates a grounded answer using only the retrieved evidence.
6. The UI displays the response along with its sources.

## Tech Stack

- Frontend: Static HTML/CSS/JavaScript with SvelteKit tooling in the repository
- Backend: FastAPI
- Database: SQLite
- Retrieval: SQLite FTS5 + vector embeddings
- AI Providers: OpenAI and Ollama
- Testing: Pytest

## Project Structure

- `app/main.py` - FastAPI routes and app entry point
- `app/agent.py` - AI orchestration and provider routing
- `app/retriever.py` - indexing, chunking, and hybrid retrieval
- `app/db.py` - SQLite persistence layer
- `app/prompts.py` - grounding rules and tool definitions
- `data/corpus/` - indexed university content
- `scripts/build_embeddings.py` - embedding build script
- `tests/` - backend retrieval and agent tests

## Why This Project Matters

University websites often contain important information, but students may struggle to find it quickly. BroncoBook AI Assistant improves accessibility by turning scattered campus pages into a conversational, source-aware experience.

For a university setting, that means:

- faster access to student support information
- more confidence in answers because sources are shown
- a better experience for admissions and campus-resource discovery
- a foundation for future student-facing AI services

## Future Improvements

Potential next steps for the project include:

- expanding the indexed university corpus
- adding authentication and role-based admin controls
- improving the frontend chat experience
- supporting more campus-specific workflows
- adding feedback collection for answer quality
- deploying the system for broader student use

## Running the Project

### Install dependencies

```bash
bun install
python3 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Configure environment

Copy `.env.example` to `.env` and add the needed API settings.

### Start the backend

```bash
bun run backend:dev
```

Then open [http://127.0.0.1:8000/](http://127.0.0.1:8000/).

## Submission Summary

BroncoBook AI Assistant is a grounded university chatbot that helps students find accurate campus information through source-cited answers, hybrid retrieval, and conversation-based interaction. It is designed to be practical, transparent, and trustworthy for student-facing use cases.
