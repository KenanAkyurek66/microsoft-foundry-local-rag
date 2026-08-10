# Local RAG AI Assistant with Microsoft Foundry Local

## Project Overview

This project is a fully local Retrieval-Augmented Generation (RAG) application that answers questions using your own local TXT documents.

It leverages the power of open-source language models:
- Document embedding, retrieval, and model inference are performed locally by this application using **Microsoft Foundry Local**.
- This project does not require a cloud LLM API during normal inference.

## Features

- Local document ingestion
- Paragraph-based chunking
- Local embeddings
- SQLite knowledge base
- Cosine similarity
- Top-K semantic retrieval
- Grounded local LLM answers
- Source citations
- Deterministic fallback for unsupported questions
- Interactive CLI

## Architecture

```text
TXT Documents
-> Chunking
-> Embeddings
-> SQLite

User Question
-> Query Embedding
-> Cosine Similarity
-> Top-K Retrieval
-> Context
-> Local LLM
-> Answer + Source
```

## Technologies

- **Python**
- **Microsoft Foundry Local**
- **qwen3-embedding-0.6b** (Embedding Model)
- **qwen2.5-1.5b** (Chat Model)
- **SQLite** (Vector & Knowledge Storage)

## Project Structure

- `app.py`: The interactive CLI application.
- `ingest.py`: The script used to ingest and chunk `.txt` documents.
- `requirements.txt`: Project dependencies.
- `documents/`: The directory where you place your `.txt` files for ingestion.
- `data/`: Stores the generated SQLite database (`knowledge.db`).
- `tests/`: Contains automated testing and early prototype checkpoints.

## Installation

Create a virtual environment:
```powershell
python -m venv .venv
```

Activate the virtual environment (Windows PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
```

Install the required dependencies:
```powershell
pip install -r requirements.txt
```

## Usage

**Step 1: Add your documents**
Add `.txt` documents containing your knowledge into the `documents/` directory.

**Step 2: Build the knowledge base**
Run the ingestion script to chunk your documents, generate embeddings, and store them in SQLite:
```powershell
python ingest.py
```

**Step 3: Start the application**
Run the main interactive application:
```powershell
python app.py
```

- Type your questions directly into the terminal to interact with the model.
- Type `exit` or `quit` to stop the application.

## Example

```text
Loading models (this might take a moment)...

========================================
Local RAG Assistant - Foundry Local
========================================

Knowledge base loaded: 9 chunks

Type your question or 'exit' to quit.

Question: Why does SQLite not need a separate server?

Answer:
SQLite does not need a separate server because it reads and writes directly to ordinary disk files without requiring an external process to manage connections and transactions.

Source:
sqlite.txt
```

## How RAG Works in This Project

1. Documents are split into paragraph chunks.
2. Chunks are converted into 1024-dimensional semantic embeddings.
3. Chunks and embeddings are stored locally in an SQLite database.
4. When a user asks a question, the query is embedded.
5. Cosine similarity calculates the distance between the query and all stored chunks.
6. The Top-K (3) most relevant chunks become context for the LLM.
7. The local LLM generates an answer strictly using only that retrieved context.

## Hallucination Control

If the context retrieved does not contain enough information to answer a user's question, the LLM will fall back gracefully and output exactly:

> *"I don't have enough information in the provided documents."*

This prevents the model from hallucinating or answering with outside knowledge.

## Testing

The project is verified against automated tests (via `run_tests.py`) covering four manually verified scenarios:
- A question about RAG
- A question about SQLite
- A question about Python
- An intentionally unrelated World Cup question (to verify hallucination fallback)

## Limitations

- Only `.txt` documents are currently supported.
- Chunking is simple and paragraph-based.
- Cosine similarity is brute-forced (O(N)), which is suitable for small datasets but may slow down for millions of records.
- No conversational memory (each question is independent).
- Answer quality depends heavily on the local model capability and the quality of the source documents.

## Future Improvements

- PDF/DOCX ingestion support.
- Token-aware chunking.
- Minimum similarity threshold for retrieval.
- Streamlit web interface.
- Larger or alternative local models depending on hardware.
- Advanced vector database storage (e.g., Qdrant, Chroma).
