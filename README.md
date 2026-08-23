# Local RAG AI Assistant with Microsoft Foundry Local

## Project Overview

This project is a Retrieval-Augmented Generation (RAG) application that answers questions grounded in your personal knowledge base. 
It leverages Microsoft Foundry Local for local embedding generation and LLM inference, while semantic retrieval is performed locally by the application using cosine similarity.

## Features

### Document Management & Processing
- **Supported Formats:** TXT, PDF, and DOCX. *(Note: Scanned/image-only PDFs without extractable text are not supported as OCR is not implemented.)*
- **Multi-Document Upload:** Seamlessly upload and index multiple files at once.
- **Incremental Indexing:** Uses SHA-256 hashing to implement smart indexing for document identity and change detection.
  - **Duplicate Document Detection:** Prevents re-indexing the exact same document.
  - **Automatic Update Detection:** Detects when a document changes and efficiently updates its chunks in the database.
- **Document Removal:** Ability to remove specific documents and their corresponding chunks from the knowledge base without affecting others.
- **Smart Chunking:** Paragraph-aware chunking targeting an approximate size of 400 words with a 50-word chunk overlap to preserve context across boundaries.

### Retrieval & Generation
- **Local Embedding Generation:** Converts text into semantic vector embeddings using local models.
- **Local LLM Inference:** Generates concise, context-grounded answers locally.
- **SQLite Knowledge Base:** Stores metadata, raw text chunks, and vector embeddings in a lightweight SQLite database.
- **Cosine Similarity Retrieval:** Computes vector distances to find the most relevant information.
- **High-Precision Context:** 
  - Retrieves the Top-K = 3 most relevant chunks.
  - Enforces a minimum retrieval similarity threshold of 0.40 to ensure context quality.
- **Deterministic Fallback:** If the retrieved context doesn't meet the similarity threshold or lacks sufficient information, the model gracefully falls back with a standard response ("I don't have enough information in the provided documents.") to prevent hallucinations.
- **Source Attribution:** Clearly cites the source document(s) used to generate the answer.

### User Interfaces
- **Streamlit Interface:** A premium, polished web application featuring a document management panel, conversation clearing, interactive chat, and dynamic visual themes.
- **CLI Application:** A fast, interactive terminal-based chat interface.

## Architecture

```text
Document (TXT, PDF, DOCX) 
-> SHA-256 Hashing (Document Identity & Change Detection)
-> Text Extraction
-> Paragraph-aware Chunking (~400 words, 50-word overlap)
-> Local Embedding Model
-> SQLite Database (Metadata, Chunks, Embeddings)

User Question 
-> Query Embedding 
-> Cosine Similarity (Threshold >= 0.40) 
-> Top-K (3) Retrieval 
-> Context Assembly 
-> Local LLM 
-> Answer + Source Attribution
```

## Main Project Files

- `streamlit_app.py`: The polished Streamlit web interface with a document management panel and chat functionality.
- `app.py`: The interactive Command Line Interface (CLI) application for asking questions from the terminal.
- `ingest.py`: The script used to index documents from the terminal.
- `document_ingestion.py`: Handles file parsing (TXT, PDF, DOCX), smart chunking (400 words, 50-word overlap), and incremental indexing logic using SHA-256.
- `retrieval.py`: Implements the cosine similarity calculation, top-K filtering, and similarity threshold logic.
- `requirements.txt`: Lists the Python dependencies required to run the project.
- `data/`: Directory where the SQLite knowledge base (`knowledge.db`) is stored.
- `run_tests.py`: The main test runner script for end-to-end verification.
- `tests/`: Contains automated unit test scripts (e.g., `test_rag.py`, `test_database.py`, `test_retrieval.py`) for verifying system behavior.

## Installation

**1. Create a virtual environment:**
```powershell
python -m venv .venv
```

**2. Activate the virtual environment (Windows PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**3. Install dependencies:**
```powershell
pip install -r requirements.txt
```

## Usage

### Using the Streamlit Interface (Recommended)

Run the Streamlit web application for a complete visual experience:
```powershell
streamlit run streamlit_app.py
```
- **Document Management:** Use the document management panel to upload multiple TXT, PDF, or DOCX files. You can view indexed documents and remove them if needed.
- **Chat:** Ask questions based on your documents.
- **Conversation Clearing:** Easily clear the chat history from the document management panel.

### Using the CLI Application

If you prefer the terminal, you can interact with your existing knowledge base via the CLI:
```powershell
python app.py
```
- Type your questions directly into the terminal to get answers.
- Type `exit` or `quit` to stop the application.

*(Note: The CLI assumes documents have been indexed either via the Streamlit interface or a dedicated ingestion script.)*

## Privacy and Local Execution

This project is built from the ground up for privacy and local execution:
- Normal document processing, embedding generation, retrieval, and LLM inference run locally through Microsoft Foundry Local.
- No cloud LLM API is required for normal inference.
- Internet access may still be required initially for installing dependencies and downloading/caching the required models.

## Testing

The project includes an automated test suite. You can run the application tests via `run_tests.py` and the unit tests located in the `tests/` directory. The test suite covers:
- Core RAG functionality and prompt adherence.
- Contextual retrieval, cosine similarity, and threshold behavior.
- End-to-end fallback mechanisms.
- Incremental indexing and document updates.

## Current Limitations

- No OCR support; scanned/image-only PDFs cannot be processed.
- No cross-session conversational memory (each question is treated independently).
- Database operations rely on basic SQLite without specialized vector indexing (e.g., HNSW), which operates in O(N) time and may slow down with extremely large datasets.

## Future Improvements

- Implementation of advanced vector indexing (e.g., Qdrant, Chroma) for scaling to larger document bases.
- Conversational memory to support follow-up questions natively.
- UI enhancements for deeper document exploration and source highlighting.

## Project Status

**Stable / Final Version**
The application successfully integrates Microsoft Foundry Local to provide a robust, private, and local-first Retrieval-Augmented Generation experience with an advanced UI and incremental indexing capabilities.
