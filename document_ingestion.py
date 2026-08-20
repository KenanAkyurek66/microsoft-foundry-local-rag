import os
import io
import json
import sqlite3
import hashlib
from pypdf import PdfReader
import docx

def extract_txt(file_bytes):
    return file_bytes.decode('utf-8', errors='replace')

def extract_pdf(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    text_pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text_pages.append(page_text)
    if not text_pages:
        raise ValueError("PDF contains no extractable text.")
    return "\n\n".join(text_pages)

def extract_docx(file_bytes):
    doc = docx.Document(io.BytesIO(file_bytes))
    paragraphs = []
    for para in doc.paragraphs:
        if para.text.strip():
            paragraphs.append(para.text.strip())
    if not paragraphs:
        raise ValueError("DOCX contains no extractable text.")
    return "\n\n".join(paragraphs)

def extract_text(filename, file_bytes):
    ext = os.path.splitext(filename)[1].lower()
    if ext == '.txt':
        return extract_txt(file_bytes)
    elif ext == '.pdf':
        return extract_pdf(file_bytes)
    elif ext == '.docx':
        return extract_docx(file_bytes)
    else:
        raise ValueError(f"Unsupported file format: {ext}")

def chunk_text(text, target_words=400, overlap_words=50):
    paragraphs = text.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    chunks = []
    current_paragraphs = []
    current_word_count = 0
    
    for p in paragraphs:
        p_words = p.split()
        p_word_count = len(p_words)
        
        if p_word_count == 0:
            continue
            
        if current_word_count + p_word_count > target_words and current_paragraphs:
            # Finalize current chunk
            chunk_text = '\n\n'.join(current_paragraphs)
            chunks.append(chunk_text)
            
            # Calculate overlap from the end of the joined chunk text
            chunk_words = chunk_text.split()
            overlap_text = " ".join(chunk_words[-overlap_words:]) if overlap_words > 0 and len(chunk_words) > overlap_words else ""
            
            # Start new chunk
            if overlap_text:
                current_paragraphs = [overlap_text, p]
                current_word_count = len(overlap_text.split()) + p_word_count
            else:
                current_paragraphs = [p]
                current_word_count = p_word_count
        else:
            current_paragraphs.append(p)
            current_word_count += p_word_count
            
    if current_paragraphs:
        chunks.append('\n\n'.join(current_paragraphs))
        
    return chunks

def setup_database(db_path):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    
    # Existing chunks table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
    ''')
    
    # New documents metadata table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS documents (
            source TEXT PRIMARY KEY,
            file_hash TEXT NOT NULL,
            file_type TEXT NOT NULL,
            indexed_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    
    conn.commit()
    return conn

def remove_document(db_path, source):
    conn = setup_database(db_path)
    cursor = conn.cursor()
    try:
        cursor.execute('DELETE FROM documents WHERE source = ?', (source,))
        cursor.execute('DELETE FROM chunks WHERE source = ?', (source,))
        conn.commit()
    finally:
        conn.close()

def index_document(filename, file_bytes, embedding_client, db_path):
    """
    Indexes a document incrementally.
    Returns a dict with 'status' (indexed, updated, skipped, error), 'filename', 'chunks_created'.
    """
    try:
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        file_ext = os.path.splitext(filename)[1].lower()
        
        conn = setup_database(db_path)
        cursor = conn.cursor()
        
        # Check if exists
        cursor.execute('SELECT file_hash FROM documents WHERE source = ?', (filename,))
        row = cursor.fetchone()
        
        if row:
            existing_hash = row[0]
            if existing_hash == file_hash:
                conn.close()
                return {"status": "skipped", "filename": filename, "chunks_created": 0}
            else:
                status = "updated"
        else:
            status = "indexed"
            
        # Extract and chunk
        text = extract_text(filename, file_bytes)
        chunks = chunk_text(text)
        
        if not chunks:
            conn.close()
            return {"status": "error", "filename": filename, "chunks_created": 0, "error": "No meaningful chunks created."}
            
        # Generate embeddings
        response = embedding_client.generate_embeddings(chunks)
        
        # Transaction: Update metadata and chunks
        cursor.execute('BEGIN TRANSACTION')
        try:
            # Always delete old metadata and chunks to prevent duplicates from legacy or updated documents
            cursor.execute('DELETE FROM documents WHERE source = ?', (filename,))
            cursor.execute('DELETE FROM chunks WHERE source = ?', (filename,))
                
            # Insert new metadata
            cursor.execute('''
                INSERT INTO documents (source, file_hash, file_type)
                VALUES (?, ?, ?)
            ''', (filename, file_hash, file_ext))
            
            # Insert new chunks
            for i, chunk_text_content in enumerate(chunks):
                emb_list = response.data[i].embedding
                emb_json = json.dumps(emb_list)
                cursor.execute('''
                    INSERT INTO chunks (source, content, embedding)
                    VALUES (?, ?, ?)
                ''', (filename, chunk_text_content, emb_json))
                
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
            
        return {"status": status, "filename": filename, "chunks_created": len(chunks)}
        
    except Exception as e:
        return {"status": "error", "filename": filename, "chunks_created": 0, "error": str(e)}
