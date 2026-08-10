import os
import sys
import glob
import json
import sqlite3
from foundry_local_sdk import Configuration, FoundryLocalManager

def chunk_text(text):
    """Split text into chunks by blank lines and trim whitespace."""
    # Split by double newline to separate paragraphs
    paragraphs = text.split('\n\n')
    chunks = [p.strip() for p in paragraphs if p.strip()]
    return chunks

def main():
    documents_dir = 'documents'
    db_path = os.path.join('data', 'knowledge.db')
    
    # 1. Setup DB
    os.makedirs('data', exist_ok=True)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
    ''')
    
    cursor.execute('DELETE FROM chunks')
    cursor.execute('DELETE FROM sqlite_sequence WHERE name="chunks"')
    conn.commit()
    
    # 2. Read and chunk documents
    txt_files = glob.glob(os.path.join(documents_dir, '*.txt'))
    if not txt_files:
        print(f"No .txt files found in {documents_dir}")
        sys.exit(1)
        
    documents = []
    for filepath in txt_files:
        filename = os.path.basename(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        
        chunks = chunk_text(text)
        for chunk in chunks:
            documents.append({
                "source": filename,
                "content": chunk
            })
            
    num_files = len(txt_files)
    num_chunks = len(documents)
    
    if num_chunks == 0:
        print("No content found in documents.")
        sys.exit(1)
        
    # 3. Setup Foundry Local
    config = Configuration(app_name="LocalRAGApp")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    
    model_alias = "qwen3-embedding-0.6b"
    model = manager.catalog.get_model(model_alias)
    if not model:
        print(f"Model '{model_alias}' not found in catalog.")
        sys.exit(1)
        
    if not model.is_cached:
        model.download()
    if not model.is_loaded:
        model.load()
        
    client = model.get_embedding_client()
    
    # 4. Generate embeddings
    contents = [doc["content"] for doc in documents]
    response = client.generate_embeddings(contents)
    
    embedding_dim = len(response.data[0].embedding)
    
    # 5. Insert into DB
    for i, doc in enumerate(documents):
        emb_list = response.data[i].embedding
        emb_json = json.dumps(emb_list)
        cursor.execute('''
            INSERT INTO chunks (source, content, embedding)
            VALUES (?, ?, ?)
        ''', (doc["source"], doc["content"], emb_json))
        
    conn.commit()
    
    # 6. Cleanup
    model.unload()
    conn.close()
    
    # 7. Print summary
    print(f"Files processed: {num_files}")
    print(f"Chunks created: {num_chunks}")
    print(f"Embedding dimension: {embedding_dim}")
    print(f"Database: {db_path}")

if __name__ == "__main__":
    main()
