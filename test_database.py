import sys
import os
import json
import sqlite3
from foundry_local_sdk import Configuration, FoundryLocalManager

def main():
    # Setup data directory
    os.makedirs('data', exist_ok=True)
    db_path = os.path.join('data', 'knowledge.db')

    # Initialize DB
    print("Initializing SQLite database...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding TEXT NOT NULL
        )
    ''')
    
    # Clear table for idempotency
    cursor.execute('DELETE FROM chunks')
    # Reset AUTOINCREMENT if needed (optional, but good for clean output)
    cursor.execute('DELETE FROM sqlite_sequence WHERE name="chunks"')
    conn.commit()

    # Initialize SDK
    print("Initializing Foundry Local SDK...")
    config = Configuration(app_name="LocalRAGApp")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    model_alias = "qwen3-embedding-0.6b"
    model = manager.catalog.get_model(model_alias)
    
    if not model:
        print(f"Model '{model_alias}' not found in catalog.")
        sys.exit(1)

    if not model.is_cached:
        print(f"Downloading '{model_alias}'... This might take a minute.")
        model.download()
    
    if not model.is_loaded:
        print(f"Loading '{model_alias}' into memory...")
        model.load()
    
    client = model.get_embedding_client()
    
    # Sample documents
    documents = [
        {"source": "python_notes.txt", "content": "Python is a high-level programming language commonly used for software development."},
        {"source": "database_notes.txt", "content": "SQLite is a lightweight serverless relational database stored in a single file."},
        {"source": "food_notes.txt", "content": "Pizza is a popular Italian dish traditionally made with dough, tomato sauce and cheese."}
    ]
    
    # Generate embeddings
    print("Generating embeddings...")
    contents = [doc["content"] for doc in documents]
    response = client.generate_embeddings(contents)
    
    # Insert into DB
    print("Inserting into database...")
    for i, doc in enumerate(documents):
        embedding_list = response.data[i].embedding
        embedding_json = json.dumps(embedding_list)
        
        cursor.execute('''
            INSERT INTO chunks (source, content, embedding)
            VALUES (?, ?, ?)
        ''', (doc["source"], doc["content"], embedding_json))
        
    conn.commit()
    
    # Read back from DB
    print("\n--- Reading from Database ---")
    cursor.execute('SELECT id, source, content, embedding FROM chunks')
    rows = cursor.fetchall()
    
    for row in rows:
        row_id, source, content, embedding_json = row
        embedding_list = json.loads(embedding_json)
        dim = len(embedding_list)
        print(f"ID: {row_id} | Source: {source} | Content: {content} | Vector Dimension: {dim}")
        
    print(f"\nTotal rows stored: {len(rows)}")
        
    # Cleanup
    print(f"\nUnloading '{model_alias}'...")
    model.unload()
    conn.close()

if __name__ == "__main__":
    main()
