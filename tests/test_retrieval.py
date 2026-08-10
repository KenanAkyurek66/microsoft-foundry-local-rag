import sys
import os
import json
import sqlite3
import math
from foundry_local_sdk import Configuration, FoundryLocalManager

def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)

def main():
    db_path = os.path.join('data', 'knowledge.db')
    if not os.path.exists(db_path):
        print(f"Error: Database file '{db_path}' does not exist. Please run test_database.py first.")
        sys.exit(1)
        
    print("Initializing SQLite database...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check if table has rows
    try:
        cursor.execute('SELECT source, content, embedding FROM chunks')
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        print("Error: Table 'chunks' does not exist in the database.")
        sys.exit(1)
        
    if not rows:
        print("Error: Table 'chunks' has no rows.")
        sys.exit(1)

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
    
    query = "Which database is lightweight and serverless?"
    print(f"\nQuery: '{query}'")
    print("Generating query embedding...")
    
    response = client.generate_embedding(query)
    query_embedding = response.data[0].embedding
    
    print("\n--- All Document Similarities ---")
    results = []
    for row in rows:
        source, content, embedding_json = row
        doc_embedding = json.loads(embedding_json)
        sim = cosine_similarity(query_embedding, doc_embedding)
        results.append({
            "source": source,
            "content": content,
            "score": sim
        })
        print(f"Source: {source} | Similarity: {sim:.4f}")
        
    # Sort results
    results.sort(key=lambda x: x["score"], reverse=True)
    
    print("\n--- Top-2 Retrieved Results ---")
    top_k = 2
    for i in range(min(top_k, len(results))):
        res = results[i]
        print(f"Rank: {i+1}")
        print(f"Source: {res['source']}")
        print(f"Score: {res['score']:.4f}")
        print(f"Content: {res['content']}\n")
        
    # Validation
    if results and results[0]["source"] == "database_notes.txt":
        print("Retrieval test PASSED")
    else:
        print("Retrieval test FAILED")
        
    # Cleanup
    print(f"\nUnloading '{model_alias}'...")
    model.unload()
    conn.close()

if __name__ == "__main__":
    main()
