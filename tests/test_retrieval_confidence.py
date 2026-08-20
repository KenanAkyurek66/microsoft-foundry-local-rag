import os
import sqlite3
from foundry_local_sdk import Configuration, FoundryLocalManager
from retrieval import retrieve, MIN_RETRIEVAL_SIMILARITY

def main():
    print("Testing Retrieval Confidence Gate...")
    db_path = os.path.join('data', 'knowledge.db')
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT source, content, embedding FROM chunks')
    rows = cursor.fetchall()
    conn.close()
    
    assert len(rows) > 0, "Knowledge base is empty"

    config = Configuration(app_name="LocalRAGApp")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance
    
    model = manager.catalog.get_model('qwen3-embedding-0.6b')
    if not model.is_cached: model.download()
    if not model.is_loaded: model.load()
    client = model.get_embedding_client()
    
    supported_tests = [
        ("What is RAG?", "rag.txt"),
        ("What is Python?", "python.txt"),
        ("Why does SQLite not need a separate server?", "sqlite.txt")
    ]
    
    unsupported_tests = [
        "Who won the 2014 FIFA World Cup?",
        "What is the capital of Japan?"
    ]
    
    try:
        # A, B, C: Supported Tests
        for q, expected_source in supported_tests:
            q_emb = client.generate_embedding(q).data[0].embedding
            results = retrieve(q_emb, rows, top_k=3)
            
            assert results, f"Failed: No results for '{q}'"
            top_score = results[0]['score']
            print(f"[{q}] Score: {top_score:.4f} (Threshold: {MIN_RETRIEVAL_SIMILARITY})")
            
            assert top_score >= MIN_RETRIEVAL_SIMILARITY, f"Failed: '{q}' failed threshold"
            assert expected_source in [r['source'] for r in results], f"Failed: Expected '{expected_source}' in top results"
            print(f" -> PASS: Threshold cleared, retrieved {expected_source}")

        # D, E: Unsupported Tests
        for q in unsupported_tests:
            q_emb = client.generate_embedding(q).data[0].embedding
            results = retrieve(q_emb, rows, top_k=3)
            
            top_score = results[0]['score'] if results else 0
            print(f"[{q}] Score: {top_score:.4f} (Threshold: {MIN_RETRIEVAL_SIMILARITY})")
            
            assert top_score < MIN_RETRIEVAL_SIMILARITY, f"Failed: Unsupported '{q}' erroneously cleared threshold"
            print(" -> PASS: Gate blocked unsupported query")
            
        print("\nALL RETRIEVAL CONFIDENCE TESTS PASSED!")
        
    finally:
        model.unload()

if __name__ == "__main__":
    main()
