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
        print(f"Error: Database file '{db_path}' does not exist.")
        sys.exit(1)
        
    print("Initializing SQLite database...")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('SELECT source, content, embedding FROM chunks')
    rows = cursor.fetchall()
    if not rows:
        print("Error: Table 'chunks' has no rows.")
        sys.exit(1)

    print("Initializing Foundry Local SDK...")
    config = Configuration(app_name="LocalRAGApp")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    # 1. Setup embedding model
    embed_model_alias = "qwen3-embedding-0.6b"
    embed_model = manager.catalog.get_model(embed_model_alias)
    if not embed_model.is_cached:
        print(f"Downloading '{embed_model_alias}'...")
        embed_model.download()
    if not embed_model.is_loaded:
        embed_model.load()
    embed_client = embed_model.get_embedding_client()
    
    # 2. Setup chat model
    chat_model_alias = "qwen2.5-1.5b"
    chat_model = manager.catalog.get_model(chat_model_alias)
    if not chat_model:
        print(f"Model '{chat_model_alias}' not found in catalog.")
        sys.exit(1)
        
    if not chat_model.is_cached:
        print(f"Downloading '{chat_model_alias}'... This might take a few minutes.")
        chat_model.download()
    if not chat_model.is_loaded:
        print(f"Loading '{chat_model_alias}' into memory...")
        chat_model.load()
    chat_client = chat_model.get_chat_client()

    # 3. Retrieval Pipeline
    question = "Which database is lightweight and serverless?"
    print("\nGenerating query embedding...")
    response = embed_client.generate_embedding(question)
    query_embedding = response.data[0].embedding
    
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
        
    results.sort(key=lambda x: x["score"], reverse=True)
    top_k = 2
    top_results = results[:top_k]
    
    # Build context string
    context_str = ""
    for res in top_results:
        context_str += f"SOURCE: {res['source']}\nCONTENT: {res['content']}\n\n"
        
    print("\n--- User Question ---")
    print(question)
    
    print("\n--- Retrieved Context ---")
    for res in top_results:
        print(f"Source: {res['source']} | Score: {res['score']:.4f}")
    
    system_instruction = (
        "Answer ONLY using the provided context.\n"
        "Do not use outside knowledge.\n"
        "If the answer cannot be found in the context, say:\n"
        "I don't have enough information in the provided documents.\n"
        "Keep the answer concise.\n"
        "You must explicitly format your response exactly like this:\n"
        "Answer: <concise answer>\n"
        "Source: <source filename>"
    )
    
    user_prompt = f"Context:\n{context_str}\nQuestion: {question}"
    
    messages = [
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_prompt}
    ]
    
    print(f"\nGenerating answer with {chat_model_alias}...")
    chat_response = chat_client.complete_chat(messages)
    answer = chat_response.choices[0].message.content.strip()
    
    print("\n--- Model Answer ---")
    print(answer)
    print("--------------------")
    
    # 5. Validation
    is_db_first = (top_results[0]["source"] == "database_notes.txt")
    has_sqlite = ("SQLite" in answer or "sqlite" in answer.lower())
    has_source = ("database_notes.txt" in answer)
    
    print()
    if is_db_first and has_sqlite and has_source:
        print("RAG test PASSED")
    else:
        print("RAG test FAILED")
        
    # 6. Cleanup
    print("\nUnloading models...")
    embed_model.unload()
    chat_model.unload()
    conn.close()

if __name__ == "__main__":
    main()
