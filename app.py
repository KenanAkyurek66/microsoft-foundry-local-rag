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
        print("Knowledge base not found. Please run 'python ingest.py' first.")
        sys.exit(1)
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute('SELECT source, content, embedding FROM chunks')
        rows = cursor.fetchall()
    except sqlite3.OperationalError:
        print("Knowledge base is empty. Please run 'python ingest.py' first.")
        sys.exit(1)
        
    if not rows:
        print("Knowledge base is empty. Please run 'python ingest.py' first.")
        sys.exit(1)
        
    print("Loading models (this might take a moment)...")
    config = Configuration(app_name="LocalRAGApp")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    embed_model_alias = "qwen3-embedding-0.6b"
    embed_model = manager.catalog.get_model(embed_model_alias)
    if not embed_model.is_cached:
        embed_model.download()
    if not embed_model.is_loaded:
        embed_model.load()
    embed_client = embed_model.get_embedding_client()
    
    chat_model_alias = "qwen2.5-1.5b"
    chat_model = manager.catalog.get_model(chat_model_alias)
    if not chat_model.is_cached:
        chat_model.download()
    if not chat_model.is_loaded:
        chat_model.load()
    chat_client = chat_model.get_chat_client()

    print("\n========================================")
    print("Local RAG Assistant - Foundry Local")
    print("========================================")
    print(f"\nKnowledge base loaded: {len(rows)} chunks\n")
    print("Type your question or 'exit' to quit.")
    
    system_instruction = (
        "Answer only from the supplied context.\n"
        "Do not use outside knowledge.\n"
        "Do not invent information.\n"
        "If the context does not contain enough information, say exactly:\n"
        "I don't have enough information in the provided documents.\n"
        "Keep the response concise and natural.\n"
        "You must format your response exactly like this:\n"
        "Answer:\n<model answer>\n\n"
        "Source:\n<source filename(s)>"
    )

    try:
        while True:
            try:
                question = input("\nQuestion: ").strip()
            except EOFError:
                break
                
            if not question:
                print("Please enter a question.")
                continue
                
            if question.lower() in ["exit", "quit"]:
                break
                
            # 1. Generate Query Embedding
            embed_response = embed_client.generate_embedding(question)
            query_embedding = embed_response.data[0].embedding
            
            # 2. Similarity Search
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
            top_k = 3
            top_results = results[:top_k]
            
            # 3. Context Construction
            context_str = ""
            for res in top_results:
                context_str += f"SOURCE: {res['source']}\nCONTENT: {res['content']}\n\n"
                
            user_prompt = f"Context:\n{context_str}\nQuestion: {question}"
            messages = [
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": user_prompt}
            ]
            
            # 4. Generate Answer
            chat_response = chat_client.complete_chat(messages)
            answer = chat_response.choices[0].message.content.strip()
            
            fallback_phrase = "I don't have enough information in the provided documents."
            if "don't have enough information" in answer.lower() or "do not have enough information" in answer.lower():
                answer = fallback_phrase
            
            print(f"\n{answer}")

    except KeyboardInterrupt:
        print()
        
    print("\nShutting down... unloading models.")
    embed_model.unload()
    chat_model.unload()
    conn.close()
    print("Goodbye!")

if __name__ == "__main__":
    main()
