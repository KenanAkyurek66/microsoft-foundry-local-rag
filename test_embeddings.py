import sys
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
    else:
        print(f"Model '{model_alias}' is already cached.")
    
    if not model.is_loaded:
        print(f"Loading '{model_alias}' into memory...")
        model.load()
    else:
        print(f"Model '{model_alias}' is already loaded.")
    
    print("Initializing embedding client...")
    client = model.get_embedding_client()
    
    sentences = [
        "Python is a programming language.",
        "Python is commonly used for software development.",
        "I like eating pizza."
    ]
    
    print("Generating embeddings...")
    response = client.generate_embeddings(sentences)
    
    embeddings = [data.embedding for data in response.data]
    
    print("\n--- Embedding Information ---")
    dim = len(embeddings[0])
    print(f"Vector Dimension: {dim}")
    
    for i, emb in enumerate(embeddings):
        print(f"Sentence {i+1} first 5 values: {emb[:5]}")
        
    print("\n--- Cosine Similarity ---")
    sim1_2 = cosine_similarity(embeddings[0], embeddings[1])
    sim1_3 = cosine_similarity(embeddings[0], embeddings[2])
    
    print(f"Sentence 1 <-> Sentence 2: {sim1_2:.4f}")
    print(f"Sentence 1 <-> Sentence 3: {sim1_3:.4f}")
    
    print(f"\nUnloading '{model_alias}'...")
    model.unload()

if __name__ == "__main__":
    main()
