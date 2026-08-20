import math
import json

# Based on empirical measurement:
# Supported documents typically score > 0.55
# Unsupported factual questions typically score < 0.35
# 0.40 provides a reliable, conservative separation.
MIN_RETRIEVAL_SIMILARITY = 0.40

def cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot_product / (mag1 * mag2)

def retrieve(query_embedding, rows, top_k=3):
    """
    Computes similarity scores for the query against all database rows.
    Returns a sorted list of dictionaries with structure:
    [{'source': str, 'content': str, 'score': float}, ...]
    """
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
    return results[:top_k]
