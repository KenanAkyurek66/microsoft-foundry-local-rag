import os
import sys
import glob
from foundry_local_sdk import Configuration, FoundryLocalManager
from document_ingestion import index_document

def main():
    documents_dir = 'documents'
    db_path = os.path.join('data', 'knowledge.db')
    
    # 1. Setup Foundry Local Embeddings
    print("Initializing embedding model...")
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
    
    # 2. Discover Documents
    supported_extensions = ('.txt', '.pdf', '.docx')
    filepaths = []
    if os.path.exists(documents_dir):
        for root, _, files in os.walk(documents_dir):
            for file in files:
                if file.lower().endswith(supported_extensions):
                    filepaths.append(os.path.join(root, file))
                    
    num_discovered = len(filepaths)
    if num_discovered == 0:
        print(f"No supported documents found in {documents_dir}")
        sys.exit(1)
        
    # 3. Index Documents
    stats = {
        "indexed": 0,
        "updated": 0,
        "skipped": 0,
        "error": 0,
        "chunks_created": 0
    }
    
    print(f"Discovered {num_discovered} documents. Starting ingestion...\n")
    for filepath in filepaths:
        filename = os.path.basename(filepath)
        try:
            with open(filepath, 'rb') as f:
                file_bytes = f.read()
                
            result = index_document(filename, file_bytes, client, db_path)
            
            status = result["status"]
            stats[status] += 1
            if status != "error":
                stats["chunks_created"] += result["chunks_created"]
                
            if status == "error":
                print(f"[ERROR] {filename}: {result.get('error', 'Unknown error')}")
            else:
                print(f"[{status.upper()}] {filename} ({result['chunks_created']} chunks)")
                
        except Exception as e:
            stats["error"] += 1
            print(f"[ERROR] Failed to read {filename}: {str(e)}")
            
    # 4. Cleanup
    model.unload()
    
    # 5. Print summary
    print("\n--- Ingestion Summary ---")
    print(f"Documents discovered: {num_discovered}")
    print(f"Indexed: {stats['indexed']}")
    print(f"Updated: {stats['updated']}")
    print(f"Skipped: {stats['skipped']}")
    if stats["error"] > 0:
        print(f"Errors: {stats['error']}")
    print(f"Chunks created: {stats['chunks_created']}")
    print(f"Database: {db_path}")

if __name__ == "__main__":
    main()
