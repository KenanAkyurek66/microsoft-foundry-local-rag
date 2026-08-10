import sys
from foundry_local_sdk import Configuration, FoundryLocalManager

def main():
    print("Initializing Foundry Local SDK...")
    config = Configuration(app_name="LocalRAGApp")
    FoundryLocalManager.initialize(config)
    manager = FoundryLocalManager.instance

    # Use a small model to make the test fast
    model_alias = "qwen2.5-0.5b"
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
    
    print("Initializing chat client...")
    client = model.get_chat_client()
    
    prompt = "Explain RAG in one sentence."
    print(f"Sending prompt: '{prompt}'")
    
    response = client.complete_chat([
        {"role": "user", "content": prompt}
    ])
    
    print("\n--- Model Response ---")
    print(response.choices[0].message.content.strip())
    print("----------------------\n")
    
    print(f"Unloading '{model_alias}'...")
    model.unload()

if __name__ == "__main__":
    main()
