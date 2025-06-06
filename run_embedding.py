# run_embedding.py

import sys
from embed_client import EmbedClient

def main():
    if len(sys.argv) < 2:
        print("Usage: python run_embedding.py \"Your text here\"")
        sys.exit(1)

    text_to_embed = sys.argv[1]
    client = EmbedClient()
    try:
        vector = client.embed_text(text_to_embed)
    except Exception as e:
        print(f"[Error] {e}")
        sys.exit(1)

    # Print out the first few dimensions for brevity, then show total length
    print(f"Embedding length: {len(vector)}")
    print("First 5 values:", vector[:5])

if __name__ == "__main__":
    main()
