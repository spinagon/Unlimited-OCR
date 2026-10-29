#!/usr/bin/env python3
"""Pre-download or verify Unlimited-OCR-2 weights and configuration from Hugging Face."""

import sys
from huggingface_hub import snapshot_download

MODEL_ID = "baidu/Unlimited-OCR-2"

def main():
    print(f"Checking / downloading model snapshot for '{MODEL_ID}'...")
    try:
        model_path = snapshot_download(
            repo_id=MODEL_ID,
            allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt", "tokenizer*"],
            ignore_patterns=["assets/*", "*.pdf", "wheel/*"],
        )
        print(f"\n[OK] Model snapshot ready at: {model_path}")
    except Exception as e:
        print(f"\n[ERROR] Failed to download model snapshot: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
