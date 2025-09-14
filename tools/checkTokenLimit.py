"""
check_token_limits.py: 
Scans JSON files in specified folders and reports those exceeding the OpenAI embeddings per-input token limit.
Usage:
    python check_token_limits.py --folders path/to/api_json path/to/auto_gen_json
"""

import os
import glob
import json
import argparse

try:
    import tiktoken
except ImportError:
    print("Please install tiktoken: pip install tiktoken")
    exit(1)

def count_tokens(text: str, model: str = "text-embedding-3-large") -> int:
    """Return number of tokens for `text` using the specified model."""
    tokenizer = tiktoken.encoding_for_model(model)
    return len(tokenizer.encode(text))

def find_oversized(folders, max_tokens=8191, model="text-embedding-3-large"):
    oversized = []
    for folder in folders:
        pattern = os.path.join(folder, "**", "*.json")
        for path in glob.glob(pattern, recursive=True):
            try:
                with open(path, "r", encoding="utf8") as f:
                    data = json.load(f)
                # dump back to string for token count
                payload = json.dumps(data)
                token_count = count_tokens(payload, model)
                if token_count > max_tokens:
                    oversized.append((path, token_count))
            except Exception as e:
                print(f"Error processing {path}: {e}")
    return oversized

def main():
    parser = argparse.ArgumentParser(
        description="Check JSON files for exceeding OpenAI embedding token limits."
    )
    parser.add_argument(
        "--folders", "-f",
        nargs="+",
        required=True,
        help="Folder paths to scan for .json files"
    )
    parser.add_argument(
        "--max-tokens", "-m",
        type=int,
        default=8191,
        help="Maximum allowed tokens per individual input"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="text-embedding-3-large",
        help="Model name for tokenization"
    )
    args = parser.parse_args()

    oversized = find_oversized(args.folders, args.max_tokens, args.model)
    if not oversized:
        print("✅ All JSON files are within the token limit.")
    else:
        print("⚠️ The following files exceed the per-input token limit:")
        for path, count in oversized:
            print(f"  • {path}: {count} tokens")

if __name__ == "__main__":
    main()
