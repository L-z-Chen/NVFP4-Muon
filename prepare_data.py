"""Download TinyShakespeare and tokenize it with the GPT-2 BPE tokenizer,
saving train.bin / val.bin as uint16 token streams (nanoGPT layout)."""

import os

import numpy as np
import requests
import tiktoken

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    raw_path = os.path.join(DATA_DIR, "input.txt")
    if not os.path.exists(raw_path):
        print("downloading tiny_shakespeare...")
        text = requests.get(URL, timeout=60).text
        with open(raw_path, "w") as f:
            f.write(text)
    else:
        text = open(raw_path).read()

    n = len(text)
    train_text = text[: int(n * 0.9)]
    val_text = text[int(n * 0.9):]

    enc = tiktoken.get_encoding("gpt2")
    train_ids = np.array(enc.encode_ordinary(train_text), dtype=np.uint16)
    val_ids = np.array(enc.encode_ordinary(val_text), dtype=np.uint16)
    print(f"train has {len(train_ids):,} tokens, val has {len(val_ids):,} tokens")

    train_ids.tofile(os.path.join(DATA_DIR, "train.bin"))
    val_ids.tofile(os.path.join(DATA_DIR, "val.bin"))
    print("wrote train.bin / val.bin to", DATA_DIR)


if __name__ == "__main__":
    main()
