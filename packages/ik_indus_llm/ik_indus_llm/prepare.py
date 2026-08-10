"""Pre-tokenize a text file into a .bin for TextDataset.

Usage:
    python -m llm_from_scratch.prepare data/my_corpus.txt data/my_corpus.bin
"""

import sys
from .data import prepare_text_bin


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m llm_from_scratch.prepare <input.txt> <output.bin>")
        sys.exit(1)
    prepare_text_bin(sys.argv[1], sys.argv[2])
