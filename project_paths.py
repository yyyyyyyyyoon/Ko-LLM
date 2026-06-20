import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

DATA_ROOT = Path(
    os.environ.get("KOLLM_DATA_ROOT", PROJECT_ROOT / "dataset")
).resolve()

OUTPUT_ROOT = Path(
    os.environ.get("KOLLM_OUTPUT_ROOT", PROJECT_ROOT / "outputs")
).resolve()

TOKENIZER_DIR = Path(
    os.environ.get("KOLLM_TOKENIZER_DIR", PROJECT_ROOT / "tokenizer_bpe_64k")
).resolve()

PACKED_DATA_DIR = DATA_ROOT / "packed_corpus_4k"

PROCESSED_DATA_DIR = DATA_ROOT / "processed"
TOKENIZER_TRAIN_DIR = DATA_ROOT / "tokenizer_train"
TOKENIZER_TRAIN_FILE = TOKENIZER_TRAIN_DIR / "tokenizer_train_data.txt"
TOKENIZER_OUTPUT_DIR = OUTPUT_ROOT / "tokenizer"

SFT_DIR = DATA_ROOT / "sft" / "summary"
SUMMARY_TRAIN_JSONL = SFT_DIR / "train.jsonl"
SUMMARY_VALID_JSONL = SFT_DIR / "valid.jsonl"
SUMMARY_TEST_JSONL = SFT_DIR / "test.jsonl"