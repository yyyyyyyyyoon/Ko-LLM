import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer


DATA_ROOT = Path(r"C:\Users\dbstj\dataset")
TOKENIZER_DIR = DATA_ROOT / "tokenizer_bpe_64k"
OUTPUT_PATH = Path("outputs") / "tokenizer_eval_report.json"
MAX_DOCS_PER_GROUP = 10000
MAX_EXAMPLES = 3

EVAL_FILES = {
    "ko": [
        DATA_ROOT / "processed" / "ko_aihub_test.jsonl",
    ],
    "en": [
        DATA_ROOT / "processed" / "enwiki_test.jsonl",
    ],
    "code": [
        DATA_ROOT / "processed" / "code_test.txt",
    ],
}

TRAIN_FALLBACK_FILES = {
    "ko": [
        DATA_ROOT / "processed" / "kowiki_train.jsonl",
        DATA_ROOT / "processed" / "ko_aihub_train.jsonl",
    ],
    "en": [
        DATA_ROOT / "processed" / "enwiki_train.jsonl",
    ],
    "code": [
        DATA_ROOT / "processed" / "code_train.txt",
    ],
}


def iter_jsonl_text(file_path: Path):
    if not file_path.exists():
        print(f"[WARN] File not found: {file_path}")
        return

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = str(obj.get("text", "")).strip() if isinstance(obj, dict) else ""
            if text:
                yield text


def iter_txt_blocks(file_path: Path):
    if not file_path.exists():
        print(f"[WARN] File not found: {file_path}")
        return

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    for block in content.split("\n\n"):
        block = block.strip()
        if block:
            yield block


def iter_texts(file_path: Path):
    if file_path.suffix == ".jsonl" or "wiki" in file_path.name:
        yield from iter_jsonl_text(file_path)
    else:
        yield from iter_txt_blocks(file_path)


def resolve_group_files(label: str):
    eval_files = [path for path in EVAL_FILES[label] if path.exists()]
    if eval_files:
        return eval_files, "holdout"

    fallback_files = [path for path in TRAIN_FALLBACK_FILES[label] if path.exists()]
    if fallback_files:
        print(f"[WARN] No holdout files found for {label}. Falling back to train files.")
        return fallback_files, "train_fallback"

    return [], "missing"


def evaluate_group(tokenizer, label: str, files):
    total_docs = 0
    total_chars = 0
    total_tokens = 0
    unk_count = 0
    roundtrip_success = 0
    roundtrip_fail = 0
    used_token_ids = Counter()
    failed_examples = []

    unk_id = tokenizer.unk_token_id
    vocab_size = tokenizer.vocab_size or len(tokenizer)

    for file_path in files:
        for text in iter_texts(file_path):
            if total_docs >= MAX_DOCS_PER_GROUP:
                break

            ids = tokenizer.encode(text, add_special_tokens=False)
            decoded = tokenizer.decode(ids, clean_up_tokenization_spaces=False)
            token_count = len(ids)

            total_docs += 1
            total_chars += len(text)
            total_tokens += token_count
            used_token_ids.update(ids)

            if unk_id is not None:
                unk_count += sum(1 for token_id in ids if token_id == unk_id)

            if decoded == text:
                roundtrip_success += 1
            else:
                roundtrip_fail += 1
                if len(failed_examples) < MAX_EXAMPLES:
                    failed_examples.append(
                        {
                            "file": str(file_path),
                            "input_preview": text[:200],
                            "decoded_preview": decoded[:200],
                        }
                    )

        if total_docs >= MAX_DOCS_PER_GROUP:
            break

    if total_docs == 0:
        print(f"\n[{label}] No documents evaluated.")
        return None

    used_vocab = len(used_token_ids)
    return {
        "label": label,
        "docs": total_docs,
        "chars": total_chars,
        "tokens": total_tokens,
        "avg_chars_per_doc": total_chars / total_docs,
        "avg_tokens_per_doc": total_tokens / total_docs,
        "avg_token_length_in_chars": total_chars / total_tokens if total_tokens else 0.0,
        "chars_per_token": total_chars / total_tokens if total_tokens else 0.0,
        "tokens_per_char": total_tokens / total_chars if total_chars else 0.0,
        "unk_count": unk_count,
        "unk_ratio": unk_count / total_tokens if total_tokens else 0.0,
        "roundtrip_success": roundtrip_success,
        "roundtrip_success_ratio": roundtrip_success / total_docs,
        "roundtrip_fail": roundtrip_fail,
        "roundtrip_fail_ratio": roundtrip_fail / total_docs,
        "used_vocab": used_vocab,
        "used_vocab_ratio": used_vocab / vocab_size if vocab_size else 0.0,
        "top_used_tokens": [
            {
                "token_id": token_id,
                "token": tokenizer.convert_ids_to_tokens([token_id])[0],
                "count": count,
            }
            for token_id, count in used_token_ids.most_common(20)
        ],
        "roundtrip_fail_examples": failed_examples,
    }


def print_result(result):
    print(f"\n[{result['label']}]")
    print(f"Docs: {result['docs']}")
    print(f"Total chars: {result['chars']:,}")
    print(f"Total tokens: {result['tokens']:,}")
    print(f"Avg chars/doc: {result['avg_chars_per_doc']:.2f}")
    print(f"Avg tokens/doc: {result['avg_tokens_per_doc']:.2f}")
    print(f"Chars/token: {result['chars_per_token']:.3f}")
    print(f"Tokens/char: {result['tokens_per_char']:.3f}")
    print(f"UNK ratio: {result['unk_ratio']:.6f}")
    print(f"Used vocab: {result['used_vocab']:,}")
    print(f"Used vocab ratio: {result['used_vocab_ratio']:.6f}")
    print(f"Round-trip success ratio: {result['roundtrip_success_ratio']:.6f}")
    print(f"Round-trip fail ratio: {result['roundtrip_fail_ratio']:.6f}")


def main():
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))

    file_plan = {}
    for label in ("ko", "en", "code"):
        files, source = resolve_group_files(label)
        file_plan[label] = {
            "source": source,
            "files": [str(path) for path in files],
        }

    print("Tokenizer:", TOKENIZER_DIR)
    print("Vocab size:", tokenizer.vocab_size)
    print("File plan:")
    for label, info in file_plan.items():
        print(f"  - {label}: {info['source']} ({len(info['files'])} files)")

    results = []
    for label in ("ko", "en", "code"):
        files = [Path(path) for path in file_plan[label]["files"]]
        result = evaluate_group(tokenizer, label, files)
        if result is not None:
            result["file_source"] = file_plan[label]["source"]
            result["files"] = file_plan[label]["files"]
            results.append(result)

    report = {
        "tokenizer_dir": str(TOKENIZER_DIR),
        "vocab_size": tokenizer.vocab_size,
        "max_docs_per_group": MAX_DOCS_PER_GROUP,
        "results": results,
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n===== Tokenizer Evaluation Result =====")
    for result in results:
        print_result(result)
    print(f"\nReport saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
