import json
import random
from pathlib import Path
from typing import Iterator, List

import torch
from transformers import AutoTokenizer

from project_paths import DATA_ROOT, TOKENIZER_DIR, PACKED_DATA_DIR

KO_FILES = [
    DATA_ROOT / "kowiki_train.jsonl",
    DATA_ROOT / "ko_aihub_train.jsonl",
]

EN_FILES = [
    DATA_ROOT / "enwiki_train.jsonl",
]

CODE_FILES = [
    DATA_ROOT / "code_train.jsonl",
]

OUTPUT_DIR = PACKED_DATA_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BLOCK_SIZE = 4096
BLOCKS_PER_FILE = 2048
EVAL_BLOCKS = 2000
SEED = 42


def iter_jsonl_texts_from_file(file_path: Path) -> Iterator[str]:
    if not file_path.exists():
        print(f"[WARN] File not found: {file_path}")
        return

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = obj.get("text", "")
            if isinstance(text, str):
                text = text.strip()
                if text:
                    yield text


class MultiFileTextSource:
    def __init__(self, files: List[Path], repeat: bool = False):
        self.files = files
        self.repeat = repeat
        self._iters = []
        self._active = []

    def _reset(self):
        self._iters = [iter_jsonl_texts_from_file(p) for p in self.files]
        self._active = [True for _ in self.files]

    def __iter__(self):
        self._reset()
        file_idx = 0

        while True:
            if not self._iters:
                return

            if not any(self._active):
                if self.repeat:
                    self._reset()
                    file_idx = 0
                else:
                    return

            checked = 0
            while checked < len(self._iters):
                i = file_idx % len(self._iters)
                file_idx += 1
                checked += 1

                if not self._active[i]:
                    continue

                try:
                    yield next(self._iters[i])
                    break
                except StopIteration:
                    self._active[i] = False
                    continue


def mixed_text_iterator(
    ko_files: List[Path],
    en_files: List[Path],
    code_files: List[Path],
) -> Iterator[tuple[str, str]]:
    """
    Phase 1:
        ko:en:code = 6:3:1

    Phase 2:
        code 데이터가 끝나면 code는 더 이상 넣지 않고,
        ko:en = 7:3으로 계속 진행

    repeat 정책:
        ko   : repeat=False
        en   : repeat=True  # ko 전체를 쓰는 동안 영어가 먼저 끝나면 반복
        code : repeat=False # 코드 중복 방지
    """

    ko_source = iter(MultiFileTextSource(ko_files, repeat=False))
    en_source = iter(MultiFileTextSource(en_files, repeat=True))
    code_source = iter(MultiFileTextSource(code_files, repeat=False))

    code_available = True

    while True:
        try:
            if code_available:
                for _ in range(6):
                    yield next(ko_source), "ko"

                for _ in range(3):
                    yield next(en_source), "en"

                try:
                    yield next(code_source), "code"
                except StopIteration:
                    code_available = False

            else:
                for _ in range(7):
                    yield next(ko_source), "ko"

                for _ in range(3):
                    yield next(en_source), "en"

        except StopIteration:
            return


def save_block_file(blocks: List[torch.Tensor], file_idx: int):
    save_path = OUTPUT_DIR / f"train_{file_idx:05d}.pt"
    data = torch.stack(blocks, dim=0)
    torch.save(data, save_path)
    print(f"[SAVE] {save_path} shape={tuple(data.shape)}")


def build_packed_dataset():
    random.seed(SEED)
    torch.manual_seed(SEED)

    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))

    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer.eos_token_id is None. EOS token is required for packing.")

    eos_id = tokenizer.eos_token_id

    token_buffer: List[int] = []
    train_blocks: List[torch.Tensor] = []
    eval_blocks: List[torch.Tensor] = []

    train_file_idx = 0
    total_blocks = 0
    total_train_blocks = 0
    total_docs = 0

    doc_counts = {
        "ko": 0,
        "en": 0,
        "code": 0,
    }

    code_exhausted_reported = False

    for text, source_name in mixed_text_iterator(
        ko_files=KO_FILES,
        en_files=EN_FILES,
        code_files=CODE_FILES,
    ):
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue

        ids.append(eos_id)
        token_buffer.extend(ids)

        total_docs += 1
        doc_counts[source_name] += 1

        if source_name != "code" and not code_exhausted_reported and doc_counts["code"] > 0:
            # code가 끝났는지 직접 알 수는 없지만, 이후 metadata에서 code 문서 수를 확인 가능
            pass

        while len(token_buffer) >= BLOCK_SIZE:
            block = token_buffer[:BLOCK_SIZE]
            token_buffer = token_buffer[BLOCK_SIZE:]

            block_tensor = torch.tensor(block, dtype=torch.long)

            if len(eval_blocks) < EVAL_BLOCKS:
                eval_blocks.append(block_tensor)
            else:
                train_blocks.append(block_tensor)
                total_train_blocks += 1

                if len(train_blocks) >= BLOCKS_PER_FILE:
                    save_block_file(train_blocks, train_file_idx)
                    train_file_idx += 1
                    train_blocks = []

            total_blocks += 1

            if total_blocks % 1000 == 0:
                print(
                    f"[PROGRESS] docs={total_docs}, "
                    f"blocks={total_blocks}, "
                    f"train_blocks={total_train_blocks}, "
                    f"train_files={train_file_idx}, "
                    f"doc_counts={doc_counts}"
                )

    if train_blocks:
        save_block_file(train_blocks, train_file_idx)

    if eval_blocks:
        eval_data = torch.stack(eval_blocks, dim=0)
        eval_path = OUTPUT_DIR / "eval.pt"
        torch.save(eval_data, eval_path)
        print(f"[SAVE] {eval_path} shape={tuple(eval_data.shape)}")

    metadata = {
        "block_size": BLOCK_SIZE,
        "phase_1_ratio": "ko:en:code = 6:3:1",
        "phase_2_ratio": "ko:en = 7:3 after code is exhausted",
        "repeat_policy": {
            "ko": False,
            "en": True,
            "code": False,
        },
        "blocks_per_file": BLOCKS_PER_FILE,
        "eval_blocks": len(eval_blocks),
        "train_blocks": total_train_blocks,
        "total_blocks": total_blocks,
        "total_docs": total_docs,
        "doc_counts": doc_counts,
        "tokenizer_dir": str(TOKENIZER_DIR),
        "ko_files": [str(p) for p in KO_FILES],
        "en_files": [str(p) for p in EN_FILES],
        "code_files": [str(p) for p in CODE_FILES],
    }

    metadata_path = OUTPUT_DIR / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[SAVE] {metadata_path}")
    print("Packed dataset build finished.")


if __name__ == "__main__":
    build_packed_dataset()