import argparse
import json
import os
import re
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import langid
import numpy as np
os.environ.setdefault("PYTORCH_NO_SHARED_MEMORY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")

from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from coref_automl.dataset_preparation import analyze_coref_quality


CACHE_ROOT = Path("/home/work/.cache/huggingface/datasets")
HF_CACHE_DIR = Path("/home/work/hongseongmin/corefer/hf_cache")
HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

TOKENIZER = AutoTokenizer.from_pretrained(
    "kakaobank/kf-deberta-base", local_files_only=True
)
PAD_ID = TOKENIZER.pad_token_id

TARGETS = (1536, 2048)

def resolve_worker_count() -> int:
    env = os.getenv("CORE_DATASET_WORKERS")
    if env:
        try:
            value = int(env)
            if value > 0:
                return value
        except ValueError:
            pass
    return max(2, (os.cpu_count() or 4))


MAX_QUALITY_WORKERS = resolve_worker_count()
PENDING_PER_TARGET = MAX_QUALITY_WORKERS * 4


class ChunkQualityBuffer:
    """Asynchronous Kiwi 품질 계산을 버퍼링하여 병렬로 처리."""

    def __init__(
        self,
        target: int,
        source: str,
        progress: Dict[int, "TargetProgress"],
        results: Dict[int, List[Dict]],
        min_pron: int,
        min_density: float,
        min_entity: float,
        executor: ThreadPoolExecutor,
        pending_limit: int = PENDING_PER_TARGET,
    ) -> None:
        self.target = target
        self.source = source
        self.progress = progress
        self.results = results
        self.min_pron = min_pron
        self.min_density = min_density
        self.min_entity = min_entity
        self.executor = executor
        self.pending_limit = max(1, pending_limit)
        self.queue: deque = deque()

    def submit(self, tokens: List[int], text: str) -> bool:
        """토큰과 텍스트를 큐에 추가하고 필요 시 처리."""
        future = self.executor.submit(analyze_coref_quality, text)
        self.queue.append((future, tokens))
        return self._drain(force=False)

    def maybe_flush(self) -> bool:
        """현재 버퍼 크기가 한계에 도달했다면 처리."""
        return self._drain(force=False)

    def finalize(self) -> bool:
        """모든 잔여 작업 처리."""
        return self._drain(force=True)

    def _drain(self, force: bool) -> bool:
        reached_limit = False
        while self.queue and (
            force
            or len(self.queue) >= self.pending_limit
            or self.queue[0][0].done()
        ):
            future, tokens = self.queue.popleft()
            quality = future.result()
            if (
                quality["pronoun_count"] >= self.min_pron
                and quality["pronoun_density"] >= self.min_density
                and quality["entity_density"] >= self.min_entity
            ):
                features = pad_features(tokens, self.target)
                features.update(
                    {
                        "source": self.source,
                        "pronoun_density": quality["pronoun_density"],
                        "entity_density": quality["entity_density"],
                        "pronoun_count": quality["pronoun_count"],
                    }
                )
                self.results[self.target].append(features)
                if self.progress[self.target].update():
                    reached_limit = True
                    # 아직 실행 중인 작업이라면 취소 시도 (이미 실행 중이면 False)
                    for future_to_cancel, _ in self.queue:
                        future_to_cancel.cancel()
                    self.queue.clear()
                    break
        return reached_limit


class TargetProgress:
    def __init__(self, target: float, desc: str):
        self.target = target
        total = None if target == float('inf') else int(target)
        self.count = 0
        self.pbar = tqdm(total=total, desc=desc, leave=True)

    def update(self, increment: int = 1) -> bool:
        if increment <= 0:
            return False
        if self.target == float('inf'):
            self.count += increment
            self.pbar.update(increment)
            return False
        remaining = max(0, int(self.target - self.count))
        delta = min(remaining, increment)
        self.count += delta
        if delta:
            self.pbar.update(delta)
        return self.count >= self.target

    def reached(self) -> bool:
        return self.target != float('inf') and self.count >= self.target


def sentence_split(text: str) -> List[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def encode_sentences(sentences: List[str]) -> List[List[int]]:
    return [TOKENIZER.encode(sent, add_special_tokens=False) for sent in sentences]


def chunk_sentences(
    tokens_cache: List[List[int]], target_len: int, min_ratio: float = 0.7, stride_ratio: float = 0.5
) -> Iterable[List[int]]:
    sentence_count = len(tokens_cache)
    if sentence_count == 0:
        return []
    min_len = int(target_len * min_ratio)
    idx = 0
    stride_steps = max(1, int(sentence_count * stride_ratio))
    while idx < sentence_count:
        current_tokens: List[int] = []
        for j in range(idx, sentence_count):
            current_tokens.extend(tokens_cache[j])
            if len(current_tokens) >= target_len:
                break
        if len(current_tokens) >= min_len:
            yield current_tokens[:target_len]
        if idx == sentence_count - 1:
            break
        increment = max(1, int(stride_steps / max(1, sentence_count))) or 1
        idx += increment


def pad_features(tokens: List[int], target_len: int) -> Dict[str, List[int]]:
    tokens = tokens[:target_len]
    pad_length = target_len - len(tokens)
    input_ids = tokens + [PAD_ID] * pad_length
    attention = [1] * len(tokens) + [0] * pad_length
    token_type = [0] * target_len
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "token_type_ids": token_type,
    }


def filter_by_quality(text: str, min_pron: int, min_density: float, min_entity: float) -> Tuple[bool, Dict]:
    quality = analyze_coref_quality(text)
    keep = (
        quality["pronoun_count"] >= min_pron
        and quality["pronoun_density"] >= min_density
        and quality["entity_density"] >= min_entity
    )
    return keep, quality


# ---------- Naver ----------

def build_naver(output_dir: Path, max_samples_per_target: float) -> Dict[int, List[Dict]]:
    data_path = (
        CACHE_ROOT
        / "daekeun-ml___naver-news-summarization-ko"
        / "default"
        / "0.0.0"
        / "fa6ec90a7beb96d182372f09b04b96797ea6588a"
        / "naver-news-summarization-ko-train.arrow"
    )
    dataset = load_dataset("arrow", data_files={"train": str(data_path)}, cache_dir=str(HF_CACHE_DIR))
    rows = dataset["train"]

    min_pron = 2
    min_density = 0.008
    min_entity = 0.55

    results = {target: [] for target in TARGETS}
    progress = {
        target: TargetProgress(max_samples_per_target, f"Naver target {target}")
        for target in TARGETS
    }

    finished_targets = set()
    with ThreadPoolExecutor(max_workers=MAX_QUALITY_WORKERS) as executor:
        buffers = {
            target: ChunkQualityBuffer(
                target=target,
                source="naver_news",
                progress=progress,
                results=results,
                min_pron=min_pron,
                min_density=min_density,
                min_entity=min_entity,
                executor=executor,
            )
            for target in TARGETS
        }

        for row in tqdm(rows, desc="Naver documents"):
            if len(finished_targets) == len(TARGETS):
                break
            doc = row["document"]
            if isinstance(doc, list):
                text = " ".join(paragraph.strip() for paragraph in doc if paragraph and paragraph.strip())
            else:
                text = str(doc).strip()
            if not text:
                continue
            sentences = sentence_split(text)
            sentence_tokens = encode_sentences(sentences)
            for target in TARGETS:
                if target in finished_targets:
                    continue
                for tokens in chunk_sentences(sentence_tokens, target):
                    decoded = TOKENIZER.decode(tokens, skip_special_tokens=True)
                    if buffers[target].submit(tokens, decoded):
                        finished_targets.add(target)
                        break
            for target in TARGETS:
                if target in finished_targets:
                    continue
                if buffers[target].maybe_flush():
                    finished_targets.add(target)
            if len(finished_targets) == len(TARGETS):
                break

        for target, buffer in buffers.items():
            if target in finished_targets:
                continue
            if buffer.finalize():
                finished_targets.add(target)

    for target in TARGETS:
        if results[target]:
            save_arrow(results[target], output_dir / f"naver_coref_{target}", target)
    return {target: len(results[target]) for target in TARGETS}


# ---------- Wikipedia ----------

def clean_wiki_line(line: str) -> str:
    line = re.sub(r"\([^)]*\)", "", line)
    line = line.strip()
    if not line:
        return ""
    lang, _ = langid.classify(line)
    if lang != "ko":
        return ""
    if len(line) < 70:
        return ""
    return line


def wiki_segment_tokens(text: str) -> List[List[int]]:
    segments = [seg.strip() for seg in text.split("\n\n") if seg.strip()]
    cleaned_tokens: List[List[int]] = []
    for seg in segments:
        cleaned_lines = []
        for raw_line in seg.split("\n"):
            cleaned = clean_wiki_line(raw_line)
            if cleaned:
                cleaned_lines.append(cleaned)
        if not cleaned_lines:
            continue
        merged = " ".join(cleaned_lines)
        cleaned_tokens.append(TOKENIZER.encode(merged, add_special_tokens=False))
    return cleaned_tokens


def wiki_chunk_from_tokens(
    cleaned_tokens: List[List[int]], target: int, min_ratio: float = 0.7
) -> Iterable[List[int]]:
    if not cleaned_tokens:
        return []
    min_len = int(target * min_ratio)
    current_tokens: List[int] = []
    for tokens in cleaned_tokens:
        if current_tokens and len(current_tokens) + len(tokens) > target:
            if len(current_tokens) >= min_len:
                yield current_tokens[:target]
            current_tokens = []
        current_tokens.extend(tokens)
    if current_tokens and len(current_tokens) >= min_len:
        yield current_tokens[:target]


def wiki_merge_chunks(text: str, target: int, min_ratio: float = 0.7) -> Iterable[List[int]]:
    cleaned_tokens = wiki_segment_tokens(text)
    yield from wiki_chunk_from_tokens(cleaned_tokens, target, min_ratio)


def build_wiki(output_dir: Path, max_samples_per_target: float) -> Dict[int, int]:
    base = CACHE_ROOT / "wikimedia___wikipedia" / "20231101.ko" / "0.0.0" / "b04c8d1ceb2f5cd4588862100d08de323dccfbaa"
    files = sorted(str(p) for p in base.glob("*.arrow"))

    min_pron = 2
    min_density = 0.01
    min_entity = 0.5

    results = {target: [] for target in TARGETS}
    progress = {
        target: TargetProgress(max_samples_per_target, f"Wiki target {target}")
        for target in TARGETS
    }

    finished_targets = set()
    with ThreadPoolExecutor(max_workers=MAX_QUALITY_WORKERS) as executor:
        buffers = {
            target: ChunkQualityBuffer(
                target=target,
                source="wiki",
                progress=progress,
                results=results,
                min_pron=min_pron,
                min_density=min_density,
                min_entity=min_entity,
                executor=executor,
            )
            for target in TARGETS
        }

        for file_path in tqdm(files, desc="Wiki files"):
            if len(finished_targets) == len(TARGETS):
                break
            dataset = load_dataset(
                "arrow",
                data_files={"train": file_path},
                cache_dir=str(HF_CACHE_DIR),
            )["train"]

            for row in tqdm(dataset, desc=f"wiki::{Path(file_path).name}", leave=False):
                if len(finished_targets) == len(TARGETS):
                    break
                text = row["text"]
                if not text or len(text) < 200:
                    continue
                cleaned_tokens = wiki_segment_tokens(text)
                if not cleaned_tokens:
                    continue
                for target in TARGETS:
                    if target in finished_targets:
                        continue
                    for tokens in wiki_chunk_from_tokens(cleaned_tokens, target):
                        decoded = TOKENIZER.decode(tokens, skip_special_tokens=True)
                        if buffers[target].submit(tokens, decoded):
                            finished_targets.add(target)
                            break
                for target in TARGETS:
                    if target in finished_targets:
                        continue
                    if buffers[target].maybe_flush():
                        finished_targets.add(target)
                if len(finished_targets) == len(TARGETS):
                    break

        for target, buffer in buffers.items():
            if target in finished_targets:
                continue
            if buffer.finalize():
                finished_targets.add(target)

    for target in TARGETS:
        if results[target]:
            save_arrow(results[target], output_dir / f"wiki_coref_segments_{target}", target)
    return {target: len(results[target]) for target in TARGETS}


# ---------- KLUE MRC ----------

def build_klue_mrc(output_dir: Path, max_samples_per_target: float) -> Dict[int, int]:
    base = CACHE_ROOT / "klue" / "mrc" / "0.0.0" / "349481ec73fff722f88e0453ca05c77a447d967c"
    dataset = load_dataset(
        "arrow",
        data_files={"train": str(base / "klue-train.arrow")},
        cache_dir=str(HF_CACHE_DIR),
    )["train"]

    min_pron = 1
    min_density = 0.0  # 최대한 관대하게
    min_entity = 0.0

    results = {target: [] for target in TARGETS}
    progress = {
        target: TargetProgress(max_samples_per_target, f"KLUE target {target}")
        for target in TARGETS
    }

    finished_targets = set()
    with ThreadPoolExecutor(max_workers=MAX_QUALITY_WORKERS) as executor:
        buffers = {
            target: ChunkQualityBuffer(
                target=target,
                source="klue_mrc_context",
                progress=progress,
                results=results,
                min_pron=min_pron,
                min_density=min_density,
                min_entity=min_entity,
                executor=executor,
            )
            for target in TARGETS
        }

        for row in tqdm(dataset, desc="KLUE MRC contexts"):
            if len(finished_targets) == len(TARGETS):
                break
            context = row["context"]
            if not context:
                continue
            sentences = sentence_split(context)
            tokens_cache = encode_sentences(sentences)
            for target in TARGETS:
                if target in finished_targets:
                    continue
                for tokens in chunk_sentences(tokens_cache, target, min_ratio=0.25):
                    decoded = TOKENIZER.decode(tokens, skip_special_tokens=True)
                    if buffers[target].submit(tokens, decoded):
                        finished_targets.add(target)
                        break
            for target in TARGETS:
                if target in finished_targets:
                    continue
                if buffers[target].maybe_flush():
                    finished_targets.add(target)

        for target, buffer in buffers.items():
            if target in finished_targets:
                continue
            buffer.finalize()

    for target in TARGETS:
        if results[target]:
            save_arrow(results[target], output_dir / f"klue_mrc_context_{target}", target)
    return {target: len(results[target]) for target in TARGETS}


# ---------- Utility ----------

def save_arrow(samples: List[Dict], directory: Path, target_len: int) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_list(samples)
    dataset.save_to_disk(str(directory))
    stats = {
        "count": len(samples),
        "token_len_avg": float(np.mean([sum(x["attention_mask"]) for x in samples])),
        "pron_density_avg": float(np.mean([x["pronoun_density"] for x in samples])),
        "entity_density_avg": float(np.mean([x["entity_density"] for x in samples])),
    }
    with open(directory / "quality.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Build high-quality coref datasets.")
    parser.add_argument("--output", default="prepared_datasets", help="Output directory")
    parser.add_argument(
        "--naver-max",
        type=int,
        default=12000,
        help="Max samples per target for Naver (0 to skip, negative for unlimited)",
    )
    parser.add_argument(
        "--wiki-max",
        type=int,
        default=15000,
        help="Max samples per target for Wiki (0 to skip, negative for unlimited)",
    )
    parser.add_argument(
        "--klue-max",
        type=int,
        default=8000,
        help="Max samples per target for KLUE MRC (0 to skip, negative for unlimited)",
    )

    args = parser.parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    if args.naver_max != 0:
        max_naver = float("inf") if args.naver_max < 0 else args.naver_max
        summary["naver"] = build_naver(output_dir, max_naver)

    if args.wiki_max != 0:
        max_wiki = float("inf") if args.wiki_max < 0 else args.wiki_max
        summary["wiki"] = build_wiki(output_dir, max_wiki)

    if args.klue_max != 0:
        max_klue = float("inf") if args.klue_max < 0 else args.klue_max
        summary["klue_mrc"] = build_klue_mrc(output_dir, max_klue)
    with open(output_dir / "build_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
class TargetProgress:
    def __init__(self, target: float, desc: str):
        total = None if target == float("inf") else int(target)
        self.target = target
        self.count = 0
        self.pbar = tqdm(total=total, desc=desc, leave=True)

    def update(self, increment: int = 1) -> bool:
        if self.target == float("inf"):
            self.pbar.update(increment)
            return False
        if self.count >= self.target:
            return True
        remaining = max(0, int(self.target - self.count))
        delta = min(remaining, increment)
        self.count += delta
        if delta > 0:
            self.pbar.update(delta)
        return self.count >= self.target
