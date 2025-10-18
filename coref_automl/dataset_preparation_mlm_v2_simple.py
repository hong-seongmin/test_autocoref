# coref_automl/dataset_preparation_mlm_v2_simple.py
"""
초간소화 데이터셋 준비 스크립트 (wiki + HPLT 전용)
- 품질 분석 없음
- 청킹 없음
- 단순 truncation만
- 빠른 데이터셋 생성
"""

from __future__ import annotations
import os
import multiprocessing
from dataclasses import dataclass
from typing import List, Optional

from datasets import load_dataset, concatenate_datasets, disable_caching, Dataset
from transformers import AutoTokenizer
from tqdm import tqdm

disable_caching()


@dataclass
class DatasetSource:
    """데이터셋 소스 설정"""
    name: str
    source: str
    subset: Optional[str]
    split: str
    domain: str
    description: str
    is_streaming: bool = False


def get_simple_dataset_sources() -> List[DatasetSource]:
    """간소화: wiki + HPLT만"""
    return [
        # Wikipedia - 긴 문서, 다양한 주제
        DatasetSource(
            name="wiki_ko",
            source="wikimedia/wikipedia",
            subset="20231101.ko",
            split="train",
            domain="encyclopedia",
            description="한국어 위키백과"
        ),

        # HPLT Korean (대규모 웹 크롤)
        DatasetSource(
            name="hplt_korean",
            source="HPLT/HPLT2.0_cleaned",
            subset="kor_Hang",
            split="train",
            domain="hplt_general",
            description="HPLT 대규모 한국어 웹 크롤",
            is_streaming=True
        ),
    ]


def simple_preprocess(
    dataset,
    source: DatasetSource,
    tokenizer,
    target_seq_len: int,
    min_length: int = 50
) -> List[str]:
    """
    초간소화 전처리
    - 텍스트 추출
    - 최소 길이 필터링
    - truncation만
    """

    processed_texts = []
    skipped = 0
    truncated = 0

    print(f"  📝 전처리 중...", flush=True)

    for item in tqdm(dataset, desc=f"  처리", unit="샘플"):
        try:
            # 도메인별 텍스트 추출
            if source.domain == "encyclopedia":
                text = item.get("text", "").strip()
            elif source.domain == "hplt_general":
                text = item.get("text", "").strip()
            else:
                text = item.get("text", "").strip()

            # 기본 필터링: 최소 길이만 체크
            if not text or len(text) < min_length:
                skipped += 1
                continue

            # 토큰 길이 체크
            tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
            token_len = len(tokens)

            # 너무 짧으면 스킵
            if token_len < 50:
                skipped += 1
                continue

            # 적절한 길이: 그대로 사용
            if token_len <= target_seq_len:
                processed_texts.append(text)
            # 긴 경우: truncate만
            else:
                truncated_ids = tokens[:target_seq_len]
                truncated_text = tokenizer.decode(truncated_ids, skip_special_tokens=True)
                processed_texts.append(truncated_text)
                truncated += 1

        except Exception as e:
            skipped += 1
            continue

    print(f"  ✅ 전처리 완료: {len(processed_texts):,}개 생성 (스킵: {skipped:,}, 잘림: {truncated:,})")
    return processed_texts


def load_simple_dataset(
    source: DatasetSource,
    tokenizer,
    target_seq_len: int,
    limit: Optional[int] = None
) -> Optional[Dataset]:
    """간소화: 단순 로드 및 전처리"""

    try:
        print(f"\n📥 {source.name}: {source.description}")

        # 데이터 로드
        load_kwargs = {"split": source.split}
        if source.subset:
            load_kwargs["name"] = source.subset

        # 스트리밍 처리
        if source.is_streaming:
            load_kwargs["streaming"] = True
            dataset_stream = load_dataset(source.source, **load_kwargs)

            # limit이 지정된 경우에만 제한
            if limit:
                print(f"  🔄 스트리밍 로드 중... (최대 {limit:,}개)")
                samples = []
                for i, sample in enumerate(dataset_stream):
                    if i >= limit:
                        break
                    samples.append(sample)

                    if (i + 1) % 10000 == 0:
                        print(f"\r  ⏳ {i+1:,}/{limit:,} ({(i+1)/limit*100:.1f}%)", end="", flush=True)

                print(f"\r  ✅ 로드 완료: {len(samples):,} 샘플" + " " * 20)
            else:
                # 무제한이어도 스트리밍은 기본 제한 (메모리 보호)
                DEFAULT_STREAMING_LIMIT = 500000
                print(f"  🔄 스트리밍 로드 중... (기본 제한: {DEFAULT_STREAMING_LIMIT:,}개)")
                samples = []
                for i, sample in enumerate(dataset_stream):
                    if i >= DEFAULT_STREAMING_LIMIT:
                        break
                    samples.append(sample)

                    if (i + 1) % 10000 == 0:
                        print(f"\r  ⏳ 로드 중: {i+1:,}/{DEFAULT_STREAMING_LIMIT:,} ({(i+1)/DEFAULT_STREAMING_LIMIT*100:.1f}%)", end="", flush=True)

                print(f"\r  ✅ 로드 완료: {len(samples):,} 샘플 (HPLT 제한 적용)" + " " * 20)

            dataset = Dataset.from_list(samples)
        else:
            dataset = load_dataset(source.source, **load_kwargs)

            # 샘플 제한
            if limit and limit < len(dataset):
                dataset = dataset.select(range(limit))

            print(f"  ✅ 로드 완료: {len(dataset):,} 샘플")

        # 전처리
        processed_texts = simple_preprocess(dataset, source, tokenizer, target_seq_len)

        if not processed_texts:
            print(f"  ⚠️ 전처리 후 샘플 없음")
            return None

        # Dataset으로 변환
        return Dataset.from_list([{"text": text} for text in processed_texts])

    except Exception as e:
        print(f"  ❌ {source.name} 로드 실패: {e}")
        return None


def prepare_simple_datasets(
    model_name: str = "kakaobank/kf-deberta-base",
    seq_lengths: List[int] = [1536, 2048],
    save_path: str = "./prepared_datasets_simple",
    max_samples_per_length: Optional[int] = None
):
    """
    초간소화 데이터셋 준비
    - wiki + HPLT만
    - 품질 분석 없음
    - 빠른 생성
    """

    os.makedirs(save_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("🚀 초간소화 데이터셋 준비 (wiki + HPLT)")
    print(f"🎯 모델: {model_name}")
    print(f"📏 시퀀스 길이: {seq_lengths}")
    print(f"💾 저장 경로: {save_path}")
    if max_samples_per_length:
        print(f"📊 최대 샘플: {max_samples_per_length:,}개/길이")
    else:
        print(f"📊 최대 샘플: 무제한")
    print("=" * 80)

    sources = get_simple_dataset_sources()

    for seq_len in seq_lengths:
        print(f"\n{'='*80}")
        print(f"🎯 시퀀스 길이: {seq_len} 토큰")
        print(f"{'='*80}")

        # 토크나이저 max_length 동적 확장
        original_max_length = tokenizer.model_max_length
        tokenizer.model_max_length = seq_len
        print(f"   Tokenizer max_length: {original_max_length} → {seq_len}")

        all_datasets = []
        total_samples = 0

        # 각 소스별 limit 계산
        if max_samples_per_length:
            limit_per_source = max_samples_per_length // len(sources)
            print(f"   소스당 제한: {limit_per_source:,}개")
        else:
            limit_per_source = None
            print(f"   소스당 제한: 없음")

        # 데이터 로드
        for source in sources:
            dataset = load_simple_dataset(source, tokenizer, seq_len, limit=limit_per_source)

            if dataset and len(dataset) > 0:
                all_datasets.append(dataset)
                total_samples += len(dataset)
                print(f"  📊 누적: {total_samples:,} 샘플")

        # 데이터 통합
        if all_datasets:
            print(f"\n🔄 데이터셋 통합 중... ({total_samples:,} 샘플)")

            combined_dataset = concatenate_datasets(all_datasets)
            combined_dataset = combined_dataset.shuffle(seed=42)

            print(f"  ✅ 통합 완료: {len(combined_dataset):,} 샘플")

            # 최종 샘플 제한
            if max_samples_per_length and len(combined_dataset) > max_samples_per_length:
                print(f"  ✂️  최대 샘플 수로 제한: {max_samples_per_length:,}")
                combined_dataset = combined_dataset.select(range(max_samples_per_length))

            # 토큰화
            print(f"\n🔤 토큰화 중... ({len(combined_dataset):,} 샘플)")

            def tokenize_function(examples):
                return tokenizer(
                    examples["text"],
                    truncation=True,
                    padding="max_length",
                    max_length=seq_len
                )

            # 시퀀스 길이에 따라 배치 크기 조정 (메모리 효율)
            if seq_len >= 2048:
                batch_size = 100
                num_proc = 4
            elif seq_len >= 1536:
                batch_size = 200
                num_proc = 8
            else:
                batch_size = 500
                num_proc = 8

            print(f"  ⚙️  토큰화 설정: batch_size={batch_size}, num_proc={num_proc}")

            tokenized_dataset = combined_dataset.map(
                tokenize_function,
                batched=True,
                batch_size=batch_size,
                remove_columns=["text"],
                num_proc=num_proc,
                desc="토큰화"
            )

            # 저장
            save_file = f"{save_path}/{model_name.replace('/', '_')}_{seq_len}_simple.arrow"
            print(f"\n💾 저장 중: {save_file}")
            tokenized_dataset.save_to_disk(save_file)

            print(f"✅ 저장 완료!")
            print(f"📊 최종: {len(tokenized_dataset):,} 샘플 × {seq_len} 토큰")

        else:
            print(f"⚠️ {seq_len} 토큰 데이터셋 생성 실패")

    print("\n" + "=" * 80)
    print("🎊 모든 데이터셋 준비 완료!")
    print(f"📂 저장 위치: {save_path}")
    print("🚀 이제 run_combined_experiment_v2.py로 훈련하세요!")
    print("=" * 80)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="초간소화 데이터셋 준비 (wiki + HPLT)")
    parser.add_argument("--model", default="kakaobank/kf-deberta-base", help="모델 이름")
    parser.add_argument(
        "--seq-len",
        nargs="+",
        type=int,
        default=[1536, 2048],
        help="시퀀스 길이들 (예: --seq-len 1536 2048)"
    )
    parser.add_argument("--save-path", default="./prepared_datasets_simple", help="저장 경로")
    parser.add_argument(
        "--max-samples",
        type=lambda x: None if x.lower() == 'none' else int(x),
        default=None,
        help="최대 샘플 수 (None=무제한, 예: --max-samples 500000)"
    )

    args = parser.parse_args()

    print(f"\n시작 설정:")
    print(f"  - 모델: {args.model}")
    print(f"  - 시퀀스 길이: {args.seq_len}")
    print(f"  - 저장 경로: {args.save_path}")
    print(f"  - 최대 샘플: {args.max_samples if args.max_samples else '무제한'}")
    print()

    prepare_simple_datasets(
        model_name=args.model,
        seq_lengths=args.seq_len,
        save_path=args.save_path,
        max_samples_per_length=args.max_samples
    )
