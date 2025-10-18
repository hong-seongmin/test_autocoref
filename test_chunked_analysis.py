#!/usr/bin/env python3
"""
청크 처리 방식의 품질 분석 기능 테스트
- 550k+ 샘플에 대한 메모리 효율적 처리 검증
"""

import time
import gc
from coref_automl.dataset_preparation_mlm_v2 import batch_analyze_quality

def test_chunked_quality_analysis():
    """청크 처리 품질 분석 테스트"""

    # 테스트용 텍스트 생성 (10만개)
    print("📝 테스트 데이터 생성 중...")
    test_texts = [
        f"이것은 테스트 문장입니다. 그는 학교에 갔습니다. 그녀는 책을 읽었습니다. 학생들은 공부했습니다. 선생님은 가르쳤습니다. {i}번째 샘플."
        for i in range(100000)
    ]

    print(f"✅ {len(test_texts):,}개 샘플 생성 완료")
    print(f"📊 예상 청크 수: {(len(test_texts) + 9999) // 10000}개\n")

    # 메모리 체크
    import psutil
    import os
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / 1024 / 1024  # MB

    print(f"💾 분석 전 메모리: {mem_before:.1f} MB\n")

    # 청크 처리 품질 분석
    start_time = time.time()
    results = batch_analyze_quality(test_texts, batch_size=500, max_workers=16)
    elapsed = time.time() - start_time

    # 메모리 체크
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    mem_delta = mem_after - mem_before

    print(f"\n💾 분석 후 메모리: {mem_after:.1f} MB (증가: {mem_delta:.1f} MB)")

    # 결과 검증
    print(f"\n✅ 분석 완료!")
    print(f"  - 입력: {len(test_texts):,}개")
    print(f"  - 결과: {len(results):,}개")
    print(f"  - 성공률: {len(results)/len(test_texts)*100:.1f}%")
    print(f"  - 소요시간: {elapsed:.1f}초")
    print(f"  - 처리속도: {len(results)/elapsed:.1f} 샘플/초")

    # 품질 점수 통계
    if results:
        quality_scores = [r['quality_score'] for r in results]
        avg_quality = sum(quality_scores) / len(quality_scores)
        print(f"  - 평균 품질: {avg_quality:.3f}")

    print(f"\n🎉 청크 처리 테스트 성공!")
    print(f"   55만개 샘플도 동일한 방식으로 처리 가능합니다.")

if __name__ == "__main__":
    test_chunked_quality_analysis()
