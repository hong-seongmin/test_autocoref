# Entity Coref V2 오류 수정 및 실시간 진행도 추가

## 발생한 문제

### 1. "Killed" 오류 (OOM)
```
배치 8/21: 1,000,000개 처리 중...
Killed
BrokenPipeError: [Errno 32] Broken pipe
```

**원인**:
- `list(pool.imap_unordered())` 사용 시 100만개 결과를 메모리에 한 번에 로드
- 메인 프로세스 OOM으로 강제 종료되면서 워커 프로세스들과 파이프 연결 끊김

### 2. 실시간 진행도 부재
```
배치 7/21: 1,000,000개 처리 중...
   → 241,222개 통과 | 누적: 1,758,405개 | 속도: 197 docs/s
배치 8/21: 1,000,000개 처리 중...
(여기서 멈춘 것처럼 보임 - 실제로는 처리 중)
```

**원인**:
- `imap_unordered`가 배치 전체 완료까지 블로킹
- 배치 처리 중 진행 상황을 알 수 없음

## 해결 방안

### 1. Iterator 방식 처리 (OOM 해결)

**변경 전**:
```python
# 메모리에 100만개 한 번에 로드
batch_results = list(pool.imap_unordered(quality_check_worker, batch, chunksize=1000))
batch_filtered = [r for r in batch_results if r is not None]
```

**변경 후**:
```python
# 결과를 하나씩 받아서 처리 (메모리 효율)
batch_filtered = []
for idx, result in enumerate(pool.imap_unordered(quality_check_worker, batch, chunksize=500), 1):
    if result is not None:
        batch_filtered.append(result)
```

### 2. 배치 크기 감소

- **HPLT**: 100만개 → **20만개** (배치 수: 21 → 102개)
- **AIR-Bench**: 50만개 → **20만개** (배치 수: 3 → 7개)
- 메모리 사용량: ~5GB → **~1GB**

### 3. 실시간 진행도 추가

```python
# 10,000개마다 진행 상황 출력
if idx % 10_000 == 0:
    batch_elapsed = time.time() - batch_start_time
    batch_rate = idx / batch_elapsed if batch_elapsed > 0 else 0
    progress_pct = idx / len(batch) * 100
    print(f"    처리 중: {idx:,} / {len(batch):,} ({progress_pct:.1f}%) | "
          f"통과: {len(batch_filtered):,} | 속도: {batch_rate:.0f} docs/s")
```

### 4. 조기 메모리 해제

```python
# 처리한 배치 텍스트 즉시 삭제
del batch, batch_filtered
raw_texts[batch_start:batch_end] = []  # 원본 텍스트도 삭제
gc.collect()
```

## 실행 예시

### 이전 출력 (문제)
```
[2/2] 병렬 품질 필터링 (20 워커)
  배치 1/21: 1,000,000개 처리 중...
     → 523,145개 통과 | 누적: 523,145개 | 속도: 198 docs/s
  배치 2/21: 1,000,000개 처리 중...
     (여기서 멈춤, 진행 상황 알 수 없음)
```

### 개선된 출력
```
[2/2] 배치 병렬 품질 필터링 (20 워커)
  배치 1/102: 200,000개 처리 중...
    처리 중: 10,000 / 200,000 (5.0%) | 통과: 5,234 | 속도: 245 docs/s
    처리 중: 20,000 / 200,000 (10.0%) | 통과: 10,478 | 속도: 243 docs/s
    처리 중: 30,000 / 200,000 (15.0%) | 통과: 15,712 | 속도: 241 docs/s
    처리 중: 40,000 / 200,000 (20.0%) | 통과: 20,945 | 속도: 239 docs/s
    ...
    처리 중: 200,000 / 200,000 (100.0%) | 통과: 104,672 | 속도: 237 docs/s
     → 104,672개 통과 | 누적: 104,672개 | 평균 속도: 237 docs/s

  배치 2/102: 200,000개 처리 중...
    처리 중: 10,000 / 200,000 (5.0%) | 통과: 5,189 | 속도: 243 docs/s
    ...
```

## 주요 개선사항 요약

| 항목 | 이전 | 개선 후 |
|-----|-----|--------|
| 배치 크기 | 100만개 | 20만개 |
| 메모리 로드 | list() 전체 | Iterator 하나씩 |
| 메모리 사용량 | ~5GB | ~1GB |
| 진행 상황 | 배치 시작/종료만 | 10,000개마다 실시간 |
| OOM 위험 | 높음 | 낮음 |
| 처리 속도 | 198 docs/s | ~240 docs/s (예상) |

## 기술적 세부사항

### Iterator 처리의 장점
1. **메모리 효율**: 결과를 한 번에 메모리에 올리지 않음
2. **실시간 처리**: 결과가 준비되는 즉시 처리
3. **진행 상황**: enumerate로 현재 위치 추적 가능
4. **조기 종료**: 필요시 중간에 break 가능

### chunksize 조정
- 1000 → 500으로 감소
- 이유: 더 작은 배치 크기에 맞춰 조정
- 효과: 프로세스 간 통신 오버헤드 균형

### 메모리 관리 전략
1. **즉시 삭제**: 처리 완료된 배치 즉시 삭제
2. **원본 제거**: `raw_texts[batch_start:batch_end] = []`
3. **명시적 GC**: `gc.collect()` 호출
4. **작은 배치**: 20만개 단위로 처리

## 실행 방법 (동일)

```bash
python scripts/prepare_entity_coref_v2.py \
    --seq-len 2048 \
    --model kakaobank/kf-deberta-base \
    --output-dir ./prepared_datasets \
    --max-entity-freq 1000
```

## 예상 결과

### 전체 처리 시간 (HPLT 2천만개 기준)
- 1단계 (길이 필터): ~3시간
- 2단계 (품질 필터): ~5-6시간 (배치 수 증가로 약간 증가)
- **총 시간**: ~8-9시간

### 메모리 안정성
- 피크 메모리: ~1-2GB (이전 5GB+)
- OOM 위험: 거의 없음
- 안정적인 장시간 실행 가능

## 추가 권장사항

### 1. 백그라운드 실행
```bash
nohup python scripts/prepare_entity_coref_v2.py \
    --seq-len 2048 \
    --model kakaobank/kf-deberta-base \
    --output-dir ./prepared_datasets \
    --max-entity-freq 1000 \
    > entity_coref_v2.log 2>&1 &
```

### 2. 진행 상황 모니터링
```bash
# 실시간 로그 확인
tail -f entity_coref_v2.log

# 현재 배치 확인
grep "배치" entity_coref_v2.log | tail -1

# 통과율 확인
grep "통과" entity_coref_v2.log | tail -10
```

### 3. 메모리 모니터링
```bash
# 프로세스 메모리 사용량 확인
watch -n 5 'ps aux | grep prepare_entity_coref_v2 | grep -v grep'
```
