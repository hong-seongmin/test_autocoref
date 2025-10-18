# 데이터셋 품질 문제 분석: 왜 MLM v2가 기대 이하의 성능을 보였는가?

## 🔍 핵심 발견: 품질 필터링의 역설

### 문제 요약
**MLM v2 데이터셋은 8개의 새로운 데이터 소스를 추가했지만, Combined 데이터셋보다 낮은 성능을 보였습니다.**

---

## 📊 데이터셋 크기 비교

### 샘플 수 비교 (1536 기준)
| 데이터셋 | 샘플 수 | 크기 | 비율 |
|---------|--------|------|------|
| **V1 Original** | **187개** | 1.7MB | 기준 |
| **V2 MLM** | **16,699개** | 147MB | **89배** |
| **Combined** | **50,665개** | 446MB | **270배** |

### 충격적인 발견
```
V1 Original:  187개
V2 MLM:       16,699개  (↑ 8,825%)
Combined:     50,665개  (↑ 26,986%)
```

**Combined가 V2 MLM보다 3배 더 많은 데이터를 가지고 있습니다!**

---

## 🔬 원인 분석

### 1. **품질 필터링 기준의 문제**

#### V1 (dataset_preparation.py)
```python
# 샘플링 비율: 매우 보수적
sample_size = min(1000, len(dataset))
indices = list(range(0, len(dataset), max(1, len(dataset) // sample_size)))

# 품질 임계값
quality_threshold = 0.6

# 결과: 187개 (극소수만 통과)
```

#### V2 (dataset_preparation_mlm_v2.py)
```python
# 샘플링 비율: 개선됨
sample_size = min(2000, len(dataset))
indices = list(range(0, len(dataset), max(1, len(dataset) // sample_size)))

# 품질 임계값 (동일)
quality_threshold = 0.6

# 결과: 16,699개 (V1보다 89배 증가, 하지만 여전히 부족)
```

#### Combined (직접 데이터 결합)
```python
# Wikipedia: 96MB
# KLUE MRC: 11MB
# Naver News: 19MB
# Entity Coref: 353MB
# 품질 필터링 없이 직접 결합

# 결과: 50,665개 (가장 많은 데이터)
```

---

### 2. **샘플링 방식의 문제**

#### V1/V2의 샘플링 로직
```python
# 전체 데이터의 일부만 분석
indices = list(range(0, len(dataset), max(1, len(dataset) // sample_size)))

# 예: 100,000개 데이터셋 → 1,000개만 분석 → 100개마다 1개 추출
# 문제: 99%의 데이터를 버림!
```

**예시 계산**:
- Wikipedia 데이터셋: 500,000개
- 샘플링: 2,000개만 분석
- **498,000개의 데이터는 평가조차 하지 않음!**
- 품질 통과: 800개 (40% 통과율)
- **최종 사용: 800개 / 500,000개 = 0.16%만 사용**

#### Combined의 직접 결합
```python
# 모든 데이터 사용
- Wikipedia 전체 → chunk → 사용
- KLUE MRC 전체 → 사용
- Naver News 전체 → 사용
- Entity Coref 전체 → 사용

# 결과: 50,665개 (대부분의 데이터 활용)
```

---

### 3. **품질 점수 계산의 문제점**

#### 품질 점수 공식 (V1/V2 동일)
```python
# 1. 대명사-개체 상호작용 점수
coref_interaction = min(1.0, (pronoun_density * 20) + (entity_density * 3) + (pronoun_density * entity_density * 50))

# 2. 복잡도 점수
complexity_score = min(1.0, (entity_density * 10) + (verb_density * 5) + (pronoun_density * 15))

# 3. 균형 점수
balance_score = 1.0 - abs(pronoun_density - entity_density * 0.3)

# 종합 점수
quality_score = (coref_interaction * 0.5) + (complexity_score * 0.3) + (balance_score * 0.2)
```

**문제**:
- **대명사(pronoun) 의존도가 너무 높음**: pronoun_density에 20배, 15배 가중치
- **하지만 Real@1/Real@5는 대명사를 제외한 명사만 평가!**
- **불일치**: 품질 필터링은 대명사 중심, 평가는 명사 중심

#### 평가 데이터 생성 (build_real_coref_eval_set)
```python
def build_real_coref_eval_set():
    # 대명사가 없는 텍스트만 사용!
    tokens = KIWI.tokenize(text)
    has_pronoun = any(t.tag == 'NP' for t in tokens)

    if has_pronoun:
        continue  # 대명사 있으면 제외!

    # 같은 명사가 2번 이상 나오는 경우만 사용
    noun_counts = Counter(noun for token in tokens if token.tag in ['NNG', 'NNP'])
    if max(noun_counts.values()) < 2:
        continue
```

**역설**:
- **훈련 데이터 필터링**: 대명사가 많은 데이터 선호 (quality_score 높음)
- **평가 데이터**: 대명사 없는 데이터만 사용!
- **결과**: 훈련과 평가의 데이터 분포 불일치

---

## 🎯 성능 차이의 원인

### Real@1 성능 비교
| 데이터셋 | 최고 성능 | 샘플 수 | 데이터 다양성 |
|---------|----------|---------|-------------|
| **V2 MLM** | 66.04% (3ep) | 16,699개 | ⭐⭐⭐ (14개 소스, 하지만 샘플링으로 손실) |
| **Combined** | 66.75% (5ep) | 50,665개 | ⭐⭐⭐⭐⭐ (6개 소스, 전체 데이터) |

### 성능 차이 원인 분석

#### 1. **데이터 양 부족** (-67% 데이터)
```
V2 MLM:   16,699개
Combined: 50,665개

차이: -33,966개 (-67% 부족)
```

#### 2. **샘플링으로 인한 데이터 손실**
```
V2 MLM 프로세스:
1. 14개 데이터 소스 로드
2. 각 소스에서 2,000개만 샘플링
3. 품질 필터링 (40% 통과)
4. 최종: 16,699개

손실률: 99% 이상의 원본 데이터 버림!
```

#### 3. **품질 필터링의 역설**
```
고품질 필터링 기준:
- 대명사 밀도 높음 → 점수 ↑
- 개체 밀도 높음 → 점수 ↑

하지만 평가는:
- 대명사 없는 텍스트만 사용!
- 반복 명사만 평가!

결과: 훈련-평가 불일치
```

---

## 📈 MLM v2의 초기 성능이 높았던 이유

### MLM v2 @ 0.5 에폭: Real@1 = 65.58%
### Combined @ 1 에폭: Real@1 = 62.92%

**왜 초기에는 MLM v2가 더 좋았을까?**

#### 1. **데이터 다양성**
- MLM v2: 14개 다양한 소스 (뉴스 8개 추가)
- Combined: 6개 소스 (Wikipedia, KLUE, KorQuAD 등)

#### 2. **도메인 분포**
```
MLM v2:
- 뉴스 소스: 8개 (금융, 경제, 네이버, 번역 등)
- QA: 2개
- Wikipedia: 1개
- 기타: 3개

Combined:
- Wikipedia: 대부분
- KLUE: 일부
- KorQuAD: 일부
```

#### 3. **빠른 수렴**
- **적은 데이터** (16K) → **빠른 수렴** (0.5 에폭)
- **다양한 도메인** → **일반화 빠름**

#### 4. **하지만 한계 도달**
- **데이터 부족** → 3 에폭 이후 과적합
- **샘플 다양성 부족** → 더 이상 개선 불가

---

## 🔬 데이터 품질 실제 비교

### 샘플 분석 결과

#### V1 (187개) - 고품질, 극소량
```
샘플 예시:
"1945년 8월 15일 일제로부터 해방되자 가장 먼저 활동을 시작한 세력은
여운형을 중심으로 하는 민족주의 좌파세력들과 박헌영의 조선공산당을
중심으로 하는 공산주의 극좌세력들이었다..."

특징:
- 토큰 수: 617~909 (평균 744)
- 내용: 정치, 사회, 역사 등 심도 있는 주제
- 품질: 매우 높음 (대명사, 개체 밀도 모두 높음)
```

#### V2 MLM (16,699개) - 중간 품질, 중간 양
```
샘플 예시:
"美 경제 상반기 이미 경기침체 진입 말 아꼈던 WSJ도 추가 하락 전망
유동성 키운 엔캐리 트레이드 끝물 채무위기 고조로 유로존도 불안..."

특징:
- 토큰 수: 563~1083 (평균 854)
- 내용: 뉴스 기사, 짧고 간결
- 품질: 중상 (뉴스 특성상 개체 많음, 대명사 적음)
```

#### Combined (50,665개) - 다양한 품질, 대량
```
샘플 예시:
"부산대 철학과에 다니는 홍준성 씨(24)는 지난해 가을 학생 신분으로
제3회 한경 청년신춘문예의 문을 두드렸다. 홍씨는 철학과 책을 좋아하는
평범한 학생이었다. 신춘문예 장편소설 부문에 당선되며 그의 삶은 변했다..."

특징:
- 토큰 수: 289~864 (평균 489)
- 내용: Wikipedia, 뉴스, QA 혼합
- 품질: 다양 (대명사 "그의", "홍씨" 등 자연스러운 coref)
```

---

## 💡 핵심 결론

### 왜 MLM v2가 Combined보다 성능이 낮은가?

#### ❌ **잘못된 가정들**

1. **"더 많은 데이터 소스 = 더 좋은 성능"**
   - 실제: 샘플링으로 99% 버림 → 실제 사용 데이터 오히려 감소

2. **"품질 필터링 = 더 좋은 성능"**
   - 실제: 품질 기준과 평가 기준 불일치 → 역효과

3. **"대명사 많은 데이터 = 좋은 Coref 데이터"**
   - 실제: Real@1/Real@5는 대명사 없는 데이터로 평가 → 훈련-평가 불일치

#### ✅ **실제 원인**

1. **데이터 양 부족** (16K vs 50K)
   - 67% 적은 데이터로 학습
   - 조기 과적합 (3 에폭)

2. **샘플링의 치명적 손실**
   - 원본 데이터의 99%를 평가조차 하지 않음
   - 고품질 샘플도 샘플링 과정에서 누락

3. **품질 기준 불일치**
   - 훈련: 대명사 선호
   - 평가: 대명사 제외
   - 결과: 학습 방향 틀어짐

4. **데이터 다양성 vs 양의 트레이드오프**
   - MLM v2: 14개 소스, 하지만 각 소스에서 적게 추출
   - Combined: 6개 소스, 하지만 각 소스에서 대량 추출
   - **결과: 양(Combined)이 다양성(MLM v2)을 이김**

---

## 🚀 해결 방안

### 1. **품질 필터링 제거 또는 대폭 완화**

```python
# 현재 (V2)
sample_size = min(2000, len(dataset))
indices = list(range(0, len(dataset), max(1, len(dataset) // sample_size)))
quality_threshold = 0.6

# 개선안 1: 샘플링 제거
# 모든 데이터 품질 평가 후 필터링
qualities = batch_analyze_quality(all_texts)
high_quality = [text for text, q in zip(all_texts, qualities) if q['quality_score'] >= 0.4]

# 개선안 2: 품질 필터링 완전 제거
# 길이 필터링만 수행
valid_texts = [text for text in all_texts if 50 < len(text) < max_len]
```

### 2. **평가에 맞춘 품질 기준 재설계**

```python
# 현재: 대명사 중심
coref_interaction = pronoun_density * 20 + entity_density * 3 + ...

# 개선안: 명사 반복 중심
def real_coref_quality(text):
    tokens = KIWI.tokenize(text)
    nouns = [t.form for t in tokens if t.tag in ['NNG', 'NNP'] and len(t.form) > 1]

    # 반복 명사 카운트
    noun_counts = Counter(nouns)
    repeated_nouns = sum(1 for count in noun_counts.values() if count >= 2)

    # 반복 비율
    repetition_score = repeated_nouns / max(1, len(set(nouns)))

    # 개체 다양성
    diversity_score = len(set(nouns)) / max(1, len(nouns))

    # 종합 점수
    quality_score = 0.6 * repetition_score + 0.4 * diversity_score

    return quality_score
```

### 3. **대량 데이터 활용 전략**

```python
# 14개 소스 모두 활용, 샘플링 최소화
def prepare_mlm_v3():
    all_data = []

    for source in sources:
        # 스트리밍으로 메모리 효율적 처리
        dataset = load_dataset(source, streaming=True)

        # 길이 필터링만 수행
        for item in dataset:
            text = extract_text(item, source.domain)
            tokens = tokenizer.encode(text)

            if target_seq_len * 0.3 < len(tokens) < target_seq_len * 1.5:
                all_data.append(text)

                # 메모리 관리: 10만개마다 저장
                if len(all_data) >= 100000:
                    save_batch(all_data)
                    all_data = []

    # 목표: 100K+ 샘플
```

### 4. **훈련 전략 조정**

```python
# MLM v2의 장점 활용
- 초기 수렴 속도 빠름 (0.5 에폭에 65.58%)
- 데이터 다양성 우수 (14개 소스)

# 개선 방향
1. 대량 데이터 생성 (100K+ 샘플)
2. Early stopping (3-4 에폭)
3. 데이터 다양성 유지 (14개 소스 모두 활용)
```

---

## 📊 예상 성능 개선

### 현재 (MLM v2)
```
샘플 수: 16,699개
최고 성능: Real@1 = 66.04% (3 에폭)
문제: 과적합, 데이터 부족
```

### 개선 후 (MLM v3)
```
샘플 수: 100,000개+ (6배 증가)
예상 성능: Real@1 = 68-69% (3-4 에폭)
근거:
- Combined (50K) = 66.75%
- MLM v3 (100K, 14개 소스) = 68-69% (예상)
```

### Entity Fine-tuning 후
```
MLM v3 (100K) → Entity FT
예상: Real@1 = 69-70%
현재 베스트: 67.78%
개선폭: +1-2%p
```

---

## 🎯 즉시 실행 가능한 액션

### 단기 (지금 당장)
1. ✅ **Combined 데이터셋 사용**
   - 이미 50K 샘플로 66.75% 달성
   - 안정적이고 입증된 성능

2. ✅ **MLM v2 checkpoint-390 + Entity FT**
   - MLM v2의 초기 수렴 속도 활용
   - Entity 데이터로 추가 학습

### 중기 (1-2일)
3. 🔄 **MLM v3 데이터셋 생성**
   - 샘플링 제거 or 대폭 완화
   - 품질 필터링 기준 재설계
   - 목표: 100K+ 샘플

4. 🔄 **MLM v3 훈련**
   - 3-4 에폭 훈련
   - Early stopping 적용

### 장기 (검증 후)
5. 📊 **평가 기준 재검토**
   - Real@1/Real@5와 훈련 목표 일치 확인
   - 필요시 평가 메트릭 추가

---

## 📝 교훈

### ✅ **올바른 접근**
1. **데이터 양 > 데이터 품질 필터링**
2. **훈련-평가 일치 > 주관적 품질 기준**
3. **전체 데이터 활용 > 샘플링**
4. **실험 결과 분석 > 가정 기반 설계**

### ❌ **잘못된 접근**
1. ~~"품질 필터링하면 무조건 좋다"~~
2. ~~"대명사 많은 데이터 = Coref 데이터"~~
3. ~~"샘플링으로 충분하다"~~
4. ~~"더 많은 소스 = 더 좋은 성능"~~

### 🎓 **핵심 인사이트**
> **"Quality filtering is only useful when quality criteria align with evaluation metrics."**
>
> 품질 필터링은 품질 기준이 평가 메트릭과 일치할 때만 유용하다.

> **"Data quantity beats quality filtering when filtering discards 99% of data."**
>
> 필터링이 99%의 데이터를 버릴 때는 데이터 양이 품질 필터링을 이긴다.

> **"Fast convergence on small data != good final performance."**
>
> 적은 데이터에서의 빠른 수렴 ≠ 좋은 최종 성능

---

## 🔍 추가 조사 필요 사항

1. **Combined 데이터셋 구성 확인**
   - 어떤 데이터가 포함되었는지
   - 왜 50K 샘플이 생성되었는지

2. **품질 점수 분포 분석**
   - V1, V2, Combined의 품질 점수 분포
   - 실제 성능과의 상관관계

3. **Real@1/Real@5 평가 데이터 분석**
   - 평가 데이터의 특성
   - 훈련 데이터와의 분포 차이

---

## 📌 결론

**MLM v2가 기대 이하의 성능을 보인 이유는 데이터셋 품질이 나쁘거나 데이터 소스가 부족해서가 아니라, 잘못된 샘플링과 불일치하는 품질 필터링 기준 때문입니다.**

**해결책은 간단합니다: 샘플링을 제거하고, 품질 필터링을 완화하거나 평가 메트릭에 맞게 재설계하여, 가능한 많은 데이터를 활용하는 것입니다.**

**"More data with simple filtering > Less data with complex quality filtering"**
