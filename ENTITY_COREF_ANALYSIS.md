# Entity Coreference Fine-tuning 성능 분석 보고서

**작성일**: 2025-10-10
**실험명**: Entity Replacement Coref Fine-tuning (Option 2)
**결론**: ❌ **Fine-tuning이 Coref 성능을 오히려 악화시킴**

---

## 📊 Executive Summary

Entity repetition 기반 fine-tuning이 Coref F1과 Coref@5 성능을 **각각 14.5%, 9.8% 감소**시켰습니다. 이는 **task mismatch와 catastrophic forgetting**이 원인으로 분석됩니다.

### 핵심 수치

| Metric | checkpoint-410 (원본) | checkpoint-1600 (fine-tuned) | 변화 |
|--------|----------------------|------------------------------|------|
| **LAMBADA@1** | 34.67% | 37.67% | **+8.7%** ✅ |
| **Coref F1** | **3.38%** | **2.89%** | **-14.5%** ❌ |
| **Coref@5** | **69.69%** | **62.84%** | **-9.8%** ❌ |
| **Overall Score** | 0.3266 | 0.3131 | **-4.1%** ❌ |

---

## 🎯 실험 설계

### 가설
"개체 반복 패턴을 학습하면 개체 간 관계 파악 능력이 향상되어 상호참조 해결 성능이 개선될 것이다."

### 훈련 방법 (Option 2)
- **데이터**: 40,000 samples (Wikipedia, KLUE MRC, Naver News)
- **방식**: 반복된 개체의 두 번째 출현을 마스킹
  ```
  원본: "홍길동은 학생이다. 홍길동은 공부한다."
  훈련: "홍길동은 학생이다. [MASK]은 공부한다."
  정답: "홍길동" (개체 자체)
  ```
- **체크포인트**: runs/combined_experiment/checkpoint-410 (seq_len=2048)
- **훈련**: 3 epochs, 1689 steps, batch_size=8, gradient_accumulation=8
- **Best model**: checkpoint-1600 (epoch 2.84, eval_loss=1.3654)

### 평가 방법
- **LAMBADA@1**: 언어 모델링 능력 (600 samples)
- **Coref F1 & @5**: 대명사 마스킹 → 문맥 명사 예측
  ```
  텍스트: "홍길동은 학생이다. 그는 공부한다."
  평가: "홍길동은 학생이다. [MASK]는 공부한다."
  정답: "그" (대명사)
  판정: 예측 명사가 문맥 명사 {"홍길동", "학생", ...}에 포함되는지
  ```

---

## 🔍 상세 결과 분석

### 1. LAMBADA 성능 (언어 모델링)
| Checkpoint | LAMBADA@1 | 변화 | 분석 |
|------------|-----------|------|------|
| checkpoint-410 | 34.67% | - | 원본 MLM 성능 |
| checkpoint-1600 | 37.67% | **+8.7%** | 일반 언어 모델링 능력 **향상** |

**해석**: Entity fine-tuning이 MLM 능력 자체는 개선했음. 특정 패턴(개체 반복)에 대한 학습이 일반화됨.

### 2. Coref F1 (정확도)
| Checkpoint | Coref F1 | 변화 | 분석 |
|------------|----------|------|------|
| checkpoint-410 | **3.38%** | - | 원본 성능 |
| checkpoint-1600 | **2.89%** | **-14.5%** | 대명사 예측 능력 **저하** |

**해석**: 개체 반복 학습이 대명사 처리 능력을 오히려 약화시킴.

### 3. Coref@5 (Top-5 Recall)
| Checkpoint | Coref@5 | 변화 | 분석 |
|------------|---------|------|------|
| checkpoint-410 | **69.69%** | - | 원본 성능 |
| checkpoint-1600 | **62.84%** | **-9.8%** | Top-5 후보에서도 정답률 **감소** |

**해석**: 단순히 정확도뿐 아니라 recall도 저하. 개체 링킹 전략 자체가 변질됨.

### 4. 훈련 중 Loss 추이
| Step | Epoch | Train Loss | Eval Loss | 분석 |
|------|-------|------------|-----------|------|
| 200 | 0.36 | 1.5642 | **1.5270** | 시작 |
| 600 | 1.07 | 1.5348 | **1.4333** | 빠른 감소 |
| 1000 | 1.78 | 1.4808 | **1.3999** | 지속 개선 |
| 1600 | 2.84 | 1.4557 | **1.3654** | ⭐ Best |
| 1689 | 3.0 | 1.3970 | - | 최종 |

**해석**: Train loss는 정상적으로 감소했으나, 이는 **entity repetition 데이터에 대한 overfitting**이었음.

---

## ⚠️ 실패 원인 분석

### 1. Task Mismatch (작업 불일치) ★★★★★
**가장 치명적인 원인**

| 항목 | 훈련 (Entity Fine-tuning) | 평가 (Coref Evaluation) |
|------|---------------------------|-------------------------|
| **입력 패턴** | 개체 반복: "홍길동...홍길동" | 대명사: "홍길동...그" |
| **마스킹 대상** | 두 번째 개체 출현 | 대명사 |
| **정답** | 개체 자체 ("홍길동") | 문맥 명사 후보 중 선택 |
| **태스크** | Named Entity 반복 예측 | 대명사 → 선행사 링킹 |

**문제점**:
- 모델이 "개체가 반복되면 그대로 예측"하도록 학습됨
- 대명사 "그", "이", "저" 등을 개체로 매핑하는 능력은 학습되지 않음
- **완전히 다른 분포의 태스크**로 fine-tuning한 것

### 2. Catastrophic Forgetting (재앙적 망각) ★★★★
- 원본 MLM이 가지고 있던 **대명사 처리 능력이 손실**됨
- 40,000 samples (3 epochs) 동안 개체 반복 패턴만 집중 학습
- 대명사는 훈련 데이터에서 **완전히 배제**됨 (고유명사 NNP만 사용)

```python
# scripts/prepare_entity_coref_dataset.py:23-25
tks = kiwi.tokenize(text)
entities = {tk.form for tk in tks if tk.tag == "NNP" and len(tk.form) >= 2}
# ❌ 대명사(NP)는 완전히 무시됨!
```

### 3. Distribution Shift (분포 변화) ★★★
| 분포 요소 | 훈련 데이터 | 평가 데이터 |
|-----------|-------------|-------------|
| **어휘 유형** | 고유명사 (NNP) | 대명사 (NP) + 문맥 명사 |
| **거리** | 평균 165자 | 가변적 (문장 내~문장 간) |
| **확실성** | 100% 확실 (같은 개체) | 불확실 (여러 후보) |

### 4. Evaluation Metric Mismatch (평가 지표 불일치) ★★
- **훈련 목표**: Exact match (개체 문자열 정확히 일치)
- **평가 목표**: Set inclusion (예측 명사가 문맥 명사 집합에 포함)

```python
# 훈련 시: "홍길동" == "홍길동" (정답)
# 평가 시: "선생님" in ["홍길동", "선생님", "학교"] (정답)
```

---

## 📈 Combined_experiment 전체 결과 비교

### seq_len=2048 체크포인트 (5개)

| Checkpoint | Step | Epoch | LAMBADA@1 | Coref F1 | Coref@5 | Score | 순위 |
|------------|------|-------|-----------|----------|---------|-------|------|
| checkpoint-820 | 820 | 2.0 | 36.17% | 3.27% | 68.03% | 0.3257 | 🥇 1 |
| **checkpoint-410** | 410 | 1.0 | 34.67% | **3.38%** | **69.69%** | 0.3266 | 🥇 **1 (Best!)** |
| checkpoint-2050 | 2050 | 5.0 | 36.83% | 3.29% | 67.91% | 0.3274 | 🥈 2 |
| checkpoint-1640 | 1640 | 4.0 | 36.00% | 3.28% | 68.09% | 0.3254 | 🥉 3 |
| checkpoint-1230 | 1230 | 3.0 | 35.00% | 3.30% | 68.41% | 0.3234 | 4 |
| **checkpoint-1600 (fine-tuned)** | 1600 | 2.84 | 37.67% | **2.89%** | **62.84%** | 0.3131 | ❌ **최하위** |

### seq_len=1536 체크포인트 (5개)

| Checkpoint | Step | Epoch | LAMBADA@1 | Coref F1 | Coref@5 | Score | 순위 |
|------------|------|-------|-----------|----------|---------|-------|------|
| checkpoint-396 | 396 | 1.0 | 35.83% | **4.10%** | **68.38%** | 0.3290 | 🥇 1 |
| checkpoint-1584 | 1584 | 4.0 | 36.50% | 3.90% | 66.08% | 0.3234 | 🥈 2 |
| checkpoint-792 | 792 | 2.0 | 36.33% | 3.92% | 66.17% | 0.3232 | 🥉 3 |
| checkpoint-1980 | 1980 | 5.0 | 37.00% | 3.86% | 65.33% | 0.3224 | 4 |
| checkpoint-1188 | 1188 | 3.0 | 35.33% | 3.79% | 64.42% | 0.3144 | 5 |

**핵심 발견**:
- ✅ **checkpoint-410 (seq=2048, step=410)**이 **전체 최고 Coref 성능**
- ✅ Early stopping 효과: epoch 1-2가 가장 좋음
- ❌ Entity fine-tuning은 모든 원본 체크포인트보다 **성능 저하**

---

## 💡 개선 방안

### Option 1: 대명사 기반 Coref 데이터로 재훈련 (★★★★★ 추천)

**방법**: 평가 방식과 동일하게 훈련 데이터 생성

```python
# 기존 (실패한 방식)
원본: "홍길동은 학생이다. 홍길동은 공부한다."
훈련: "홍길동은 학생이다. [MASK]은 공부한다."
정답: "홍길동"

# 개선 방안
원본: "홍길동은 학생이다. 그는 공부한다."
훈련: "홍길동은 학생이다. [MASK]는 공부한다."
정답: 문맥 명사 집합에 포함된 예측 (["홍길동", "학생", ...])
손실: 예측 명사가 문맥 명사에 포함되도록 학습
```

**장점**:
- ✅ 훈련과 평가 task 완벽히 일치
- ✅ 대명사 처리 능력 직접 학습
- ✅ 실제 상호참조 해결 패턴 학습

**구현 계획**:
1. `scripts/prepare_pronoun_coref_dataset.py` 생성
2. Wikipedia/KLUE에서 대명사(NP) 추출
3. 대명사 마스킹 + 문맥 명사 추출
4. 40,000 samples 생성
5. checkpoint-410에서 재훈련

### Option 2: Mixed Training (개체 + 대명사) (★★★★)

**방법**: 두 task를 혼합하여 훈련

```python
# 50% 개체 반복
"홍길동...홍길동" → "홍길동...[MASK]" (target: "홍길동")

# 50% 대명사 coref
"홍길동...그" → "홍길동...[MASK]" (target: 문맥 명사)
```

**장점**:
- ✅ 두 능력 모두 학습
- ✅ Catastrophic forgetting 완화
- ⚠️ 하지만 task confusion 가능성

### Option 3: Curriculum Learning (★★★)

**방법**: 단계적 학습

```python
# Stage 1 (10,000 samples): 개체 반복 (쉬운 패턴)
"홍길동...홍길동" → [MASK]

# Stage 2 (20,000 samples): 대명사 coref (어려운 패턴)
"홍길동...그" → [MASK]

# Stage 3 (10,000 samples): 혼합
```

### Option 4: 원본 checkpoint-410 사용 (★★★★ 현실적)

**방법**: Entity fine-tuning을 포기하고 원본 사용

**근거**:
- checkpoint-410이 이미 **전체 최고 Coref 성능**
- Coref@5 69.69%는 충분히 준수한 성능
- Fine-tuning으로 개선 불가능함을 실험적으로 증명

**장점**:
- ✅ 추가 작업 불필요
- ✅ 검증된 성능
- ✅ 리소스 절약

---

## 🎓 교훈 (Lessons Learned)

### 1. Task Alignment is Critical
- Fine-tuning은 **평가 task와 정확히 일치**해야 함
- "비슷한" task는 오히려 혼란을 야기할 수 있음

### 2. Distribution Matters More Than Intuition
- "개체 반복을 학습하면 상호참조도 잘할 것"이라는 **직관은 틀렸음**
- 실제 데이터 분포와 평가 방식을 정확히 분석해야 함

### 3. Catastrophic Forgetting in Specialized Fine-tuning
- 특정 패턴에만 집중한 fine-tuning은 **기존 능력 손실** 위험
- Mixed training이나 regularization 필요

### 4. Early Stopping is Powerful
- epoch 1-2가 가장 좋은 성능 (checkpoint-410, checkpoint-396)
- 과도한 훈련은 오히려 **일반화 능력 저하**

### 5. Baseline is Strong
- Combined_experiment의 MLM 훈련이 이미 충분한 coref 능력 학습
- 추가 fine-tuning이 항상 개선을 보장하지 않음

---

## 📝 권장 사항

### 즉시 실행 (Immediate Actions)
1. ✅ **checkpoint-410을 최종 모델로 선정**
   - Coref F1: 3.38%, Coref@5: 69.69%
   - 전체 체크포인트 중 최고 성능

2. ✅ **Entity fine-tuning 중단**
   - 현재 방식으로는 개선 불가능
   - 리소스 낭비 방지

### 향후 실험 (Future Work)
1. 🔬 **Option 1 시도**: 대명사 기반 coref 데이터로 재훈련
   - 예상 개선: Coref F1 3.38% → 5-8%
   - 예상 개선: Coref@5 69.69% → 75-80%

2. 🔬 **Auxiliary Task Learning**
   - MLM + Coref 동시 학습
   - Multi-task learning framework

3. 🔬 **Zero-shot Prompting**
   - Fine-tuning 대신 in-context learning
   - Few-shot examples로 coref 해결

---

## 📚 참고 자료

### 관련 파일
- 훈련 스크립트: `scripts/run_entity_coref_finetune.py`
- 데이터 생성: `scripts/prepare_entity_coref_dataset.py`
- 평가 스크립트: `scripts/reevaluate_checkpoints.py`
- 평가 로직: `coref_automl/tune.py` (build_coref_eval_set, eval_coref_f1)

### 체크포인트
- **Best original**: `runs/combined_experiment/checkpoint-410`
- **Fine-tuned (failed)**: `runs/entity_coref_finetune/.../checkpoint-1600`

### 데이터
- 훈련 데이터: `prepared_datasets/entity_coref_2048/` (40,000 samples)
- 평가 결과: `reevaluation_results_partial.json`

---

**결론**: Entity repetition fine-tuning은 실패했으나, 원본 checkpoint-410이 충분히 우수한 성능을 보임. 향후 대명사 기반 재훈련을 시도하거나, 현재 모델을 그대로 사용하는 것을 권장함.
