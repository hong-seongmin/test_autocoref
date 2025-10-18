# KoCoNovel

**KoCoNovel**은 50개의 한국 현대 및 근대 소설을 기반으로 한 문자 상호참조(Character Coreference) 데이터셋입니다.

자세한 내용은 논문을 참고하세요: [KoCoNovel: Annotated Dataset of Character Coreference in Korean Novels](https://arxiv.org/abs/2404.01140)

## 출처
- GitHub: https://github.com/storidient/koconovel
- arXiv 논문: https://arxiv.org/abs/2404.01140

## 업데이트 내역
- **[2025-02]** CoNLL 및 JSONL 형식으로 데이터셋 제공
- **[2025-02]** 모든 직접 인용문에 대한 화자 할당 주석 추가

---

## 말뭉치(Corpus)

말뭉치는 [Wikisource](https://ko.wikisource.org/wiki/)의 퍼블릭 도메인 텍스트에서 가져왔습니다. 전처리 과정에서 오타와 잘못된 줄 바꿈을 수정하고 현대 한국어 문법에 맞게 철자를 조정했습니다. 소설 목록은 `list_of_novels.csv` 파일에서 확인할 수 있습니다.

## 데이터 및 주석

KoCoNovel은 문법적으로 교정된 소설 텍스트와 4가지 옵션의 문자 상호참조 주석, 그리고 모든 직접 인용문에 대한 화자 주석을 포함합니다.

### 주석 옵션

- **[Reader/Omniscient]** 전지적 작가 또는 독자의 관점
  - **Omniscient (전지적)**: 서사 전체에 대한 완전한 지식 기반
  - **Reader (독자)**: 장면 수준의 제한된 정보 기반

- **[Separate/Overlapped]** 복수 개체 처리 방식
  - **Separate (분리)**: 복수 개체를 개별 개체와 별도로 처리 (예: ['우리'], ['나'], ['너'])
  - **Overlapped (중첩)**: 복수 개체를 개별 개체 참조의 중첩으로 표현 (예: ['우리', '나'], ['우리', '너'])

## 데이터 형식

### JSONL 형식
상호참조 클러스터와 화자 식별 간의 관계를 쉽게 분석할 수 있도록 설계되었습니다.

**text.jsonl 구조:**
```json
{"doc_id": "20_0", "text": "소설 본문..."}
```

**coref.jsonl 구조:**
```json
{
  "doc_id": "20_0",
  "omniscent_separate": [[["577", "578", "그", "C"], ["553", "556", "박춘수", "C"]]],
  "omniscent_overlapped": [...],
  "reader_separate": [...],
  "reader_overlapped": [...],
  "speakers": []
}
```

### CoNLL 형식
e2e-coref, s2e-coref, LingMess와 같은 기존 상호참조 해결 모델과 호환되는 표준 형식입니다.

## 디렉토리 구조

```
koconovel/
├── data/
│   ├── conll/                      # CoNLL 형식
│   │   ├── omniscent_overlapped/   # 전지적 + 중첩
│   │   ├── omniscent_separate/     # 전지적 + 분리
│   │   ├── reader_overlapped/      # 독자 + 중첩
│   │   └── reader_separate/        # 독자 + 분리
│   └── jsonl/                      # JSONL 형식
│       └── [50개 소설 디렉토리]
├── list_of_novels.csv              # 소설 목록
├── LICENSE                         # CC BY-SA 4.0
└── README.md
```

## 데이터 통계

| 항목 | 값 |
|------|-----|
| 총 소설 수 | 50편 |
| 데이터 형식 | CoNLL, JSONL |
| 주석 유형 | 4가지 |
| 주석 작업 | NER, 상호참조, 화자 식별 |

## 사용 예시

### Python으로 JSONL 읽기
```python
import json

# 텍스트 읽기
with open('data/jsonl/20_Anemone/text.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        print(f"Doc ID: {data['doc_id']}")
        print(f"Text: {data['text'][:100]}...")

# 상호참조 주석 읽기
with open('data/jsonl/20_Anemone/coref.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        coref_data = json.loads(line)
        print(f"Omniscient-Separate: {coref_data['omniscent_separate']}")
```

## 라이선스

이 데이터셋은 [Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/) 라이선스에 따라 제공됩니다.

## 인용

이 데이터셋을 사용하는 경우 다음과 같이 인용해 주세요:

```bibtex
@misc{kim2024koconovel,
      title={KoCoNovel: Annotated Dataset of Character Coreference in Korean Novels},
      author={Kyuhee Kim and Surin Lee and Sangah Lee},
      year={2024},
      eprint={2404.01140},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


