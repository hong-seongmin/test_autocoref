# 한국어 상호참조(Coreference) 데이터셋 모음

이 디렉토리는 한국어 상호참조 해결(Coreference Resolution) 연구를 위한 여러 데이터셋을 포함하고 있습니다.

## 📁 데이터셋 구조

```
dataset/
├── ecmt/              # ECMT 상호참조 데이터셋 (CoNLL-U 형식)
├── koconovel/         # KoCoNovel 문학 텍스트 상호참조 데이터셋
├── kobooknlp/         # KoBookNLP (KoCoNovel과 동일한 데이터)
└── README.md          # 이 문서
```

## 📊 데이터셋 개요

### 1. ECMT (Entity-Centric Machine Translation)
- **경로**: `ecmt/`
- **형식**: CoNLL-U (CoNLL Universal Dependencies)
- **내용**: 한국어 상호참조 주석이 포함된 텍스트
- **파일**:
  - `ko_ecmt-corefud-train.conllu` (481,333 줄, ~33.8 MB)
  - `ko_ecmt-corefud-dev.conllu` (52,982 줄, ~3.7 MB)
- **특징**: 개체 정보, 토큰 범위, 의존 관계 포함
- **문서**: [ecmt/README.md](ecmt/README.md)

### 2. KoCoNovel
- **경로**: `koconovel/`
- **출처**: [GitHub](https://github.com/storidient/koconovel) | [논문](https://arxiv.org/abs/2404.01140)
- **형식**: CoNLL, JSONL
- **내용**: 50개 한국 현대/근대 소설의 문자 상호참조 주석
- **주석 유형**: 4가지 (관점 2 × 복수처리 2)
  - omniscient/reader × separate/overlapped
- **주석 작업**:
  - 문자 개체명 인식 (NER)
  - 상호참조 해결
  - 화자 식별
- **라이선스**: CC BY-SA 4.0
- **문서**: [koconovel/README.md](koconovel/README.md)

### 3. KoBookNLP
- **경로**: `kobooknlp/`
- **출처**: [GitHub](https://github.com/storidient/KoBookNLP)
- **⚠️ 중요**: KoCoNovel과 **동일한 데이터셋**
- **차이점**:
  - KoBookNLP: NLP 라이브러리로 시작
  - KoCoNovel: 데이터셋 중심
  - 데이터 내용은 완전히 동일
- **문서**: [kobooknlp/README.md](kobooknlp/README.md)

## 🔍 데이터셋 비교

| 데이터셋 | 형식 | 크기 | 도메인 | 주석 유형 |
|---------|------|------|--------|-----------|
| ECMT | CoNLL-U | ~37 MB | 일반 | 개체, 토큰, 의존관계 |
| KoCoNovel | CoNLL, JSONL | 50편 소설 | 문학 | NER, 상호참조, 화자 |
| KoBookNLP | CoNLL, JSONL | 50편 소설 | 문학 | NER, 상호참조, 화자 |

## 🚀 사용 방법

### ECMT 데이터 읽기
```python
def read_conllu(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = []
        current_sent = []
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                continue
            elif not line:
                if current_sent:
                    sentences.append(current_sent)
                    current_sent = []
            else:
                fields = line.split('\t')
                current_sent.append(fields)
        if current_sent:
            sentences.append(current_sent)
    return sentences

# 사용 예시
train_data = read_conllu('ecmt/ko_ecmt-corefud-train.conllu')
```

### KoCoNovel/KoBookNLP JSONL 읽기
```python
import json

# 텍스트 읽기
with open('koconovel/data/jsonl/20_Anemone/text.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line)
        print(f"Doc ID: {data['doc_id']}")
        print(f"Text: {data['text'][:100]}...")

# 상호참조 주석 읽기
with open('koconovel/data/jsonl/20_Anemone/coref.jsonl', 'r', encoding='utf-8') as f:
    for line in f:
        coref_data = json.loads(line)
        print(f"Coreferences: {coref_data['omniscent_separate']}")
```

## 📝 인용

### ECMT
ECMT 데이터셋의 인용 정보는 원본 출처를 참고하세요.

### KoCoNovel/KoBookNLP
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

## 📄 라이선스

- **ECMT**: 원본 출처의 라이선스 참조
- **KoCoNovel**: [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)
- **KoBookNLP**: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) 또는 CC BY-SA 4.0

## 🔗 관련 리소스

### 공식 리포지토리
- [KoCoNovel GitHub](https://github.com/storidient/koconovel)
- [KoBookNLP GitHub](https://github.com/storidient/KoBookNLP)

### 관련 도구 및 표준
- [CoNLL-U Format](https://universaldependencies.org/format.html)
- [Universal Dependencies](https://universaldependencies.org/)

## 📚 추가 정보

각 데이터셋에 대한 자세한 정보는 해당 디렉토리의 README 파일을 참조하세요:
- [ECMT 상세 문서](ecmt/README.md)
- [KoCoNovel 상세 문서](koconovel/README.md)
- [KoBookNLP 상세 문서](kobooknlp/README.md)

## 📊 데이터셋 통계 요약

| 구분 | ECMT | KoCoNovel/KoBookNLP |
|------|------|---------------------|
| **총 크기** | ~37 MB | 50편 소설 |
| **형식** | CoNLL-U | CoNLL + JSONL |
| **주석** | 개체, 의존관계 | NER, 상호참조, 화자 |
| **언어** | 한국어 | 한국어 |
| **도메인** | 일반 | 문학 (소설) |
