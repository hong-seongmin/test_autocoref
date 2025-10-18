# KoBookNLP

> **중요**: KoBookNLP는 KoCoNovel 데이터셋과 동일합니다. 두 리포지토리는 같은 데이터를 제공합니다.

**KoBookNLP/KoCoNovel**은 50개의 한국 현대 및 근대 소설을 기반으로 한 NLP 라이브러리 및 문자 상호참조 데이터셋입니다.

## 출처
- **KoBookNLP GitHub**: https://github.com/storidient/KoBookNLP
- **KoCoNovel GitHub**: https://github.com/storidient/koconovel
- **논문**: [KoCoNovel: Annotated Dataset of Character Coreference in Korean Novels](https://arxiv.org/abs/2404.01140)

## 데이터셋 개요

이 디렉토리는 KoBookNLP/KoCoNovel 데이터셋을 포함하고 있으며, 한국어 문학 텍스트를 위한 NLP 라이브러리에서 사용됩니다.

### 주요 기능
- **문자 개체명 인식(NER)**: 등장인물 식별
- **상호참조 해결(Coreference Resolution)**: 동일한 개체를 가리키는 표현들을 연결
- **화자 식별(Speaker Identification)**: 직접 인용문의 화자 표시

### 데이터 출처
- **말뭉치**: [Wikisource](https://ko.wikisource.org/wiki/)의 퍼블릭 도메인 한국 소설
- **전처리**: 맞춤법 교정, 줄 바꿈 수정, 현대 한국어 문법에 맞춘 철자 조정
- **소설 목록**: `list_of_novels.csv` 참조

## 주석 유형

4가지 주석 옵션을 제공합니다:

- **[Reader/Omniscient]** 관점의 차이
  - **Omniscient (전지적)**: 서사 전체에 대한 완전한 지식 기반
  - **Reader (독자)**: 장면 수준의 제한된 정보 기반

- **[Separate/Overlapped]** 복수 개체 처리 방식
  - **Separate (분리)**: 복수 개체를 개별 개체와 별도로 처리
  - **Overlapped (중첩)**: 복수 개체를 개별 개체 참조의 중첩으로 표현

## 데이터 형식

### JSONL 형식
- 상호참조 클러스터와 화자 식별 간의 관계 분석에 최적화
- `text.jsonl`: 소설 본문
- `coref.jsonl`: 상호참조 및 화자 주석

### CoNLL 형식
- e2e-coref, s2e-coref, LingMess 등 기존 모델과 호환
- 4가지 주석 유형별 디렉토리 제공

## 데이터 통계

| 항목 | 값 |
|------|-----|
| 총 소설 수 | 50편 |
| 데이터 형식 | CoNLL, JSONL |
| 주석 유형 | 4가지 |
| 주석 작업 | NER, 상호참조, 화자 식별 |

## KoBookNLP vs KoCoNovel

두 리포지토리는 **동일한 데이터셋**을 제공합니다:
- **KoBookNLP**: NLP 라이브러리로 시작하여 데이터셋을 포함
- **KoCoNovel**: 데이터셋에 초점을 맞춘 리포지토리
- 데이터 내용과 형식은 완전히 동일함

## 라이선스

[Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/) 또는 [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)

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

## 추가 정보

자세한 사용 방법과 데이터 구조는 KoCoNovel 리포지토리를 참고하세요:
- GitHub: https://github.com/storidient/koconovel
- 논문: https://arxiv.org/abs/2404.01140


