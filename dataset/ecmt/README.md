# ECMT (Entity-Centric Machine Translation) Korean Coreference Dataset

## 개요
ECMT는 한국어 상호참조(Coreference Resolution)를 위한 CoNLL-U 형식의 데이터셋입니다.

## 데이터 구성

### 파일 목록
- `ko_ecmt-corefud-train.conllu` - 학습용 데이터 (481,333 줄)
- `ko_ecmt-corefud-dev.conllu` - 검증용 데이터 (52,982 줄)

### 데이터 형식
데이터는 CoNLL-U (CoNLL Universal Dependencies) 형식으로 제공됩니다.

#### CoNLL-U 형식 구조
각 토큰은 다음과 같은 10개의 필드를 포함합니다:
1. **ID**: 토큰 인덱스
2. **FORM**: 단어 형태
3. **LEMMA**: 기본형
4. **UPOS**: Universal POS 태그
5. **XPOS**: 언어별 POS 태그
6. **FEATS**: 형태론적 특징
7. **HEAD**: 구문 의존 관계의 헤드
8. **DEPREL**: 의존 관계 유형
9. **DEPS**: 추가 의존 관계
10. **MISC**: 기타 정보 (Entity, SpaceAfter, TokenRange 등)

#### 상호참조 주석
- **Entity**: 개체 정보가 `Entity=(eid-etype-head-other)` 형식으로 표기됨
- 예시: `Entity=(e67050--2|TokenRange=0:4)` 또는 `Entity=e67050)`

### 데이터 샘플
```
# newdoc id = 1430
# global.Entity = eid-etype-head-other
# newpar id = 1
# sent_id = 1430:1:1
# text = 9·11 테러(September 11 attacks)는 2001년 9월 11일에 미국에서 벌어진 항공기 납치 동시다발 자살 테러로 뉴욕의 110층짜리 세계무역센터(WTC) 쌍둥이 빌딩이 무너지고, 버지니아 주 알링턴 군의 미국 국방부 펜타곤이 공격받은 대참사이다.
1	9·11	9+·+1+1	NUM	nnc	_	2	nummod	_	Entity=(e67050--2|TokenRange=0:4
2	테러	테러	NOUN	ncn	_	36	dislocated	_	Entity=e67050)|SpaceAfter=No|TokenRange=5:7
```

## 통계

| 구분 | 학습 데이터 | 검증 데이터 |
|------|------------|------------|
| 총 라인 수 | 481,333 | 52,982 |
| 파일 크기 | ~33.8 MB | ~3.7 MB |

## 사용 방법

### Python에서 CoNLL-U 파일 읽기
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
train_data = read_conllu('ko_ecmt-corefud-train.conllu')
dev_data = read_conllu('ko_ecmt-corefud-dev.conllu')
```

## 라이선스
해당 데이터셋의 라이선스는 원본 출처를 참고하시기 바랍니다.

## 관련 리소스
- CoNLL-U 형식 상세 정보: https://universaldependencies.org/format.html
