# coref_automl/coref_utils.py
from typing import List, Tuple
from kiwipiepy import Kiwi

KIWI = Kiwi()

# Kiwi 품사 태그 기준
PRONOUN_POS = {"NP"}        # 대명사
NOUN_POS = {"NNG", "NNP"}   # 일반명사/고유명사


def clean_token_str(s: str) -> str:
    """fill-mask의 subword(##) 제거 + 공백 정리"""
    return s.replace("##", "").strip()


def is_noun(s: str) -> bool:
    """명사 포함 여부 판정(형태소 하나라도 명사면 True)"""
    s = clean_token_str(s)
    if not s:
        return False
    anal = KIWI.analyze(s, top_n=1)
    if not anal or not anal[0] or not anal[0][0]:
        return False
    return any(t.tag in NOUN_POS for t in anal[0][0])


def mask_first_pronoun(text: str, mask_token: str = "[MASK]") -> str:
    """문장 내 첫 대명사(NP)를 mask_token으로 치환; 없으면 원문 반환"""
    toks = KIWI.tokenize(text)
    for t in toks:
        if t.tag in PRONOUN_POS:
            return text[: t.start] + mask_token + text[t.end :]
    return text


def josa_attach(noun: str, josa: str) -> str:
    """
    후보가 조사와 함께 등장해야 할 때 받침 유무에 따라 을/를, 은/는, 이/가 교정(간단 규칙).
    """
    def has_final(c):
        code = ord(c) - 0xAC00
        return (code % 28) != 0

    if not noun:
        return noun + josa

    last = noun[-1]
    padchim = has_final(last)
    mapping = {
        "을/를": "을" if padchim else "를",
        "은/는": "은" if padchim else "는",
        "이/가": "이" if padchim else "가",
    }
    return noun + mapping.get(josa, josa)

