# coref_automl/textutils.py
from __future__ import annotations
from typing import List, Dict, Any, Sequence, Tuple, Optional
import itertools
import math

from transformers import PreTrainedTokenizerBase


def truncate_around_mask(
    text: str,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    keep_for_mask: int = 8,
) -> str:
    """
    긴 문장을 토큰 기준으로 [MASK] 주변만 남겨 max_length를 만족하도록 잘라낸 후,
    special tokens를 포함해 문자열로 복원한다.

    - keep_for_mask: [MASK] 주변 최소 여유 토큰 수(컨텍스트가 거의 없을 때 대비)
    """
    mask_token = tokenizer.mask_token
    if mask_token not in text:
        return text  # [MASK]가 없으면 그대로

    # 토큰화(스페셜 토큰 제외)하여 [MASK] 위치 탐색
    enc = tokenizer(text, add_special_tokens=False)
    ids: List[int] = enc["input_ids"]
    mask_id = tokenizer.mask_token_id
    # 혹시 토큰화 중 [MASK]가 손실되지 않도록, decode 후 다시 검사할 수도 있지만,
    # 여기서는 원문에 [MASK] 있으므로 mask_id를 직접 삽입하여 정렬한다.
    # 안전하게는 다시 인코딩 시 [MASK]가 특수토큰으로 들어가므로 별도 처리:
    # → 아래에서 재인코딩 방식 사용

    # 재인코딩: [MASK]를 강제 특수토큰으로 포함
    enc2 = tokenizer(text, add_special_tokens=False, return_tensors=None)
    ids2: List[int] = enc2["input_ids"]
    # [MASK]가 subword 등으로 쪼개지지 않도록, tokenizer는 일반적으로 mask_token_id 하나로 매핑
    # 혹시 못 찾으면 전량 반환
    try:
        mpos = ids2.index(tokenizer.mask_token_id)
    except ValueError:
        return text

    # 스페셜 토큰 2개([CLS], [SEP])가 추가될 수 있으니 여유를 둔다
    budget = max(8, max_length - 2)
    left_budget = budget // 2
    right_budget = budget - left_budget

    left_ids = ids2[max(0, mpos - left_budget): mpos]
    right_ids = ids2[mpos + 1: mpos + 1 + right_budget]
    new_ids = [tokenizer.cls_token_id] + left_ids + [tokenizer.mask_token_id] + right_ids + [tokenizer.sep_token_id]
    # 문자열 복원(스페셜 토큰은 유지해야 [MASK]가 살아있음)
    s = tokenizer.decode(new_ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
    # 혹시 토크나이저가 공백/스페셜 주위 정리를 해서 [MASK]가 사라질 가능성 최소화
    if tokenizer.mask_token not in s:
        # 최후 보정: 원문에서 [MASK]가 있던 위치 근방으로 슬라이스(문자 단위)
        i = text.find(mask_token)
        half = max(0, (len(text) - len(mask_token)) // 2)
        start = max(0, i - half)
        end = min(len(text), i + len(mask_token) + half)
        return text[start:end]
    return s


def flatten_fillmask_outputs(outs: Any) -> List[Dict[str, Any]]:
    """
    fill-mask pipeline 결과가 단일 리스트 또는 [[...]](배치)로 나올 수 있다.
    양쪽 모두를 안전하게 리스트[dict] 형태로 평탄화한다.
    """
    if not isinstance(outs, list):
        return []
    if len(outs) == 0:
        return []
    if isinstance(outs[0], list):
        # 배치
        return list(itertools.chain.from_iterable(outs))
    # 단일
    return outs

