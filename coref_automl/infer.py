# coref_automl/infer.py
import torch
import numpy as np
from typing import List, Dict
from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM
from .coref_utils import mask_first_pronoun, is_noun, clean_token_str

CANDIDATE_MODELS = [
    "kakaobank/kf-deberta-base",
    "kykim/bert-kor-base",
    "google-bert/bert-base-multilingual-cased",
]


def make_fill(model_name):
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForMaskedLM.from_pretrained(model_name)
    device = 0 if torch.cuda.is_available() else -1
    return pipeline("fill-mask", model=mdl, tokenizer=tok, device=device), tok


def ensemble_fill_for_pronoun(text: str, top_k=5, models: List[str] = None) -> Dict:
    """
    문장에서 첫 대명사(NP)를 모델별 마스크 토큰으로 치환한 뒤,
    각 모델의 fill-mask 결과를 취합, '명사'만 필터링하여 앙상블 상위 후보를 반환.
    """
    models = models or CANDIDATE_MODELS
    pools = {}
    masked_map = {}

    for m in models:
        pipe, tok = make_fill(m)
        masked = mask_first_pronoun(text, mask_token=tok.mask_token)
        masked_map[m] = masked

        outs = pipe(masked, top_k=100)  # 넉넉히 뽑고
        for o in outs:
            token = clean_token_str(o["token_str"])
            if not token or not is_noun(token):
                continue
            pools.setdefault(token, []).append(float(o["score"]))

    agg = [(k, float(np.mean(v)), len(v)) for k, v in pools.items()]
    # (평균 score, 합의 모델 수)로 내림차순 정렬
    agg.sort(key=lambda x: (x[1], x[2]), reverse=True)
    cands = [{"token": k, "score": s, "models_agree": n} for k, s, n in agg[:top_k]]
    any_masked = masked_map[models[0]]
    return {"masked": any_masked, "candidates": cands}


if __name__ == "__main__":
    # 간단 실행 예시
    text = "영희는 학교에 갔다. 그는 새로운 것을 배웠다."
    print(ensemble_fill_for_pronoun(text, top_k=5))

