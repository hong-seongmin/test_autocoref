# -*- coding: utf-8 -*-
"""
v4: AutoML 시도별 코어프 비교평가/집계/점수화 내장 (완성본)

핵심 기능
- (훈련) H100 최적화, 배치 자동 최대화, 데이터로더 병렬, 포지션 확장(2048)
- (평가) 공정비교(둘 다 512), 같은 모델 내 512/1024/2048 비교, 롱컨텍스트 합성 세트(가변 gap) 평가
- (AutoML) 각 trial 마다 훈련→평가→JSON/CSV 저장, 가중합 composite score로 최적 탐색
- (수집) 완료된 trial_* 폴더 일괄 스캔해 leaderboard CSV 재생성

사용 예시
1) 단일 훈련+평가:
uv run span_mlm_coref_probe.py --model klue/roberta-large --out_dir runs/x \
  --total_tokens 3e8 --schedule 0.6,0.25,0.15 --seqs 512,1024,2048 \
  --eval_every_steps 2000 --log_every_steps 50 --dataloader_workers 24 \
  --do_eval_after_train 1 --base_for_eval klue/roberta-large \
  --fair_max_tokens 512 --long_gaps 1200,1800,2400 --long_n 50

2) AutoML + 평가 + 집계:
uv run span_mlm_coref_probe.py --automl 1 --n_trials 20 \
  --model klue/roberta-large --out_dir runs/automl_spanmlm \
  --total_tokens 1e8 --do_eval_after_train 1 --base_for_eval klue/roberta-large \
  --fair_max_tokens 512 --long_gaps 1200,1800,2400 --long_n 30 \
  --score_weights 0.6,0.4

3) 훈련 없이 저장된 체크포인트만 평가:
uv run span_mlm_coref_probe.py --eval_only 1 \
  --ft_dir runs/spanmlm_kor/final --base_for_eval klue/roberta-large \
  --fair_max_tokens 512 --long_gaps 1200,1800,2400 --long_n 50

4) AutoML 결과 폴더 일괄 수집(leaderboard CSV 재생성):
uv run span_mlm_coref_probe.py --collect 1 --out_dir runs/automl_spanmlm
"""
import os, re, math, time, json, argparse, random, string, inspect, warnings, csv, glob, sys
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Subset

from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForMaskedLM, Trainer, TrainingArguments,
    PreTrainedModel, PreTrainedTokenizerBase, DataCollatorWithPadding
)
from transformers.trainer import TrainerCallback, TrainerState, TrainerControl
import optuna

# ============ 기본 최적화 플래그 ============
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
torch.set_float32_matmul_precision("high")
try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass



# ============ 전처리 유틸 ============
HANGUL_RANGE = re.compile(r"[가-힣]")
WS_MULTI = re.compile(r"\s+")
PUNCT_TR = str.maketrans({k:" " for k in string.punctuation})

def normalize_text(s: str) -> str:
    if not isinstance(s, str): return ""
    s = s.replace("\u200b", " ").replace("\t", " ").replace("\xa0", " ")
    s = WS_MULTI.sub(" ", s).strip()
    return s

def hangul_ratio(s: str) -> float:
    if not s: return 0.0
    total = len(s)
    kr = len(HANGUL_RANGE.findall(s))
    return kr / max(1, total)

def simple_dedup_key(s: str) -> str:
    s = s.lower().translate(PUNCT_TR)
    s = WS_MULTI.sub(" ", s).strip()
    if len(s) < 20: return ""
    return s

def available_cuda():
    return torch.cuda.is_available(), torch.cuda.device_count()

# ============ 데이터 소스 로더 ============
def load_wiki_texts(limit:int=None) -> List[str]:
    ds = load_dataset("wikimedia/wikipedia", "20231101.ko", split="train")
    out=[]; add=out.append
    for ex in ds:
        t = normalize_text(ex.get("text",""))
        if t: add(t)
        if limit and len(out)>=limit: break
    return out

def load_hplt_texts(limit:int=None) -> List[str]:
    ds = load_dataset("HPLT/HPLT2.0_cleaned", "kor_Hang", split="train")
    out=[]; add=out.append
    for ex in ds:
        t = normalize_text(ex.get("text",""))
        if t: add(t)
        if limit and len(out)>=limit: break
    return out

def load_novel_texts(limit:int=None) -> List[str]:
    ds = load_dataset("werty1248/Korean-1930-Novel-Scene-Summarize", split="train")
    out=[]; add=out.append
    for ex in ds:
        t = normalize_text(ex.get("text",""))
        if t: add(t)
        if limit and len(out)>=limit: break
    return out

def load_news_texts(limit:int=None) -> List[str]:
    ds = load_dataset("daekeun-ml/naver-news-summarization-ko", split="train")
    out=[]; add=out.append
    for ex in ds:
        t = normalize_text(ex.get("document",""))
        if t: add(t)
        if limit and len(out)>=limit: break
    return out

def build_corpus(
    wiki_ratio=0.4, news_ratio=0.3, hplt_ratio=0.2, novel_ratio=0.1,
    total_docs=120_000, min_kr=0.6
) -> List[str]:
    n_wiki  = int(total_docs * wiki_ratio)
    n_news  = int(total_docs * news_ratio)
    n_hplt  = int(total_docs * hplt_ratio)
    n_novel = int(total_docs * novel_ratio)
    print(f"[build_corpus] target docs: wiki={n_wiki}, news={n_news}, hplt={n_hplt}, novel={n_novel}")

    wiki  = load_wiki_texts(n_wiki)
    news  = load_news_texts(n_news)
    hplt  = load_hplt_texts(n_hplt)
    novel = load_novel_texts(n_novel)
    pool = wiki + news + hplt + novel

    seen=set(); cleaned=[]
    for t in pool:
        if len(t)<50: continue
        if hangul_ratio(t) < min_kr: continue
        k = simple_dedup_key(t)
        if not k or k in seen: continue
        seen.add(k); cleaned.append(t)
    random.shuffle(cleaned)
    print(f"[build_corpus] cleaned={len(cleaned)} (dedup from {len(pool)})")
    return cleaned

# ============ 토크나이징 & 패킹 ============
def tokenize_and_pack(texts: List[str], tokenizer: AutoTokenizer,
                      block_size:int=512, stride:int=0, keep_short=False) -> Dict[str, List[int]]:
    tokenizer.model_max_length = int(1e9)  # 경고 억제
    all_ids=[]
    for t in texts:
        ids = tokenizer.encode(t, add_special_tokens=False)
        if tokenizer.eos_token_id is not None:
            all_ids.extend(ids+[tokenizer.eos_token_id])
        else:
            all_ids.extend(ids)

    blocks = {"input_ids": [], "attention_mask": []}
    i=0
    step = (block_size - stride) if stride>0 else block_size
    while i + block_size <= len(all_ids):
        chunk = all_ids[i:i+block_size]
        blocks["input_ids"].append(chunk)
        blocks["attention_mask"].append([1]*block_size)
        i += step

    if keep_short and i < len(all_ids):
        chunk = all_ids[i:]
        if len(chunk) > 32:
            pad = block_size - len(chunk)
            blocks["input_ids"].append(chunk + [tokenizer.pad_token_id]*pad)
            blocks["attention_mask"].append([1]*len(chunk) + [0]*pad)
    print(f"[pack] total blocks={len(blocks['input_ids'])}, block_size={block_size}, stride={stride}")
    return blocks

class PackedDataset(Dataset):
    def __init__(self, arrs: Dict[str, List[List[int]]]):
        self.input_ids = arrs["input_ids"]
        self.attn = arrs["attention_mask"]
    def __len__(self): return len(self.input_ids)
    def __getitem__(self, i):
        return {"input_ids": torch.tensor(self.input_ids[i]),
                "attention_mask": torch.tensor(self.attn[i])}

# ============ Span-MLM 콜레이터 ============
# ===== Span-MLM Collator (robust) =====
@dataclass
class DataCollatorForSpanMLM(DataCollatorWithPadding):
    tokenizer: PreTrainedTokenizerBase
    mlm_probability: float = 0.15
    mean_span_length: float = 3.0
    mask_token: Optional[int] = None
    repl_upper: int = 0  # safe upper bound for random replacement vocab ids

    def __post_init__(self):
        if self.mask_token is None:
            self.mask_token = self.tokenizer.mask_token_id

    def _sample_spans(self, L: int) -> List[Tuple[int,int]]:
        n_to_mask = max(1, int(self.mlm_probability * L))
        spans=[]; covered=0
        while covered < n_to_mask:
            start = random.randrange(0, L)
            p = 1.0 / self.mean_span_length
            span_len = max(1, np.random.geometric(p))
            end = min(L, start+span_len)
            spans.append((start, end))
            covered += (end-start)
        return spans

    def _apply_span_mask(self, ids: torch.Tensor, attn: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        labels = ids.clone()
        L = ids.size(0)
        special = set(self.tokenizer.all_special_ids)
        spans = self._sample_spans(L)
        mask_positions=set()
        for s,e in spans:
            for j in range(s,e):
                if j>=L: break
                if ids[j].item() in special or attn[j].item()==0: continue
                mask_positions.add(j)
        for j in range(L):
            if j not in mask_positions:
                labels[j] = -100
            else:
                r = random.random()
                if r < 0.8:
                    ids[j] = self.mask_token
                elif r < 0.9:
                    upper = self.repl_upper or self.tokenizer.vocab_size
                    ids[j] = random.randrange(int(upper))
                else:
                    pass
        return ids, labels

    def _coerce_features(self, features):
        # dict-of-batch → list-of-dict
        if isinstance(features, dict):
            keys = list(features.keys())
            first = features[keys[0]]
            B = first.size(0) if isinstance(first, torch.Tensor) else len(first)
            out=[]
            for i in range(B):
                item={}
                bad=False
                for k, v in features.items():
                    vi = None
                    if isinstance(v, torch.Tensor):
                        vi = v[i]
                    elif isinstance(v, (list, tuple)):
                        vi = v[i] if i < len(v) else None
                    else:
                        vi = v[i] if hasattr(v, "__getitem__") else None
                    if vi is None:
                        bad=True; break
                    if not isinstance(vi, torch.Tensor):
                        vi = torch.tensor(vi)
                    item[k] = vi
                if not bad:
                    out.append(item)
            features = out

        # list로 통일
        if not isinstance(features, (list, tuple)):
            features = [features]

        cleaned=[]
        for f in features:
            if f is None:
                continue
            if isinstance(f, dict):
                ids = f.get("input_ids", None)
                attn = f.get("attention_mask", None)
            elif isinstance(f, (list, tuple)):
                ids = f[0] if len(f)>0 else None
                attn = f[1] if len(f)>1 else None
            else:
                ids, attn = f, None

            if ids is None:
                continue
            if not isinstance(ids, torch.Tensor):
                try:
                    ids = torch.tensor(ids)
                except Exception:
                    continue
            if attn is None:
                attn = torch.ones_like(ids)
            elif not isinstance(attn, torch.Tensor):
                try:
                    attn = torch.tensor(attn)
                except Exception:
                    attn = torch.ones_like(ids)
            cleaned.append({"input_ids": ids, "attention_mask": attn})
        return cleaned


    def __call__(self, features: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        feats = self._coerce_features(features)
        # 혹시라도 비정상 샘플이 섞이면 필터링
        feats = [f for f in feats if "input_ids" in f and "attention_mask" in f]
        if len(feats)==0:
            raise RuntimeError("Empty batch after coercion in DataCollatorForSpanMLM")

        input_ids = torch.stack([f["input_ids"] for f in feats])
        attention_mask = torch.stack([f["attention_mask"] for f in feats])

        out_ids=[]; labels=[]
        for i in range(input_ids.size(0)):
            ids, lab = self._apply_span_mask(input_ids[i].clone(), attention_mask[i])
            out_ids.append(ids); labels.append(lab)

        return {
            "input_ids": torch.stack(out_ids),
            "attention_mask": attention_mask,
            "labels": torch.stack(labels)
        }


# ============ RoBERTa 포지션 확장 ============
def expand_roberta_positions(model, target_seq_len: int = 2048, slack: int = 512):
    """
    target_seq_len: 평가 목표 길이(예: 2048)
    slack: 치환으로 늘어나는 여유분(기본 512)
    """
    emb = model.roberta.embeddings
    pad = getattr(emb, "padding_idx", 1)
    max_len = int(target_seq_len + slack)
    new_embed_len = int(max_len + pad + 1)  # HF RoBERTa는 pad+1 포함

    # 1) position_embeddings 확장
    old_w = emb.position_embeddings.weight.data
    old_len, H = old_w.size()
    if new_embed_len > old_len:
        new_w = old_w.new_empty((new_embed_len, H))
        new_w[:old_len] = old_w
        new_w[old_len:] = old_w[-1]
        with torch.no_grad():
            emb.position_embeddings.weight = torch.nn.Parameter(new_w)

    # 2) 버퍼 재등록 길이도 new_embed_len로 맞추기
    device = emb.position_embeddings.weight.device
    emb.register_buffer("token_type_ids",
                        torch.zeros((1, new_embed_len), dtype=torch.long, device=device),
                        persistent=False)
    emb.register_buffer("position_ids",
                        torch.arange(new_embed_len, dtype=torch.long, device=device).unsqueeze(0),
                        persistent=False)

    # 3) config도 갱신
    model.config.max_position_embeddings = new_embed_len
    return model


# ============ 코어프 프록시 유틸 ============

PRON_LIST = {"그","그녀","그것","그는","그녀는","그것은","그를","그녀를","그의","그녀의","그가","그녀가","그에게","그녀에게"}
PRON_LIST |= {"그들","그들은","그들을","그들의","그들에게"}
SENT_SPLIT = re.compile(r'([.!?。…\n]+)\s*')

def split_sentences(text:str)->List[str]:
    parts = SENT_SPLIT.split(text)
    sents=[]; buff=""
    for p in parts:
        if not p: continue
        buff += p
        if SENT_SPLIT.match(p):
            if buff.strip(): sents.append(buff.strip())
            buff=""
    if buff.strip(): sents.append(buff.strip())
    return sents if sents else [text]

def pron_spans(text:str)->List[Tuple[str, Tuple[int,int]]]:
    out=[]; i=0
    sorted_pron = sorted(PRON_LIST, key=len, reverse=True)
    while i<len(text):
        for p in sorted_pron:
            if text.startswith(p, i):
                out.append((p, (i, i+len(p))))
                i+=len(p); break
        else:
            i+=1
    return out

def _last_korean(s):
    for ch in reversed(s):
        if ord('가')<=ord(ch)<=ord('힣'): return ch
    return None
def _has_jongseong(ch):
    if ch is None: return False
    i = ord(ch)-ord('가')
    return i%28 != 0
def _fix_particle(ante:str, particle_hint:str)->str:
    if not particle_hint: return ""
    first = particle_hint[0]
    jong = _has_jongseong(_last_korean(ante))
    if first in "은는": return ("은" if jong else "는")+particle_hint[1:]
    if first in "이가": return ("이" if jong else "가")+particle_hint[1:]
    if first in "을를": return ("을" if jong else "를")+particle_hint[1:]
    return particle_hint

# 조사/스톱워드 정리
KOR_PARTICLE_TAIL = re.compile(r"(은|는|이|가|을|를|의|에|에게|에서|으로|로|과|와|도|만|뿐|께서|조차|까지|부터)$")
STOPWORDS = {"정말","아직","그리고","그러나","하지만","배경","설명","위에","환영","데이터","최종","논문","문서는"}

def _strip_particles(w: str) -> str:
    if not w: return ""
    w = WS_MULTI.sub("", w)
    # 여러 번 붙은 조사도 반복적으로 제거
    prev = None
    while prev != w:
        prev = w
        w = KOR_PARTICLE_TAIL.sub("", w)
    return w

def _norm_token(w: Optional[str]) -> str:
    if not w: return ""
    w = re.sub(r"[^가-힣A-Za-z0-9]", "", w)  # 기호 제거
    w = _strip_particles(w)
    return w


def _candidate_mentions(text:str)->List[str]:
    toks = re.findall(r"[가-힣A-Za-z0-9]{2,}", text)
    seen=set(); out=[]
    for w in toks:
        if w in PRON_LIST: 
            continue
        base = _norm_token(w)
        if not base or base in STOPWORDS:
            continue
        if base in seen:
            continue
        seen.add(base); out.append(base)
    return out


def _truncate_text_by_tokens(tokenizer, text:str, max_tokens:int)->str:
    if max_tokens is None or max_tokens<=0: return text
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens: return text
    ids = ids[:max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)

def _build_filler(tokenizer, target_tokens:int)->str:
    # 문장부호 위주로 채워서 후보 엔티티에 안 걸리게
    unit = " … …"
    s = unit
    # 토큰 길이가 target_tokens에 도달할 때까지 반복
    while len(tokenizer.encode(s, add_special_tokens=False)) < target_tokens:
        s += unit
    return s.strip()


def _insert_filler_between_head_and_tail(tokenizer, text:str, gap_tokens:int)->str:
    sents = split_sentences(text)
    pron_idx = None
    for i, s in enumerate(sents):
        if pron_spans(s):
            pron_idx = i; break
    if pron_idx is None or pron_idx==0:
        # 안전하게: 텍스트 앞에 그대로 두고 뒤에 filler 삽입
        head = " ".join(sents[:1])
        tail = " ".join(sents[1:])
    else:
        head = " ".join(sents[:pron_idx])
        tail = " ".join(sents[pron_idx:])
    filler = _build_filler(tokenizer, gap_tokens)
    return (head + " " + filler + " " + tail).strip()

# ===== Long-gap synthetic set for robustness =====
_PERSONS = ["민수","철수","지수","영희","수진","하람","현우","세준","영준","도윤","서준","하준"]
_OBJECTS = ["노트북","문서","시제품","보고서","프로토타입","초안","수정본","장비"]
_PRON_PAIRS = [
    ("그는", "PER"), ("그녀는", "PER"), ("그것은", "OBJ"),
]

def _repeat_to_tokens(tokenizer, target_tokens:int, unit:str=" 그리고 테스트 문장"):
    """토크나이저 기준으로 target_tokens에 최대한 맞춰 채우는 더미 텍스트 생성."""
    s = ""
    while True:
        ids = tokenizer(s, add_special_tokens=False)["input_ids"]
        if len(ids) >= target_tokens: break
        s += unit
    return s

def build_long_gap_set(tokenizer, gaps: List[int], n_per_gap:int=30, seed:int=123) -> List[Dict[str, Any]]:
    """
    gaps: 토큰 길이 기준 갭(선행지시어와 대명사 사이 토큰 수)
    n_per_gap: 갭별 샘플 개수
    """
    rng = random.Random(seed)
    exs=[]
    for g in gaps:
        for _ in range(n_per_gap):
            prn, kind = rng.choice(_PRON_PAIRS)
            if kind=="PER":
                ante = rng.choice(_PERSONS)
                # 간단한 인칭 문장 템플릿 (조사 자동 교정은 proxy가 해줌)
                head = f"{ante}은 회의에 참석했다."
                tail = f"{prn} 발표를 맡았다."
            else:
                ante = rng.choice(_OBJECTS)
                head = f"{ante}이(가) 창고에 도착했다."
                tail = f"{prn} 아직 검수 중이다."

            filler = _repeat_to_tokens(tokenizer, g)
            text = f"{head} {filler} {tail}"

            ex = {"text": text, "expect": {prn: ante}}
            exs.append(ex)
    return exs

def eval_all_checkpoints(out_dir:str, base_for_eval:str, fair_max_tokens:int, long_gaps:str, long_n:int):
    os.makedirs(out_dir, exist_ok=True)
    gaps = [int(x) for x in long_gaps.split(",") if x.strip()]
    rows=[]
    header=["trial","ckpt_name","global_step","fair_acc","long_avg_acc","composite","path"]

    print(f"[eval_ckpt_all] scanning {out_dir}"); sys.stdout.flush()
    for tdir in sorted(glob.glob(os.path.join(out_dir, "trial_*"))):
        trial_id = int(os.path.basename(tdir).split("_")[-1])
        print(f"[eval_ckpt_all] trial={trial_id} dir={tdir}"); sys.stdout.flush()
        ckpt_dirs=[]
        final_dir = os.path.join(tdir, "final")
        if os.path.exists(final_dir):
            ckpt_dirs.append(("final", final_dir, None))
        for c in sorted(glob.glob(os.path.join(tdir, "checkpoint-*"))):
            name = os.path.basename(c)
            step = None
            try:
                step = int(name.split("-")[-1])
            except Exception:
                pass
            ckpt_dirs.append((name, c, step))

        for name, ft, step in ckpt_dirs:
            print(f"[eval] start name={name} step={step} path={ft}")
            sys.stdout.flush()
            rep = eval_coref_compare(
                base_model_name=base_for_eval,
                ft_dir=ft,
                device="cuda" if torch.cuda.is_available() else "cpu",
                fair_max_tokens=fair_max_tokens,
                long_gaps=gaps,
                long_n=long_n
            )
            print(f"[eval] done  name={name} step={step}"); sys.stdout.flush()
            save_json(os.path.join(ft, "eval_coref.json"), rep)
            composite, fair, long_avg = compute_composite_score(rep)  # acc 기반
            rows.append([trial_id, name, step if step is not None else "", fair, long_avg, composite, ft])

    # CSV 저장
    csv_path = os.path.join(out_dir, "automl_ckpt_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"[collect] checkpoint leaderboard saved: {csv_path}")


def eval_all_trials(out_dir:str, base_for_eval:str, fair_max_tokens:int, long_gaps:str, long_n:int):
    gaps = [int(x) for x in long_gaps.split(",") if x.strip()]
    for tdir in sorted(glob.glob(os.path.join(out_dir, "trial_*"))):
        ft = os.path.join(tdir, "final")
        if not os.path.exists(ft): 
            continue
        rep = eval_coref_compare(
            base_model_name=base_for_eval,
            ft_dir=ft,
            device="cuda" if torch.cuda.is_available() else "cpu",
            fair_max_tokens=fair_max_tokens,
            long_gaps=gaps,
            long_n=long_n
        )
        save_json(os.path.join(tdir, "eval_coref.json"), rep)
    collect_automl_results(out_dir)


def evaluate_coref_suite(model, tokenizer,
                         long_gaps:List[int]=(600,1200,1800,2400),
                         long_n:int=30) -> Dict[str, Any]:
    """
    기본 짧은 세트(COREf_PROXY_SET) + 롱갭 합성세트까지 일괄 평가.
    반환: {'short':{acc,f1,n}, 'long':{gap: {acc,f1,n}}, 'long_avg': {...}}
    """
    out = {}
    # Short set
    short = coref_proxy_score(model, tokenizer, COREf_PROXY_SET)
    out["short"] = {"acc": short["acc"], "f1": short["f1"], "n": sum(len(x["expect"]) for x in COREf_PROXY_SET)}

    # Long-gap sets (사전 형태를 반환하는 build_long_gap_set에 맞춰 호출)
    long_sets = build_long_gap_set(tokenizer, COREf_PROXY_SET, gaps=long_gaps, n=long_n)
    lg = {}
    for g, exs in long_sets.items():
        m = coref_proxy_score(model, tokenizer, exs, max_tokens=2048)
        lg[str(g)] = {"acc": m["acc"], "f1": m["f1"], "n": sum(len(x["expect"]) for x in exs)}
    out["long"] = lg

    # 평균 롱 점수
    if lg:
        acc_avg = sum(v["acc"] for v in lg.values()) / len(lg)
        f1_avg  = sum(v["f1"]  for v in lg.values()) / len(lg)
        out["long_avg"] = {"acc": acc_avg, "f1": f1_avg}
    return out


def save_trial_metrics(base_dir:str, trial_num:int, params:Dict[str,Any], metrics:Dict[str,Any], model_dir:str, error:str=None):
    os.makedirs(base_dir, exist_ok=True)
    row = {
        "trial": trial_num,
        "params": params,
        "metrics": metrics,
        "model_dir": model_dir,
        "error": error
    }
    with open(os.path.join(base_dir, "automl_trials.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False)+"\n")

    # 베스트(단기/장기) 갱신
    def _update_best(key:str, score:float, file:str):
        path = os.path.join(base_dir, file)
        best = None
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    best = json.load(f)
            except Exception:
                best = None
        if (best is None) or (score > best.get("best_score",-1)):
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"best_score":score, "trial":trial_num, "params":params, "model_dir":model_dir, "metrics":metrics}, f, ensure_ascii=False, indent=2)

    _update_best("short_f1", metrics["short"]["f1"], "best_by_short_f1.json")
    if "long_avg" in metrics:
        _update_best("long_avg_f1", metrics["long_avg"]["f1"], "best_by_longavg_f1.json")


@torch.no_grad()
def _mask_logprob_sum(model, tokenizer, text, start, end, replacement):
    enc = tokenizer(text, return_offsets_mapping=True, return_tensors="pt", truncation=False)
    offs = enc["offset_mapping"][0].tolist()
    ids  = enc["input_ids"][0].tolist()
    attn = enc["attention_mask"][0].tolist()

    mask_idxs = [i for i,(s,e) in enumerate(offs) if not (s==0 and e==0) and s<end and e>start]
    if not mask_idxs: return -1e9

    cand_ids = tokenizer(replacement, add_special_tokens=False)["input_ids"]
    if not cand_ids: return -1e9

    l, r = min(mask_idxs), max(mask_idxs)
    mask_id = tokenizer.mask_token_id
    new_ids = ids[:l] + [mask_id]*len(cand_ids) + ids[r+1:]
    new_att = attn[:l] + [1]*len(cand_ids) + attn[r+1:]

    # ======= 안전 크롭(마스크 구간 포함) =======
    try:
        pe = getattr(getattr(model, "roberta", model).embeddings, "position_ids")
        max_len = pe.size(1)  # 여유 포함 확장된 길이
    except Exception:
        max_len = getattr(model.config, "max_position_embeddings", 4096)

    if len(new_ids) > max_len:
        need = max_len
        # 마스크 시작이 창 중앙쯤 오도록 좌측 오프셋 계산
        left = max(0, min(l - (need // 2), len(new_ids) - need))
        right = left + need
        l_shift = l - left
        new_ids = new_ids[left:right]
        new_att = new_att[left:right]
        pos = list(range(l_shift, l_shift + len(cand_ids)))
    else:
        pos = list(range(l, l + len(cand_ids)))
    # =========================================

    logits = model(torch.tensor([new_ids], device=model.device),
                   attention_mask=torch.tensor([new_att], device=model.device)).logits
    logprobs = torch.log_softmax(logits[0, pos, :], dim=-1)
    cand = torch.tensor(cand_ids, device=model.device).unsqueeze(-1)
    tok_lp = torch.gather(logprobs, 1, cand).squeeze(-1)
    return float(tok_lp.sum().item())


PRON_KIND = {
    "그는":"PER","그녀는":"PER","그가":"PER","그녀가":"PER","그의":"PER","그녀의":"PER",
    "그들은":"GRP","그들":"GRP",
    "그것":"OBJ","그것은":"OBJ"
}
PRON_KIND.update({
    "그들":"GRP","그들은":"GRP","그들을":"GRP","그들의":"GRP","그들에게":"GRP"
})
PERSON_TITLES = {}
OBJECT_LEX = {}
GROUP_LEX = {}

def _is_personish(w:str)->bool:
    if w in PERSON_TITLES: return True
    return bool(re.fullmatch(r"[가-힣]{2,4}", w)) and (w not in OBJECT_LEX) and (w not in STOPWORDS)

def _is_objectish(w:str)->bool:
    return (w in OBJECT_LEX)

def _is_groupish(w:str)->bool:
    return (w in GROUP_LEX) or w.endswith("들")

def filter_by_pronoun_kind(pron:str, cands:List[str])->List[str]:
    kind = PRON_KIND.get(pron,"")
    if kind=="PER":
        f = [w for w in cands if _is_personish(w)]
        return f if f else cands
    if kind=="OBJ":
        f = [w for w in cands if _is_objectish(w)]
        return f if f else cands
    if kind=="GRP":
        f = [w for w in cands if _is_groupish(w) or _is_personish(w)]
        return f if f else cands
    return cands


def coref_proxy_predict(model, tokenizer, text:str, max_tokens:int=None)->Dict[str,str]:
    # 전체 텍스트를 한 번만 자르고(토큰 기준), 그 위에서 전역 위치로 평가
    full_text = _truncate_text_by_tokens(tokenizer, text, max_tokens) if max_tokens else text
    preds={}
    sents = split_sentences(full_text)
    global_seen=[]
    cursor = 0  # 전역 오프셋(문자 인덱스)

    for sent in sents:
        ps = pron_spans(sent)
        for p, (s,e) in ps:
            # 현재 문장 내 후보 + 이전 문장 누적 후보(=거리 반영을 위해 '전역 문맥'을 사용)
            pre_cands = _candidate_mentions(sent[:s])
            cand_pool = []
            for c in global_seen + pre_cands:
                if c not in cand_pool:
                    cand_pool.append(c)
            cand_pool = filter_by_pronoun_kind(p, cand_pool)

            # 조사 보정
            tail = ""
            if e < len(sent):
                m = re.match(r"[은는이가을를에게]*", sent[e:e+2])
                tail = m.group(0) if m else ""

            # 전역 문자 인덱스로 변환
            g_start = cursor + s
            g_end   = cursor + e

            best=None; best_lp=-1e9
            for cand in cand_pool:
                rep = cand + _fix_particle(cand, tail)
                # 핵심: 'sent'가 아니라 'full_text'를 넣어서 전체 문맥으로 MLM 점수 평가
                lp = _mask_logprob_sum(model, tokenizer, full_text, g_start, g_end, rep)
                if lp > best_lp:
                    best_lp = lp
                    best = (p, cand)
            if best:
                preds[best[0]] = best[1]

        # 문장 처리 후 전역 후보 갱신
        for m in _candidate_mentions(sent):
            if m not in global_seen:
                global_seen.append(m)
        cursor += len(sent)  # 다음 문장 시작 전역 오프셋로 이동

    return preds



def coref_proxy_score(model, tokenizer, eval_set:List[Dict[str,Any]], max_tokens:int=None)->Dict[str,Any]:
    details=[]; correct=0; total=0
    for ex in eval_set:
        pred = coref_proxy_predict(model, tokenizer, ex["text"], max_tokens=max_tokens)
        for prn, gold in ex["expect"].items():
            total += 1
            got_raw = pred.get(prn, None)
            got = _norm_token(got_raw) if got_raw is not None else None

            # gold을 리스트로 정규화
            gold_list = gold if isinstance(gold, list) else [gold]
            gold_norm = {_norm_token(g) for g in gold_list}

            ok = (got in gold_norm) if got is not None else False
            if ok: correct += 1
            details.append({"pronoun":prn,"gold":gold,"pred":got_raw,"ok":bool(ok)})
    acc = correct / max(1,total)
    f1 = acc
    return {"acc":acc, "f1":f1, "n": total, "details": details}


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

# ============ 평가 세트 생성(롱컨텍스트) ============
def build_long_gap_set(tokenizer, base_set:List[Dict[str,Any]], gaps:List[int], n:int=30, seed:int=42):
    rnd = random.Random(seed)
    # pronoun가 존재하는 샘플만 사용
    candidates = [ex for ex in base_set if len(ex["expect"])>0]
    out = {}
    for g in gaps:
        chosen = [candidates[i % len(candidates)] for i in range(n)]  # 순환 샘플링
        items=[]
        for ex in chosen:
            new_text = _insert_filler_between_head_and_tail(tokenizer, ex["text"], g)
            items.append({"text": new_text, "expect": ex["expect"]})
        out[g] = items
    return out  # {gap: [examples...]}

# ============ 평가 리포트 ============
def eval_coref_compare(base_model_name:str, ft_dir:str, device:str="cuda",
                       fair_max_tokens:int=512, long_gaps:List[int]=[1200,1800,2400],
                       long_n:int=30):
    # 로딩
    base_tok = AutoTokenizer.from_pretrained(base_model_name, use_fast=True)
    base = AutoModelForMaskedLM.from_pretrained(base_model_name).to(device).eval()
    ft_tok = AutoTokenizer.from_pretrained(ft_dir, use_fast=True)
    ft = AutoModelForMaskedLM.from_pretrained(ft_dir).to(device).eval()


    # ★ 평가 안정화: 토크나이저/포지션 확장
    ft_tok.model_max_length = int(1e9)  # 오프셋/인코딩 시 불필요한 트렁케이션 방지
    ft = expand_roberta_positions(ft, target_seq_len=2048, slack=512)
    # 우리가 아래에서 1024/2048까지 평가하므로 필요 길이 = 2048
    need_len = 2048
    try:
        if getattr(ft.config, "max_position_embeddings", 0) < (need_len + 2):
            ft = expand_roberta_positions(ft, target_seq_len=2048, slack=128)
    except Exception as e:
        print(f"[warn] position expand failed: {e}")

    # 공정 비교(둘 다 max_tokens=512)
    fair = {
        "base@512": coref_proxy_score(base, base_tok, COREf_PROXY_SET, max_tokens=fair_max_tokens) | {"name":"base@512"},
        "ft@512":   coref_proxy_score(ft, ft_tok, COREf_PROXY_SET,   max_tokens=fair_max_tokens) | {"name":"ft@512"},
    }

    # 같은 모델 내 512 vs 확장 길이 비교(퇴행 체크)
    same_model_512_vs_1024 = {
        "ft@512":  coref_proxy_score(ft, ft_tok, COREf_PROXY_SET, max_tokens=512)  | {"name":"ft@512"},
        "ft@1024": coref_proxy_score(ft, ft_tok, COREf_PROXY_SET, max_tokens=1024) | {"name":"ft@1024"},
    }
    same_model_512_vs_2048 = {
        "ft@512":  coref_proxy_score(ft, ft_tok, COREf_PROXY_SET, max_tokens=512)  | {"name":"ft@512"},
        "ft@2048": coref_proxy_score(ft, ft_tok, COREf_PROXY_SET, max_tokens=2048) | {"name":"ft@2048"},
    }

    # 롱컨텍스트 합성 세트
    long_sets = build_long_gap_set(ft_tok, COREf_PROXY_SET, gaps=long_gaps, n=long_n)
    long_results = {}
    for g, exs in long_sets.items():
        # 롱 세트는 확장 길이로만 평가(최대 2048)
        long_results[g] = coref_proxy_score(ft, ft_tok, exs, max_tokens=2048) | {"gap":g, "name":f"ft@2048_gap{g}"}

    report = {
        "fair_compare": fair,
        "same_model_compare_512_vs_1024": same_model_512_vs_1024,
        "same_model_compare_512_vs_2048": same_model_512_vs_2048,
        "long_context_synth": long_results,
    }
    return report

def print_eval_report(report:Dict[str,Any]):
    print("\n== 공정 비교 (둘 다 max_tokens=512) ==")
    print(json.dumps(report["fair_compare"], ensure_ascii=False, indent=2))
    print("\n== 같은 모델 내 512 vs 확장 길이 비교 (롱컨텍스트 효과) ==")
    print(json.dumps(report["same_model_compare_512_vs_1024"], ensure_ascii=False, indent=2))
    print(json.dumps(report["same_model_compare_512_vs_2048"], ensure_ascii=False, indent=2))
    print("\n== 롱컨텍스트 합성 세트 (긴 갭) ==")
    print(json.dumps(report["long_context_synth"], ensure_ascii=False, indent=2))

# ============ Trainer 확장 ============
class CorefProbeTrainer(Trainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix: str = "eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        ckpt_metric = coref_proxy_score(self.model, self.tokenizer, COREf_PROXY_SET)
        metrics[f"{metric_key_prefix}_coref_acc"] = ckpt_metric["acc"]
        metrics[f"{metric_key_prefix}_coref_f1"]  = ckpt_metric["f1"]
        self.log(metrics)
        return metrics

class ETACallback(TrainerCallback):
    def __init__(self): self.start=None; self.last_step=0; self.last_time=None
    def on_train_begin(self, args, state:TrainerState, control:TrainerControl, **kwargs):
        self.start=time.time(); self.last_time=self.start; self.last_step=0
    def on_step_end(self, args, state:TrainerState, control:TrainerControl, **kwargs):
        now=time.time(); steps_done=state.global_step
        steps_total = state.max_steps if state.max_steps>0 else (state.num_train_epochs*state.num_steps_per_epoch)
        dt = now-self.last_time; ds = steps_done-self.last_step
        if ds>0 and dt>0:
            spd = ds/dt
            remain = (steps_total-steps_done)/max(1e-6, spd)
            hrs=int(remain//3600); mins=int((remain%3600)//60)
            print(f"[ETA] step {steps_done}/{steps_total} ~ {hrs}h {mins}m remaining (speed {spd:.2f} step/s)")
        self.last_time=now; self.last_step=steps_done

# ============ TrainingArguments 호환 ============
def make_training_args(**kwargs):
    sig = inspect.signature(TrainingArguments.__init__)
    params = sig.parameters
    filtered = {k:v for k,v in kwargs.items() if k in params}

    if "bf16" in params and "bf16" in kwargs:
        filtered["bf16"] = kwargs["bf16"]
    if "fp16" in params and "fp16" in kwargs:
        filtered["fp16"] = kwargs["fp16"]
    if "optim" in params and "optim" in kwargs:
        filtered["optim"] = kwargs["optim"]
    if "dataloader_num_workers" in params and "dataloader_num_workers" in kwargs:
        filtered["dataloader_num_workers"] = kwargs["dataloader_num_workers"]
    if "dataloader_pin_memory" in params and "dataloader_pin_memory" in kwargs:
        filtered["dataloader_pin_memory"] = kwargs["dataloader_pin_memory"]
    if "dataloader_persistent_workers" in params and "dataloader_persistent_workers" in kwargs:
        filtered["dataloader_persistent_workers"] = kwargs["dataloader_persistent_workers"]
    if "dataloader_prefetch_factor" in params and "dataloader_prefetch_factor" in kwargs:
        filtered["dataloader_prefetch_factor"] = kwargs["dataloader_prefetch_factor"]

    if "evaluation_strategy" in params and "evaluation_strategy" in kwargs:
        filtered["evaluation_strategy"] = kwargs["evaluation_strategy"]
        if kwargs.get("evaluation_strategy") == "steps" and "eval_steps" in params and "eval_steps" in kwargs:
            filtered["eval_steps"] = kwargs["eval_steps"]
    else:
        if "evaluate_during_training" in params:
            filtered["evaluate_during_training"] = True
            if "eval_steps" in params and "eval_steps" in kwargs:
                filtered["eval_steps"] = kwargs["eval_steps"]

    return TrainingArguments(**filtered)

# ============ 학습 시도(한 구간) ============
def try_train_once(model, tokenizer, train_ds, eval_ds, collator, out_dir, block_size,
                   eval_every_steps, log_every_steps, lr, weight_decay, warmup_ratio,
                   per_device_bs, grad_accum, fp16, bf16, optim_name,
                   dataloader_workers, compile_flag, seed):
    os.makedirs(out_dir, exist_ok=True)

    try:
        model.config._attn_implementation = "sdpa"
    except Exception:
        pass

    if compile_flag:
        try:
            model = torch.compile(model, mode="default", backend="inductor")
        except Exception as e:
            print(f"[warn] torch.compile 실패: {e}")

    args = make_training_args(
        output_dir=out_dir,
        overwrite_output_dir=True,
        num_train_epochs=1.0,
        per_device_train_batch_size=per_device_bs,
        per_device_eval_batch_size=max(1, per_device_bs),
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_ratio=warmup_ratio,
        logging_steps=log_every_steps,
        evaluation_strategy="steps",
        eval_steps=eval_every_steps,
        save_steps=eval_every_steps,
        save_total_limit=3,
        dataloader_num_workers=dataloader_workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=True,
        dataloader_drop_last=True, 
        fp16=bool(fp16 and not bf16),
        bf16=bool(bf16),
        optim=optim_name,
        report_to=[],
        lr_scheduler_type="cosine",
        seed=seed
    )

    trainer = CorefProbeTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        callbacks=[ETACallback()]
    )
    print(f"[train] block={block_size}, bs={per_device_bs}, ga={grad_accum}, "
          f"fp16={bool(fp16 and not bf16)}, bf16={bool(bf16)}, optim={optim_name}, "
          f"workers={dataloader_workers}")
    trainer.train()
    trainer.evaluate()
    return trainer

# ============ 메인 훈련 루프 ============
def run_train(
    model_name:str,
    out_dir:str,
    total_tokens: int = int(3e8),
    schedule: List[float] = [0.6,0.25,0.15],
    seqs: List[int] = [512,1024,2048],
    eval_every_steps: int = 2000,
    log_every_steps: int = 50,
    lr: float = 5e-5,
    weight_decay: float = 0.01,
    mask_prob: float = 0.15,
    mean_span: float = 3.0,
    start_bs: int = 256,
    grad_accum: int = 32,
    warmup_ratio: float = 0.01,
    fp16: bool = True,
    seed:int=42,
    dataloader_workers:int=8,
    eval_subset_blocks:int=4096,
    compile_flag:int=0
):
    assert len(schedule)==len(seqs), "schedule과 seqs 길이가 같아야 합니다"
    assert abs(sum(schedule)-1.0)<1e-6, "schedule 합이 1이어야 합니다"
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.sep_token or "[PAD]"
    tokenizer.model_max_length = int(1e9)

    corpus = build_corpus(total_docs=120_000)

    model = AutoModelForMaskedLM.from_pretrained(model_name)

    bf16_possible = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    bf16_use = bf16_possible
    fp16_use = fp16 and not bf16_use

    gc_enabled = False
    if hasattr(model, "gradient_checkpointing_enable"):
        if gc_enabled: model.gradient_checkpointing_enable()
        else:
            model.gradient_checkpointing_disable = getattr(model, "gradient_checkpointing_disable", lambda: None)
            model.gradient_checkpointing_disable()

    has_cuda, ndev = available_cuda()
    print(f"[device] cuda={has_cuda}, ndev={ndev}, bf16={bf16_use}, fp16={fp16_use}")

    optim_name = "adamw_torch_fused" if has_cuda else "adamw_torch"

    for frac, block_size in zip(schedule, seqs):
        if block_size > model.config.max_position_embeddings:
            print(f"[pos-expand] expanding to {block_size}")
            model = expand_roberta_positions(model, target_seq_len=block_size)

        target_tokens = int(total_tokens * frac)
        stride = 128 if block_size<=1024 else 256

        packed = tokenize_and_pack(corpus, tokenizer, block_size=block_size, stride=stride, keep_short=False)
        tokens_total = len(packed["input_ids"]) * block_size
        if tokens_total > target_tokens:
            need_blocks = max(1, target_tokens // block_size)
            idx = list(range(len(packed["input_ids"]))); random.shuffle(idx); idx = idx[:need_blocks]
            packed = {
                "input_ids":[packed["input_ids"][i] for i in idx],
                "attention_mask":[packed["attention_mask"][i] for i in idx],
            }
        train_ds = PackedDataset(packed)

        eval_ds = train_ds
        if eval_subset_blocks>0 and len(train_ds) > eval_subset_blocks:
            eval_idx = list(range(eval_subset_blocks))
            eval_ds = Subset(train_ds, eval_idx)

        collator = DataCollatorForSpanMLM(
            tokenizer=tokenizer, mlm_probability=mask_prob, mean_span_length=mean_span
        )
        collator.repl_upper = model.get_input_embeddings().weight.size(0)

        bs = max(1, start_bs)
        eff_tokens_per_update = bs * max(1, ndev) * grad_accum
        ga = max(1, eff_tokens_per_update // (bs * max(1, ndev)))
        if ga < 1: ga = 1

        while True:
            try:
                step_out_dir = os.path.join(out_dir, f"blk{block_size}_bs{bs}_ga{ga}_gc{int(gc_enabled)}")
                trainer = try_train_once(
                    model, tokenizer, train_ds, eval_ds, collator, step_out_dir, block_size,
                    eval_every_steps, log_every_steps, lr, weight_decay, warmup_ratio,
                    bs, ga, fp16_use, bf16_use, optim_name,
                    dataloader_workers, compile_flag, seed
                )
                model = trainer.model
                break
            except RuntimeError as e:
                msg = str(e).lower()
                oom = ("out of memory" in msg) or ("cuda error" in msg) or ("cublas" in msg)
                if oom:
                    torch.cuda.empty_cache()
                    if not gc_enabled and hasattr(model, "gradient_checkpointing_enable"):
                        print("[OOM] gradient checkpointing ON으로 전환 후 재시도")
                        model.gradient_checkpointing_enable()
                        gc_enabled = True
                        continue
                    new_bs = bs//2
                    if new_bs < 1:
                        raise RuntimeError("메모리 부족: batch size=1, gc=ON에서도 불가") from e
                    scale = max(1, bs//new_bs)
                    ga = max(ga * scale, ga+1)
                    bs = new_bs
                    print(f"[OOM] bs 낮춤 → bs={bs}, ga={ga}")
                    continue
                else:
                    raise e

    final_dir = os.path.join(out_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir); tokenizer.save_pretrained(final_dir)
    print(f"[done] saved to {final_dir}")
    return final_dir

# ============ 저장/집계 유틸 ============
def _ensure_dir(p): os.makedirs(p, exist_ok=True)

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def append_row_csv(path, header:list, row:list):
    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if not exists: w.writerow(header)
        w.writerow(row)

def compute_composite_score(report, weights=(0.6,0.4)):
    fair = report["fair_compare"]["ft@512"]["acc"]
    longs = [v["acc"] for v in report["long_context_synth"].values()]
    long_avg = sum(longs) / max(1, len(longs))
    return weights[0]*fair + weights[1]*long_avg, fair, long_avg

def make_study(study_name, storage, direction="maximize"):
    if storage:
        return optuna.create_study(
            study_name=study_name,
            storage=storage,
            direction=direction,
            load_if_exists=True  # 핵심: 이어하기
        )
    else:
        # 기존 메모리 모드(이어하기 불가)
        return optuna.create_study(direction=direction)

def trial_already_done(tmp_dir):
    final_dir = os.path.join(tmp_dir, "final")
    rep_path = os.path.join(tmp_dir, "eval_coref.json")
    return os.path.exists(final_dir) and os.path.exists(rep_path)

def load_cached_target(rep_path, score_weights=(0.7,0.3)):
    with open(rep_path, "r", encoding="utf-8") as f:
        rep = json.load(f)
    if "short" in rep and "long_avg" in rep:
        return score_weights[0]*rep["short"]["f1"] + score_weights[1]*rep["long_avg"]["f1"], rep
    else:
        fair = rep["fair_compare"]["ft@512"]["f1"]
        longs = [v["f1"] for v in rep["long_context_synth"].values()]
        long_avg = sum(longs)/max(1,len(longs))
        return score_weights[0]*fair + score_weights[1]*long_avg, rep


# ===== AutoML =====
def automl_search(
    model_name:str,
    out_dir:str,
    n_trials:int=20,
    total_tokens:int=int(1e8),
    seed:int=42,
    eval_long_gaps:List[int]=(600,1200,1800),
    eval_long_n:int=30,
    study_name:Optional[str]=None,
    storage:Optional[str]=None,
    skip_if_done:int=1,
    backfill_from_fs:int=0,
    score_weights=(0.6,0.4)
):
    # 커리큘럼(문자 라벨 -> 시퀀스/스케줄 매핑)
    CURRIC = {
        "512-1024-2048": {"seqs":[512,1024,2048], "sched":[0.6,0.25,0.15]},
        "512-2048":      {"seqs":[512,2048],      "sched":[0.7,0.3]},
        "1024-2048":     {"seqs":[1024,2048],     "sched":[0.5,0.5]},
    }

    def objective(trial: optuna.Trial):
        # 하이퍼파라미터 샘플링
        lr = trial.suggest_float("lr", 4e-5, 6e-5, log=True)
        wd = trial.suggest_float("weight_decay", 0.02, 0.08)
        mask_prob = trial.suggest_float("mask_prob", 0.17, 0.20)
        mean_span = trial.suggest_float("mean_span", 3.0, 4.8)
        start_bs = trial.suggest_categorical("start_bs", [256])
        grad_accum = trial.suggest_categorical("grad_accum", [32])
        dataloader_workers = trial.suggest_categorical("dataloader_workers", [12,24])
        compile_flag = trial.suggest_categorical("compile_flag", [0])
        curriculum_key = trial.suggest_categorical("curriculum", ["1024-2048"])
        seqs = CURRIC[curriculum_key]["seqs"]
        sched = CURRIC[curriculum_key]["sched"]

        tmp_dir = os.path.join(out_dir, f"trial_{trial.number}")
        os.makedirs(tmp_dir, exist_ok=True)

        params_rec = {
            "lr": lr, "weight_decay": wd, "mask_prob": mask_prob, "mean_span": mean_span,
            "start_bs": start_bs, "grad_accum": grad_accum, "dataloader_workers": dataloader_workers,
            "compile_flag": compile_flag, "curriculum": curriculum_key, "seqs": seqs, "sched": sched,
            "total_tokens": int(total_tokens), "seed": seed
        }

        # 이미 완료된 trial이면 캐시 점수 사용(이어달리기)
        if skip_if_done and trial_already_done(tmp_dir):
            # 저장된 eval_coref.json에서 목표 스코어를 weights로 재계산
            target, metrics = load_cached_target(os.path.join(tmp_dir, "eval_coref.json"),
                                                 score_weights)
            save_trial_metrics(out_dir, trial.number, params_rec, metrics, model_dir=os.path.join(tmp_dir,"final"))
            return float(target)

        try:
            # 학습 실행 (커리큘럼 반영)
            run_train(
                model_name=model_name, out_dir=tmp_dir,
                total_tokens=int(total_tokens), schedule=sched, seqs=seqs,
                eval_every_steps=1000, log_every_steps=100,
                lr=lr, weight_decay=wd, mask_prob=mask_prob, mean_span=mean_span,
                start_bs=start_bs, grad_accum=grad_accum, fp16=True, seed=seed,
                dataloader_workers=dataloader_workers, eval_subset_blocks=2048,
                compile_flag=compile_flag
            )

            # 최종 체크포인트 로드 & 평가
            model = AutoModelForMaskedLM.from_pretrained(os.path.join(tmp_dir, "final"))
            tok = AutoTokenizer.from_pretrained(os.path.join(tmp_dir, "final"))
            metrics = evaluate_coref_suite(model, tok, long_gaps=list(eval_long_gaps), long_n=eval_long_n)
            # ★ 각 트라이얼 폴더에 eval_coref.json 저장 (resume/skip용)
            save_json(os.path.join(tmp_dir, "eval_coref.json"), metrics)
            # ★ 가중치 반영해 목적함수 계산
            target = score_weights[0]*metrics["short"]["f1"] + score_weights[1]*metrics["long_avg"]["f1"]
            save_trial_metrics(out_dir, trial.number, params_rec, metrics, model_dir=os.path.join(tmp_dir, "final"))
            return float(target)

        except RuntimeError as e:
            torch.cuda.empty_cache()
            err = str(e)
            save_trial_metrics(out_dir, trial.number, params_rec, metrics={"short":{"f1":0,"acc":0,"n":0}}, model_dir="", error=err)
            return 0.0
        except Exception as e:
            err = str(e)
            save_trial_metrics(out_dir, trial.number, params_rec, metrics={"short":{"f1":0,"acc":0,"n":0}}, model_dir="", error=err)
            return 0.0

    # Optuna 스터디(이어달리기: load_if_exists=True)
    study = make_study(
        study_name=(os.path.basename(out_dir) if study_name is None else study_name),
        storage=storage,
        direction="maximize"
    )
    study.optimize(objective, n_trials=n_trials)

    # 최종 요약 저장
    print("[AutoML] best_score:", study.best_value)
    print("[AutoML] best_params:", study.best_trial.params)
    with open(os.path.join(out_dir, "automl_best.json"), "w", encoding="utf-8") as f:
        json.dump({"best_params":study.best_trial.params, "best_score":study.best_value}, f, ensure_ascii=False, indent=2)



# ============ Leaderboard 재수집 ============
def collect_automl_results(out_dir:str, score_weights=(0.6,0.4)):
    os.makedirs(out_dir, exist_ok=True)
    leaderboard_csv = os.path.join(out_dir, "automl_summary.csv")

    # 1) automl_trials.jsonl 읽어 trial→마지막 파라미터 매핑
    trial_params = {}
    trials_log = os.path.join(out_dir, "automl_trials.jsonl")
    if os.path.exists(trials_log):
        with open(trials_log, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except Exception:
                    continue
                t = int(row.get("trial"))
                params = row.get("params", {}) or {}
                trial_params[t] = params  # 같은 trial이면 마지막 항목으로 덮어씀
    # hparams.json이 있으면 그걸 우선 사용(가독성↑)
    for tdir in glob.glob(os.path.join(out_dir, "trial_*")):
        try:
            tid = int(os.path.basename(tdir).split("_")[-1])
        except Exception:
            continue
        hp_path = os.path.join(tdir, "hparams.json")
        if os.path.exists(hp_path):
            try:
                with open(hp_path, "r", encoding="utf-8") as f:
                    trial_params[tid] = json.load(f)
            except Exception:
                pass

    # 2) 각 trial의 eval_coref.json 읽어 점수 집계
    header = [
        "trial","lr","weight_decay","mask_prob","mean_span",
        "start_bs","grad_accum","dataloader_workers","compile_flag",
        "curriculum","seqs","schedule","total_tokens","seed",
        "ft512_acc","long_avg_acc","composite_score"
    ]
    rows = []
    for tdir in sorted(glob.glob(os.path.join(out_dir, "trial_*"))):
        try:
            trial_id = int(os.path.basename(tdir).split("_")[-1])
        except Exception:
            continue
        eval_path = os.path.join(tdir, "eval_coref.json")
        if not os.path.exists(eval_path):
            continue

        with open(eval_path, "r", encoding="utf-8") as f2:
            report = json.load(f2)
        composite, fair, long_avg = compute_composite_score(report, weights=score_weights)

        p = trial_params.get(trial_id, {})
        # 리스트/튜플 파라미터는 보기 좋게 문자열화
        seqs      = p.get("seqs", "")
        schedule  = p.get("sched", p.get("schedule",""))
        if isinstance(seqs, (list, tuple)): seqs = ",".join(map(str, seqs))
        if isinstance(schedule, (list, tuple)): schedule = ",".join(map(str, schedule))

        rows.append([
            trial_id,
            p.get("lr",""), p.get("weight_decay",""), p.get("mask_prob",""), p.get("mean_span",""),
            p.get("start_bs",""), p.get("grad_accum",""), p.get("dataloader_workers",""), p.get("compile_flag",""),
            p.get("curriculum",""),
            seqs, schedule, p.get("total_tokens",""), p.get("seed",""),
            fair, long_avg, composite
        ])

    # 3) CSV 저장
    with open(leaderboard_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"[collect] leaderboard saved: {leaderboard_csv}")


# ============ CLI ============
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="klue/roberta-large")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--total_tokens", type=float, default=3e8)
    ap.add_argument("--schedule", type=str, default="0.6,0.25,0.15")
    ap.add_argument("--seqs", type=str, default="512,1024,2048")
    ap.add_argument("--eval_every_steps", type=int, default=2000)
    ap.add_argument("--log_every_steps", type=int, default=50)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--mask_prob", type=float, default=0.15)
    ap.add_argument("--mean_span", type=float, default=3.0)
    ap.add_argument("--start_bs", type=int, default=256)
    ap.add_argument("--grad_accum", type=int, default=32)
    ap.add_argument("--warmup_ratio", type=float, default=0.01)
    ap.add_argument("--fp16", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--automl", type=int, default=0)
    ap.add_argument("--n_trials", type=int, default=20)
    ap.add_argument("--dataloader_workers", type=int, default=8)
    ap.add_argument("--eval_subset_blocks", type=int, default=4096)
    ap.add_argument("--compile_flag", type=int, default=0)
    ap.add_argument("--eval_long_gaps", type=str, default="600,1200,1800,2400")
    ap.add_argument("--eval_long_n", type=int, default=30)
    # v4 추가: 평가/집계 관련
    ap.add_argument("--do_eval_after_train", type=int, default=0)
    ap.add_argument("--base_for_eval", type=str, default="klue/roberta-large")
    ap.add_argument("--fair_max_tokens", type=int, default=512)
    ap.add_argument("--long_gaps", type=str, default="1200,1800,2400")
    ap.add_argument("--long_n", type=int, default=30)
    ap.add_argument("--score_weights", type=str, default="0.6,0.4")

    ap.add_argument("--study_name", type=str, default=None)
    ap.add_argument("--storage", type=str, default=None)  # e.g., sqlite:///.../optuna.db
    ap.add_argument("--resume", type=int, default=0)
    ap.add_argument("--skip_if_done", type=int, default=1)
    ap.add_argument("--backfill_from_fs", type=int, default=0)
    ap.add_argument("--set_tf32", type=int, default=0)

    # 평가 전용 실행
    ap.add_argument("--eval_only", type=int, default=0)
    ap.add_argument("--eval_all", type=int, default=0)
    ap.add_argument("--eval_ckpt_all", type=int, default=0)
    ap.add_argument("--ft_dir", type=str, default="")

    # 리더보드 재수집
    ap.add_argument("--collect", type=int, default=0)
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    w = tuple(float(x) for x in args.score_weights.split(","))
    # TF32 설정 (파싱 이후)
    if args.set_tf32:
        try:
            torch.backends.cuda.matmul.fp32_precision = "high"   # 'ieee'/'highest'도 가능
            torch.backends.cudnn.conv.fp32_precision = "tf32"
        except Exception as e:
            warnings.warn(f"TF32 setting failed: {e}")

    schedule = [float(x) for x in args.schedule.split(",")]
    seqs = [int(x) for x in args.seqs.split(",")]

    if args.collect:
        collect_automl_results(args.out_dir, score_weights=w)
        exit(0)

    if args.eval_ckpt_all:
        eval_all_checkpoints(
            out_dir=args.out_dir,
            base_for_eval=args.base_for_eval,
            fair_max_tokens=args.fair_max_tokens,
            long_gaps=args.long_gaps,
            long_n=args.long_n
        )
        exit(0)


    if args.eval_only:
        assert args.ft_dir, "--ft_dir 를 지정하세요"
        rep = eval_coref_compare(
            base_model_name=args.base_for_eval,
            ft_dir=args.ft_dir,
            device="cuda" if torch.cuda.is_available() else "cpu",
            fair_max_tokens=args.fair_max_tokens,
            long_gaps=[int(x) for x in args.long_gaps.split(",") if x.strip()],
            long_n=args.long_n
        )
        print_eval_report(rep)
        # JSON도 저장
        save_json(os.path.join(args.out_dir, "eval_coref.json"), rep)
        exit(0)

    if args.eval_all:
        eval_all_trials(
            out_dir=args.out_dir,
            base_for_eval=args.base_for_eval,
            fair_max_tokens=args.fair_max_tokens,
            long_gaps=args.long_gaps,
            long_n=args.long_n
        )
        exit(0)


    if args.automl:
        long_gaps = [int(x) for x in args.eval_long_gaps.split(",")] if args.eval_long_gaps else []
        automl_search(
            model_name=args.model,
            out_dir=args.out_dir,
            n_trials=args.n_trials,
            total_tokens=int(args.total_tokens),
            seed=args.seed,
            eval_long_gaps=long_gaps,
            eval_long_n=args.eval_long_n,
            study_name=args.study_name,
            storage=args.storage,
            skip_if_done=args.skip_if_done,
            backfill_from_fs=args.backfill_from_fs,
            score_weights=w,
        )
    else:
        final_dir = run_train(
            model_name=args.model, out_dir=args.out_dir,
            total_tokens=int(args.total_tokens),
            schedule=schedule, seqs=seqs,
            eval_every_steps=args.eval_every_steps,
            log_every_steps=args.log_every_steps,
            lr=args.lr, weight_decay=args.weight_decay,
            mask_prob=args.mask_prob, mean_span=args.mean_span,
            start_bs=args.start_bs, grad_accum=args.grad_accum,
            warmup_ratio=args.warmup_ratio, fp16=bool(args.fp16),
            seed=args.seed, dataloader_workers=args.dataloader_workers,
            eval_subset_blocks=args.eval_subset_blocks,
            compile_flag=args.compile_flag
        )
        if args.do_eval_after_train:
            rep = eval_coref_compare(
                base_model_name=args.base_for_eval,
                ft_dir=final_dir,
                device="cuda" if torch.cuda.is_available() else "cpu",
                fair_max_tokens=args.fair_max_tokens,
                long_gaps=[int(x) for x in args.long_gaps.split(",") if x.strip()],
                long_n=args.long_n
            )
            print_eval_report(rep)
            save_json(os.path.join(args.out_dir, "eval_coref.json"), rep)
