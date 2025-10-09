# coref_automl/probe.py
import os, math, gc, random, time, torch, warnings
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
warnings.filterwarnings("ignore")

MODELS = [
    "kakaobank/kf-deberta-base",
    "kykim/bert-kor-base",
    "google-bert/bert-base-multilingual-cased",
]

def load_ko_lambada_split(prefer=("validation","dev","test")):
    """
    Ko-LAMBADA는 보통 test만 제공됨. 위 우선순위로 시도.
    (환경변수 HUGGINGFACE_HUB_TOKEN / HF_TOKEN 사용 가능)
    """
    hf_token = os.getenv("HUGGINGFACE_HUB_TOKEN") or os.getenv("HF_TOKEN")
    last_err = None
    for sp in prefer:
        try:
            kwargs = dict(split=sp)
            if hf_token:
                kwargs["token"] = hf_token
            return load_dataset("thunder-research-group/SNU_Ko-LAMBADA", **kwargs)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Ko-LAMBADA split 로드 실패: {last_err}")

def sample_ko_lambada(num=64, seed=42):
    ds = load_ko_lambada_split(prefer=("validation","dev","test"))
    rnd = random.Random(seed)
    idxs = [rnd.randrange(0, len(ds)) for _ in range(min(num, len(ds)))]
    items = []
    for i in idxs:
        ex = ds[i]
        text = ex.get("text") or ex.get("context") or ""
        target = ex.get("target") or ex.get("answer") or ""
        if not text or not target:
            continue
        # 말미가 target이면 말미를 [MASK]로, 아니면 포함 시 교체, 아니면 끝에 추가
        if text.strip().endswith(target):
            masked = text[: len(text) - len(target)] + "[MASK]"
        else:
            masked = text.replace(target, "[MASK]") if target in text else (text + " [MASK]")
        items.append(masked)
    # 혹시 비었으면 최소 1개는 보장(실데이터 기반, 빈문장 회피용)
    return items if items else ["한국어 문맥에서 [MASK]을(를) 예측합니다."]

def try_step(model, tokenizer, texts, seq_len):
    """
    실데이터 배치로 forward+backward 1 step 수행하여
    메모리/시간을 현실적으로 측정.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=seq_len,
        return_tensors="pt"
    ).to(device)
    labels = inputs["input_ids"].clone()
    # 15% 랜덤 마스킹, PAD=-100
    probs = torch.rand_like(labels.float())
    labels[probs > 0.15] = -100
    labels[(labels == tokenizer.pad_token_id)] = -100

    out = model(**inputs, labels=labels)
    loss = out.loss
    loss.backward()
    if device == "cuda":
        torch.cuda.synchronize()

def find_max_bs(model, tokenizer, seq_len, start=2, limit=256, texts=None):
    """
    지수 탐색으로 OOM이 나지 않는 최대 per-device batch size를 찾는다.
    accelerate.find_executable_batch_size는 버전별 시그니처 상이하므로 사용하지 않음.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if texts is None:
        texts = sample_ko_lambada(num=512)

    def runner(bs):
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
        batch = texts[:bs]
        try_step(model, tokenizer, batch, seq_len)

    bs = start
    last_ok = None
    # 1) 증가 단계: 2배씩 올리며 성공하는 최대치 탐색
    while bs <= limit:
        try:
            runner(bs)
            last_ok = bs
            bs *= 2
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                break
            raise

    # 2) 실패 구간에서 이분/선형 축소 탐색(보수적으로 하향)
    if last_ok is None:
        bs = max(1, start // 2)
        while bs >= 1:
            try:
                runner(bs); last_ok = bs; break
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    bs //= 2
                else:
                    raise
    return last_ok or 1

def measure_step_time(model, tokenizer, seq_len, bs, reps=8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    texts = sample_ko_lambada(num=max(64, bs))
    # 워밍업 2회
    for _ in range(2):
        try_step(model, tokenizer, texts[:bs], seq_len)
        model.zero_grad(set_to_none=True)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        try_step(model, tokenizer, texts[:bs], seq_len)
        model.zero_grad(set_to_none=True)
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / reps

def gpu_info():
    if not torch.cuda.is_available():
        return "-"
    p = torch.cuda.get_device_properties(0)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    return f"{p.name} ({int(total)} MB)"

def main():
    print(f"CUDA: {torch.cuda.is_available()} | GPU: {gpu_info()}")
    for name in MODELS:
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForMaskedLM.from_pretrained(name)

        max_len_tok = tokenizer.model_max_length
        max_pos = getattr(model.config, "max_position_embeddings", None) or max_len_tok
        # 모델 최대 길이 정보가 1e30 같은 sentinel이면 max_pos 사용
        seq_len = min(512, max_len_tok if max_len_tok != int(1e30) else max_pos)

        max_bs = find_max_bs(model, tokenizer, seq_len)
        step_t = measure_step_time(model, tokenizer, seq_len, max_bs, reps=6)

        # 전역 토큰 타겟(경험상 1024~2048 사이가 안정) → 권장 grad_acc 산정
        target_tokens = 1024
        eff_tokens = max_bs * seq_len
        base_grad_acc = max(1, math.ceil(target_tokens / eff_tokens))

        # 1.2s/step 이하 유지 기준으로 grad_acc 상한 추정(체감치)
        grad_acc = base_grad_acc
        while True:
            est = step_t * (grad_acc / base_grad_acc)
            if est > 1.2 or grad_acc >= 128:
                break
            grad_acc *= 2
        grad_range = (max(1, grad_acc // 2), max(1, grad_acc))

        print("\n====================")
        print(f"모델: {name}")
        print(f"토크나이저 최대 길이: {max_len_tok}")
        print(f"config.max_position_embeddings: {max_pos}")
        print(f"측정 seq_len: {seq_len}")
        print(f"최대 per-device batch size(실데이터 기반): {max_bs}")
        print(f"평균 step time@bs={max_bs}: {step_t:.3f}s")
        print(f"권장 gradient_accumulation_steps(≤1.2s/step 기준): {grad_range[0]}~{grad_range[1]}")
        print("====================")
        # 모델 객체 정리
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()

