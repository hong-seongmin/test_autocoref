import argparse
import json
from pathlib import Path
from typing import List

import torch
from safetensors.torch import load_file
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer, pipeline

from coref_automl.tune import (
    build_coref_eval_set,
    build_eval_from_lambada,
)
from coref_automl.tune import clip_around_mask
from coref_automl.coref_utils import is_noun


def expand_position_embeddings(model: AutoModelForMaskedLM, seq_len: int) -> None:
    hidden_size = model.config.hidden_size
    position_embed = getattr(model.deberta.embeddings, "position_embeddings", None)
    if position_embed is not None and position_embed.weight is not None:
        old_num, dim = position_embed.weight.shape
        if seq_len > old_num:
            new_embed = torch.nn.Embedding(seq_len, dim)
            new_embed.weight.data[:old_num] = position_embed.weight.data.clone()
            new_embed.weight.data[old_num:] = position_embed.weight.data[-1:].repeat(seq_len - old_num, 1)
            model.deberta.embeddings.position_embeddings = new_embed.to(model.device)
    rel_embeddings = getattr(model.deberta.encoder, "rel_embeddings", None)
    if rel_embeddings is None or rel_embeddings.weight is None:
        raise NotImplementedError("Cannot expand relative embeddings for this model.")

    old_rel_num, rel_dim = rel_embeddings.weight.shape
    if seq_len > old_rel_num:
        new_rel = torch.nn.Embedding(seq_len, rel_dim)
        new_rel.weight.data[:old_rel_num] = rel_embeddings.weight.data.clone()
        new_rel.weight.data[old_rel_num:] = rel_embeddings.weight.data[-1:].repeat(seq_len - old_rel_num, 1)
        model.deberta.encoder.rel_embeddings = new_rel.to(model.device)
    model.config.max_position_embeddings = seq_len


def fill_and_filter(fill, texts, k, mask_token, batch_size, seq_len, desc):
    preds_all = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc, leave=False):
        chunk = texts[start : start + batch_size]
        clipped = [clip_around_mask(t, mask_token, seq_len=seq_len) for t in chunk]
        outs = fill(clipped, top_k=max(50, k), batch_size=len(clipped))
        for item in outs:
            cand = []
            for p in item:
                token_str = p.get("token_str", "").strip().replace("##", "")
                if token_str and is_noun(token_str):
                    cand.append(token_str)
                if len(cand) >= k:
                    break
            preds_all.append(cand)
    return preds_all


def evaluate_checkpoint(checkpoint: Path, seq_len: int, batch_size: int, device: int) -> dict:
    config = AutoConfig.from_pretrained(checkpoint)
    config.max_position_embeddings = max(getattr(config, "max_position_embeddings", seq_len), seq_len)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    tokenizer.model_max_length = seq_len
    tokenizer.init_kwargs["model_max_length"] = seq_len
    model = AutoModelForMaskedLM.from_config(config)
    expand_position_embeddings(model, seq_len)
    state_dict = load_file(checkpoint / "model.safetensors")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        print(f"[WARN] Unexpected keys {unexpected} in {checkpoint}")

    fill = pipeline(
        "fill-mask",
        model=model,
        tokenizer=tokenizer,
        device=device,
        tokenizer_kwargs={"truncation": True, "max_length": seq_len},
    )
    mask_token = tokenizer.mask_token or "[MASK]"

    eval_lbd = build_eval_from_lambada(limit=600)
    eval_coref = build_coref_eval_set(limit=800, max_seq_len=seq_len)

    # LAMBADA
    masked = [it["masked"] for it in eval_lbd]
    golds = [it["target"] for it in eval_lbd]
    preds = fill_and_filter(fill, masked, k=1, mask_token=mask_token, batch_size=batch_size, seq_len=seq_len, desc=f"LAMBADA ({checkpoint.name})")
    correct = 0
    for g, cands in zip(golds, preds):
        if not cands:
            continue
        if g in cands or any((g in c) or (c in g) for c in cands):
            correct += 1
    lbd_top1 = correct / max(1, len(eval_lbd))

    # Coref
    masked_coref = [it["masked"] for it in eval_coref]
    ctx_nouns = [set(it["context_nouns"]) for it in eval_coref]
    preds_coref = fill_and_filter(fill, masked_coref, k=5, mask_token=mask_token, batch_size=batch_size, seq_len=seq_len, desc=f"Coref ({checkpoint.name})")

    tp = fp = fn = 0
    ok = 0
    for ctx, cands in zip(ctx_nouns, preds_coref):
        pred_set = set(cands)
        ctx_set = ctx
        if any(c in ctx_set for c in cands):
            ok += 1
        tp += len(pred_set & ctx_set)
        fp += len(pred_set - ctx_set)
        fn += len(ctx_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    coref_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    coref_top5 = ok / max(1, len(eval_coref))
    score = 0.4 * coref_f1 + 0.3 * coref_top5 + 0.3 * lbd_top1

    return {
        "checkpoint": str(checkpoint),
        "seq_len": seq_len,
        "lambada_top1": lbd_top1,
        "coref_f1": coref_f1,
        "coref_top5": coref_top5,
        "score": score,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate checkpoints on coref/LAMBADA metrics.")
    parser.add_argument("--checkpoint", action="append", required=True, help="Path to a checkpoint directory (can be repeated)")
    parser.add_argument("--seq-len", type=int, required=True, help="Sequence length to evaluate at")
    parser.add_argument("--output", type=str, default="checkpoint_eval.jsonl", help="Where to store evaluation results")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for evaluation pipeline")
    args = parser.parse_args()

    checkpoints: List[Path] = [Path(p) for p in args.checkpoint]
    device = 0 if torch.cuda.is_available() else -1

    results = []
    total = len(checkpoints)
    for idx, ckpt in enumerate(checkpoints, 1):
        if not ckpt.exists():
            print(f"[WARN] checkpoint not found: {ckpt}")
            continue
        print(f"[{idx}/{total}] Evaluating {ckpt}...")
        record = evaluate_checkpoint(ckpt, args.seq_len, args.batch_size, device)
        print(
            f"    LAMBADA@1={record['lambada_top1']:.4f} | "
            f"Coref F1={record['coref_f1']:.4f} | Coref@5={record['coref_top5']:.4f} | score={record['score']:.4f}"
        )
        results.append(record)

    output_path = Path(args.output)
    with output_path.open("w", encoding="utf-8") as f:
        for record in results:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    main()
