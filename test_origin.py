from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
from coref_automl.tune import (
    build_eval_from_lambada,
    build_coref_eval_set,
    eval_lambada_topk,
    eval_coref_f1,
    eval_coref_recall_topk,
)


def main():
    model_name = "kakaobank/kf-deberta-base"
    tok = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    mdl = AutoModelForMaskedLM.from_pretrained(model_name, local_files_only=True)
    fill = pipeline("fill-mask", model=mdl, tokenizer=tok, device=0)

    for seq_len in (1536, 2048):
        eval_lbd = build_eval_from_lambada(limit=600)
        l_top1 = eval_lambada_topk(
            fill,
            eval_lbd,
            tok.mask_token or "[MASK]",
            k=1,
            batch_size=32,
            seq_len=seq_len,
        )

        eval_coref = build_coref_eval_set(limit=800, max_seq_len=seq_len)
        coref_f1 = eval_coref_f1(
            fill,
            eval_coref,
            tok.mask_token or "[MASK]",
            k=5,
            batch_size=32,
            seq_len=seq_len,
        )
        coref_top5 = eval_coref_recall_topk(
            fill,
            eval_coref,
            tok.mask_token or "[MASK]",
            k=5,
            batch_size=32,
            seq_len=seq_len,
        )

        score = 0.4 * coref_f1 + 0.3 * coref_top5 + 0.3 * l_top1
        print(f"\n=== {seq_len} tokens ===")
        print(f"LAMBADA top@1 : {l_top1:.6f}")
        print(f"Coref F1      : {coref_f1:.6f}")
        print(f"Coref top@5   : {coref_top5:.6f}")
        print(f"Combined score: {score:.6f}")


if __name__ == "__main__":
    main()
