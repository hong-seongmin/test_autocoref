# coref_automl/sweep.py
from __future__ import annotations
import argparse
import os
import shlex
import subprocess
import sys
import time
from itertools import product
from typing import List

from .bus import BUS

DEFAULT_MODELS = [
    "kakaobank/kf-deberta-base",
    "kykim/bert-kor-base",
    "google-bert/bert-base-multilingual-cased",
]

def run_cmd(cmd: List[str], env=None) -> int:
    print("[RUN]", " ".join(shlex.quote(c) for c in cmd))
    try:
        p = subprocess.run(cmd, env=env)
        return p.returncode
    except KeyboardInterrupt:
        return 130

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=str, default=",".join(DEFAULT_MODELS), help="comma-separated model ids")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seeds", type=str, default="42")
    ap.add_argument("--train-limits", type=str, default="60000")
    ap.add_argument("--uv", action="store_true", help="use `uv run` prefix")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    limits = [int(x) for x in args.train_limits.split(",") if x.strip()]

    jobs = list(product(models, seeds, limits))
    total_jobs = len(jobs)

    # Sweep 시작 신호
    BUS.log(event="sweep_begin", ts=time.time(), total_jobs=total_jobs)

    results = []
    ok = 0
    for i, (m, s, L) in enumerate(jobs, start=1):
        BUS.log(event="sweep_job_begin", model=m, seed=s, train_limit=L, ts=time.time())
        env = os.environ.copy()
        # 동일 BUS 경로 유지
        if "COREF_BUS" in os.environ:
            env["COREF_BUS"] = os.environ["COREF_BUS"]

        base = ["python", "-m", "coref_automl.tune", "--model", m, "--trials", str(args.trials), "--seed", str(s), "--train-limit", str(L)]
        cmd = (["uv", "run"] + base) if args.uv else base

        start_time = time.time()
        ret = run_cmd(cmd, env=env)
        end_time = time.time()

        job_result = {
            "job_id": i,
            "model": m,
            "seed": s,
            "train_limit": L,
            "trials": args.trials,
            "status_code": ret,
            "duration": end_time - start_time,
            "started_at": start_time,
            "ended_at": end_time
        }

        if ret == 0:
            ok += 1
        else:
            print(f"[WARN] run failed (ret={ret}): {m} seed={s} limit={L}", file=sys.stderr)

        results.append(job_result)
        BUS.log(event="sweep_job_end", model=m, seed=s, train_limit=L, ts=time.time(), rc=ret)

    # 종합 결과 저장
    import json
    result_file = "runs/sweep_comprehensive_results.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": {
                "total_jobs": total_jobs,
                "successful_jobs": ok,
                "failed_jobs": total_jobs - ok,
                "models_tested": list(set(m for m, s, L in jobs)),
                "seeds_used": list(set(s for m, s, L in jobs)),
                "train_limits_used": list(set(L for m, s, L in jobs))
            },
            "jobs": results
        }, f, indent=2, ensure_ascii=False)

    print(f"[DONE] {ok}/{total_jobs} jobs succeeded")
    print(f"[RESULTS] Comprehensive results saved to {result_file}")

if __name__ == "__main__":
    main()
