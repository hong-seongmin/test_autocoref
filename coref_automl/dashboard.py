# coref_automl/dashboard.py
from __future__ import annotations
import math
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from rich import box
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .bus import BusTail, BUS_PATH

console = Console()

# ────────────────────────────────────────────────────────────────────────────────
# 유틸
# ────────────────────────────────────────────────────────────────────────────────

def fmt_num(x: Optional[float], nd=4, empty="-"):
    if x is None:
        return empty
    try:
        return f"{x:.{nd}f}"
    except Exception:
        return empty

def fmt_sci(x: Optional[float], empty="-"):
    if x is None:
        return empty
    try:
        return f"{x:.2e}"
    except Exception:
        return empty

def fmt_eta(sec: Optional[float], empty="-"):
    if sec is None or sec != sec or sec < 0 or math.isinf(sec):
        return empty
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

class EMA:
    def __init__(self, beta: float = 0.9):
        self.beta = beta
        self._y = None

    def update(self, x: float) -> float:
        if self._y is None:
            self._y = x
        else:
            self._y = self.beta * self._y + (1 - self.beta) * x
        return self._y

# ────────────────────────────────────────────────────────────────────────────────
# 상태 컨테이너
# ────────────────────────────────────────────────────────────────────────────────

@dataclass
class ProgressState:
    model: str = "-"
    trial: Optional[int] = None
    step: int = 0
    total_steps: Optional[int] = None
    loss: Optional[float] = None
    lr: Optional[float] = None
    top1: Optional[float] = None
    top5: Optional[float] = None
    step_time: Optional[float] = None
    step_time_ema: Optional[float] = None
    samples_per_s: Optional[float] = None
    tokens_per_s: Optional[float] = None
    # HP/데이터셋
    hp: Dict[str, Any] = field(default_factory=dict)
    dataset_meta: Dict[str, Any] = field(default_factory=dict)
    # ETA
    trial_eta_sec: Optional[float] = None
    # 집계용
    seq_len_for_tp: Optional[int] = None

@dataclass
class StudyState:
    total_trials: Optional[int] = None
    begun: int = 0
    finished: int = 0
    trial_times: List[float] = field(default_factory=list)

    @property
    def remaining_trials(self) -> Optional[int]:
        if self.total_trials is None:
            return None
        return max(0, self.total_trials - self.finished)

    @property
    def eta_sec(self) -> Optional[float]:
        if self.total_trials is None:
            return None
        if not self.trial_times:
            return None
        avg = sum(self.trial_times) / len(self.trial_times)
        rem = self.remaining_trials
        if rem is None:
            return None
        return rem * avg

@dataclass
class SweepState:
    total_jobs: Optional[int] = None
    begun: int = 0
    finished: int = 0
    job_times: List[float] = field(default_factory=list)

    @property
    def remaining_jobs(self) -> Optional[int]:
        if self.total_jobs is None:
            return None
        return max(0, self.total_jobs - self.finished)

    @property
    def eta_sec(self) -> Optional[float]:
        if self.total_jobs is None or not self.job_times:
            return None
        avg = sum(self.job_times) / len(self.job_times)
        rem = self.remaining_jobs
        if rem is None:
            return None
        return rem * avg

@dataclass
class SystemState:
    gpu_name: str = "-"
    util: Optional[float] = None
    temp: Optional[float] = None
    mem_used: Optional[float] = None
    mem_total: Optional[float] = None
    power: Optional[float] = None
    clock: Optional[float] = None
    cuda_alloc: Optional[float] = None

@dataclass
class EvalState:
    lbd_top1: Optional[float] = None
    coref_f1: Optional[float] = None
    coref_top5: Optional[float] = None

@dataclass
class DatasetProgressState:
    stage: str = ""
    completed: int = 0
    total: int = 0
    progress: float = 0.0
    seq_len: Optional[int] = None
    domain: str = ""

# ────────────────────────────────────────────────────────────────────────────────
# 대시보드 상태 + 이벤트 처리
# ────────────────────────────────────────────────────────────────────────────────

class DashboardState:
    def __init__(self):
        self.progress = ProgressState()
        self.study = StudyState()
        self.sweep = SweepState()
        self.system = SystemState()
        self.eval = EvalState()
        self.dataset_progress = DatasetProgressState()
        self._trial_begin_ts: Optional[float] = None
        self._job_begin_ts: Optional[float] = None
        self._ema_step = EMA(beta=0.9)

    def on_event(self, ev: Dict[str, Any]):
        # section 기반
        sec = ev.get("section")
        if sec == "progress":
            self._on_progress(ev)
            return
        if sec == "throughput":
            self._on_throughput(ev)
            return
        if sec == "system":
            self._on_system(ev)
            return
        if sec == "eval_stream":
            self._on_eval(ev)
            return
        if sec == "dataset_progress":
            self._on_dataset_progress(ev)
            return

        # event 기반
        evt = ev.get("event")
        if evt == "trial_begin":
            self._trial_begin_ts = ev.get("ts", time.time())
            self.study.begun += 1
            if "study_trials_total" in ev:
                try:
                    self.study.total_trials = int(ev["study_trials_total"])
                except Exception:
                    pass
            return
        if evt == "trial_end":
            t1 = ev.get("ts", time.time())
            if self._trial_begin_ts is not None:
                self.study.trial_times.append(max(0.0, t1 - self._trial_begin_ts))
            self.study.finished += 1
            self._trial_begin_ts = None
            return
        if evt == "sweep_begin":
            # 새 sweep 시작 시 상태 초기화
            self.study = StudyState()
            self.sweep = SweepState()
            if "total_jobs" in ev:
                try:
                    self.sweep.total_jobs = int(ev["total_jobs"])
                except Exception:
                    pass
            return
        if evt == "sweep_job_begin":
            self._job_begin_ts = ev.get("ts", time.time())
            self.sweep.begun += 1
            return
        if evt == "sweep_job_end":
            t1 = ev.get("ts", time.time())
            if self._job_begin_ts is not None:
                self.sweep.job_times.append(max(0.0, t1 - self._job_begin_ts))
            self.sweep.finished += 1
            self._job_begin_ts = None
            return

    # 세부 섹션 처리
    def _on_progress(self, ev: Dict[str, Any]):
        p = self.progress
        p.model = ev.get("model", p.model)
        p.trial = ev.get("trial", p.trial)
        p.step = int(ev.get("step", p.step) or 0)
        p.total_steps = ev.get("total_steps", p.total_steps)
        p.loss = ev.get("loss", p.loss)
        p.lr = ev.get("lr", p.lr)
        p.top1 = ev.get("top1", p.top1)
        p.top5 = ev.get("top5", p.top5)
        # HP/데이터셋
        hp = ev.get("hp")
        if isinstance(hp, dict):
            p.hp = hp
            if "max_length" in hp and isinstance(hp["max_length"], int):
                p.seq_len_for_tp = hp["max_length"]
            elif "seq_len" in hp and isinstance(hp["seq_len"], int):
                p.seq_len_for_tp = hp["seq_len"]
        dm = ev.get("dataset_meta")
        if isinstance(dm, dict):
            p.dataset_meta = dm
        # ETA(현재 trial)
        st_ema = ev.get("step_time_ema", p.step_time_ema)
        if st_ema is not None:
            p.step_time_ema = st_ema
        st = ev.get("step_time")
        if st is not None:
            p.step_time = float(st)
            p.step_time_ema = self._ema_step.update(float(st))
        if p.total_steps and p.step_time_ema:
            remain = max(0, int(p.total_steps) - int(p.step))
            p.trial_eta_sec = p.step_time_ema * remain
        else:
            p.trial_eta_sec = None

        # Throughput(전송해주는 경우)
        thr = ev.get("throughput", {})
        if isinstance(thr, dict):
            p.samples_per_s = thr.get("samples_per_s", p.samples_per_s)
            p.tokens_per_s = thr.get("tokens_per_s", p.tokens_per_s)

    def _on_throughput(self, ev: Dict[str, Any]):
        p = self.progress
        st = ev.get("step_time")
        if st is not None:
            p.step_time = float(st)
            p.step_time_ema = self._ema_step.update(float(st))
        if "samples_per_s" in ev:
            p.samples_per_s = ev["samples_per_s"]
        if "tokens_per_s" in ev:
            p.tokens_per_s = ev["tokens_per_s"]
        if p.total_steps and p.step_time_ema:
            remain = max(0, int(p.total_steps) - int(p.step))
            p.trial_eta_sec = p.step_time_ema * remain

    def _on_system(self, ev: Dict[str, Any]):
        s = self.system
        s.gpu_name = ev.get("gpu_name", s.gpu_name)
        s.util = ev.get("util", s.util)
        s.temp = ev.get("temp", s.temp)
        s.mem_used = ev.get("mem_used", s.mem_used)
        s.mem_total = ev.get("mem_total", s.mem_total)
        s.power = ev.get("power", s.power)
        s.clock = ev.get("clock", s.clock)
        s.cuda_alloc = ev.get("cuda_alloc", s.cuda_alloc)

    def _on_eval(self, ev: Dict[str, Any]):
        e = self.eval
        if "lbd_top1" in ev: e.lbd_top1 = ev["lbd_top1"]
        if "coref_f1" in ev: e.coref_f1 = ev["coref_f1"]
        if "coref_top5" in ev: e.coref_top5 = ev["coref_top5"]

    def _on_dataset_progress(self, ev: Dict[str, Any]):
        dp = self.dataset_progress
        dp.stage = ev.get("stage", dp.stage)
        dp.completed = ev.get("completed", dp.completed)
        dp.total = ev.get("total", dp.total)
        dp.progress = ev.get("progress", dp.progress)
        dp.seq_len = ev.get("seq_len", dp.seq_len)
        dp.domain = ev.get("domain", dp.domain)

# ────────────────────────────────────────────────────────────────────────────────
# 렌더러
# ────────────────────────────────────────────────────────────────────────────────

def render_progress(st: DashboardState) -> Panel:
    p = st.progress
    t = Table(expand=True, box=box.SIMPLE_HEAVY, show_header=True, title="Progress & ETA")
    t.add_column("model"); t.add_column("trial"); t.add_column("step"); t.add_column("total")
    t.add_column("progress"); t.add_column("ETA(trial)")
    t.add_column("loss"); t.add_column("top@1"); t.add_column("top@5"); t.add_column("lr")

    # progress bar 텍스트
    if p.total_steps:
        perc = clamp((p.step / max(1, p.total_steps)) * 100.0, 0.0, 100.0)
        bar_full = 20
        filled = int(round(perc / 100.0 * bar_full))
        bar = "<" + "█" * filled + " " * (bar_full - filled) + f">  {perc:0.1f}%"
    else:
        bar = "-"

    t.add_row(
        str(p.model),
        "-" if p.trial is None else str(p.trial),
        str(p.step),
        "-" if p.total_steps is None else str(p.total_steps),
        bar,
        fmt_eta(p.trial_eta_sec),
        fmt_num(p.loss, nd=4),
        fmt_num(p.top1, nd=3),
        fmt_num(p.top5, nd=3),
        fmt_sci(p.lr),
    )
    return Panel(t)

def render_throughput(st: DashboardState) -> Panel:
    p = st.progress
    t = Table(expand=True, box=box.SIMPLE, title="Throughput")
    t.add_column("step_time(s)"); t.add_column("EMA(s)"); t.add_column("samples/s"); t.add_column("tokens/s")
    t.add_row(
        fmt_num(p.step_time, nd=3),
        fmt_num(p.step_time_ema, nd=3),
        fmt_num(p.samples_per_s, nd=1),
        fmt_num(p.tokens_per_s, nd=1),
    )
    return Panel(t)

def render_hparams(st: DashboardState) -> Panel:
    p = st.progress
    hp = p.hp or {}
    t = Table(expand=True, box=box.MINIMAL_DOUBLE_HEAD, title="Hyperparameters / Dataset")
    t.add_column("param"); t.add_column("value")
    def add(name, key):
        val = hp.get(key, "-")
        t.add_row(name, str(val))
    # 자주 보는 핵심들
    add("seq_len", "seq_len" if "seq_len" in hp else "max_length")
    add("per_device_bs", "per_device_bs")
    add("grad_acc", "grad_acc")
    add("effective_tokens/update", "effective_tokens_per_update")
    add("warmup_ratio", "warmup_ratio")
    add("lr", "lr")
    add("weight_decay", "weight_decay")
    add("min_prob", "min_prob")
    add("max_prob", "max_prob")
    add("bf16", "bf16")

    dm = p.dataset_meta or {}
    if dm:
        t.add_row("dataset.corpus", dm.get("corpus", "-"))
        t.add_row("dataset.subset", dm.get("subset", "-"))
        t.add_row("dataset.split", dm.get("split", "-"))
        if "train_limit" in hp:
            t.add_row("dataset.train_limit", str(hp["train_limit"]))

    # 데이터셋 진행 상황 추가
    dp = st.dataset_progress
    if dp.stage and dp.total > 0:
        t.add_row("📊 dataset_progress", f"{dp.stage}: {dp.completed}/{dp.total} ({dp.progress:.1%})")
        if dp.seq_len:
            t.add_row("📏 seq_length", str(dp.seq_len))
        if dp.domain:
            t.add_row("🏷️ domain", dp.domain)

    return Panel(t)

def render_eval(st: DashboardState) -> Panel:
    e = st.eval
    t = Table(expand=True, box=box.SIMPLE_HEAVY, title="Evaluation (fill-mask nouns)")
    t.add_column("metric"); t.add_column("value")
    t.add_row("LAMBADA top@1", fmt_num(e.lbd_top1, nd=3))
    t.add_row("Coref F1", fmt_num(e.coref_f1, nd=3))
    t.add_row("Coref top@5", fmt_num(e.coref_top5, nd=3))
    return Panel(t)

def render_system(st: DashboardState) -> Panel:
    s = st.system
    t = Table(expand=True, box=box.MINIMAL_DOUBLE_HEAD, title="GPU / System")
    t.add_column("gpu_name"); t.add_column("util(%)"); t.add_column("temp(°C)")
    t.add_column("mem_used/total(MB)"); t.add_column("power(W)"); t.add_column("clock(MHz)"); t.add_column("cuda_alloc(MB)")
    mem = "-"
    if s.mem_used is not None and s.mem_total is not None:
        mem = f"{int(s.mem_used)}/{int(s.mem_total)}"
    t.add_row(
        s.gpu_name,
        "-" if s.util is None else str(int(s.util)),
        "-" if s.temp is None else str(int(s.temp)),
        mem,
        "-" if s.power is None else fmt_num(s.power, nd=1),
        "-" if s.clock is None else str(int(s.clock)),
        "-" if s.cuda_alloc is None else str(int(s.cuda_alloc)),
    )
    return Panel(t)

def render_etas(st: DashboardState) -> Panel:
    t = Table(expand=True, box=box.SIMPLE, title="ETAs (All Levels)")
    t.add_column("Scope"); t.add_column("Remaining"); t.add_column("ETA")
    # 현재 Trial
    pr = st.progress
    rem_steps = "-" if pr.total_steps is None else f"{max(0, pr.total_steps - pr.step)} steps"
    t.add_row("Current Trial", rem_steps, fmt_eta(pr.trial_eta_sec))
    # 모델 Study
    rem_trials = "-" if st.study.remaining_trials is None else f"{st.study.remaining_trials} trials"
    t.add_row("Model Study (all trials)", rem_trials, fmt_eta(st.study.eta_sec))
    # 풀 스윕
    rem_jobs = "-" if st.sweep.remaining_jobs is None else f"{st.sweep.remaining_jobs} jobs"
    t.add_row("Full Sweep", rem_jobs, fmt_eta(st.sweep.eta_sec))
    return Panel(t)

def build_layout(st: DashboardState) -> Layout:
    layout = Layout()
    layout.split_row(
        Layout(name="L", ratio=3),
        Layout(name="R", ratio=2),
    )
    # 좌측: 상단 진행, 하단 하이퍼/평가
    layout["L"].split_column(
        Layout(name="progress", ratio=2),
        Layout(name="throughput", ratio=1),
        Layout(name="hparams", ratio=2),
        Layout(name="eval", ratio=2),
    )
    # 우측: 시스템 + ETA
    layout["R"].split_column(
        Layout(name="system", ratio=3),
        Layout(name="etas", ratio=2),
    )

    layout["progress"].update(render_progress(st))
    layout["throughput"].update(render_throughput(st))
    layout["hparams"].update(render_hparams(st))
    layout["eval"].update(render_eval(st))
    layout["system"].update(render_system(st))
    layout["etas"].update(render_etas(st))
    return layout

# ────────────────────────────────────────────────────────────────────────────────
# GPU 로컬 폴링(선택적) — BUS에 system 이벤트가 안 오더라도 최소한 채움
# ────────────────────────────────────────────────────────────────────────────────

def update_gpu_info(st: DashboardState, interval: float = 0.5):
    if not torch.cuda.is_available():
        return
    device = 0
    try:
        name = torch.cuda.get_device_name(device)
        st.system.gpu_name = name
    except Exception:
        return
    def _update():
        while True:
            try:
                free, total = torch.cuda.mem_get_info(device)
                used = total - free
                st.system.mem_used = used / (1024 * 1024)
                st.system.mem_total = total / (1024 * 1024)

                # 추가 GPU 정보
                st.system.util = torch.cuda.utilization(device)
                st.system.temp = torch.cuda.temperature(device) if hasattr(torch.cuda, 'temperature') else None
                st.system.power = torch.cuda.power_draw(device) if hasattr(torch.cuda, 'power_draw') else None
                st.system.clock = torch.cuda.clock_rate(device) if hasattr(torch.cuda, 'clock_rate') else None
                st.system.cuda_alloc = (total - free) / total * 100 if total > 0 else 0
            except Exception:
                pass
            time.sleep(interval)
    th = threading.Thread(target=_update, daemon=True)
    th.start()

# ────────────────────────────────────────────────────────────────────────────────
# 메인
# ────────────────────────────────────────────────────────────────────────────────

def main():
    import time
    print(f"[dashboard] tailing BUS: {BUS_PATH}")
    st = DashboardState()
    update_gpu_info(st, interval=1.0)

    tail = BusTail(BUS_PATH, poll_sec=0.2)
    last_update = 0
    update_interval = 1.0  # 최소 1초 간격 업데이트

    with Live(build_layout(st), refresh_per_second=8, screen=False) as live:
        for ev in tail:
            try:
                st.on_event(ev)
                now = time.time()
                if now - last_update >= update_interval:
                    live.update(build_layout(st))
                    last_update = now
            except Exception as e:
                # 이벤트 처리 중 문제는 로그로만 남기고 계속 진행
                console.log(f"[yellow]warn[/yellow]: failed to handle event: {e}")

if __name__ == "__main__":
    main()

