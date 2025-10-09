# coref_automl/callback.py
from __future__ import annotations
import time
from typing import Optional, Dict, Any
from transformers import TrainerCallback
from .bus import BUS

class LiveMetricsCallback(TrainerCallback):
    """
    학습 중 실시간 메트릭을 BUS(ndjson)로 송신하여 CLI 대시보드에 표시.
    - on_log: loss, lr, step, total_steps, step_time, throughput (progress)
    - on_evaluate: top1, top5 (progress)
    - on_train_begin: HP/데이터셋 메타(대시보드가 상단 패널에 노출)
    """

    def __init__(
        self,
        model_name: str,
        trial_id: int | str = "-",
        hp: Optional[Dict[str, Any]] = None,
        dataset_meta: Optional[Dict[str, Any]] = None,
        total_steps: Optional[int] = None,
        emit_every_steps: int = 10,
    ):
        self.model_name = model_name
        self.trial_id = trial_id
        self.hp = hp or {}
        self.dataset_meta = dataset_meta or {}
        self.total_steps = total_steps
        self.emit_every_steps = max(1, int(emit_every_steps))
        self._last_t = None

    def _send_throughput(self, state, dt: float):
        per_device_bs = self.hp.get("per_device_bs")
        grad_acc = self.hp.get("grad_acc")
        seq_len = self.hp.get("seq_len") or self.hp.get("max_length")
        samples_per_update = None
        tokens_per_update = None
        if per_device_bs is not None and grad_acc is not None:
            samples_per_update = int(per_device_bs) * int(grad_acc)
            if seq_len is not None:
                tokens_per_update = samples_per_update * int(seq_len)

        BUS.log(
            section="throughput",
            model=self.model_name,
            trial=self.trial_id,
            step=int(state.global_step),
            step_time=dt,
            samples_per_s=(samples_per_update / dt) if (samples_per_update and dt) else None,
            tokens_per_s=(tokens_per_update / dt) if (tokens_per_update and dt) else None,
        )

    def on_train_begin(self, args, state, control, **kwargs):
        self._last_t = time.time()
        BUS.log(
            section="progress",
            model=self.model_name,
            trial=self.trial_id,
            step=int(state.global_step),
            total_steps=self.total_steps,
            hp=self.hp,
            dataset_meta=self.dataset_meta,
        )

    def on_step_end(self, args, state, control, **kwargs):
        now = time.time()
        if self._last_t is not None:
            dt = max(1e-6, now - self._last_t)
            if state.global_step % self.emit_every_steps == 0:
                self._send_throughput(state, dt)
        self._last_t = now

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None:
            return
        now = time.time()
        dt = None
        if self._last_t is not None:
            dt = max(1e-6, now - self._last_t)
        self._last_t = now

        per_device_bs = self.hp.get("per_device_bs")
        grad_acc = self.hp.get("grad_acc")
        seq_len = self.hp.get("seq_len") or self.hp.get("max_length")
        samples_per_update = None
        tokens_per_update = None
        if per_device_bs is not None and grad_acc is not None:
            samples_per_update = int(per_device_bs) * int(grad_acc)
            if seq_len is not None:
                tokens_per_update = samples_per_update * int(seq_len)

        BUS.log(
            section="progress",
            model=self.model_name,
            trial=self.trial_id,
            step=int(state.global_step),
            total_steps=self.total_steps,
            loss=float(logs.get("loss", logs.get("training_loss", 0.0))),
            lr=float(logs.get("learning_rate", 0.0)) if "learning_rate" in logs else None,
            step_time=dt,
            throughput={
                "samples_per_s": (samples_per_update / dt) if (samples_per_update and dt) else None,
                "tokens_per_s": (tokens_per_update / dt) if (tokens_per_update and dt) else None,
            } if dt else {},
            hp=self.hp,
            dataset_meta=self.dataset_meta,
        )

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is None:
            return
        BUS.log(
            section="progress",
            model=self.model_name,
            trial=self.trial_id,
            step=int(state.global_step),
            total_steps=self.total_steps,
            top1=float(metrics.get("top1", -1.0)),
            top5=float(metrics.get("top5", -1.0)),
        )
