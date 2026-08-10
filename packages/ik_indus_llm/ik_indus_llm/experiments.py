"""Experiment registry — persistent log of every run with hypothesis tracking.

This is the "experiment" table from the system prompt. Every model
version and every meaningful change gets recorded with:
  - hypothesis (what we expect to improve)
  - change (what we did)
  - baseline (what we compare against)
  - result (what actually happened)
  - decision (keep / revert / investigate)

Stored as a JSONL file for human-inspectability. Can be backed by
SQLite or any external DB if the project grows.
"""

from __future__ import annotations
import json
import platform
import subprocess
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def _git_sha() -> Optional[str]:
    """Best-effort current git SHA. Returns None if not in a git repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        return out.decode().strip()
    except Exception:
        return None


def _env_metadata() -> Dict[str, Any]:
    """Snapshot of the environment for reproducibility."""
    md = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "timestamp": time.time(),
    }
    try:
        import torch
        md["torch"] = torch.__version__
        md["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            md["cuda_version"] = torch.version.cuda
            md["device_count"] = torch.cuda.device_count()
            md["device_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    md["git_sha"] = _git_sha()
    return md


@dataclass
class Experiment:
    """A single recorded experiment."""
    id: str
    name: str
    status: str  # "planned" | "running" | "completed" | "failed" | "interrupted"
    created_at: float

    # Scientific method
    hypothesis: str = ""
    change: str = ""
    baseline: str = ""
    result: str = ""
    decision: str = ""  # "" | "keep" | "revert" | "investigate"

    # Technical metadata
    model_version: str = ""
    parameter_count: int = 0
    dataset_version: str = ""
    dataset_hash: str = ""
    tokenizer_version: str = ""
    architecture: str = ""
    hyperparameters: Dict[str, Any] = field(default_factory=dict)

    # Results
    training_loss: Optional[float] = None
    validation_loss: Optional[float] = None
    perplexity: Optional[float] = None
    benchmark_results: Dict[str, Any] = field(default_factory=dict)

    # Operational
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration_seconds: Optional[float] = None
    checkpoint_path: Optional[str] = None
    failure_reason: Optional[str] = None
    environment: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)


class ExperimentRegistry:
    """Append-only log of experiments.

    Storage: JSONL file (one experiment per line). Easy to grep, diff,
    and load. Swap with a database if/when scale demands it.
    """
    def __init__(self, path: str = "experiments.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._cache: List[Experiment] = []
        self._load()

    def _load(self):
        if not self.path.exists():
            return
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            d = json.loads(line)
            self._cache.append(Experiment(**d))

    def _append(self, exp: Experiment):
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(exp)) + "\n")

    def create(
        self,
        name: str,
        hypothesis: str = "",
        change: str = "",
        baseline: str = "",
        model_version: str = "",
        dataset_version: str = "",
        tags: Optional[List[str]] = None,
    ) -> Experiment:
        exp = Experiment(
            id=str(uuid.uuid4())[:8],
            name=name,
            status="planned",
            created_at=time.time(),
            hypothesis=hypothesis,
            change=change,
            baseline=baseline,
            model_version=model_version,
            dataset_version=dataset_version,
            environment=_env_metadata(),
            tags=tags or [],
        )
        self._cache.append(exp)
        self._append(exp)
        return exp

    def start(self, exp: Experiment, hyperparameters: Optional[Dict] = None) -> Experiment:
        exp.status = "running"
        exp.start_time = time.time()
        if hyperparameters:
            exp.hyperparameters.update(hyperparameters)
        self._rewrite()
        return exp

    def complete(
        self,
        exp: Experiment,
        training_loss: Optional[float] = None,
        validation_loss: Optional[float] = None,
        perplexity: Optional[float] = None,
        benchmark_results: Optional[Dict] = None,
        checkpoint_path: Optional[str] = None,
        decision: str = "",
        result: str = "",
    ) -> Experiment:
        exp.status = "completed"
        exp.end_time = time.time()
        exp.duration_seconds = (exp.end_time - (exp.start_time or exp.end_time))
        if training_loss is not None:
            exp.training_loss = training_loss
        if validation_loss is not None:
            exp.validation_loss = validation_loss
        if perplexity is not None:
            exp.perplexity = perplexity
        if benchmark_results:
            exp.benchmark_results.update(benchmark_results)
        if checkpoint_path:
            exp.checkpoint_path = checkpoint_path
        if decision:
            exp.decision = decision
        if result:
            exp.result = result
        self._rewrite()
        return exp

    def fail(self, exp: Experiment, reason: str) -> Experiment:
        exp.status = "failed"
        exp.failure_reason = reason
        exp.end_time = time.time()
        exp.duration_seconds = (exp.end_time - (exp.start_time or exp.end_time))
        self._rewrite()
        return exp

    def interrupt(self, exp: Experiment, reason: str = "session terminated") -> Experiment:
        exp.status = "interrupted"
        exp.failure_reason = reason
        exp.end_time = time.time()
        exp.duration_seconds = (exp.end_time - (exp.start_time or exp.end_time))
        self._rewrite()
        return exp

    def _rewrite(self):
        # Rewrite the whole file (small registry, OK to be simple)
        with self.path.open("w") as f:
            for e in self._cache:
                f.write(json.dumps(asdict(e)) + "\n")

    # ---- queries ----
    def list(self, status: Optional[str] = None, tag: Optional[str] = None) -> List[Experiment]:
        out = list(self._cache)
        if status:
            out = [e for e in out if e.status == status]
        if tag:
            out = [e for e in out if tag in e.tags]
        return out

    def latest(self) -> Optional[Experiment]:
        return self._cache[-1] if self._cache else None

    def by_model(self, model_version: str) -> List[Experiment]:
        return [e for e in self._cache if e.model_version == model_version]

    def summary(self) -> Dict[str, Any]:
        n_total = len(self._cache)
        by_status = {}
        for e in self._cache:
            by_status[e.status] = by_status.get(e.status, 0) + 1
        return {"total": n_total, "by_status": by_status, "path": str(self.path)}
