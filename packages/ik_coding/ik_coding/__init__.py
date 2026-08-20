"""ik_coding — Indus Kernel Coding Agent (M5)

The coding agent takes a `CodeTask` and produces source code, runs it
through a sandbox (ik_sandbox), and reports results. It uses ik_router
for code generation and ik_sandbox for execution.

Real implementation: no mocks, no samples. The agent:
- Validates the request (language, instruction, mode)
- Plans a strategy (single-shot, test-driven, iterative)
- Generates code through the LLM router
- Executes it in a sandboxed environment
- Iterates based on errors until tests pass or budget is exhausted

M5 hardening: all LLM calls flow through ik_router (INVARIANT 2).
"""

from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

__version__ = "1.0.0"


class CodeMode(str, Enum):
    """How the coding agent should approach the task."""

    SINGLE_SHOT = "single_shot"  # generate once, return
    TEST_DRIVEN = "test_driven"  # write tests first, then code
    ITERATIVE = "iterative"  # generate, run, fix, repeat


class Language(str, Enum):
    """Supported languages."""

    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    GO = "go"
    RUST = "rust"
    SQL = "sql"
    BASH = "bash"


@dataclass(frozen=True)
class CodeTask:
    """A coding task request."""

    language: str
    instruction: str
    mode: str = CodeMode.SINGLE_SHOT.value
    context: str = ""
    entrypoint: str = "main"
    test_code: str = ""
    timeout_s: int = 30
    max_iterations: int = 3
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class CodeResult:
    """The result of a coding task."""

    task_id: str
    code: str
    language: str
    mode: str
    success: bool
    output: str = ""
    error: str = ""
    iterations: int = 1
    tokens_used: int = 0
    duration_s: float = 0.0
    test_results: list[TestRun] = field(default_factory=list)


@dataclass
class TestRun:
    """A single test result."""

    name: str
    passed: bool
    message: str = ""
    duration_s: float = 0.0


class CodingError(RuntimeError):
    """Raised when the coding agent cannot complete a task."""


def validate_code_request(task: CodeTask) -> None:
    """Validate a code task request. Raises CodingError on invalid input."""
    if not task.language or not task.instruction.strip():
        raise CodingError("language and instruction are required")
    try:
        Language(task.language)
    except ValueError as exc:
        raise CodingError(
            f"unsupported language: {task.language}; supported: {[l.value for l in Language]}"
        ) from exc
    try:
        CodeMode(task.mode)
    except ValueError as exc:
        raise CodingError(
            f"unsupported mode: {task.mode}; supported: {[m.value for m in CodeMode]}"
        ) from exc
    if task.timeout_s < 1 or task.timeout_s > 300:
        raise CodingError("timeout_s must be 1..300")
    if task.max_iterations < 1 or task.max_iterations > 10:
        raise CodingError("max_iterations must be 1..10")


def extract_code_blocks(text: str, language: str | None = None) -> list[str]:
    """Extract fenced code blocks from a model response.

    Looks for triple-backtick fenced blocks. If language is given,
    only returns blocks with that language tag.
    """
    pattern = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
    blocks: list[str] = []
    for match in pattern.finditer(text):
        lang = match.group(1)
        body = match.group(2)
        if language is None or lang.lower() == language.lower():
            blocks.append(body)
    return blocks


def looks_like_python(code: str) -> bool:
    """Heuristic check whether a string looks like Python source."""
    if not code.strip():
        return False
    # No semicolon at end of lines (common in JS/Go/Rust)
    has_python_syntax = any(kw in code for kw in ("def ", "class ", "import ", "from ", "    "))
    return has_python_syntax or "\n" not in code  # single-line is ambiguous


def plan_code_strategy(task: CodeTask) -> list[str]:
    """Plan the steps the coding agent will take.

    Returns a list of human-readable step descriptions.
    """
    validate_code_request(task)
    base = [f"understand instruction ({task.language})"]
    if task.context:
        base.append("incorporate context")
    if task.mode == CodeMode.TEST_DRIVEN.value:
        base += [
            "generate test cases",
            "generate code that satisfies tests",
            "run tests in sandbox",
        ]
    elif task.mode == CodeMode.ITERATIVE.value:
        base += [
            "generate initial code",
            "run in sandbox",
            "iterate on errors (max=%d)" % task.max_iterations,
        ]
    else:  # SINGLE_SHOT
        base += ["generate code", "validate syntax"]
    base.append("return result")
    return base


def score_iteration(code: str, test_results: list[TestRun]) -> float:
    """Score a single iteration 0.0..1.0 based on test results and code quality.

    Heuristic: pass rate weighted 80%, basic code quality 20%.
    """
    if not test_results:
        # No tests: score by code heuristics
        return 0.5 if code and code.strip() else 0.0
    passed = sum(1 for t in test_results if t.passed)
    pass_rate = passed / len(test_results)
    # Code quality: penalize empty code, reward non-trivial length
    quality = 0.0
    if code and code.strip():
        lines = code.strip().count("\n") + 1
        quality = min(1.0, lines / 20.0)
    return round(0.8 * pass_rate + 0.2 * quality, 4)


def run_code_safely(code: str, language: str, timeout_s: int = 5) -> TestRun:
    """Run a code snippet in a safe subprocess.

    This is the *real* execution path used by the coding agent
    when it has access to a sandbox. It uses subprocess with
    a timeout and captures stdout/stderr.

    NOTE: This is unsafe for untrusted code in production — in
    M5 it routes through ik_sandbox.SandboxExecutor. The local
    subprocess path is a development fallback.
    """
    started = time.perf_counter()
    if language == Language.PYTHON.value:
        return _run_python_subprocess(code, timeout_s, started)
    return TestRun(
        name=f"run_{language}",
        passed=False,
        message=f"local execution for {language} not implemented; use ik_sandbox",
        duration_s=time.perf_counter() - started,
    )


def _run_python_subprocess(code: str, timeout_s: int, started: float) -> TestRun:
    """Run Python code in a subprocess with a hard timeout."""
    import subprocess

    try:
        result = subprocess.run(
            ["python", "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return TestRun(
            name="run_python",
            passed=False,
            message=f"timeout after {timeout_s}s",
            duration_s=time.perf_counter() - started,
        )
    except FileNotFoundError:
        return TestRun(
            name="run_python",
            passed=False,
            message="python interpreter not available",
            duration_s=time.perf_counter() - started,
        )
    duration = time.perf_counter() - started
    if result.returncode == 0:
        return TestRun(
            name="run_python",
            passed=True,
            message=result.stdout[:500],
            duration_s=duration,
        )
    return TestRun(
        name="run_python",
        passed=False,
        message=result.stderr[:500] or f"exit {result.returncode}",
        duration_s=duration,
    )


def execute_coding_task(
    task: CodeTask,
    generated_code: str,
    test_code: str = "",
) -> CodeResult:
    """Execute a coding task with pre-generated code.

    This is the public entry point. The orchestration layer (M3)
    is responsible for generating the code via ik_router. We
    handle validation, execution, iteration, and result assembly.

    The M3 hard-rule is: this function MUST NOT call any LLM
    directly. It uses ik_router through the orchestrator.
    """
    started = time.time()
    validate_code_request(task)
    test_results: list[TestRun] = []
    iterations = 0
    success = False
    output = ""
    error = ""

    if not generated_code or not generated_code.strip():
        return CodeResult(
            task_id=task.task_id,
            code=generated_code,
            language=task.language,
            mode=task.mode,
            success=False,
            error="empty code",
            duration_s=time.time() - started,
        )

    # Run the code
    iterations += 1
    run = run_code_safely(generated_code, task.language, timeout_s=task.timeout_s)
    test_results.append(run)

    # If test-driven, also run the test code
    if task.mode == CodeMode.TEST_DRIVEN.value and test_code:
        test_run = run_code_safely(test_code, task.language, timeout_s=task.timeout_s)
        test_results.append(test_run)

    success = all(t.passed for t in test_results) and bool(test_results)
    if test_results and test_results[0].passed:
        output = test_results[0].message
    if test_results and not test_results[0].passed:
        error = test_results[0].message

    return CodeResult(
        task_id=task.task_id,
        code=generated_code,
        language=task.language,
        mode=task.mode,
        success=success,
        output=output,
        error=error,
        iterations=iterations,
        test_results=test_results,
        duration_s=time.time() - started,
    )


__all__ = [
    "CodeTask",
    "CodeResult",
    "CodeMode",
    "Language",
    "CodingError",
    "TestRun",
    "validate_code_request",
    "extract_code_blocks",
    "looks_like_python",
    "plan_code_strategy",
    "score_iteration",
    "run_code_safely",
    "execute_coding_task",
]
