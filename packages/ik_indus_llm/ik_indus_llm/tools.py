"""Built-in tools for the Indus agent.

Each tool has:
  - name:        short identifier
  - description: shown to the LLM in the system prompt
  - signature:   JSON schema for the arguments
  - run(args) -> str: the actual implementation

Indus can call any of these via the ReAct loop. The tool descriptions
are designed to be unambiguous so even a small model can pick correctly.
"""

from __future__ import annotations
import ast
import math
import operator
import json
import os
import subprocess
import tempfile
from typing import Any, Callable, Dict, List

from .model import Indus


class Tool:
    """A tool the agent can call."""
    def __init__(self, name: str, description: str, signature: Dict[str, Any],
                 fn: Callable[[Dict[str, Any]], str]):
        self.name = name
        self.description = description
        self.signature = signature
        self.fn = fn

    def call(self, args: Dict[str, Any]) -> str:
        if not isinstance(args, dict):
            return "ERROR: tool arguments must be a JSON object"
        for key, expected in self.signature.items():
            if key not in args:
                return f"ERROR: missing required argument '{key}'"
            if expected == "string" and not isinstance(args[key], str):
                return f"ERROR: argument '{key}' must be a string"
        try:
            return self.fn(args)
        except Exception as e:
            return f"ERROR: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Calculator — safe AST-based arithmetic evaluator
# ---------------------------------------------------------------------------

_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}
_SAFE_FUNCS = {
    "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
    "sqrt": math.sqrt, "log": math.log, "log2": math.log2, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "pi": math.pi, "e": math.e,
}


def _safe_eval(expr: str) -> Any:
    tree = ast.parse(expr, mode="eval")
    return _eval_node(tree.body)


def _eval_node(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported op: {type(node.op).__name__}")
        return op(_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"unsupported unary op: {type(node.op).__name__}")
        return op(_eval_node(node.operand))
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _SAFE_FUNCS:
            fn = _SAFE_FUNCS[node.func.id]
            if callable(fn):
                return fn(*[_eval_node(a) for a in node.args])
            return fn  # constant
        raise ValueError(f"unsupported call: {ast.dump(node.func)}")
    if isinstance(node, ast.Name):
        if node.id in _SAFE_FUNCS:
            v = _SAFE_FUNCS[node.id]
            return v() if callable(v) else v
        raise ValueError(f"unknown name: {node.id}")
    raise ValueError(f"unsupported node: {ast.dump(node)}")


def calculator_tool(args: Dict[str, Any]) -> str:
    """Evaluate a math expression safely."""
    expr = args.get("expression", "")
    if not expr:
        return "ERROR: empty expression"
    return str(_safe_eval(expr))


# ---------------------------------------------------------------------------
# Python sandbox — execute a short snippet and return stdout
# ---------------------------------------------------------------------------

def python_tool(args: Dict[str, Any]) -> str:
    """Execute Python in an isolated Docker container.

    This is intentionally *not* an in-process ``exec`` sandbox.  Docker must
    be available and the image must already be present/pullable by the host.
    The container has no network, a read-only root filesystem, a temporary
    /tmp, strict memory/CPU/process limits, and a hard timeout.
    """
    code = args.get("code", "")
    if not isinstance(code, str) or not code.strip():
        return "ERROR: empty code"
    if len(code) > 16_000:
        return "ERROR: code exceeds 16,000 character limit"

    image = os.environ.get("INDUS_PYTHON_SANDBOX_IMAGE", "python:3.12-alpine")
    cmd = [
        "docker", "run", "--rm",
        "--network", "none",
        "--read-only",
        "--tmpfs", "/tmp:rw,nosuid,nodev,noexec,size=32m",
        "--memory", "128m",
        "--cpus", "0.5",
        "--pids-limit", "64",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        image,
        "python", "-I", "-S", "-c", code,
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=8,
            check=False,
        )
    except FileNotFoundError:
        return "ERROR: Docker is required for the Python tool"
    except subprocess.TimeoutExpired:
        return "ERROR: Python execution timed out after 8 seconds"
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip()
        return f"ERROR: sandbox exit {proc.returncode}: {err[:4000]}"
    return proc.stdout[:12000] or "(no output)"


# ---------------------------------------------------------------------------
# String tools — simple text utilities
# ---------------------------------------------------------------------------

def reverse_tool(args: Dict[str, Any]) -> str:
    return args.get("text", "")[::-1]


def word_count_tool(args: Dict[str, Any]) -> str:
    text = args.get("text", "")
    return str(len(text.split()))


# ---------------------------------------------------------------------------
# Built-in registry
# ---------------------------------------------------------------------------

def default_tools() -> List[Tool]:
    return [
        Tool(
            name="calculator",
            description=("Evaluate a math expression and return the result. "
                         "Input: {\"expression\": \"<expr>\"}  e.g. 2 + 3 * 4"),
            signature={"expression": "string"},
            fn=calculator_tool,
        ),
        Tool(
            name="python",
            description=("Execute a short Python snippet and return its stdout. "
                         "Input: {\"code\": \"print(2+2)\"}"),
            signature={"code": "string"},
            fn=python_tool,
        ),
        Tool(
            name="reverse_string",
            description="Reverse a string. Input: {\"text\": \"hello\"}",
            signature={"text": "string"},
            fn=reverse_tool,
        ),
        Tool(
            name="word_count",
            description="Count words in a text. Input: {\"text\": \"<text>\"}",
            signature={"text": "string"},
            fn=word_count_tool,
        ),
    ]


class ToolRegistry:
    """A name -> Tool map the agent uses to look up and call tools."""
    def __init__(self, tools: List[Tool] = None):
        self.tools = {t.name: t for t in (tools or default_tools())}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool

    def call(self, name: str, args: Dict[str, Any]) -> str:
        if name not in self.tools:
            return f"ERROR: unknown tool '{name}'"
        return self.tools[name].call(args)

    def descriptions(self) -> str:
        """Render all tools as a system-prompt block."""
        out = ["Available tools:"]
        for t in self.tools.values():
            out.append(f"  - {t.name}: {t.description}")
        return "\n".join(out)

    def names(self) -> List[str]:
        return list(self.tools.keys())
