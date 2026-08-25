"""LLM Router — single ingress for every LLM call.

Real implementation. No mocks, no demo mode.

Provider backends (in priority order):
1. Native Indus (local torch model, indus_tiny_v0.3.0.pt) — if checkpoint
   is available and the user explicitly opts in (e.g. model_hint starts
   with "indus/")
2. External API providers via LiteLLM (openai, anthropic, google, etc.) —
   if their env var is set

The router fails loud with ConfigurationError if no backend can serve
the request.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from ik_router.budget import get_budget_enforcer
from ik_router.cache import get_cache
from ik_router.errors import ConfigurationError
from ik_router.fallback import get_fallback_chain
from ik_router.policy import ModelCandidate, get_policy_engine
from ik_router.types import (
    EmbedRequest,
    EmbedResponse,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    MessageRole,
    ToolCall,
)

logger = logging.getLogger(__name__)

_NATIVE_PREFIX = "indus/"


class LLMRouter:
    """The LLM Router. Single ingress for every LLM call in the kernel."""

    def __init__(self) -> None:
        self.policy = get_policy_engine()
        self.cache = get_cache()
        self.budget = get_budget_enforcer()
        self.fallback_chain = get_fallback_chain()
        self._litellm = None
        self._indus_local = None
        self._init_litellm()

    # ------------------------------------------------------------------
    # Backend configuration
    # ------------------------------------------------------------------
    def _init_litellm(self) -> None:
        try:
            import litellm
        except ImportError as e:
            raise ConfigurationError(
                "litellm is not installed. Install with: uv pip install litellm"
            ) from e
        self._litellm = litellm
        litellm.drop_params = True
        litellm.telemetry = False

    def _indus_checkpoint(self) -> str | None:
        """Return path to the native Indus checkpoint if it exists."""
        candidate = os.environ.get("INDUS_LLM_CHECKPOINT")
        if candidate and os.path.isfile(candidate):
            return candidate
        default = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "ik_indus_llm",
                "ik_indus_llm",
                "artifacts",
                "checkpoints",
                "pretrain",
                "indus_tiny_v0.3.0.pt",
            )
        )
        return default if os.path.isfile(default) else None

    def _has_external_provider(self) -> bool:
        keys = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GOOGLE_API_KEY",
            "AZURE_API_KEY",
            "COHERE_API_KEY",
            "MISTRAL_API_KEY",
            "GROQ_API_KEY",
            "TOGETHER_API_KEY",
            "FIREWORKS_API_KEY",
            "DEEPINFRA_API_KEY",
            "OPENROUTER_API_KEY",
            "LITELLM_API_KEY",
            # NVIDIA NIM via build.nvidia.com — uses nvapi-* keys
            "NVIDIA_NIM_API_KEY",
            "INDUS_LLM_API_KEY",
        ]
        return any(os.environ.get(k) for k in keys)

    def _has_native_provider(self) -> bool:
        return self._indus_checkpoint() is not None

    def is_configured(self) -> bool:
        """Return True if at least one backend is available."""
        return self._has_external_provider() or self._has_native_provider()

    def _load_indus_local(self):
        """Lazy-load the Indus local runtime."""
        if self._indus_local is not None:
            return self._indus_local
        checkpoint = self._indus_checkpoint()
        if not checkpoint:
            raise ConfigurationError("Indus local checkpoint is not configured")
        try:
            from ik_indus_llm.runtime import IndusLLMRuntime
        except ImportError as e:
            raise ConfigurationError(
                "ik_indus_llm/torch is required for the native Indus provider"
            ) from e
        self._indus_local = IndusLLMRuntime(checkpoint)
        return self._indus_local

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    async def complete(self, req: LLMRequest) -> LLMResponse:
        """Run a completion. Real backend call (no mocks)."""
        # Validate the request up-front. Failing early prevents noisy errors
        # from downstream providers (e.g. Anthropic's "non-system message"
        # complaint) and surfaces a clear, actionable error to the caller.
        if not req.messages:
            raise ValueError("LLMRequest.messages must not be empty")
        if not any(
            m.role != MessageRole.SYSTEM and (m.content or "").strip() for m in req.messages
        ):
            raise ValueError(
                "LLMRequest.messages must contain at least one non-system message "
                "with non-empty content"
            )
        if not self.is_configured():
            raise ConfigurationError(
                "No LLM backend available. Set one of: OPENAI_API_KEY, "
                "ANTHROPIC_API_KEY, GOOGLE_API_KEY, AZURE_API_KEY, "
                "COHERE_API_KEY, MISTRAL_API_KEY, GROQ_API_KEY, "
                "TOGETHER_API_KEY, FIREWORKS_API_KEY, DEEPINFRA_API_KEY, "
                "OPENROUTER_API_KEY, LITELLM_API_KEY, or install the "
                "native Indus checkpoint."
            )
        started = time.perf_counter()

        # 1. Cache lookup
        if not req.bypass_cache:
            cached = self.cache.get(req)
            if cached is not None:
                cached.latency_ms = int((time.perf_counter() - started) * 1000)
                return cached

        # 2. Policy
        capability_set = set(req.capability_requirements)
        candidate = self.policy.select(req.model_hint, capability_set, req.max_cost_cents)

        # 3. Estimate cost
        est_tokens = self._count_message_tokens(req) + (req.max_tokens or 256)
        est_cost = self._compute_cost_from_candidate(candidate, est_tokens // 2, est_tokens // 2)

        # 4. Budget reservation (transactional)
        reservation_id = self.budget.reserve_with_id(req.tenant_id, est_cost, est_tokens)

        # 5. Fallback chain call
        try:
            response = await self.fallback_chain.execute(
                req,
                primary=candidate.model_id,
                call_fn=self._call_one,
            )
        except Exception:
            self.budget.release(reservation_id)
            self.policy.mark_unhealthy(candidate.model_id)
            raise

        # 6. Reconcile against actual usage
        self.budget.reconcile(reservation_id, response.cost_cents, response.usage.total_tokens)

        # 7. Cache
        self.cache.set(req, response)

        # 8. Telemetry
        elapsed = int((time.perf_counter() - started) * 1000)
        logger.info(
            "llm_call model=%s tokens=%d cost_cents=%d latency_ms=%d cache_hit=%s fallback=%s",
            response.model_used,
            response.usage.total_tokens,
            response.cost_cents,
            elapsed,
            response.cache_hit,
            response.fallback_used,
        )
        return response

    async def _call_one(self, req: LLMRequest, model: str) -> LLMResponse:
        """Call one model. Dispatches to native or external."""
        if model.startswith(_NATIVE_PREFIX):
            return await self._call_indus_local(req, model)
        return await self._call_litellm(req, model)

    async def _call_litellm(self, req: LLMRequest, model: str) -> LLMResponse:
        """Call LiteLLM (external provider)."""
        msgs = [{"role": m.role.value, "content": m.content} for m in req.messages]
        kwargs: dict[str, Any] = {"model": model, "messages": msgs}
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        if req.max_tokens is not None:
            kwargs["max_tokens"] = req.max_tokens
        if req.top_p is not None:
            kwargs["top_p"] = req.top_p
        if req.stop:
            kwargs["stop"] = req.stop
        if req.tools:
            kwargs["tools"] = [{"type": "function", "function": t.model_dump()} for t in req.tools]
        if req.tool_choice:
            kwargs["tool_choice"] = req.tool_choice
        if req.response_format:
            kwargs["response_format"] = req.response_format.model_dump(exclude_none=True)
        if req.stream:
            raise NotImplementedError("streaming not yet supported; use ik_streaming (M4)")
        # Special handling for NVIDIA NIM: rewrite the model name so LiteLLM
        # knows to use the nvidia_nim provider, and inject the api_key/base_url.
        if model.startswith("nvidia/") or self._is_nim_model(model):
            nvidia_model = self._resolve_nim_model(model)
            kwargs["model"] = f"nvidia_nim/{nvidia_model}"
            kwargs["api_key"] = os.environ.get(
                "NVIDIA_NIM_API_KEY",
                os.environ.get("INDUS_LLM_API_KEY", ""),
            )
            base = os.environ.get(
                "NVIDIA_NIM_API_BASE",
                os.environ.get(
                    "INDUS_LLM_BASE_URL",
                    "https://integrate.api.nvidia.com/v1",
                ),
            )
            kwargs["api_base"] = base
        # If api_key was set in the env, let LiteLLM pick it up
        if "api_key" not in kwargs:
            for env_key in (
                "OPENAI_API_KEY",
                "ANTHROPIC_API_KEY",
                "GOOGLE_API_KEY",
                "NVIDIA_NIM_API_KEY",
                "INDUS_LLM_API_KEY",
            ):
                v = os.environ.get(env_key)
                if v:
                    kwargs["api_key"] = v
                    break
        resp = await self._litellm.acompletion(**kwargs)
        return self._litellm_to_response(resp, model)

    def _is_nim_model(self, model: str) -> bool:
        """Return True if `model` is a known NVIDIA NIM model identifier."""
        if not model:
            return False
        if "/" in model and model.split("/", 1)[0] == "nvidia":
            return True
        # Common Nemotron / NVIDIA-only model names
        nim_only_prefixes = ("nemotron-", "llama-3.1-nemotron", "llama-3.3-nemotron")
        return any(model.startswith(p) for p in nim_only_prefixes)

    def _resolve_nim_model(self, model: str) -> str:
        """Translate a model hint into NVIDIA NIM canonical form.

        Accepts:
          - "nvidia/nemotron-3-ultra-550b-a55b" → "nvidia/nemotron-3-ultra-550b-a55b"
          - "nemotron-3-ultra-550b-a55b"        → "nvidia/nemotron-3-ultra-550b-a55b"
        LiteLLM expects the form "nvidia_nim/<vendor/model-name>", so we
        keep the full nvidia/... prefix here.
        """
        if model.startswith("nvidia/"):
            return model
        return f"nvidia/{model}"

    async def _call_indus_local(self, req: LLMRequest, model: str) -> LLMResponse:
        """Call the local Indus model (real, no mock).

        Extracts the last user message, calls runtime.generate, wraps the
        result. Token counts are estimated via tiktoken; cost is 0 (local).
        """
        runtime = self._load_indus_local()
        last_user = next((m for m in reversed(req.messages) if m.role == MessageRole.USER), None)
        if last_user is None:
            raise ConfigurationError("Indus native provider requires a user message")
        prompt = last_user.content
        if not prompt.strip():
            raise ConfigurationError("Indus native provider requires non-empty prompt")
        max_new = req.max_tokens or 128
        temperature = req.temperature if req.temperature is not None else 0.7
        # Run the real model in a thread (PyTorch is CPU-bound)
        import asyncio

        out = await asyncio.to_thread(
            runtime.generate,
            prompt,
            max_new_tokens=max_new,
            temperature=temperature,
        )
        prompt_tokens = self._count_text_tokens(prompt, model)
        completion_tokens = self._count_text_tokens(out, model)
        return LLMResponse(
            model_used=model,
            provider="indus-local",
            content=out,
            role=MessageRole.ASSISTANT,
            finish_reason="stop",
            usage=LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            cost_cents=0,
            latency_ms=0,
        )

    def _litellm_to_response(self, resp: Any, model: str) -> LLMResponse:
        """Convert a LiteLLM response to our LLMResponse."""
        choice = resp.choices[0]
        msg = choice.message
        usage = resp.usage
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        tool_calls = None
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            import json as _json

            tool_calls = []
            for tc in msg.tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = _json.loads(args)
                    except _json.JSONDecodeError:
                        args = {"raw": args}
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args or {}))
        return LLMResponse(
            model_used=model,
            provider=model.split("/")[0] if "/" in model else "unknown",
            content=msg.content or "",
            role=MessageRole.ASSISTANT,
            tool_calls=tool_calls,
            finish_reason=choice.finish_reason,
            usage=LLMUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            cost_cents=self._compute_cost(model, prompt_tokens, completion_tokens),
            latency_ms=0,
        )

    async def embed(self, req: EmbedRequest) -> EmbedResponse:
        """Embed text via LiteLLM (real)."""
        if not self._has_external_provider():
            raise ConfigurationError("No external provider API key found; required for embeddings.")
        texts = [req.input] if isinstance(req.input, str) else req.input
        resp = await self._litellm.aembedding(model=req.model, input=texts)
        return EmbedResponse(
            embeddings=[d["embedding"] for d in resp["data"]],
            model=req.model,
            usage=LLMUsage(
                prompt_tokens=resp["usage"]["prompt_tokens"],
                completion_tokens=0,
                total_tokens=resp["usage"]["prompt_tokens"],
            ),
            cost_cents=0,
        )

    def _count_text_tokens(self, text: str, model: str) -> int:
        """Count tokens in text (real, tiktoken-backed)."""
        try:
            import tiktoken

            try:
                encoding = tiktoken.encoding_for_model(model)
            except KeyError:
                encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except ImportError:
            return max(1, len(text) // 4)

    def _count_message_tokens(self, req: LLMRequest) -> int:
        return (
            sum(
                self._count_text_tokens(m.content, req.model_hint or "gpt-4o-mini")
                for m in req.messages
            )
            + 4
        )

    def _compute_cost(self, model: str, prompt_tokens: int, completion_tokens: int) -> int:
        for c in self.policy.candidates:
            if c.model_id == model:
                return self._compute_cost_from_candidate(c, prompt_tokens, completion_tokens)
        return 0

    def _compute_cost_from_candidate(
        self, candidate: ModelCandidate, prompt_tokens: int, completion_tokens: int
    ) -> int:
        input_cost = (prompt_tokens / 1000) * candidate.cost_per_1k_input_cents
        output_cost = (completion_tokens / 1000) * candidate.cost_per_1k_output_cents
        return int(input_cost + output_cost)


_router: LLMRouter | None = None


def get_router() -> LLMRouter:
    """Return the cached router (singleton)."""
    global _router
    if _router is None:
        _router = LLMRouter()
    return _router
