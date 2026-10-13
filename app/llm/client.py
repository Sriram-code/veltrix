"""JSON-mode LLM client with an automatic provider chain. Temperature 0.

Order:
  1. LLM_PROVIDER, if its key/URL is configured
  2. any paid provider with a key: openai, anthropic, gemini
  3. groq   - free tier serving open-weight models (GROQ_API_KEY, free sign-up)
  4. ollama - fully local open-source model (reachable OLLAMA_BASE_URL, no key)
If a call fails, the next provider is tried (up to LLM_MAX_ATTEMPTS).
Every call logs the number of corpus tokens it carries."""
from __future__ import annotations

import json
import os
import re
import threading
import time

import httpx

from app.config import HARD_CORPUS_TOKEN_LIMIT, settings
from app.logging_setup import log

DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
    "gemini": "gemini-2.0-flash",
    "groq": "openai/gpt-oss-120b",
    "ollama": "qwen2.5:3b",
}
MODEL_ENV = {"groq": "GROQ_MODEL", "ollama": "OLLAMA_MODEL"}
ALL = ("openai", "anthropic", "gemini", "groq", "ollama")


class LLMError(RuntimeError):
    pass


class CorpusBudgetExceeded(RuntimeError):
    pass


def _key(provider: str) -> str:
    return {
        "openai": os.getenv("OPENAI_API_KEY", ""),
        "anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
        "gemini": os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY", ""),
        "groq": os.getenv("GROQ_API_KEY", ""),
    }.get(provider, "").strip()


def _ollama_url() -> str:
    return os.getenv("OLLAMA_BASE_URL", "").strip().rstrip("/")


def _ollama_reachable() -> bool:
    if not _ollama_url():
        return False
    try:
        return httpx.get(f"{_ollama_url()}/api/tags", timeout=3).status_code == 200
    except Exception:
        return False


def resolve_chain(explicit: str | None = None, ollama_ok: bool | None = None) -> list[str]:
    """Ordered list of usable providers."""
    explicit = (explicit if explicit is not None else settings.llm_provider or "").lower().strip()
    if explicit == "google":
        explicit = "gemini"
    order = ([explicit] if explicit in ALL else []) + [p for p in ALL if p != explicit]
    if ollama_ok is None:
        ollama_ok = _ollama_reachable()
    return [p for p in order if (ollama_ok if p == "ollama" else bool(_key(p)))]


def _parse_json(text: str) -> dict:
    m = re.search(r"\{.*\}", (text or "").strip(), re.S)
    if not m:
        raise LLMError("no JSON in LLM output")
    return json.loads(m.group(0))


class LLMClient:
    def __init__(self, provider: str | None = None, model: str | None = None):
        self._explicit = provider
        self._model_override = model
        self.max_attempts = int(os.getenv("LLM_MAX_ATTEMPTS", "2"))
        self.chain = resolve_chain(provider)
        log("llm_chain", chain=self.chain or ["none"])
        if not self.chain:
            log("llm_unconfigured", hint="set a paid key, GROQ_API_KEY, or start Ollama (see README)")
        if "ollama" in self.chain:
            threading.Thread(target=self._prepare_ollama, daemon=True).start()

    @property
    def provider(self) -> str:
        return self.chain[0] if self.chain else "none"

    def model_for(self, provider: str) -> str:
        explicit = (self._explicit or settings.llm_provider or "").lower()
        if provider == explicit and (self._model_override or settings.llm_model):
            return self._model_override or settings.llm_model
        return os.getenv(MODEL_ENV.get(provider, "_none_"), "") or DEFAULT_MODELS[provider]

    def complete_json(self, system: str, user: str, *, purpose: str, corpus_tokens: int,
                      request_id: str = "") -> dict:
        if corpus_tokens > HARD_CORPUS_TOKEN_LIMIT:
            raise CorpusBudgetExceeded(corpus_tokens)
        if not self.chain:
            raise LLMError("no LLM provider configured")
        last: Exception | None = None
        for provider in self.chain[: max(1, self.max_attempts)]:
            t0, ok = time.time(), False
            try:
                out = _parse_json(getattr(self, f"_{provider}")(system, user, self.model_for(provider)))
                ok = True
                return out
            except Exception as e:  # fall through to the next provider
                last = e
            finally:
                log("llm_call", request_id=request_id, purpose=purpose, provider=provider,
                    model=self.model_for(provider), corpus_tokens=corpus_tokens, ok=ok,
                    ms=int((time.time() - t0) * 1000))
        raise LLMError(f"all providers failed: {type(last).__name__}")

    # ---------------------------------------------------------------- transport
    def _post(self, provider, url, headers, body, timeout=None):
        r = httpx.post(url, headers=headers, json=body, timeout=timeout or settings.llm_timeout_s)
        if r.status_code >= 400:
            raise LLMError(f"{provider} HTTP {r.status_code}")
        return r.json()

    def _openai_compatible(self, provider, base, system, user, model):
        data = self._post(provider, f"{base}/chat/completions", {"Authorization": f"Bearer {_key(provider)}"},
                          {"model": model, "temperature": 0, "seed": 7, "response_format": {"type": "json_object"},
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": user}]})
        return data["choices"][0]["message"]["content"]

    # ---------------------------------------------------------------- providers
    def _openai(self, system, user, model):
        return self._openai_compatible("openai", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
                                       system, user, model)

    def _groq(self, system, user, model):
        return self._openai_compatible("groq", os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
                                       system, user, model)

    def _anthropic(self, system, user, model):
        data = self._post("anthropic", "https://api.anthropic.com/v1/messages",
                          {"x-api-key": _key("anthropic"), "anthropic-version": "2023-06-01"},
                          {"model": model, "max_tokens": 1200, "temperature": 0, "system": system,
                           "messages": [{"role": "user",
                                         "content": user + "\n\nRespond with a single JSON object only."}]})
        return "".join(b.get("text", "") for b in data.get("content", []))

    def _gemini(self, system, user, model):
        data = self._post("gemini", f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                          {"x-goog-api-key": _key("gemini")},
                          {"systemInstruction": {"parts": [{"text": system}]},
                           "contents": [{"role": "user", "parts": [{"text": user}]}],
                           "generationConfig": {"temperature": 0, "responseMimeType": "application/json"}})
        return data["candidates"][0]["content"]["parts"][0]["text"]

    def _ollama(self, system, user, model):
        data = self._post("ollama", f"{_ollama_url()}/api/chat", {},
                          {"model": model, "stream": False, "format": "json", "keep_alive": "30m",
                           "options": {"temperature": 0, "seed": 7,
                                       "num_ctx": int(os.getenv("OLLAMA_NUM_CTX", "8192"))},
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": user}]},
                          timeout=float(os.getenv("OLLAMA_TIMEOUT_S", "28")))
        return data["message"]["content"]

    def _prepare_ollama(self):
        """Pull the model if missing, then load it so the first chat isn't slow."""
        model = self.model_for("ollama")
        try:
            tags = httpx.get(f"{_ollama_url()}/api/tags", timeout=5).json()
            have = {m.get("name", "").removesuffix(":latest") for m in tags.get("models", [])}
            if model.removesuffix(":latest") not in have:
                log("ollama_pull_start", model=model)
                httpx.post(f"{_ollama_url()}/api/pull", json={"model": model, "stream": False}, timeout=1800)
            httpx.post(f"{_ollama_url()}/api/generate", timeout=300,
                       json={"model": model, "prompt": "ok", "keep_alive": "30m", "options": {"num_predict": 1}})
            log("ollama_ready", model=model)
        except Exception as e:
            log("ollama_prepare_failed", model=model, err=type(e).__name__)
