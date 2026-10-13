"""Provider chain: paid key first, then free Groq, then local Ollama. No network used."""
import pytest

from app.llm import client as C

KEYS = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "GEMINI_API_KEY", "GROQ_API_KEY"]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for k in KEYS:
        monkeypatch.delenv(k, raising=False)


def test_paid_key_wins(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("GROQ_API_KEY", "y")
    assert C.resolve_chain("openai", ollama_ok=True) == ["openai", "groq", "ollama"]


def test_no_paid_key_uses_groq(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "y")
    assert C.resolve_chain("openai", ollama_ok=False) == ["groq"]


def test_no_keys_uses_local_ollama():
    assert C.resolve_chain("openai", ollama_ok=True) == ["ollama"]


def test_nothing_configured():
    assert C.resolve_chain("openai", ollama_ok=False) == []


def test_explicit_free_provider_first(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    assert C.resolve_chain("ollama", ollama_ok=True)[0] == "ollama"


def test_falls_through_to_next_provider_on_error(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "y")
    monkeypatch.setattr(C, "_ollama_reachable", lambda: True)
    monkeypatch.setattr(C.LLMClient, "_prepare_ollama", lambda self: None)
    llm = C.LLMClient("openai")
    assert llm.chain == ["groq", "ollama"]

    def boom(self, *a):
        raise C.LLMError("down")

    monkeypatch.setattr(C.LLMClient, "_groq", boom)
    monkeypatch.setattr(C.LLMClient, "_ollama", lambda self, s, u, m: '{"ok": true}')
    assert llm.complete_json("s", "u", purpose="t", corpus_tokens=10) == {"ok": True}


def test_budget_guard_before_any_call():
    llm = C.LLMClient.__new__(C.LLMClient)
    llm.chain = ["ollama"]
    with pytest.raises(C.CorpusBudgetExceeded):
        llm.complete_json("s", "u", purpose="t", corpus_tokens=3001)
