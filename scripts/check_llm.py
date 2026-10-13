"""Shows which LLM provider the service will use and makes one tiny test call.
Usage: docker compose run --rm app python scripts/check_llm.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.llm.client import LLMClient  # noqa: E402
from app.logging_setup import setup_logging  # noqa: E402

setup_logging("./logs/check_llm.log")
llm = LLMClient()
print("provider chain:", llm.chain or "NONE - set a key or start Ollama")
if "ollama" in llm.chain:
    print("preparing Ollama model (first time downloads it, a few minutes)...")
    llm._prepare_ollama()
if llm.chain:
    print("model:", llm.model_for(llm.chain[0]))
    print(llm.complete_json('Reply with JSON {"ok": true}.', "ping", purpose="check", corpus_tokens=0))
