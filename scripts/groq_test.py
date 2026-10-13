"""Shows the exact Groq error and which models your key can use."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app.config  # noqa: E402,F401  (loads .env)
import httpx  # noqa: E402

key = os.getenv("GROQ_API_KEY", "")
print("key loaded:", bool(key))
print("starts with gsk_:", key.startswith("gsk_"))
print("length:", len(key))

base = "https://api.groq.com/openai/v1"
headers = {"Authorization": "Bearer " + key}

r = httpx.get(base + "/models", headers=headers, timeout=15)
print()
print("GET /models ->", r.status_code)
if r.status_code == 200:
    ids = sorted(m["id"] for m in r.json()["data"])
    print("models you can use:")
    for i in ids:
        print("   ", i)
else:
    print(r.text[:500])

model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
body = {
    "model": model,
    "temperature": 0,
    "response_format": {"type": "json_object"},
    "messages": [
        {"role": "system", "content": "Reply with JSON."},
        {"role": "user", "content": "Return ok as true in JSON."},
    ],
}
r = httpx.post(base + "/chat/completions", headers=headers,
               json=body, timeout=20)
print()
print("POST chat (" + model + ") ->", r.status_code)
print(r.text[:500])