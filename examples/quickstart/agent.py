"""A tiny agent: one chat completion through any OpenAI-compatible API."""

import os

import httpx


def answer(prompt: str) -> str:
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    key = os.environ.get("OPENAI_API_KEY")
    response = httpx.post(
        f"{base_url}/chat/completions",
        headers={"authorization": f"Bearer {key}"} if key else {},
        json={
            "model": os.environ.get("QUICKSTART_MODEL", "gpt-4o-mini"),
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=120,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]
