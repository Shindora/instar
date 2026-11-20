from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping

DEFAULT_UPSTREAMS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com",
    "gemini": "https://generativelanguage.googleapis.com",
}
DEFAULT_MODEL_ENV: dict[str, str] = {
    "OPENAI_BASE_URL": "openai",
    "ANTHROPIC_BASE_URL": "anthropic",
    "GOOGLE_GEMINI_BASE_URL": "gemini",
}


def resolve_upstreams(
    declared: Mapping[str, str] | None = None,
    model_env: Mapping[str, str] | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Upstream name to base URL.

    A model variable already set where instar starts (a company gateway, say) replaces the
    default for its upstream, so the proxy chains through it. Declared upstreams win over both.
    """
    environ = os.environ if environ is None else environ
    upstreams = dict(DEFAULT_UPSTREAMS)
    for var, name in {**DEFAULT_MODEL_ENV, **(model_env or {})}.items():
        if environ.get(var):
            upstreams[name] = environ[var]
    return {**upstreams, **(declared or {})}


def model_url(proxy_url: str, session: str, upstream: str) -> str:
    return f"{proxy_url.rstrip('/')}/s/{session}/{upstream}"


def judge_url(proxy_url: str, session: str, upstream: str) -> str:
    return f"{proxy_url.rstrip('/')}/j/{session}/{upstream}"
