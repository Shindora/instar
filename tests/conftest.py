from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from instar.server import serve_in_thread as serve
from tests.fixtures.mock_upstream import MockUpstream

if TYPE_CHECKING:
    from collections.abc import Iterator


@pytest.fixture
def mock_upstream() -> MockUpstream:
    return MockUpstream()


@pytest.fixture
def upstream_url(mock_upstream: MockUpstream) -> Iterator[str]:
    with serve(mock_upstream.build_app()) as url:
        yield url


@pytest.fixture
def model_env(upstream_url: str, monkeypatch: pytest.MonkeyPatch) -> str:
    monkeypatch.setenv("OPENAI_BASE_URL", f"{upstream_url}/v1")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", upstream_url)
    monkeypatch.setenv("GOOGLE_GEMINI_BASE_URL", upstream_url)
    return upstream_url
