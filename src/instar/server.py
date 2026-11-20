from __future__ import annotations

import socket
import threading
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING

import uvicorn

if TYPE_CHECKING:
    from collections.abc import Generator

    from starlette.applications import Starlette

KEEP_ALIVE_S = 120


@contextmanager
def serve_in_thread(app: Starlette, host: str = "127.0.0.1") -> Generator[str]:
    """Serve an app on a free port from a background thread and yield its base URL.

    A thread keeps uvicorn away from the caller's signal handlers and event loop. Leaving the
    block waits for in-flight requests, then runs the app's lifespan shutdown.
    """
    sock = socket.socket()
    sock.bind((host, 0))
    port = sock.getsockname()[1]
    config = uvicorn.Config(
        app, timeout_keep_alive=KEEP_ALIVE_S, log_level="warning", access_log=False
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    thread.start()
    try:
        while not server.started:
            if not thread.is_alive():
                raise RuntimeError("server thread exited before it started")
            time.sleep(0.005)
        yield f"http://{host}:{port}"
    finally:
        server.should_exit = True
        thread.join()
        sock.close()
