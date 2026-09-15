from __future__ import annotations

import os

_IN_ACTIONS = os.environ.get("GITHUB_ACTIONS") == "true"


def log(level: str, message: str) -> None:
    if _IN_ACTIONS:
        print(f"::{level}::{message}", flush=True)
    else:
        print(f"[{level}] {message}", flush=True)


def info(message: str) -> None:
    print(message, flush=True)


def set_output(name: str, value: str) -> None:
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as file:
            file.write(f"{name}={value}\n")
