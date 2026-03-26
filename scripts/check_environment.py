from __future__ import annotations

import importlib
import sys


REQUIRED_MODULES = [
    "fastapi",
    "pydantic",
    "pydantic_settings",
    "sqlalchemy",
]

OPTIONAL_MODULES = [
    "openai",
    "langchain_openai",
    "langchain_text_splitters",
    "mem0",
]


def probe(modules: list[str]) -> list[str]:
    missing: list[str] = []
    for module_name in modules:
        try:
            importlib.import_module(module_name)
        except Exception:
            missing.append(module_name)
    return missing


if __name__ == "__main__":
    print(f"Python: {sys.version}")

    missing_required = probe(REQUIRED_MODULES)
    missing_optional = probe(OPTIONAL_MODULES)

    if missing_required:
        print("Missing required modules:", ", ".join(missing_required))
    else:
        print("Required modules: OK")

    if missing_optional:
        print("Missing optional modules:", ", ".join(missing_optional))
    else:
        print("Optional modules: OK")

    if missing_required:
        raise SystemExit(1)
