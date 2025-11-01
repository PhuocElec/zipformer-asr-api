from __future__ import annotations
import inspect
import json
from pathlib import Path
from typing import Any
from app.core.settings import Settings

SAMPLE_PATH = Path(".env.example")
OVERWRITE = True

SENSITIVE_SUFFIXES = ("_TOKEN", "_SECRET", "_PASSWORD", "_PASS")
SENSITIVE_EXACT = {"API_KEY", "AUTH_TOKEN", "BEARER_TOKEN"}

def is_sensitive(env_key: str) -> bool:
    u = env_key.upper()
    return u in SENSITIVE_EXACT or any(u.endswith(suf) for suf in SENSITIVE_SUFFIXES)

def _to_env_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float, str)):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return json.dumps(list(value), ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return str(value)

def generate_env_example(filepath: Path = SAMPLE_PATH, overwrite: bool = OVERWRITE) -> None:
    fields = Settings.model_fields
    lines: list[str] = []

    for name, f in fields.items():
        env_key = name
        required = f.is_required()
        default = None if required else f.default

        value = "" if required else _to_env_value(default)

        if is_sensitive(env_key) and default is None:
            value = ""

        lines.append(f"{env_key}={value}")

    if filepath.exists() and not overwrite:
        raise FileExistsError(f"{filepath} already exists. Enable OVERWRITE to replace it.")
    filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Generated {filepath} with {len(fields)} variables.")

if __name__ == "__main__":
    assert inspect.isclass(Settings), "Settings class not found"
    generate_env_example()
