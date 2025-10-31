from __future__ import annotations
import inspect
from pathlib import Path
from typing import Any
from app.core.settings import Settings

SAMPLE_PATH = Path(".env.example")
OVERWRITE = True

def _to_env_value(field_name: str, value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple, set)):
        return ",".join(str(x) for x in value)
    return str(value)

def generate_env_example(filepath: Path = SAMPLE_PATH, overwrite: bool = OVERWRITE) -> None:
    fields = Settings.model_fields
    lines: list[str] = []

    for name, f in fields.items():
        required = f.is_required()
        default = None if required else f.default
        value = "" if required else _to_env_value(name, default)
        if name.upper().endswith("TOKEN") or "KEY" in name.upper() or "SECRET" in name.upper():
            value = ""
        lines.append(f"{name}={value}")

    if filepath.exists() and not overwrite:
        raise FileExistsError(f"{filepath} already exists. Enable OVERWRITE to replace it.")
    filepath.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated {filepath} with {len(fields)} variables.")

if __name__ == "__main__":
    assert inspect.isclass(Settings), "Settings class not found"
    generate_env_example()
