from pathlib import Path

import yaml

from dcv_benchmark.constants import PROMPTS_DIR


def load_prompt_text(path: str, key: str) -> str:
    """
    Extracts a specific prompt template from a YAML configuration file.

    This helper allows prompts to be organized in YAML dictionaries. It first
    checks if the `path` exists as provided; if not, it attempts to resolve it
    relative to the global `PROMPTS_DIR`.

    Args:
        path (str): Path to the YAML file (e.g. "prompts/system_prompts.yaml").
        key (str): The specific key within the YAML file to retrieve
            (e.g. "default_rag").

    Returns:
        str: The raw prompt text.

    Raises:
        FileNotFoundError: If the file cannot be located.
        KeyError: If the requested key is missing from the file.
    """
    file_path = Path(path)

    if not file_path.exists():
        candidate = PROMPTS_DIR / file_path.name
        if candidate.exists():
            file_path = candidate
        else:
            raise FileNotFoundError(f"Prompt file not found: {path}")

    with open(file_path, encoding="utf-8") as f:
        # Allow .yaml or .yml
        if file_path.suffix not in [".yaml", ".yml"]:
            raise ValueError("Prompt file must be .yaml or .yml")

        data: dict[str, str] = yaml.safe_load(f)

    if key not in data:
        raise KeyError(f"Key '{key}' not found in {file_path}")

    return data[key]
