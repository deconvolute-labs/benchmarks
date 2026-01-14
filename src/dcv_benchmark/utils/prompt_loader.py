from pathlib import Path

import yaml

from dcv_benchmark.constants import PROMPTS_DIR


def load_prompt_text(path: str, key: str) -> str:
    """
    Loads a specific prompt string from a YAML file.

    Args:
        path: The path to the file containing the prompt.
        key: The specific key in the file containing the prompt.

    Returns:
        The selected prompt as a string.
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
