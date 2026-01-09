from pathlib import Path

import yaml


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
        raise FileNotFoundError(f"Prompt file not found: {path}")

    with open(file_path, encoding="utf-8") as f:
        if file_path.suffix in [".yaml", ".yml"]:
            data: dict[str, str] = yaml.safe_load(f)
        else:
            raise ValueError("Prompt file must be .yaml")

    if key not in data:
        raise KeyError(f"Key '{key}' not found in {path}")

    return data[key]
