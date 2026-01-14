from pathlib import Path
from typing import Any

from dcv_benchmark.constants import PROJECT_ROOT


def _sanitize_config_paths(data: Any) -> Any:
    """
    Recursively converts absolute paths inside the config dict to relative paths
    based on the PROJECT_ROOT.
    """
    if isinstance(data, dict):
        return {k: _sanitize_config_paths(v) for k, v in data.items()}
    if isinstance(data, list):
        return [_sanitize_config_paths(v) for v in data]

    if isinstance(data, str):
        # If it looks like an absolute path, make it relative
        try:
            path = Path(data)
            if path.is_absolute() and path.is_relative_to(PROJECT_ROOT):
                return str(path.relative_to(PROJECT_ROOT))
        except (ValueError, OSError):
            # Not a path or can't be relativized
            pass

    return data
