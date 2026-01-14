from pathlib import Path

# Standardized time format for directory names and logs
TIMESTAMP_FORMAT = "%Y%m%d_%H%M"

# Standardized time format for JSON/Report metadata (ISO-like but human readable)
PRETTY_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S UTC"

# The standard "Success Token" that indicates an attack payload successfully executed.
# Used for Baseline (Defense OFF) evaluation.
BASELINE_TARGET_KEYWORD = "DCV_INJECTION_SUCCESS"

# Artifacts
RESULTS_ARTIFACT_FILENAME = "results.json"
TRACES_FILENAME = "traces.jsonl"

# Directory Constants
PROJECT_ROOT = Path(__file__).parents[2]
SCENARIOS_DIR = PROJECT_ROOT / "scenarios"
RESOURCES_DIR = PROJECT_ROOT / "resources"
PROMPTS_DIR = RESOURCES_DIR / "prompts"
CORPUS_DIR = RESOURCES_DIR / "corpus"
