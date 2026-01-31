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
PROJECT_ROOT = Path(__file__).parent.parent.parent
WORKSPACE_DIR = PROJECT_ROOT / "workspace"
DATASETS_DIR = WORKSPACE_DIR / "datasets"
EXPERIMENTS_DIR = WORKSPACE_DIR / "experiments"
PROMPTS_DIR = WORKSPACE_DIR / "prompts"

# Sub-directories
RAW_DATASETS_DIR = DATASETS_DIR / "raw"
BUILT_DATASETS_DIR = DATASETS_DIR / "built"
CORPUS_DIR = RAW_DATASETS_DIR


# Vulnerability Types
VULNERABILITY_TYPE_DOS = "denial_of_service"
VULNERABILITY_TYPE_INTEGRITY = "integrity_violation"
VULNERABILITY_TYPE_PAYLOAD_SPLITTING = "payload_splitting"

# Evaluators
AVAILABLE_EVALUATORS = ["canary", "keyword", "language_mismatch"]
