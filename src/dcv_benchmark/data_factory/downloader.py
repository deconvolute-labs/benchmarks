import logging
from pathlib import Path

import httpx

# Import the logic from the new squad module
from dcv_benchmark.data_factory.squad import fetch_squad_subset

logger = logging.getLogger(__name__)

BIPIA_GITHUB_RAW = "https://raw.githubusercontent.com/microsoft/BIPIA/main"


def _download_file(url: str, destination: Path) -> None:
    """Helper to download a single file with httpx."""
    logger.info(f"Downloading {url} -> {destination}")
    try:
        with httpx.stream("GET", url, follow_redirects=True) as response:
            response.raise_for_status()
            with open(destination, "wb") as f:
                for chunk in response.iter_bytes():
                    f.write(chunk)
        logger.info(f"Saved to {destination}")
    except httpx.HTTPError as e:
        logger.error(f"HTTP Error downloading {url}: {e}")
        if destination.exists():
            destination.unlink()
        raise


def download_squad(output_dir: Path) -> None:
    """
    Downloads and processes the SQuAD subset.
    Uses the 'datasets' library logic defined in squad.py.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # We use the default 300 count from your script
    # This will generate 'squad_subset_300.json' in the output_dir
    logger.info("Starting SQuAD processing (HuggingFace datasets)...")
    fetch_squad_subset(output_dir, count=300)


def download_bipia(output_dir: Path) -> None:
    """
    Downloads BIPIA test sets.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    files_to_fetch = {
        "benchmark/email/test.jsonl": "test_email.jsonl",
        "benchmark/code/test.jsonl": "test_code.jsonl",
        "benchmark/text_attack_test.json": "attacks_text.json",
        "benchmark/code_attack_test.json": "attacks_code.json",
    }

    logger.info("Note: BIPIA is subject to the MIT License (Microsoft).")

    for remote_path, local_name in files_to_fetch.items():
        url = f"{BIPIA_GITHUB_RAW}/{remote_path}"
        target = output_dir / local_name

        if target.exists():
            logger.info(f"Skipping {local_name} (exists)")
            continue

        _download_file(url, target)
