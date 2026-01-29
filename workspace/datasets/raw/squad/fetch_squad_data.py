import json
import logging
import random
from pathlib import Path

# Requires: pip install datasets tqdm
from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
COUNT = 300
OUTPUT_DIR = Path("resources/corpus")
OUTPUT_FILE = OUTPUT_DIR / "squad_subset_300.json"
SEED = 42  # For reproducibility


def fetch_squad_subset(count: int = 300):
    """
    Fetches a randomized, diverse subset of SQuAD (v1.1) validation set.
    """
    logger.info("Loading SQuAD (validation split)...")

    # Load the FULL validation split (it's small enough to fit in memory ~10k items)
    # We turn off streaming to allow easy shuffling
    dataset = load_dataset("squad", split="validation", streaming=False)

    # Convert to list to shuffle
    all_samples = list(dataset)
    logger.info(
        f"Loaded {len(all_samples)} total samples. Shuffling with seed {SEED}..."
    )

    # Shuffle to mix topics (Biology, History, Sports, etc.)
    random.seed(SEED)
    random.shuffle(all_samples)

    clean_samples = []
    seen_contexts = set()

    # Select samples, preferring UNIQUE contexts to maximize diversity
    # If we run out of unique contexts, we allow duplicates (unlikely for 300 samples)
    with tqdm(total=count) as pbar:
        for row in all_samples:
            if len(clean_samples) >= count:
                break

            context = row["context"]

            # OPTIONAL: Dedup logic.
            # If we want strictly 300 different paragraphs, uncomment the next two lines.
            # if context in seen_contexts:
            #     continue

            seen_contexts.add(context)

            clean_samples.append(
                {
                    "id": row["id"],
                    "query": row["question"],
                    "reference_answer": row["answers"]["text"][0]
                    if row["answers"]["text"]
                    else None,
                    # In our pipeline, this Context becomes the "Gold Chunk"
                    "source_document": context,
                    "title": row["title"],
                }
            )
            pbar.update(1)

    # Save to disk
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "source": "huggingface/squad",
                    "split": "validation",
                    "strategy": "random_shuffled",
                    "seed": SEED,
                    "count": len(clean_samples),
                    "unique_topics_covered": len(
                        set(s["title"] for s in clean_samples)
                    ),
                    "description": "Diverse corpus of Question+Context pairs.",
                },
                "data": clean_samples,
            },
            f,
            indent=2,
        )

    logger.info(f"Successfully saved {len(clean_samples)} samples to {OUTPUT_FILE}")
    logger.info(f"Unique Topics covered: {len(set(s['title'] for s in clean_samples))}")


if __name__ == "__main__":
    fetch_squad_subset(COUNT)
