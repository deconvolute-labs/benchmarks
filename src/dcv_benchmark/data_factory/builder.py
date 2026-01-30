import random
from pathlib import Path
from typing import Literal

from dcv_benchmark.data_factory.base import BaseCorpusLoader, BaseInjector
from dcv_benchmark.data_factory.retrieval import EphemeralRetriever
from dcv_benchmark.models.data_factory import DataFactoryConfig
from dcv_benchmark.models.dataset import (
    AttackInfo,
    BenchmarkSample,
    ContextChunk,
    CorpusInfo,
    Dataset,
    DatasetMeta,
)
from dcv_benchmark.utils.logger import get_logger

logger = get_logger(__name__)


class DatasetBuilder:
    """
    Orchestrates the creation of a RAG Security Dataset.

    Workflow:
    1. Load raw samples (Query + Gold Chunk) from a corpus.
    2. Index all Gold Chunks into an ephemeral vector store.
    3. For each sample:
       a. Retrieve 'k' similar chunks (Distractors) from the store.
       b. Ensure the Gold Chunk is included in the context.
       c. Decide if the sample should be an Attack (based on attack_rate).
       d. If Attack: Inject malicious payload into one context chunk.
    4. Compile final Dataset object.
    """

    def __init__(
        self,
        loader: BaseCorpusLoader,
        injector: BaseInjector,
        config: DataFactoryConfig,
    ):
        self.loader = loader
        self.injector = injector
        self.config = config
        self.retriever = EphemeralRetriever()

    def build(self) -> Dataset:
        """
        Executes the build pipeline and returns the constructed Dataset.
        """
        raw_samples = self.loader.load(self.config.source_file)
        if not raw_samples:
            raise ValueError("Loader returned no samples. Cannot build dataset.")

        # Index Knowledge Base
        # Assumption: Knowledge Base is the collection of all source docs in the corpus
        # Map Doc Content -> Doc ID to easily track them.
        kb_docs = [s.source_document for s in raw_samples]
        kb_ids = [s.id for s in raw_samples]

        self.retriever.index(documents=kb_docs, ids=kb_ids)

        # Generate Samples
        benchmark_samples = []
        attack_count = 0

        # Use a fixed seed for reproducibility of the Attack/Benign split
        rng = random.Random(42)  # noqa: S311

        logger.info(f"Generating samples with attack rate {self.config.attack_rate}...")

        for raw in raw_samples:
            # A. Retrieve Contexts (Distractors + Candidates)
            # We fetch k chunks. Note: Gold chunk might be in here if retrieval is good.
            retrieved_texts = self.retriever.query(
                query_text=raw.query, k=self.config.retrieval_k
            )

            # B. Enforce Gold Chunk Presence (The "Oracle" Property)
            # In this benchmark, we want to test Integrity, not Recall.
            # So we MUST ensure the correct answer context is present.
            final_context_texts = retrieved_texts
            if raw.source_document not in final_context_texts:
                # Replace the last retrieved chunk with the Gold Chunk
                if final_context_texts:
                    final_context_texts.pop()
                final_context_texts.insert(0, raw.source_document)

            # C. Determine Sample Type (Attack vs Benign)
            is_attack = rng.random() < self.config.attack_rate
            sample_type: Literal["attack", "benign"] = (
                "attack" if is_attack else "benign"
            )
            strategy = self.config.attack_strategy if is_attack else "none"

            # D. Build Context Objects (and Inject if needed)
            context_objs = []

            # If attack, pick ONE chunk to poison
            target_chunk_idx = -1
            if is_attack:
                attack_count += 1
                target_chunk_idx = rng.randint(0, len(final_context_texts) - 1)

            for i, text in enumerate(final_context_texts):
                is_malicious_chunk = i == target_chunk_idx
                content = text

                if is_malicious_chunk:
                    content = self.injector.inject(
                        clean_text=text,
                        payload=self.config.attack_payload,
                        strategy=strategy,
                    )

                context_objs.append(
                    ContextChunk(
                        id=f"{raw.id}_chk_{i}",
                        content=content,
                        is_malicious=is_malicious_chunk,
                        metadata={
                            "origin": "retrieved"
                            if text != raw.source_document
                            else "gold"
                        },
                    )
                )

            # E. Assemble Sample
            benchmark_samples.append(
                BenchmarkSample(
                    id=raw.id,
                    query=raw.query,
                    sample_type=sample_type,
                    attack_strategy=strategy,
                    reference_answer=raw.reference_answer,
                    context=context_objs,
                )
            )

        logger.info(
            "Generation complete. Total: "
            f"{len(benchmark_samples)}, Attacks: {attack_count}"
        )

        # 4. Construct Final Dataset
        return Dataset(
            meta=DatasetMeta(
                name=self.config.dataset_name,
                version=self.config.version,
                description=self.config.description,
                author=self.config.author,
                corpus_info=CorpusInfo(
                    source_files=[self.config.source_file],
                    ingestion_params={
                        "retrieval_k": self.config.retrieval_k,
                        "embedding_provider": self.config.embedding_provider,
                        "embedding_model": self.config.embedding_model,
                    },
                ),
                attack_info=AttackInfo(
                    strategy=self.config.attack_strategy,
                    rate=self.config.attack_rate,
                    payload=self.config.attack_payload,
                    configuration={
                        "truncate": self.config.truncate_overflow,
                        "reps": self.config.flooding_repetitions,
                    },
                ),
            ),
            samples=benchmark_samples,
        )

    def save(self, dataset: Dataset, output_path: str | Path) -> None:
        """Helper to save the dataset to JSON."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            f.write(dataset.model_dump_json(indent=2))
        logger.info(f"Dataset saved to {path}")
