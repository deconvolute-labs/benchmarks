import datetime
import re
from pathlib import Path

from dcv_benchmark.analytics.reporter import ReportGenerator
from dcv_benchmark.constants import (
    AVAILABLE_EVALUATORS,
    BASELINE_TARGET_KEYWORD,
    TIMESTAMP_FORMAT,
)
from dcv_benchmark.evaluators.base import BaseEvaluator
from dcv_benchmark.evaluators.canary import CanaryEvaluator
from dcv_benchmark.evaluators.keyword import KeywordEvaluator
from dcv_benchmark.evaluators.language import LanguageMismatchEvaluator
from dcv_benchmark.models.dataset import Dataset
from dcv_benchmark.models.evaluation import SecurityEvaluationResult
from dcv_benchmark.models.experiments_config import ExperimentConfig
from dcv_benchmark.models.responses import TargetResponse
from dcv_benchmark.models.traces import TraceItem
from dcv_benchmark.targets.basic_rag import BasicRAG
from dcv_benchmark.targets.basic_rag_guard import BasicRAGGuard
from dcv_benchmark.utils.dataset_loader import DatasetLoader
from dcv_benchmark.utils.logger import get_logger, print_run_summary

logger = get_logger(__name__)


class ExperimentRunner:
    def __init__(self, output_dir: str | Path = "results"):
        self.output_dir = Path(output_dir)

    def _validate_baseline_payload(self, dataset: Dataset) -> None:
        """Helper to validate dataset payload for Keyword evaluation."""
        attack_info = dataset.meta.attack_info
        if not attack_info:
            logger.warning(
                "Dataset metadata is missing 'attack_info'. Skipping validation."
            )
            return

        pattern = re.compile(rf"\b{re.escape(BASELINE_TARGET_KEYWORD)}\b")
        if not pattern.search(attack_info.payload):
            error_msg = (
                "Configuration Mismatch! \n"
                f"Evaluator expects: '{BASELINE_TARGET_KEYWORD}'\n"
                f"Dataset payload: '{attack_info.payload}'"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        logger.info(
            f"Confirmed: Dataset payload contains target '{BASELINE_TARGET_KEYWORD}'."
        )

    def run(
        self,
        experiment_config: ExperimentConfig,
        limit: int | None = None,
        debug_traces: bool = False,
    ) -> Path:
        """
        Executes the experiment loop.
        Returns the path to the run directory.

        Args:
            experiment_config: The experiment configuration data.
            limit: Optional integer to limit the number of samples to process.
            debug_traces: If True, include full content (query, context, response)
                in traces.

        Returns:
            The path to the dir which contains the results.
        """
        start_time = datetime.datetime.now()

        # Setup run directory
        run_id = start_time.strftime(TIMESTAMP_FORMAT)
        run_dir = self.output_dir / f"run_{run_id}"

        logger.info(f"Starting Run: {run_id}")

        # Initialize components
        # We assume the dataset path is relative to the project root
        logger.info("Initializing components ...")

        dataset_path_or_name = experiment_config.input.dataset_name
        if not dataset_path_or_name:
            # Fallback to default path:
            # workspace/datasets/built/{config.name}/dataset.json
            from dcv_benchmark.constants import BUILT_DATASETS_DIR

            fallback_path = BUILT_DATASETS_DIR / experiment_config.name / "dataset.json"
            if not fallback_path.exists():
                error_msg = (
                    "No dataset path provided and default path not found: "
                    f"{fallback_path}\n"
                    "Please provide 'input.dataset_name' in config or ensure the "
                    "default dataset exists."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)

            logger.info(f"No dataset provided. Using default path: {fallback_path}")
            dataset_path_or_name = str(fallback_path)

        dataset: Dataset = DatasetLoader(dataset_path_or_name).load()

        logger.info(f"Loaded dataset: {dataset.meta.name} (v{dataset.meta.version})")
        logger.info(f"Description: {dataset.meta.description}")

        # Select pipeline
        target: BasicRAG | BasicRAGGuard | None = None
        target_name = experiment_config.target.name
        if target_name == "basic_rag":
            target = BasicRAG(experiment_config.target)
        elif target_name == "basic_rag_guard":
            target = BasicRAGGuard(experiment_config.target)
        else:
            raise ValueError(
                f"Unsupported target: '{target_name}'. "
                "Available targets: 'basic_rag', 'basic_rag_guard'."
            )

        # Evaluator Setup
        eval_config = experiment_config.evaluator

        if eval_config is None:
            error_msg = (
                "Missing Configuration: No evaluator specified.\nYou must explicitly"
                " define an 'evaluator' section in your experiment YAML.\n"
                f"Available types: {', '.join(AVAILABLE_EVALUATORS)}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Initialize Evaluator
        evaluator: BaseEvaluator

        if eval_config.type == "canary":
            logger.info("Evaluator: Canary Defense Integrity")
            evaluator = CanaryEvaluator()

        elif eval_config.type == "keyword":
            # For Keyword evaluator, we still validate the dataset payload
            self._validate_baseline_payload(dataset)

            kw = eval_config.target_keyword or BASELINE_TARGET_KEYWORD
            logger.info(f"Evaluator: Keyword (Target: '{kw}')")
            evaluator = KeywordEvaluator(target_keyword=kw)

        elif eval_config.type == "language_mismatch":
            logger.info(
                "Evaluator: Language Mismatch "
                f"(Expected: {eval_config.expected_language})"
            )
            try:
                evaluator = LanguageMismatchEvaluator(
                    expected_language=eval_config.expected_language,
                    strict=eval_config.strict,
                )
            except ImportError as e:
                logger.error("Missing dependencies for Language Evaluator.")
                raise e
        else:
            # Should be caught by Pydantic, but good for safety
            raise ValueError(f"Unknown evaluator type: {eval_config.type}")

        # Prepare output file
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=True)

        traces_path = run_dir / "traces.jsonl"
        logger.info(f"Dataset: {len(dataset.samples)} samples. Output: {traces_path}")

        # Execution loop
        count = 0
        success_count = 0
        total_samples = len(dataset.samples)
        if limit:
            total_samples = min(total_samples, limit)

        # Log progress every 10% or every 1 sample if less than 10
        log_interval = max(1, total_samples // 10)

        with open(traces_path, "w", encoding="utf-8") as f:
            for sample in dataset.samples:
                if limit and count >= limit:
                    logger.info(f"Limit of {limit} reached.")
                    break

                if (count + 1) % log_interval == 0 or (count + 1) == total_samples:
                    pct = ((count + 1) / total_samples) * 100
                    logger.info(
                        f"Progress: {count + 1}/{total_samples} "
                        f"({pct:.0f}%) samples processed."
                    )

                logger.debug(
                    f"Processing Sample {count + 1}/{total_samples} "
                    f"(ID: {sample.id}) [{sample.sample_type}]"
                )

                if sample.sample_type == "attack":
                    logger.debug(f"  > Strategy: {sample.attack_strategy}")

                # Log step: Invocation
                logger.debug("  > Invoking Target...")

                try:
                    # Invoke Target. For Generator-only tests (Canary),
                    # we often force context from the sample
                    forced_context = (
                        [c.content for c in sample.context] if sample.context else None
                    )

                    t0 = datetime.datetime.now()

                    response: TargetResponse = target.invoke(
                        user_query=sample.query, forced_context=forced_context
                    )

                    latency = (datetime.datetime.now() - t0).total_seconds()

                    # Evaluate
                    logger.debug("  > Evaluating Response...")
                    eval_result: SecurityEvaluationResult = evaluator.evaluate(
                        response=response, sample=sample
                    )

                    logger.debug(
                        f"Eval result: {eval_result.model_dump_json(indent=2)}"
                    )

                    if eval_result.passed:
                        logger.debug(f"Sample {sample.id}: Passed!")
                        success_count += 1
                    else:
                        logger.debug(f"Sample {sample.id}: Failed!")

                    trace = TraceItem(
                        sample_id=sample.id,
                        sample_type=sample.sample_type,
                        attack_strategy=sample.attack_strategy,
                        user_query=sample.query if debug_traces else None,
                        response=response,
                        evaluation=eval_result,
                        latency_seconds=latency,
                    )

                    if not debug_traces:
                        # Clear sensitive/bulky content from trace
                        trace.response.content = None
                        trace.response.raw_content = None
                        trace.response.used_context = []

                    f.write(trace.model_dump_json() + "\n")
                    f.flush()

                except Exception as e:
                    logger.error(
                        f"❌ Error processing sample {sample.id}: {e}", exc_info=True
                    )

                count += 1

        # End of loop
        end_time = datetime.datetime.now()

        logger.info(f"✅ Run Complete. Processed {count} samples.")

        logger.info("Generating report...")
        reporter = ReportGenerator(run_dir)
        reporter.generate(
            config=experiment_config, start_time=start_time, end_time=end_time
        )

        print_run_summary(
            total=count,
            success=success_count,
            duration=end_time - start_time,
            artifacts_path=str(run_dir),
        )

        return run_dir
