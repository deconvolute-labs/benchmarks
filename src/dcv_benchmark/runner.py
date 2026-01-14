import datetime
import re
from pathlib import Path

from dcv_benchmark.analytics.reporter import ReportGenerator
from dcv_benchmark.constants import BASELINE_TARGET_KEYWORD, TIMESTAMP_FORMAT
from dcv_benchmark.evaluators.canary import CanaryEvaluator
from dcv_benchmark.evaluators.keyword import KeywordEvaluator
from dcv_benchmark.models.dataset import Dataset
from dcv_benchmark.models.evaluation import SecurityEvaluationResult
from dcv_benchmark.models.experiments_config import ExperimentConfig
from dcv_benchmark.models.responses import TargetResponse
from dcv_benchmark.models.traces import TraceItem
from dcv_benchmark.targets.basic_rag import BasicRAG
from dcv_benchmark.utils.dataset_loader import DatasetLoader
from dcv_benchmark.utils.logger import get_logger, print_run_summary

logger = get_logger()


class ExperimentRunner:
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self, experiment_config: ExperimentConfig, limit: int | None = None
    ) -> Path:
        """
        Executes the experiment loop.
        Returns the path to the run directory.

        Args:
            experiment_config: The experiment configuration data.
            limit: Optional integer to limit the number of samples to process.

        Returns:
            The path to the dir which contains the results.
        """
        start_time = datetime.datetime.now()

        # Setup run directory
        run_id = start_time.strftime(TIMESTAMP_FORMAT)
        run_dir = self.output_dir / f"run_{run_id}"
        run_dir.mkdir()

        logger.info(f"Starting Run: {run_id}")

        # Initialize components
        # We assume the dataset path is relative to the project root
        logger.info("Initializing components...")
        dataset: Dataset = DatasetLoader(experiment_config.input.dataset_path).load()

        logger.info(f"Loaded dataset: {dataset.meta.name} (v{dataset.meta.version})")
        logger.info(f"Description: {dataset.meta.description}")
        target = BasicRAG(experiment_config.target)

        canary_layer = next(
            (
                layer
                for layer in experiment_config.target.defense.layers
                if layer.type == "canary"
            ),
            None,
        )
        canary_enabled = canary_layer.enabled if canary_layer else False

        if canary_enabled:
            logger.info("Defense Enabled: Using CanaryEvaluator.")
            evaluator: CanaryEvaluator | KeywordEvaluator = CanaryEvaluator()
        else:
            logger.info("Defense Disabled: Using KeywordEvaluator (Baseline Mode).")

            # Ensure the dataset actually contains the payload we are looking for.
            attack_info = dataset.meta.attack_info

            if not attack_info:
                # If no attack info exists, we can't verify the payload.
                # This might happen if using a legacy dataset or a purely benign one.
                logger.warning(
                    "Dataset metadata is missing 'attack_info'. "
                    "Skipping payload validation."
                )

            else:
                pattern = re.compile(rf"\b{re.escape(BASELINE_TARGET_KEYWORD)}\b")

                if not pattern.search(attack_info.payload):
                    # CRITICAL: The dataset uses a different payload
                    # (or a variation like ...SUCCESSFUL)
                    error_msg = (
                        "Configuration Mismatch! \n"
                        "The Evaluator expects the exact token: "
                        f"'{BASELINE_TARGET_KEYWORD}'\n"
                        f"But the dataset payload is: '{attack_info.payload}'\n"
                        "-> Please regenerate the dataset with the correct payload."
                    )
                    logger.error(error_msg)
                    raise ValueError(error_msg)

                logger.info(
                    "Confirmed: Dataset payload contains target "
                    f"'{BASELINE_TARGET_KEYWORD}'."
                )

            evaluator = KeywordEvaluator(target_keyword=BASELINE_TARGET_KEYWORD)

        # Prepare output file
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
                        user_query=sample.query,
                        response=response,
                        evaluation=eval_result,
                        latency_seconds=latency,
                    )

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
