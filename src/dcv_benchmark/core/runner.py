import datetime
from pathlib import Path

from dcv_benchmark.analytics.reporter import ReportGenerator
from dcv_benchmark.constants import TIMESTAMP_FORMAT
from dcv_benchmark.core.factories import create_evaluator, create_target, load_dataset
from dcv_benchmark.models.config.experiment import ExperimentConfig
from dcv_benchmark.models.evaluation import (
    BaseEvaluationResult,
)
from dcv_benchmark.models.responses import TargetResponse
from dcv_benchmark.models.traces import TraceItem
from dcv_benchmark.utils.logger import (
    get_logger,
    print_dataset_header,
    print_experiment_header,
    print_run_summary,
)

logger = get_logger(__name__)


class ExperimentRunner:
    def __init__(self, output_dir: str | Path = "results"):
        self.output_dir = Path(output_dir)

    def run(
        self,
        experiment_config: ExperimentConfig,
        limit: int | None = None,
        debug_traces: bool = False,
    ) -> Path:
        """
        Executes the experiment loop.
        Returns the path to the run directory.
        """
        start_time = datetime.datetime.now()
        run_id = start_time.strftime(TIMESTAMP_FORMAT)
        run_dir = self.output_dir / f"run_{run_id}"

        print_experiment_header(experiment_config.model_dump())
        logger.info(f"Starting Run: {run_id}")
        logger.info("Initializing components ...")

        # 1. Load Dataset
        dataset = load_dataset(experiment_config)
        print_dataset_header(experiment_config.input.model_dump())

        # 2. Create Target
        target = create_target(experiment_config)

        # 3. Create Evaluator
        evaluator = create_evaluator(
            experiment_config.evaluator, target=target, dataset=dataset
        )

        # Prepare output
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

                logger.debug("  > Invoking Target...")

                try:
                    forced_context = (
                        [c.content for c in sample.context] if sample.context else None
                    )

                    t0 = datetime.datetime.now()

                    response: TargetResponse = target.invoke(
                        user_query=sample.query, forced_context=forced_context
                    )

                    latency = (datetime.datetime.now() - t0).total_seconds()

                    logger.debug("  > Evaluating Response...")
                    eval_result: BaseEvaluationResult = evaluator.evaluate(
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
