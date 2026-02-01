import datetime
from pathlib import Path

from dcv_benchmark.analytics.calculators.security import SecurityMetricsCalculator
from dcv_benchmark.analytics.reporter import ReportGenerator
from dcv_benchmark.constants import TIMESTAMP_FORMAT
from dcv_benchmark.core.factories import (
    create_experiment_evaluators,
    create_target,
    load_dataset,
)
from dcv_benchmark.models.config.experiment import ExperimentConfig
from dcv_benchmark.models.responses import TargetResponse
from dcv_benchmark.models.traces import TraceItem
from dcv_benchmark.utils.logger import (
    ExperimentProgressLogger,
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
        start_time = datetime.datetime.now()
        run_name = (
            f"{experiment_config.name}_{experiment_config.version.replace('.', '-')}_"
            f"{start_time.strftime(TIMESTAMP_FORMAT)}"
        )
        run_dir = self.output_dir / run_name

        print_experiment_header(experiment_config.model_dump())
        logger.info(f"Starting Run: {run_name}")
        logger.info("Initializing components ...")

        # Load Dataset
        dataset = load_dataset(experiment_config)
        print_dataset_header(dataset.meta)

        # Create Target
        target = create_target(experiment_config)

        # Create Evaluators (Strict Auto-Config)
        evaluators = create_experiment_evaluators(experiment_config, target, dataset)

        if not evaluators:
            logger.warning(
                "⚠️ No evaluators were created! No metrics will be generated."
            )

        # Prepare output
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=True)
        traces_path = run_dir / "traces.jsonl"
        logger.info(f"Dataset: {len(dataset.samples)} samples. Saving traces to:")
        logger.info(f"{traces_path}")

        # Execution loop
        count = 0
        success_count = 0

        total_samples = len(dataset.samples)
        if limit:
            total_samples = min(total_samples, limit)

        # Initialize Progress Logger
        progress_logger = ExperimentProgressLogger(total_samples)
        progress_logger.start()

        with open(traces_path, "w", encoding="utf-8") as f:
            for sample in dataset.samples:
                if limit and count >= limit:
                    logger.info(f"Limit of {limit} reached.")
                    break

                # Update Progress
                progress_logger.log_progress(count, success_count)

                logger.debug(
                    f"Processing Sample {count + 1}/{total_samples} "
                    f"(ID: {sample.id}) [{sample.sample_type}]"
                )

                try:
                    forced_context = (
                        [c.content for c in sample.context] if sample.context else None
                    )

                    t0 = datetime.datetime.now()

                    response: TargetResponse = target.invoke(
                        user_query=sample.query, forced_context=forced_context
                    )

                    latency = (datetime.datetime.now() - t0).total_seconds()

                    # Evaluation Loop
                    eval_results = {}
                    sample_passed_all = True

                    for eval_name, evaluator in evaluators.items():
                        # We pass the response. If content is "Blocked",
                        # evaluator handles it.
                        res = evaluator.evaluate(response=response, sample=sample)
                        eval_results[eval_name] = res
                        if not res.passed:
                            sample_passed_all = False

                    if sample_passed_all:
                        success_count += 1

                    trace = TraceItem(
                        sample_id=sample.id,
                        sample_type=sample.sample_type,
                        attack_strategy=sample.attack_strategy,
                        user_query=sample.query if debug_traces else None,
                        response=response,
                        evaluations=eval_results,
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
        duration = (end_time - start_time).total_seconds()

        # Quick Calculation for Summary
        # We perform a calculation here to display the stats immediately
        # The reporter will do it again for the full report, which is fine.
        calculator = SecurityMetricsCalculator()
        try:
            metrics = calculator.calculate(traces_path)
            print_run_summary(
                metrics=metrics.global_metrics,
                duration=duration,
                artifacts_path=str(run_dir),
            )
        except Exception as e:
            logger.warning(f"Could not print summary table: {e}")

        # Report generation (Full Artifacts)
        logger.info("Generating full report artifacts...")
        reporter = ReportGenerator(run_dir)
        reporter.generate(
            config=experiment_config, start_time=start_time, end_time=end_time
        )

        logger.info(f"Detailed results saved to: {run_dir}")

        return run_dir
