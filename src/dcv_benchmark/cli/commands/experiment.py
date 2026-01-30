import argparse
from typing import Any

from dcv_benchmark.cli.experiments import run_experiment


def handle_run(args: argparse.Namespace) -> None:
    """
    Handles the 'experiment run' command.
    """
    run_experiment(
        config_path_str=args.config, limit=args.limit, debug_traces=args.debug_traces
    )


def register_experiment_commands(subparsers: Any, parent_parser: Any) -> None:
    """Registers the 'experiment' subcommand group."""
    exp_parser = subparsers.add_parser(
        "experiment", help="Experiment execution tools", parents=[parent_parser]
    )
    exp_subs = exp_parser.add_subparsers(dest="experiment_command", required=True)

    # Run Command
    run_parser = exp_subs.add_parser(
        "run", help="Execute an experiment from a config file", parents=[parent_parser]
    )
    run_parser.add_argument(
        "config", help="Path to the experiment.yaml configuration file"
    )
    run_parser.add_argument(
        "--limit", type=int, help="Limit execution to N samples (for debugging)"
    )
    run_parser.add_argument(
        "--debug-traces",
        action="store_true",
        help="Enable verbose logging and full-text traces",
    )
    run_parser.set_defaults(func=handle_run)
