import argparse
import sys

from dcv_benchmark.cli.data import register_data_commands
from dcv_benchmark.cli.experiments import register_experiment_commands
from dcv_benchmark.utils.logger import setup_logging


def main() -> None:
    setup_logging()

    parser = argparse.ArgumentParser(
        prog="dcv-benchmark",
        description=(
            "Deconvolute AI Benchmarking Tool\n"
            "Evaluate RAG security and robustness against adversarial attacks."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global arguments
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging globally."
    )

    # Create subparsers for the top-level commands (data, experiment)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Register modules
    register_data_commands(subparsers)
    register_experiment_commands(subparsers)

    # Parse arguments
    args = parser.parse_args()

    # Pre-initialize logger for CLI parsing phases if needed,
    # though individual handlers (run/data) often re-init to set specific levels.
    if args.command == "data":
        setup_logging(level="DEBUG" if args.debug else "INFO")

    # Execute the mapped function
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
