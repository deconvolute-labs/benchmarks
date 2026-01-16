import argparse

from dcv_benchmark.cli.data import register_data_cli
from dcv_benchmark.cli.run import register_run_cli
from dcv_benchmark.utils.logger import setup_logger


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deconvolute Benchmark: Security Evaluation"
    )

    # Global arguments
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging globally."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # Register subcommands
    register_run_cli(subparsers)
    register_data_cli(subparsers)

    args = parser.parse_args()

    # Pre-initialize logger for CLI parsing phases if needed,
    # though individual handlers (run/data) often re-init to set specific levels.
    if args.command == "data":
        setup_logger(level="DEBUG" if args.debug else "INFO")

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
