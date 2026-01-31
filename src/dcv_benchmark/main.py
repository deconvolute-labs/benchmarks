import argparse
import sys

from dcv_benchmark.cli.commands.data import register_data_commands
from dcv_benchmark.cli.commands.experiment import register_experiment_commands
from dcv_benchmark.utils.logger import setup_logging


def main() -> None:
    # Create a parent parser for global arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging globally."
    )

    # Setup the main parser
    parser = argparse.ArgumentParser(
        prog="dcv-benchmarks",
        description=(
            "Deconvolute Labs Benchmarking Tool\n"
            "Evaluate the Deconvolute SDK for RAG security and robustness against "
            "adversarial attacks."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parent_parser],  # Allow --debug at root level
    )

    # Create subparsers for the top-level commands
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Register modules, passing the parent_parser to inherit flags
    register_data_commands(subparsers, parent_parser)
    register_experiment_commands(subparsers, parent_parser)

    # Parse arguments
    args = parser.parse_args()

    # Setup logging based on the parsed debug flag
    setup_logging(level="DEBUG" if args.debug else "INFO")

    # Execute the mapped function
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
