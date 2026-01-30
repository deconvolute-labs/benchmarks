import argparse
import sys

from dcv_benchmark.cli.commands.data import register_data_commands
from dcv_benchmark.cli.commands.experiment import register_experiment_commands
from dcv_benchmark.utils.logger import setup_logging


def main() -> None:
    # Create a parent parser for global arguments
    # This parser is NOT used to parse the final args directly, but to be inherited.
    # add_help=False is crucial for parent parsers to avoid conflict with -h/--help
    # in children.
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging globally."
    )

    # 2. Setup the main parser
    parser = argparse.ArgumentParser(
        prog="dcv-benchmark",
        description=(
            "Deconvolute AI Benchmarking Tool\n"
            "Evaluate RAG security and robustness against adversarial attacks."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[parent_parser],  # Allow --debug at root level too
    )

    # Create subparsers for the top-level commands (data, experiment)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Register modules, passing the parent_parser to inherit flags
    register_data_commands(subparsers, parent_parser)
    register_experiment_commands(subparsers, parent_parser)

    # Parse arguments
    args = parser.parse_args()

    # Setup logging based on the parsed debug flag
    # Since --debug is in the parent parser, it should be available in args regardless
    # of where it was placed
    # (as long as the subparser inherited it).
    setup_logging(level="DEBUG" if args.debug else "INFO")

    # Execute the mapped function
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
