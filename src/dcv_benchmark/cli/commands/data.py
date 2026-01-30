import argparse
from typing import Any

from dcv_benchmark.cli.data import build_data, download_data


def handle_download(args: argparse.Namespace) -> None:
    """
    Handles the 'data download' command.
    """
    download_data(source=args.source, output_dir=args.output_dir)


def handle_build(args: argparse.Namespace) -> None:
    """
    Handles the 'data build' command.
    """
    build_data(config_path_str=args.config, name=args.name, overwrite=args.overwrite)


def register_data_commands(subparsers: Any, parent_parser: Any) -> None:
    """Registers the 'data' subcommand group."""
    data_parser = subparsers.add_parser(
        "data",
        help="Data Factory tools",
        parents=[parent_parser],  # Inherit global flags like --debug
        add_help=False,  # Let the parent parser handle help if needed, but usually we
        # want help here.
        # Wait, if we use parents, we usually don't suppress help unless we want to
        # avoid duplicates.
        # Standard pattern is parents=[parent], and we CAN have help.
    )
    # Re-enable help since we might have suppressed it above or want to ensure it shows
    # up.
    # Actually, if we use a parent parser with add_help=False for the global flags,
    # we should be fine.
    # The parent_parser passed in should likely be configured with add_help=False.

    data_subs = data_parser.add_subparsers(dest="data_command", required=True)

    # --- Download Command ---
    dl_parser = data_subs.add_parser(
        "download", help="Fetch raw datasets (SQuAD, BIPIA)", parents=[parent_parser]
    )
    dl_parser.add_argument(
        "source",
        choices=["squad", "bipia"],
        help="Name of the source dataset to download",
    )
    dl_parser.add_argument(
        "--output-dir",
        "-o",
        help="Override default output directory (workspace/datasets/raw/ ...)",
    )
    dl_parser.set_defaults(func=handle_download)

    # --- Build Command ---
    build_parser = data_subs.add_parser(
        "build", help="Generate/Inject a dataset from a recipe", parents=[parent_parser]
    )
    build_parser.add_argument(
        "config", help="Path to the dataset configuration file (YAML)"
    )
    build_parser.add_argument(
        "--name", help="Name for the built dataset (overrides config name)"
    )
    build_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing dataset if it exists",
    )
    build_parser.set_defaults(func=handle_build)
