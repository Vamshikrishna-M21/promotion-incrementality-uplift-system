from __future__ import annotations

import argparse
import json

from promo_uplift.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Promotion incrementality and uplift decision system")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML config file.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    summary = run_pipeline(args.config)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

