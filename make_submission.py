"""Production CLI to train the final ensemble and generate submission files.

Usage:
    python make_submission.py --data-dir data --out-dir .

Outputs:
    validation_predictions.csv      (load_id,predicted_rate)
    december_chart_inputs.csv       (pickup,delivery,distance,equipment,weight,date,predicted_rate)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from src.config import PipelineConfig
from src.pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("make_submission")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train freight rate production model and write challenge submission files."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Path to directory containing input CSVs (default: data)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Path to directory where submission files are written (default: .)",
    )
    parser.add_argument(
        "--expected-rows",
        type=int,
        default=12_000,
        help="Expected validation row count (default: 12000)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger.info("Starting freight rate production pipeline...")
    logger.info("Data directory: %s", args.data_dir.resolve())
    logger.info("Output directory: %s", args.out_dir.resolve())

    try:
        val_path, dec_path = run_pipeline(
            data_dir=args.data_dir,
            output_dir=args.out_dir,
            expected_rows=args.expected_rows,
        )
        logger.info("Submission successfully generated!")
        logger.info("Validation predictions: %s", val_path)
        logger.info("December chart inputs: %s", dec_path)
    except Exception as exc:
        logger.error("Pipeline execution failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
