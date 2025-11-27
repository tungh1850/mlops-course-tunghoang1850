"""
Script to create new data versions for testing DVC pipeline.
This simulates receiving new data by creating different samples from the original dataset.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def create_data_version(
    source_path: Path,
    output_path: Path,
    sample_fraction: float = 0.3,
    random_state: int = 42,
) -> None:
    """
    Create a new version of the dataset by sampling.

    Args:
        source_path: Path to the original dataset
        output_path: Path where the new version will be saved
        sample_fraction: Fraction of data to sample (default: 0.3 = 30%)
        random_state: Random seed for reproducibility
    """
    logger.info(f"Loading data from {source_path}")
    df = pd.read_csv(source_path)

    logger.info(f"Original dataset shape: {df.shape}")

    # Sample the data
    df_sample = df.sample(frac=sample_fraction, random_state=random_state)
    logger.info(f"Sampled dataset shape: {df_sample.shape}")

    # Save the new version
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_sample.to_csv(output_path, index=False)
    logger.info(f"New data version saved to {output_path}")
    logger.info(
        f"Reduction: {len(df)} → {len(df_sample)} rows ({sample_fraction*100}%)"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Create a new data version for testing DVC pipeline"
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/facts_dataset.csv",
        help="Path to source dataset",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/facts_dataset.csv",
        help="Path to output dataset (overwrites source by default)",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.3,
        help="Fraction of data to sample (e.g., 0.3 for 30%%)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    source_path = Path(args.source)
    output_path = Path(args.output)

    if not source_path.exists():
        logger.error(f"Source file not found: {source_path}")
        return

    create_data_version(
        source_path=source_path,
        output_path=output_path,
        sample_fraction=args.fraction,
        random_state=args.seed,
    )


if __name__ == "__main__":
    main()
