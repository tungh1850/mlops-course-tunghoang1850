"""
DVC Data Preparation Script
Handles preprocessing before model training
"""

import os

import pandas as pd
import yaml


def load_params():
    """Load parameters from params.yaml"""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)


def prepare_data():
    """Prepare and preprocess data for training"""
    params = load_params()

    print("📥 Loading raw data...")
    df = pd.read_csv("data/facts_dataset.csv")
    print(f"Raw data shape: {df.shape}")

    # Handle missing values
    print(
        f"🔧 Handling missing values ({params['preprocess']['missing_value_strategy']})..."
    )
    if params["preprocess"]["missing_value_strategy"] == "drop":
        df = df.dropna()
    elif params["preprocess"]["missing_value_strategy"] == "fill_mean":
        numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    print(f"After preprocessing: {df.shape}")

    # Create processed data directory
    os.makedirs("data/processed", exist_ok=True)

    # For now, save the same data (in real pipelines, you'd transform here)
    df.to_csv("data/processed/all_data.csv", index=False)
    print("✓ Processed data saved")

    return df


if __name__ == "__main__":
    prepare_data()
