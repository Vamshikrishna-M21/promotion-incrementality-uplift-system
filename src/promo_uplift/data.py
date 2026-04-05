from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [f"f{i}" for i in range(12)]
LABEL_COLUMNS = ["treatment", "conversion", "visit", "exposure"]
ALL_COLUMNS = FEATURE_COLUMNS + LABEL_COLUMNS


@dataclass(slots=True)
class DatasetPaths:
    raw_path: Path
    processed_dir: Path

    @property
    def train_path(self) -> Path:
        return self.processed_dir / "train.csv.gz"

    @property
    def valid_path(self) -> Path:
        return self.processed_dir / "valid.csv.gz"

    @property
    def test_path(self) -> Path:
        return self.processed_dir / "test.csv.gz"


def ensure_directories(paths: DatasetPaths) -> None:
    paths.raw_path.parent.mkdir(parents=True, exist_ok=True)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)


def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    subprocess.run(["curl", "-L", url, "-o", str(destination)], check=True)
    return destination


def _sample_chunk(chunk: pd.DataFrame, rng: np.random.Generator, probability: float) -> pd.DataFrame:
    mask = rng.random(len(chunk)) < probability
    return chunk.loc[mask, ALL_COLUMNS]


def build_criteo_sample(
    raw_path: Path,
    output_dir: Path,
    sample_size: int,
    total_rows_hint: int,
    chunk_size: int,
    seed: int,
) -> DatasetPaths:
    paths = DatasetPaths(raw_path=raw_path, processed_dir=output_dir)
    ensure_directories(paths)
    if paths.train_path.exists() and paths.valid_path.exists() and paths.test_path.exists():
        return paths

    rng = np.random.default_rng(seed)
    probability = min(1.0, sample_size / total_rows_hint)
    sampled_parts: list[pd.DataFrame] = []

    for chunk in pd.read_csv(raw_path, compression="gzip", chunksize=chunk_size):
        sampled = _sample_chunk(chunk, rng, probability)
        if not sampled.empty:
            sampled_parts.append(sampled)

    if not sampled_parts:
        raise ValueError("No rows were sampled from the raw dataset.")

    sampled_df = pd.concat(sampled_parts, ignore_index=True)
    if len(sampled_df) > sample_size:
        sampled_df = sampled_df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    train_df, valid_df, test_df = split_train_valid_test(sampled_df, seed=seed)
    train_df.to_csv(paths.train_path, index=False, compression="gzip")
    valid_df.to_csv(paths.valid_path, index=False, compression="gzip")
    test_df.to_csv(paths.test_path, index=False, compression="gzip")
    return paths


def split_train_valid_test(
    frame: pd.DataFrame,
    seed: int,
    train_fraction: float = 0.6,
    valid_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if train_fraction + valid_fraction >= 1.0:
        raise ValueError("Train and validation fractions must sum to less than 1.")

    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_rows = len(shuffled)
    train_end = int(n_rows * train_fraction)
    valid_end = int(n_rows * (train_fraction + valid_fraction))
    return (
        shuffled.iloc[:train_end].reset_index(drop=True),
        shuffled.iloc[train_end:valid_end].reset_index(drop=True),
        shuffled.iloc[valid_end:].reset_index(drop=True),
    )


def load_split(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip")
