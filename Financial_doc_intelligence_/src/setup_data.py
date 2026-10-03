"""Download the public datasets used by the pipeline."""

import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.paths import RAW

DOWNLOADS = [
    (
        "https://raw.githubusercontent.com/patronus-ai/financebench/main/data/financebench_open_source.jsonl",
        RAW / "financebench" / "financebench_open_source.jsonl",
    ),
    (
        "https://raw.githubusercontent.com/JarFraud/FraudDetection/master/data_FraudDetection_JAR2020.csv",
        RAW / "jar_fraud" / "data_FraudDetection_JAR2020.csv",
    ),
    (
        "https://raw.githubusercontent.com/JarFraud/FraudDetection/master/AAER_firm_year.csv",
        RAW / "jar_fraud" / "AAER_firm_year.csv",
    ),
]


def count_rows(path: Path) -> int:
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for line in f if line.strip())
    return len(pd.read_csv(path))


def download(url: str, dest: Path, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)
    return dest


def main(force: bool = False) -> None:
    for url, dest in DOWNLOADS:
        download(url, dest, force=force)
        print(f"OK  {dest.relative_to(RAW.parent.parent)}  rows={count_rows(dest):,}")


def data_present() -> bool:
    return all(dest.exists() for _, dest in DOWNLOADS)


if __name__ == "__main__":
    main(force="--force" in sys.argv)
