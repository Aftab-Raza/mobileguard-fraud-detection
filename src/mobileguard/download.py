"""Reproducible downloader for the public PaySim Parquet mirror."""

from __future__ import annotations

from pathlib import Path

import requests

PAYSIM_URL = (
    "https://huggingface.co/datasets/purulalwani/"
    "Synthetic-Financial-Datasets-For-Fraud-Detection/resolve/"
    "refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)
PAYSIM_EXPECTED_BYTES = 273_667_706


def download_paysim(
    destination: str | Path,
    *,
    overwrite: bool = False,
    url: str = PAYSIM_URL,
) -> Path:
    """Stream PaySim to disk and use an atomic rename on success."""

    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not overwrite:
        return output

    partial = output.with_suffix(f"{output.suffix}.part")
    if partial.exists():
        partial.unlink()

    with requests.get(url, stream=True, timeout=(15, 120)) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    if url == PAYSIM_URL and partial.stat().st_size != PAYSIM_EXPECTED_BYTES:
        actual = partial.stat().st_size
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Dataset size check failed: expected {PAYSIM_EXPECTED_BYTES}, got {actual}"
        )

    partial.replace(output)
    return output

