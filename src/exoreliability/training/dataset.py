"""PyTorch access to a built KOI dataset (see ``data.examples``).

Model input per view is a ``[channels, bins]`` float32 tensor:

* channel 0 — **flux**: stored bin medians of ``flux - 1``; empty bins filled by linear
  interpolation between the nearest observed bins (circular in phase for the global
  view, edge-held for the local view), then normalised per example;
* channel 1 — **observation mask** (optional, default on): 1 where the bin held data,
  0 where the value was interpolated.

Normalisation (per example and per view, from observed bins only, so no statistic is
shared across examples or splits):

* ``"depth"`` (default): subtract the median, divide by (median − minimum), so the
  out-of-transit level is 0 and the deepest bin is −1 (as in Shallue & Vanderburg 2018).
  Removes absolute depth/SNR scale — a strongly class-dependent variable in DR25 (see
  docs/data_contract.md §9) — so the model must rely on shape. Falls back to the robust
  standard deviation if the view has no dip.
* ``"none"``: raw ``flux - 1``.

Arrays are memory-mapped; each item is read on demand, so nothing is loaded wholesale
onto the GPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from exoreliability.data.examples import STATUS_OK, PreparedDataset, load_dataset

Normalization = Literal["depth", "none"]
ViewName = Literal["global", "local"]

DEFAULT_METADATA = (
    "example_index",
    "kepid",
    "kepoi_name",
    "split",
    "label_name",
    "koi_disposition",
    "koi_pdisposition",
    "koi_period",
    "koi_time0bk",
    "koi_duration",
    "koi_depth",
    "koi_model_snr",
    "koi_kepmag",
    "n_kois_on_star",
)


def fill_missing(
    values: np.ndarray, count: np.ndarray, *, circular: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate empty bins; return (filled float32 values, observed float32 mask).

    With no observed bins the result is all zeros (such examples are excluded upstream by
    the coverage thresholds).
    """
    observed = (count > 0) & np.isfinite(values)
    out = np.asarray(values, dtype=np.float64).copy()
    n = out.size
    if not observed.any():
        return np.zeros(n, np.float32), np.zeros(n, np.float32)
    if not observed.all():
        idx = np.arange(n)
        if circular:
            out[~observed] = np.interp(idx[~observed], idx[observed], out[observed], period=n)
        else:
            out[~observed] = np.interp(idx[~observed], idx[observed], out[observed])
    return out.astype(np.float32), observed.astype(np.float32)


def normalize_view(values: np.ndarray, observed: np.ndarray, method: Normalization) -> np.ndarray:
    if method == "none":
        return values.astype(np.float32)
    obs = values[observed > 0]
    if obs.size == 0:
        return values.astype(np.float32)
    med = float(np.median(obs))
    scale = med - float(obs.min())
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.4826 * float(np.median(np.abs(obs - med))) or 1.0
    return ((values - med) / scale).astype(np.float32)


class KOIDataset(Dataset[dict[str, Any]]):
    """Examples of one split (or all) from a built dataset directory.

    By default only examples with ``example_status == "ok"`` and a supervised label
    (``training_status == "train_eligible"``) are included; pass ``trainable_only=False``
    to also get excluded KOIs (e.g. candidates, label = NaN) for inference/research.
    """

    def __init__(
        self,
        root: Path | PreparedDataset,
        split: str | None = None,
        *,
        views: Sequence[ViewName] = ("global",),
        include_mask: bool = True,
        normalization: Normalization = "depth",
        trainable_only: bool = True,
        return_metadata: bool = False,
        metadata_columns: Sequence[str] = DEFAULT_METADATA,
    ) -> None:
        self.data = root if isinstance(root, PreparedDataset) else load_dataset(Path(root))
        ex = self.data.examples
        keep = ex.example_status == STATUS_OK
        if trainable_only:
            keep &= ex.training_status == "train_eligible"
        if split is not None:
            if split not in ("train", "val", "test"):
                raise ValueError(f"unknown split {split!r}")
            keep &= ex.split == split
        self.examples: pd.DataFrame = ex[keep].reset_index(drop=True)
        self.indices = self.examples["example_index"].to_numpy(np.int64)
        if not set(views) <= {"global", "local"} or not views:
            raise ValueError(
                f"views must be a non-empty subset of ('global', 'local'), got {views}"
            )
        self.views = tuple(views)
        self.include_mask = include_mask
        self.normalization = normalization
        self.return_metadata = return_metadata
        self.metadata_columns = tuple(metadata_columns)

    def __len__(self) -> int:
        return len(self.indices)

    def _view(self, name: ViewName, i: int) -> torch.Tensor:
        flux = self.data.global_flux if name == "global" else self.data.local_flux
        count = self.data.global_count if name == "global" else self.data.local_count
        filled, observed = fill_missing(
            np.asarray(flux[i]), np.asarray(count[i]), circular=name == "global"
        )
        channels = [normalize_view(filled, observed, self.normalization)]
        if self.include_mask:
            channels.append(observed)
        return torch.from_numpy(np.stack(channels))

    def __getitem__(self, item: int) -> dict[str, Any]:
        i = int(self.indices[item])
        row = self.examples.iloc[item]
        label = row["label"]
        out: dict[str, Any] = {name: self._view(name, i) for name in self.views}
        out["label"] = torch.tensor(
            float("nan") if pd.isna(label) else float(label), dtype=torch.float32
        )
        if self.return_metadata:
            out["meta"] = {c: _plain(row[c]) for c in self.metadata_columns}
        return out

    @property
    def input_shapes(self) -> dict[str, tuple[int, int]]:
        c = 2 if self.include_mask else 1
        return {
            "global": (c, self.data.global_flux.shape[1]),
            "local": (c, self.data.local_flux.shape[1]),
        }


def _plain(value: Any) -> Any:
    """Collate-friendly scalar: NaN for missing numbers, '' for missing strings."""
    if value is None or value is pd.NA:
        return float("nan")
    if isinstance(value, np.generic):
        return value.item()
    return value


def make_dataloader(
    dataset: KOIDataset,
    *,
    batch_size: int = 32,
    shuffle: bool = False,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader[dict[str, Any]]:
    """CPU DataLoader with a seeded shuffle order (``shuffle=True`` for training)."""
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
    )
