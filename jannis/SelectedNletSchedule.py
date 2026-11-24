from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ms2deepscore.train_new_model.pair_data_persistence import SELECTED_PAIR_INCHIKEYS_FILENAME, SELECTED_PAIR_INDICES_FILENAME, _atomic_save_npy

@dataclass(frozen=True)
class SelectedNletSchedule:
    """Compact representation of an expanded selected N-let schedule.

    ``nlet_indices`` contains N integer indices into ``inchikeys`` for each
    scheduled N-let occurrence. Repeated selections remain repeated rows.
    The anchor is at position 0.
    """

    inchikeys: np.ndarray
    nlet_indices: np.ndarray

    def __post_init__(self):
        if self.nlet_indices.ndim != 2 or self.nlet_indices.shape[1] < 3:
            raise ValueError("nlet_indices must have shape (n_nlets, N), N >= 3.")

        if np.any(self.nlet_indices < 0) or np.any(self.nlet_indices >= len(self.inchikeys)):
            raise ValueError("nlet_indices contains invalid indices.")

    def __len__(self) -> int:
        return int(self.nlet_indices.shape[0])

    def __getitem__(self, index: int):
        return tuple(
            str(self.inchikeys[int(i)])
            for i in self.nlet_indices[index]
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    @classmethod
    def from_nlets(
        cls,
        nlets: Sequence[tuple[int, ...]],
        inchikeys: Sequence[str],
        n: int,
    ) -> "SelectedNletSchedule":
        inchikey_array = np.asarray(inchikeys, dtype="U14")
        nlet_indices = np.asarray(nlets, dtype=np.int32).reshape(-1, n)

        return cls(inchikey_array, nlet_indices)

def selected_nlet_schedule_exists(folder: str | Path) -> bool:
    folder = Path(folder)
    paths = [
        folder / SELECTED_PAIR_INDICES_FILENAME,
        folder / SELECTED_PAIR_INCHIKEYS_FILENAME,
    ]

    existing = [path.exists() for path in paths]
    if any(existing) and not all(existing):
        raise ValueError(f"Incomplete selected-N-let schedule in {folder}.")

    return all(existing)

def save_selected_nlet_schedule(
    folder: str | Path,
    schedule: SelectedNletSchedule,
) -> None:
    """Persist the selected N-let schedule in compact, mmap-friendly arrays."""
    folder = Path(folder)

    _atomic_save_npy(
        folder / SELECTED_PAIR_INDICES_FILENAME,
        np.asarray(schedule.nlet_indices, dtype=np.int32),
    )
    _atomic_save_npy(
        folder / SELECTED_PAIR_INCHIKEYS_FILENAME,
        np.asarray(schedule.inchikeys, dtype="S14"),
    )


def load_selected_nlet_schedule(
    folder: str | Path,
) -> SelectedNletSchedule:
    """Load the selected N-let schedule without expanding Python tuples."""
    folder = Path(folder)

    if not selected_nlet_schedule_exists(folder):
        raise FileNotFoundError(
            f"No complete selected-N-let schedule found in {folder}."
        )

    nlet_indices = np.load(
        folder / SELECTED_PAIR_INDICES_FILENAME,
        mmap_mode="r",
        allow_pickle=False,
    )
    inchikeys_bytes = np.load(
        folder / SELECTED_PAIR_INCHIKEYS_FILENAME,
        allow_pickle=False,
    )
    inchikeys = inchikeys_bytes.astype("U14")

    return SelectedNletSchedule(inchikeys, nlet_indices)