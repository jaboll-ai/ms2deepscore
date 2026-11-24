from collections import Counter, defaultdict
import heapq
from typing import List, Union
from matchms import Spectrum
import numpy as np


import hashlib
import json
from pathlib import Path

from tqdm import tqdm

from SpectrumNletGenerator import SpectrumNletGenerator
from SelectedNletSchedule import SelectedNletSchedule, load_selected_nlet_schedule, save_selected_nlet_schedule, selected_nlet_schedule_exists
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore
from ms2deepscore.fingerprint_similarity_computations import (
    compute_fingerprint_similarity_matrix,
    compute_tanimoto_similarity_per_bin,
)
from ms2deepscore.train_new_model import SpectrumPairGenerator
from ms2deepscore.train_new_model.inchikey_pair_selection import (
    compute_fingerprints_for_training,
)
from ms2deepscore.train_new_model.pair_data_persistence import (
    PAIR_DATA_FORMAT_VERSION,
    _jsonable,
    candidate_data_exists,
    load_candidate_pair_data,
    prepare_pair_data_folder,
    save_candidate_pair_data,
    spectra_structure_signature,
)

def build_candidate_data_manifest(
    *,
    kind: str,
    settings,
    spectra_signature_1: str,
    spectra_signature_2: str | None = None,
) -> dict:
    """Manifest for fingerprint calculation and candidate preparation."""

    candidate_settings = {
        "fingerprint_type": settings.fingerprint_type,
        "fingerprint_nbits": settings.fingerprint_nbits,
        "max_pairs_per_bin": settings.max_pairs_per_bin,
        "same_prob_bins": np.asarray(settings.same_prob_bins).tolist(),
        "include_diagonal": (
            settings.include_diagonal if kind == "within_set" else False
        ),
        "random_seed": settings.random_seed,
    }

    manifest = {
        "format_version": PAIR_DATA_FORMAT_VERSION,
        "kind": kind,
        "spectra_signature_1": spectra_signature_1,
        "candidate_settings": candidate_settings,
    }

    if spectra_signature_2 is not None:
        manifest["spectra_signature_2"] = spectra_signature_2

    return _jsonable(manifest)

def build_selection_data_manifest(
    *,
    candidate_manifest: dict,
    settings,
) -> dict:
    """Manifest for pair selection, dependent on prepared candidate data."""

    balancing_settings = {
        "average_inchikey_sampling_count": settings.average_inchikey_sampling_count,
        "max_inchikey_sampling": settings.max_inchikey_sampling,
        "max_pair_resampling": settings.max_pair_resampling,
        "random_seed": settings.random_seed,
    }

    manifest = {
        "format_version": PAIR_DATA_FORMAT_VERSION,
        "candidate_manifest": candidate_manifest,
        "balancing_settings": balancing_settings,
    }

    return _jsonable(manifest)

def prepare_candidate_data(
    spectra: List[Spectrum],
    settings: SettingsMS2Deepscore,
    pair_data_folder: Union[str, Path, None] = None,
):
    """Prepare or load candidate pairs grouped by Tanimoto bins."""

    if settings.random_seed is not None:
        np.random.seed(settings.random_seed)

    candidate_folder = None

    candidate_manifest = build_candidate_data_manifest(
        kind="within_set",
        settings=settings,
        spectra_signature_1=spectra_structure_signature(spectra),
    )

    if pair_data_folder is not None:
        base_folder = Path(pair_data_folder)

        candidate_id = hashlib.sha256(
            json.dumps(
                candidate_manifest,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:16]

        candidate_folder = prepare_pair_data_folder(
            base_folder / "candidates" / candidate_id,
            candidate_manifest,
        )

    # Load cached candidate data if available.
    if candidate_folder is not None and candidate_data_exists(candidate_folder):
        print(f"Loading candidate pair/score data from {candidate_folder}")

        (
            available_pairs_per_bin_matrix,
            available_scores_per_bin_matrix,
            inchikeys14_unique,
            fingerprints,
        ) = load_candidate_pair_data(candidate_folder)

    else:
        fingerprints, inchikeys14_unique = compute_fingerprints_for_training(
            spectra,
            settings.fingerprint_type,
            settings.fingerprint_nbits,
        )

        if len(inchikeys14_unique) < settings.batch_size:
            raise ValueError(
                "The number of unique inchikeys must be larger than the batch size."
            )

        available_pairs_per_bin_matrix, available_scores_per_bin_matrix = (
            compute_tanimoto_similarity_per_bin(
                fingerprints,
                settings.max_pairs_per_bin,
                fingerprint_type=settings.fingerprint_type,
                selection_bins=settings.same_prob_bins,
                include_diagonal=settings.include_diagonal,
            )
        )

        if candidate_folder is not None:
            print(f"Saving candidate pair/score data to {candidate_folder}")

            save_candidate_pair_data(
                candidate_folder,
                available_pairs_per_bin_matrix,
                available_scores_per_bin_matrix,
                inchikeys14_unique,
                fingerprints,
            )

    if len(inchikeys14_unique) < settings.batch_size:
        raise ValueError(
            "The number of unique inchikeys must be larger than the batch size."
        )

    return (
        available_pairs_per_bin_matrix,
        available_scores_per_bin_matrix,
        inchikeys14_unique,
        fingerprints,
        candidate_manifest,
    )

def balanced_selection_of_nlets_per_bin(
    available_pairs_per_bin_matrix: np.ndarray,
    settings: SettingsMS2Deepscore,
    n: int,
    strategy: list[int],
):
    """Select balanced N-lets from the predefined Tanimoto bins.

    Bin ordering follows settings.same_prob_bins.

    Each N-let contains one anchor and n-1 compounds selected
    according to the specified strategy.

    Returns tuples of compound indices, with the anchor first.
    """

    assert n >= 3, "N must be at least 3."
    assert len(strategy) == len(
        settings.same_prob_bins
    ), "Strategy length must match the number of similarity bins."
    assert all(
        isinstance(x, (int, np.integer)) and x >= 0 for x in strategy
    ), "Strategy entries must be non-negative integers."
    assert sum(strategy) == n - 1, "Strategy must sum to N - 1."

    num_bins, num_inchikeys, _ = available_pairs_per_bin_matrix.shape

    assert num_bins == len(
        strategy
    ), "Candidate matrix bin count does not match strategy."

    # Target average number of times each InChIKey is sampled.
    required_number_of_nlets = int(
        settings.average_inchikey_sampling_count * num_inchikeys / n
    )

    inchikey_counts = np.zeros(num_inchikeys, dtype=np.int64)

    selected_nlets, inchikey_counts = select_balanced_nlets(
        available_pairs_per_bin_matrix,
        inchikey_counts,
        required_number_of_nlets,
        strategy,
        settings.max_pair_resampling,
        settings.max_inchikey_sampling,
        settings.random_seed,
    )

    return selected_nlets

def select_balanced_nlets(
    available_pairs_per_bin_matrix: np.ndarray,
    inchikey_counts: np.ndarray,
    required_number_of_nlets: int,
    strategy: list[int],
    max_resampling: int,
    max_inchikey_count: int,
    random_seed: int | None = None,
):
    """Select balanced N-lets using a configurable bin strategy.

    The least frequently sampled anchors are prioritized.

    For each anchor, compounds are selected from the requested
    Tanimoto bins, prioritizing low InChIKey counts and low
    anchor-pair resampling counts.

    Parameters
    ----------
    available_pairs_per_bin_matrix:
        Candidate indices with shape
        (num_bins, num_inchikeys, max_pairs_per_bin).
        Invalid entries are marked -1.

    inchikey_counts:
        Global sampling frequency of each InChIKey.

    required_number_of_nlets:
        Target number of N-lets to select.

    strategy:
        Number of compounds selected per bin.
        The anchor is not included.

    max_resampling:
        Maximum number of times an anchor-candidate pair
        can be selected.

    max_inchikey_count:
        Maximum number of times an InChIKey can appear.

    random_seed:
        Seed for randomized tie-breaking.

    Returns
    -------
    selected_nlets:
        Integer array of shape (n_nlets, N), containing compound indices.

    inchikey_counts:
        Updated global sampling frequencies.
    """

    rng = np.random.default_rng(random_seed)
    num_bins, num_inchikeys, _ = available_pairs_per_bin_matrix.shape
    if required_number_of_nlets < 0:
        raise ValueError("Required number of N-lets cannot be negative.")

    # Precompute valid candidates for each anchor in each bin.
    candidates = []

    for bin_idx in range(num_bins):
        candidates_per_anchor = []
        for anchor in range(num_inchikeys):
            row = available_pairs_per_bin_matrix[bin_idx, anchor]
            valid = np.unique(row[(row >= 0) & (row < num_inchikeys) & (row != anchor)])
            candidates_per_anchor.append(valid)
        candidates.append(candidates_per_anchor)

    # Count how often each compound is used as an anchor.
    anchor_counts = np.zeros(num_inchikeys, dtype=np.int64)
    # Track anchor-candidate pair resampling.
    pair_counts = Counter()
    # Preselect anchors that have enough candidates in every bin.
    eligible_anchors = []

    for anchor in range(num_inchikeys):
        if all(
            len(candidates[bin_idx][anchor]) >= amount
            for bin_idx, amount in enumerate(strategy)
        ):
            eligible_anchors.append(anchor)

    if not eligible_anchors and required_number_of_nlets > 0:
        raise ValueError(
            "No anchors have enough candidates for the requested strategy."
        )

    # Min-heap: least-used anchors first.
    anchor_heap = [
        (anchor_counts[a], inchikey_counts[a], rng.random(), a)
        for a in eligible_anchors
    ]

    heapq.heapify(anchor_heap)

    selected_nlets = []

    def pair_key(a, b):
        return (min(a, b), max(a, b))

    def select_for_anchor(anchor):
        """Try to construct one complete N-let for an anchor."""

        if inchikey_counts[anchor] >= max_inchikey_count:
            return None

        nlet = [anchor]
        used = {anchor}

        for bin_idx, amount in enumerate(strategy):

            if amount == 0:
                continue

            available = [
                int(candidate)
                for candidate in candidates[bin_idx][anchor]
                if candidate not in used
                and inchikey_counts[candidate] < max_inchikey_count
                and pair_counts[pair_key(anchor, int(candidate))] < max_resampling
            ]

            if len(available) < amount:
                return None

            # Randomized tie-breaking.
            rng.shuffle(available)

            # Prefer least frequently sampled compounds and pairs.
            available.sort(
                key=lambda candidate: (
                    inchikey_counts[candidate],
                    pair_counts[pair_key(anchor, candidate)],
                )
            )

            chosen = available[:amount]

            nlet.extend(chosen)
            used.update(chosen)

        return tuple(nlet)

    with tqdm(
        total=required_number_of_nlets,
        desc="Balanced sampling of N-lets",
    ) as progress_bar:

        # Temporarily skipped anchors are retried after each round.
        # This avoids abandoning an anchor just because a different
        # candidate combination is currently unavailable.
        while len(selected_nlets) < required_number_of_nlets:

            deferred = []
            selected_this_round = 0

            while anchor_heap and len(selected_nlets) < required_number_of_nlets:

                _, _, _, anchor = heapq.heappop(anchor_heap)

                nlet = select_for_anchor(anchor)

                if nlet is None:
                    continue

                selected_nlets.append(nlet)
                selected_this_round += 1
                progress_bar.update(1)

                anchor_counts[anchor] += 1

                # Update global sampling frequencies.
                for compound in nlet:
                    inchikey_counts[compound] += 1

                # Update anchor-candidate resampling frequencies.
                for compound in nlet[1:]:
                    pair_counts[pair_key(anchor, compound)] += 1

                # Reinsert anchor with updated priority after this round.
                deferred.append(anchor)

            if len(selected_nlets) >= required_number_of_nlets:
                break

            if selected_this_round == 0:
                raise ValueError(
                    f"Only {len(selected_nlets)} of "
                    f"{required_number_of_nlets} N-lets could be selected. "
                    "Increase sampling limits or reduce "
                    "average_inchikey_sampling_count."
                )

            # Rebuild priorities using updated counts.
            anchor_heap = [
                (
                    anchor_counts[a],
                    inchikey_counts[a],
                    rng.random(),
                    a,
                )
                for a in deferred
                if inchikey_counts[a] < max_inchikey_count
            ]

            heapq.heapify(anchor_heap)

    selected_nlets_array = np.asarray(
        selected_nlets,
        dtype=np.int32,
    ).reshape(-1, sum(strategy) + 1)

    return selected_nlets_array, inchikey_counts

def convert_to_selected_nlets_list(
    selected_nlet_indices: np.ndarray,
    inchikeys14_unique: list[str],
    fingerprints,
    settings: SettingsMS2Deepscore,
) -> list[tuple[tuple[str, ...], np.ndarray]]:
    """Convert selected N-let indices into InChIKeys and similarity matrices.

    Parameters
    ----------
    selected_nlet_indices:
        Integer array of shape (n_nlets, N), containing indices into
        inchikeys14_unique. The anchor is at position 0.

    inchikeys14_unique:
        InChIKeys corresponding to the fingerprint indices.

    fingerprints:
        Fingerprints returned by prepare_candidate_data().

    settings:
        MS2DeepScore settings, including fingerprint_type.

    Returns
    -------
    list:
        Each entry contains:
            (tuple_of_inchikeys, similarity_matrix)

        similarity_matrix has shape (N, N).
    """

    if len(fingerprints) != len(inchikeys14_unique):
        raise ValueError(
            "Fingerprint count must match InChIKey count."
        )

    selected_nlets = []

    for nlet_indices in selected_nlet_indices:
        indices = np.asarray(nlet_indices, dtype=np.int64)

        if len(indices) < 3:
            raise ValueError("N-lets must contain at least 3 compounds.")

        if np.any(indices < 0) or np.any(indices >= len(inchikeys14_unique)):
            raise ValueError("N-let contains an invalid compound index.")

        if len(np.unique(indices)) != len(indices):
            raise ValueError("N-let contains duplicate compounds.")

        # Convert compound indices to InChIKeys.
        nlet_inchikeys = tuple(
            inchikeys14_unique[i] for i in indices
        )

        # Select fingerprints for this N-let.
        # Keep the original representation for the selected fingerprint type.
        if isinstance(fingerprints, np.ndarray):
            nlet_fingerprints = fingerprints[indices]
        else:
            nlet_fingerprints = [
                fingerprints[i] for i in indices
            ]

        # Calculate full pairwise Tanimoto similarity matrix.
        similarity_matrix = compute_fingerprint_similarity_matrix(
            nlet_fingerprints,
            nlet_fingerprints,
            fingerprint_type=settings.fingerprint_type,
        )

        similarity_matrix = np.asarray(
            similarity_matrix,
            dtype=np.float32,
        )

        if similarity_matrix.shape != (len(indices), len(indices)):
            raise ValueError("Invalid similarity matrix shape.")

        if not np.all(np.isfinite(similarity_matrix)):
            raise ValueError("Similarity matrix contains non-finite values.")

        selected_nlets.append(
            (nlet_inchikeys, similarity_matrix)
        )

    return selected_nlets

def select_nlets(
    available_pairs_per_bin_matrix,
    inchikeys14_unique,
    candidate_manifest,
    settings,
    n: int,
    strategy: list[int],
    pair_data_folder=None,
):
    """Select balanced N-lets with independent hashed caching."""

    if n < 3:
        raise ValueError("N must be at least 3.")
    if len(strategy) != len(settings.same_prob_bins):
        raise ValueError("Strategy length must match number of bins.")
    if sum(strategy) != n - 1:
        raise ValueError("Strategy must select exactly n - 1 compounds.")

    selection_folder = None

    if pair_data_folder is not None:
        selection_manifest = build_selection_data_manifest(
            candidate_manifest=candidate_manifest,
            settings=settings,
        )
        selection_manifest.update({
            "selection_type": "nlet",
            "n": n,
            "strategy": list(strategy),
        })

        selection_id = hashlib.sha256(
            json.dumps(
                selection_manifest, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest()[:16]

        selection_folder = prepare_pair_data_folder(
            Path(pair_data_folder) / "selection" / selection_id,
            selection_manifest,
        )

        if selected_nlet_schedule_exists(selection_folder):
            print(f"Loading selected N-lets from {selection_folder}")
            return load_selected_nlet_schedule(selection_folder)

    selected_nlets = balanced_selection_of_nlets_per_bin(
        available_pairs_per_bin_matrix,
        settings,
        n=n,
        strategy=strategy,
    )

    if selection_folder is not None:
        schedule = SelectedNletSchedule.from_nlets(
            selected_nlets,
            inchikeys14_unique,
            n,
        )
        save_selected_nlet_schedule(selection_folder, schedule)
        return schedule

    return selected_nlets

def create_spectrum_nlet_generator(
    spectra: List[Spectrum], settings: SettingsMS2Deepscore,
    n: int, strategy: list[int],
    pair_data_folder: Union[str, Path, None] = None,
) -> SpectrumNletGenerator:
    """Create a balanced N-let generator with independent hashed caching."""

    # Stage 1: Prepare or load candidate data.
    (
        available_pairs_per_bin_matrix,
        available_scores_per_bin_matrix,
        inchikeys14_unique,
        fingerprints,
        candidate_manifest,
    ) = prepare_candidate_data(spectra, settings, pair_data_folder)

    # Stage 2: Select or load cached N-lets.
    selection = select_nlets(
        available_pairs_per_bin_matrix,
        inchikeys14_unique,
        candidate_manifest,
        settings,
        n=n,
        strategy=strategy,
        pair_data_folder=pair_data_folder,
    )

    # Stage 3: Pass the compact selection directly to the generator.
    return SpectrumNletGenerator(
        selected_inchikey_nlets=selection,
        inchikeys14_unique=inchikeys14_unique,
        fingerprints=fingerprints,
        fingerprint_type=settings.fingerprint_type,
        spectra=spectra,
        shuffle=settings.shuffle,
        random_seed=settings.random_seed,
    )