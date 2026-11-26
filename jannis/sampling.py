from collections import Counter
import heapq
from math import isqrt
from typing import Any, List, Union
from matchms import Spectrum
import numpy as np


import hashlib
import json
from pathlib import Path

from tqdm import tqdm

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
from SpectrumNletGenerator import SpectrumNletGenerator
from SelectedNletSchedule import SelectedNletSchedule, load_selected_nlet_schedule, save_selected_nlet_schedule, selected_nlet_schedule_exists
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore

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

### Helper ###
def count_candidate_nlet_bins(
    candidate_indices: np.ndarray,
    fingerprints,
    fingerprint_type: str,
    similarity_bins,
) -> tuple[np.ndarray, np.ndarray]:
    """Count all unique pairwise similarities in each Tanimoto bin."""

    indices = np.asarray(candidate_indices, dtype=np.int64)

    if isinstance(fingerprints, np.ndarray):
        selected_fingerprints = fingerprints[indices]
    else:
        selected_fingerprints = [fingerprints[i] for i in indices]

    similarity_matrix = compute_fingerprint_similarity_matrix(
        selected_fingerprints,
        selected_fingerprints,
        fingerprint_type=fingerprint_type,
    )

    # Extract unique pairs, excluding diagonal.
    i, j = np.triu_indices(len(indices), k=1)
    similarities = similarity_matrix[i, j]

    # Convert your existing bin definitions to histogram edges.
    sorted_bins = sorted(similarity_bins, key=lambda b: b[0])
    edges = np.array(
        [bin_pair[0] for bin_pair in sorted_bins]
        + [sorted_bins[-1][1]],
        dtype=float,
    )

    counts, _ = np.histogram(similarities, bins=edges)

    return similarity_matrix, counts

def propose_random_nlet(context: dict) -> np.ndarray:
    """Propose one N-let using the existing cached candidate bins."""

    available_pairs_per_bin_matrix: np.ndarray = context["available_pairs_per_bin_matrix"]
    n: int = context["n"]
    rng: np.random.Generator = context["rng"]
    _, num_molecules, _ = available_pairs_per_bin_matrix.shape

    for _ in range(100):
        # Choose a random anchor.
        anchor = int(rng.integers(num_molecules))

        # Get its candidate molecules from ALL bins.
        partners = available_pairs_per_bin_matrix[:, anchor, :].ravel()

        # Remove invalid entries, the anchor, and duplicates.
        partners = np.unique(
            partners[(partners >= 0) & (partners != anchor)]
        )

        if len(partners) < n - 1:
            continue

        # Randomly choose N-1 distinct partners.
        chosen = rng.choice(partners, size=n - 1, replace=False)

        return np.concatenate(([anchor], chosen)).astype(np.int32)

    raise ValueError("Could not find an anchor with enough candidates.")

def calculate_reuse_penalty(context: dict) -> float:
    """Calculate normalized molecule reuse penalty for a candidate."""

    candidate: np.ndarray = context["candidate"]
    molecule_counts: np.ndarray = context["molecule_counts"]

    candidate_usage = molecule_counts[candidate].mean()
    average_usage = molecule_counts.mean()

    reuse_penalty = (
        (candidate_usage - average_usage)
        / (candidate_usage + average_usage + 1.0)
    )

    return float(reuse_penalty)

### matrix_balanced ###
def propose_matrix_balanced_nlet(context: dict) -> np.ndarray:
    """Propose one N-let for matrix-balanced sampling.

    Strategy:
    - Pick a random anchor.
    - Prefer anchor bins that are currently underrepresented globally.
    - Within each bin, prefer molecules that have been used less often.
    - First try to take one partner per bin.
    - If more partners are needed, reuse bins in the same priority order.

    This is only a proposal heuristic. Final quality is still decided by
    the strategies score function.
    """
    available_pairs_per_bin_matrix: np.ndarray = context["available_pairs_per_bin_matrix"]
    selected_counts: np.ndarray = context["selected_counts"]
    molecule_counts: np.ndarray = context["molecule_counts"]
    n: int = context["n"]
    rng: np.random.Generator = context["rng"]
    max_tries: int = context.get("max_tries", 100)
    num_bins, num_molecules, _ = available_pairs_per_bin_matrix.shape

    for _ in range(max_tries):
        anchor = int(rng.integers(num_molecules))

        # Collect valid unique partners per bin for this anchor.
        candidates_per_bin = []
        nonempty_bins = []

        for bin_idx in range(num_bins):
            partners = available_pairs_per_bin_matrix[bin_idx, anchor]
            partners = np.unique(
                partners[
                    (partners >= 0)
                    & (partners < num_molecules)
                    & (partners != anchor)
                ]
            ).astype(np.int32)

            candidates_per_bin.append(partners)

            if len(partners) > 0:
                nonempty_bins.append(bin_idx)

        if not nonempty_bins:
            continue

        # Global bin priority: least represented bins first.
        bin_priority = np.asarray(nonempty_bins, dtype=np.int32)
        rng.shuffle(bin_priority)
        bin_priority = sorted(bin_priority, key=lambda b: selected_counts[b])

        selected = [anchor]
        used = {anchor}

        def pick_partner(partners: np.ndarray) -> int | None:
            """Pick a low-reuse partner with random tie-breaking."""
            available = np.array(
                [p for p in partners if int(p) not in used],
                dtype=np.int32,
            )
            if len(available) == 0:
                return None

            usage = molecule_counts[available]
            best = available[usage == usage.min()]

            return int(rng.choice(best))

        # Pass 1: try one partner from each prioritized bin.
        for bin_idx in bin_priority:
            if len(selected) >= n:
                break

            partner = pick_partner(candidates_per_bin[bin_idx])
            if partner is None:
                continue

            selected.append(partner)
            used.add(partner)

        # Pass 2: if still short, reuse bins in the same priority order.
        while len(selected) < n:
            added_any = False

            for bin_idx in bin_priority:
                if len(selected) >= n:
                    break

                partner = pick_partner(candidates_per_bin[bin_idx])
                if partner is None:
                    continue

                selected.append(partner)
                used.add(partner)
                added_any = True

            if not added_any:
                break

        if len(selected) == n:
            return np.asarray(selected, dtype=np.int32)

    raise ValueError("Could not find an anchor with enough distinct candidates.")

def score_matrix_balanced(context: dict) -> float:
    """Score global similarity balance and molecule reuse."""
    candidate_counts: np.ndarray = context["candidate_counts"]
    selected_counts: np.ndarray = context["selected_counts"]

    num_bins = len(selected_counts)
    pairs_per_nlet = candidate_counts.sum()

    # 1. Current deviation from a uniform distribution.
    deviation = selected_counts - selected_counts.sum() / num_bins

    # 2. Candidate's contribution to that deviation.
    contribution = candidate_counts - pairs_per_nlet / num_bins

    # 3. Change in squared imbalance.
    improvement = (
        2 * np.dot(deviation, contribution)
        + np.dot(contribution, contribution)
    )

    # Normalize to approximately [-1, 1].
    scale = (
        2 * np.linalg.norm(deviation) * pairs_per_nlet
        + pairs_per_nlet**2
    )

    balance_error = improvement / scale

    # 4. Combined score (lower is better).
    return float(balance_error)

### natural ###
def propose_natural_nlet(context: dict) -> np.ndarray:
    """Sample N distinct molecules uniformly at random."""

    n: int = context["n"]
    rng: np.random.Generator = context["rng"]
    num_molecules: int = len(context["fingerprints"])

    return rng.choice(
        num_molecules,
        size=n,
        replace=False,
    ).astype(np.int32)

### cluster_background ###
def propose_cluster_background_nlet(context: dict) -> np.ndarray:
    """Propose an N-let with a similar cluster and dissimilar background."""

    candidate_matrix: np.ndarray = context["available_pairs_per_bin_matrix"]
    similarity_bins = context["similarity_bins"]
    n: int = context["n"]
    rng: np.random.Generator = context["rng"]
    max_tries: int = context.get("max_tries", 100)

    _, num_molecules, _ = candidate_matrix.shape

    cluster_size = n // 2
    background_size = n - cluster_size

    # Prefer high similarity for cluster, low similarity for background.
    high_bins = sorted(
        range(len(similarity_bins)),
        key=lambda i: similarity_bins[i][0],
        reverse=True,
    )
    low_bins = list(reversed(high_bins))

    for _ in range(max_tries):
        anchor = int(rng.integers(num_molecules))
        used = {anchor}
        cluster = [anchor]
        background = []

        def sample_from_bins(bin_order, amount):
            selected = []

            for bin_idx in bin_order:
                if len(selected) >= amount:
                    break

                partners = candidate_matrix[bin_idx, anchor]
                available = np.unique(
                    partners[
                        (partners >= 0)
                        & (partners < num_molecules)
                    ]
                )
                available = [int(p) for p in available if int(p) not in used]
                rng.shuffle(available)

                for partner in available:
                    if len(selected) >= amount:
                        break

                    selected.append(partner)
                    used.add(partner)

            return selected

        # Prefer the most similar available molecules.
        cluster.extend(
            sample_from_bins(high_bins, cluster_size - 1)
        )

        if len(cluster) != cluster_size:
            continue

        # Prefer the least similar available molecules.
        background.extend(
            sample_from_bins(low_bins, background_size)
        )

        if len(background) != background_size:
            continue

        return np.asarray(cluster + background, dtype=np.int32)

    raise ValueError(
        "Could not propose a cluster-background N-let "
        "with enough distinct molecules."
    )

def score_cluster_background(context: dict) -> float:
    """Score an N-let containing a similar cluster and dissimilar background."""

    candidate: np.ndarray = context["candidate"]
    similarity_matrix: np.ndarray = context["similarity_matrix"]

    n = len(candidate)
    cluster_size = n // 2

    # 1. Similarity within the cluster: HIGH is good.
    cluster_matrix = similarity_matrix[:cluster_size, :cluster_size]
    i, j = np.triu_indices(cluster_size, k=1)
    cluster_similarities = cluster_matrix[i, j]

    cluster_error = np.mean((1.0 - cluster_similarities) ** 2)

    # 2. Similarity between cluster and background: LOW is good.
    cross_similarities = similarity_matrix[:cluster_size, cluster_size:]
    cross_error = np.mean(cross_similarities ** 2)

    # 3. Similarity within background: LOW is good.
    background_matrix = similarity_matrix[cluster_size:, cluster_size:]
    i, j = np.triu_indices(n - cluster_size, k=1)
    background_similarities = background_matrix[i, j]

    background_error = np.mean(background_similarities ** 2)

    # Equal weight for the two broad objectives:
    # cluster cohesion versus background separation.
    structure_error = (
        0.5 * cluster_error
        + 0.25 * cross_error
        + 0.25 * background_error
    )

    return float(structure_error)

### multi_clusters ###
def propose_multiple_clusters_nlet(context: dict) -> np.ndarray:
    """Propose several locally similar clusters with distant cluster anchors."""

    candidate_matrix: np.ndarray = context["available_pairs_per_bin_matrix"]
    fingerprints = context["fingerprints"]
    fingerprint_type: str = context["fingerprint_type"]
    similarity_bins = context["similarity_bins"]
    n: int = context["n"]
    rng: np.random.Generator = context["rng"]
    max_tries: int = context.get("max_tries", 100)

    if n < 6:
        raise ValueError("Multiple-cluster sampling requires N >= 6.")

    _, num_molecules, _ = candidate_matrix.shape

    num_clusters = isqrt(n)
    cluster_size = n // num_clusters
    num_background = n % num_clusters

    # Bin order is based on similarity, not array index.
    high_bins = sorted(
        range(len(similarity_bins)),
        key=lambda i: similarity_bins[i][0],
        reverse=True,
    )
    low_bins = list(reversed(high_bins))

    def get_partners(anchor, bin_order, used):
        """Collect unused partners, prioritizing the requested bin order."""
        result = []

        for bin_idx in bin_order:
            partners = candidate_matrix[bin_idx, anchor]
            valid = partners[
                (partners >= 0)
                & (partners < num_molecules)
                & (partners != anchor)
            ]

            valid = np.unique(valid)
            rng.shuffle(valid)

            result.extend(
                int(p) for p in valid if int(p) not in used
            )

        return list(dict.fromkeys(result))

    def choose_distant_anchor(anchors, used):
        """Prefer a molecule dissimilar to all existing cluster anchors."""

        # Collect distant candidates from existing anchors.
        pool = []

        for anchor in anchors:
            partners = get_partners(anchor, low_bins, used)
            pool.extend(partners[:64])

        pool = np.unique(pool)

        if len(pool) == 0:
            return None

        # Compare candidate fingerprints with ALL existing anchors.
        if isinstance(fingerprints, np.ndarray):
            pool_fps = fingerprints[pool]
            anchor_fps = fingerprints[np.asarray(anchors)]
        else:
            pool_fps = [fingerprints[i] for i in pool]
            anchor_fps = [fingerprints[i] for i in anchors]

        similarities = compute_fingerprint_similarity_matrix(
            pool_fps,
            anchor_fps,
            fingerprint_type=fingerprint_type,
        )

        # Minimize similarity to the closest existing anchor.
        worst_similarity = np.max(similarities, axis=1)
        best = pool[np.isclose(worst_similarity, worst_similarity.min())]

        return int(rng.choice(best))

    for _ in range(max_tries):
        used = set()
        anchors = []
        clusters = []
        success = True

        # Construct each cluster.
        for cluster_idx in range(num_clusters):

            if cluster_idx == 0:
                anchor = int(rng.integers(num_molecules))
            else:
                anchor = choose_distant_anchor(anchors, used)

            if anchor is None:
                success = False
                break

            used.add(anchor)
            anchors.append(anchor)

            # Find similar molecules around this cluster's anchor.
            partners = get_partners(anchor, high_bins, used)

            if len(partners) < cluster_size - 1:
                success = False
                break

            chosen = partners[:cluster_size - 1]

            cluster = [anchor] + chosen
            clusters.extend(cluster)
            used.update(chosen)

        if not success:
            continue

        # Add background molecules distant from cluster anchors.
        background = []

        for _ in range(num_background):
            partner = choose_distant_anchor(anchors, used)

            if partner is None:
                success = False
                break

            background.append(partner)
            used.add(partner)

        if success:
            return np.asarray(
                clusters + background,
                dtype=np.int32,
            )

    raise ValueError(
        "Could not propose a multiple-cluster N-let "
        "with enough distinct molecules."
    )

def score_multiple_clusters(context: dict) -> float:
    """Score cluster cohesion, cluster separation, and background separation."""

    similarity_matrix: np.ndarray = context["similarity_matrix"]
    n: int = context["n"]

    if n < 6:
        raise ValueError("Multiple-cluster sampling requires N >= 6.")

    num_clusters = isqrt(n)
    cluster_size = n // num_clusters
    num_background = n % num_clusters
    clustered_count = num_clusters * cluster_size

    # 1. Identify each cluster by its position in the N-let.
    cluster_ids = np.repeat(np.arange(num_clusters), cluster_size)

    # 2. Extract every unique pair.
    i, j = np.triu_indices(n, k=1)
    similarities = similarity_matrix[i, j]

    # Both molecules belong to clusters.
    both_clustered = (i < clustered_count) & (j < clustered_count)

    # Both molecules belong to the SAME cluster.
    within_cluster = np.zeros(len(i), dtype=bool)
    within_cluster[both_clustered] = (
        cluster_ids[i[both_clustered]]
        == cluster_ids[j[both_clustered]]
    )

    # Molecules belong to DIFFERENT clusters.
    between_clusters = both_clustered & ~within_cluster

    # At least one molecule belongs to the background.
    involves_background = ~both_clustered

    # 3. Structural errors, each in [0, 1].
    cohesion_error = np.mean(
        (1.0 - similarities[within_cluster]) ** 2
    )

    separation_error = np.mean(
        similarities[between_clusters] ** 2
    )

    if num_background > 0:
        background_error = np.mean(
            similarities[involves_background] ** 2
        )
    else:
        background_error = None

    # Equal weighting of the available structural objectives.
    errors = [cohesion_error, separation_error]

    if background_error is not None:
        errors.append(background_error)

    structure_error = float(np.mean(errors))

    return float(structure_error)

### similarity_chain ###
def propose_similarity_chain_nlet(context: dict) -> np.ndarray:
    """Propose a chain with neighboring similarities near a shared target."""

    candidate_matrix: np.ndarray = context["available_pairs_per_bin_matrix"]
    similarity_bins = context["similarity_bins"]
    n: int = context["n"]
    rng: np.random.Generator = context["rng"]
    target_similarity: float = context["target_similarity"]
    max_tries: int = context.get("max_tries", 100)

    _, num_molecules, _ = candidate_matrix.shape

    if n < 3:
        raise ValueError("Similarity chains require N >= 3.")

    # Prefer bins whose midpoint is closest to the target.
    bin_order = sorted(
        range(len(similarity_bins)),
        key=lambda i: abs(
            np.mean(similarity_bins[i]) - target_similarity
        ),
    )

    for _ in range(max_tries):
        anchor = int(rng.integers(num_molecules))

        chain = [anchor]
        used = {anchor}

        while len(chain) < n:
            previous = chain[-1]
            chosen = None

            # Search nearby similarity bins first.
            for bin_idx in bin_order:
                partners = candidate_matrix[bin_idx, previous]

                available = np.unique(
                    partners[
                        (partners >= 0)
                        & (partners < num_molecules)
                        & (partners != previous)
                    ]
                )

                available = available[
                    ~np.isin(available, list(used))
                ]

                if len(available) == 0:
                    continue

                chosen = int(rng.choice(available))
                break

            if chosen is None:
                break

            chain.append(chosen)
            used.add(chosen)

        if len(chain) == n:
            return np.asarray(chain, dtype=np.int32)

    raise ValueError("Could not propose a complete similarity chain.")

def score_similarity_chain(context: dict) -> float:
    """Score adjacent similarity, consistency, and structural drift."""

    similarity_matrix: np.ndarray = context["similarity_matrix"]
    target_similarity: float = context["target_similarity"]
    n: int = context["n"]

    # 1. Similarity between consecutive chain members.
    adjacent = np.diag(similarity_matrix, k=1)

    target_error = np.mean(
        (adjacent - target_similarity) ** 2
    )

    # 2. Adjacent similarities should be consistent.
    consistency_error = np.var(adjacent)

    # 3. Similarity should decrease as chain distance increases.
    # Compare relationships originating from the same molecule.
    drift_errors = []

    for start in range(n - 2):
        similarities = similarity_matrix[start, start + 1:]

        violations = np.maximum(
            similarities[1:] - similarities[:-1],
            0.0,
        )

        drift_errors.extend(violations ** 2)

    drift_error = (
        float(np.mean(drift_errors))
        if drift_errors
        else 0.0
    )

    # All three errors are nonnegative and lower is better.
    return float(np.mean([target_error, consistency_error, drift_error]))

### selection ###
SAMPLING_STRATEGIES = {
    "matrix_balanced": {
        "propose": propose_matrix_balanced_nlet,
        "score": score_matrix_balanced,
    },
    "natural": {
        "propose": propose_natural_nlet,
        "score": None,
    },
    "cluster_background": {
        "propose": propose_cluster_background_nlet,
        "score": score_cluster_background,
    },
    "multi_clusters": {
        "propose": propose_multiple_clusters_nlet,
        "score": score_multiple_clusters,
    },
    "similarity_chain": {
        "propose": propose_similarity_chain_nlet,
        "score": score_similarity_chain,
    }
}
def select_nlets_by_strategy(
    strategy: str,
    available_pairs_per_bin_matrix: np.ndarray,
    fingerprints,
    fingerprint_type: str,
    similarity_bins,
    n: int,
    required_number_of_nlets: int,
    random_seed: int | None = None,
    num_proposals: int = 32,
    reuse_weight: float = 0.01,
    target_similarity_range: tuple[float, float] = (0.3, 0.8),
) -> np.ndarray:
    """Select N-lets using strategy-specific proposal and scoring functions."""
    if num_proposals < 1:
        raise ValueError("num_proposals must be at least 1.")
    if strategy not in SAMPLING_STRATEGIES:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

    propose = SAMPLING_STRATEGIES[strategy]["propose"]
    score_candidate = SAMPLING_STRATEGIES[strategy]["score"]

    num_molecules = len(fingerprints)

    # Shared state and configuration.
    context = {
        "available_pairs_per_bin_matrix": available_pairs_per_bin_matrix,
        "fingerprints": fingerprints,
        "fingerprint_type": fingerprint_type,
        "similarity_bins": similarity_bins,
        "n": n,
        "rng": np.random.default_rng(random_seed),
        "selected_counts": np.zeros(len(similarity_bins), dtype=np.int64),
        "molecule_counts": np.zeros(num_molecules, dtype=np.int64),
        "reuse_weight": reuse_weight,
    }

    selected_nlets = []

    for _ in tqdm(range(required_number_of_nlets), desc=f"{strategy} sampling"):
        best_candidate = None
        best_counts = None
        best_score = float("inf")
        context["target_similarity"] = float(
            context["rng"].uniform(*target_similarity_range)
        )

        for _ in range(num_proposals):

            # Strategy-specific proposal.
            context["candidate"] = propose(context)

            # Shared similarity evaluation.
            context["similarity_matrix"], context["candidate_counts"] = (
                count_candidate_nlet_bins(
                    context["candidate"],
                    fingerprints,
                    fingerprint_type,
                    similarity_bins,
                )
            )

            if score_candidate is None:
                best_candidate = context["candidate"]
                best_counts = context["candidate_counts"]
                break
            # Strategy-specific score.
            score = score_candidate(context)
            score += context["reuse_weight"] * calculate_reuse_penalty(context)

            if score < best_score:
                best_score = score
                best_candidate = context["candidate"]
                best_counts = context["candidate_counts"]

        if best_candidate is None:
            raise ValueError("Could not select a valid N-let.")

        selected_nlets.append(best_candidate)

        # Update shared state after choosing the winner.
        context["selected_counts"] += best_counts
        context["molecule_counts"][best_candidate] += 1

    return np.asarray(selected_nlets, dtype=np.int32).reshape(-1, n)


def select_nlets(
    available_pairs_per_bin_matrix,
    inchikeys14_unique,
    candidate_manifest,
    settings,
    n: int,
    strategy: list[int] | str,
    pair_data_folder=None,
    fingerprints=None,
    num_proposals: int = 32,
    reuse_weight: float = 0.01,
    target_similarity_range: tuple[float, float] = (0.3, 0.8),
):
    """Select balanced N-lets with independent hashed caching."""

    if n < 3:
        raise ValueError("N must be at least 3.")
    if isinstance(strategy, str):
        if strategy not in SAMPLING_STRATEGIES:
            raise ValueError(f"Unknown sampling strategy: {strategy}")
        if fingerprints is None:
            raise ValueError("Fingerprints are required for string strategies.")
    else:
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
            "strategy": strategy,
        })
        if isinstance(strategy, str):
            selection_manifest.update({
                "num_proposals": num_proposals,
                "reuse_weight": reuse_weight,
                "target_similarity_range": target_similarity_range,
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

    if isinstance(strategy, str):
        required_number_of_nlets = int(
            settings.average_inchikey_sampling_count
            * len(inchikeys14_unique)
            / n
        )

        selected_nlets = select_nlets_by_strategy(
            strategy=strategy,
            available_pairs_per_bin_matrix=available_pairs_per_bin_matrix,
            fingerprints=fingerprints,
            fingerprint_type=settings.fingerprint_type,
            similarity_bins=settings.same_prob_bins,
            n=n,
            required_number_of_nlets=required_number_of_nlets,
            random_seed=settings.random_seed,
            num_proposals=num_proposals,
            reuse_weight=reuse_weight,
            target_similarity_range=target_similarity_range,
        )
    else:
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
    n: int, strategy: list[int] | str,
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
        fingerprints=fingerprints,  # Add this
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