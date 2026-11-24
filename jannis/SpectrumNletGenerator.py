import json
from collections import Counter, defaultdict

import numpy as np
from matchms import Spectrum

from SelectedNletSchedule import SelectedNletSchedule
from ms2deepscore.fingerprint_similarity_computations import compute_fingerprint_similarity_matrix


class SpectrumNletGenerator:
    def __init__(
        self,
        selected_inchikey_nlets: SelectedNletSchedule | np.ndarray,
        inchikeys14_unique: list[str],
        fingerprints: np.ndarray,
        fingerprint_type: str,
        spectra: list[Spectrum],
        shuffle: bool = True,
        random_seed: int = 0,
    ):
        """Generate spectrum N-lets with pairwise similarity matrices on demand.

        Parameters
        ----------
        selected_inchikey_nlets:
            SelectedNletSchedule or integer array of shape (n_nlets, N).
            Each row contains compound indices, with the anchor at position 0.

        inchikeys14_unique:
            InChIKeys in the same order as the fingerprints.

        fingerprints:
            Molecular fingerprints used to calculate Tanimoto similarities.

        fingerprint_type:
            Fingerprint type used for similarity calculations.

        spectra:
            Available spectra for sampling.

        shuffle:
            Whether to shuffle the N-let sampling order.

        random_seed:
            Seed for reproducible spectrum sampling and shuffling.
        """
        self.selected_inchikey_nlets = selected_inchikey_nlets
        self.spectra = spectra
        self.fingerprints = fingerprints
        self.fingerprint_type = fingerprint_type
        self.shuffle = shuffle
        self.random_nr_generator = np.random.default_rng(random_seed)
        self._idx = 0

        # Handle compact schedules and regular NumPy arrays.
        if isinstance(selected_inchikey_nlets, SelectedNletSchedule):
            self.nlet_indices = selected_inchikey_nlets.nlet_indices
            self.inchikeys = selected_inchikey_nlets.inchikeys

            if not np.array_equal(self.inchikeys, inchikeys14_unique):
                raise ValueError("Schedule InChIKeys do not match fingerprint ordering.")
        else:
            self.nlet_indices = selected_inchikey_nlets
            self.inchikeys = np.asarray(inchikeys14_unique)

        if self.nlet_indices.ndim != 2 or self.nlet_indices.shape[1] < 3:
            raise ValueError("N-let indices must have shape (n_nlets, N), N >= 3.")

        if len(self.nlet_indices) == 0:
            raise ValueError("No N-lets provided.")

        if len(self.fingerprints) != len(self.inchikeys):
            raise ValueError("Fingerprint count must match InChIKey count.")

        self.n_let_size = self.nlet_indices.shape[1]

        # Shuffle indices, not the schedule itself.
        self._nlet_order = np.arange(len(self.nlet_indices))

        if self.shuffle:
            self.random_nr_generator.shuffle(self._nlet_order)

        # Map InChIKeys to available spectrum indices.
        self.spectra_by_inchikey = defaultdict(list)

        for i, spectrum in enumerate(self.spectra):
            inchikey = spectrum.get("inchikey")

            if inchikey is not None:
                self.spectra_by_inchikey[inchikey[:14]].append(i)

    def __iter__(self):
        return self

    def __next__(self):
        # Reshuffle after completing the schedule.
        if self._idx >= len(self.nlet_indices):
            self._idx = 0

            if self.shuffle:
                self.random_nr_generator.shuffle(self._nlet_order)

        # Get the next N-let's compound indices.
        nlet_index = self._nlet_order[self._idx]
        indices = self.nlet_indices[nlet_index]

        # Resolve InChIKeys.
        inchikeys = [str(self.inchikeys[i]) for i in indices]

        similarity_matrix = self._compute_similarity_matrix(indices)

        # Select a spectrum for each compound.
        spectra = [
            self._get_spectrum_with_inchikey(
                inchikey,
                self.random_nr_generator,
            )
            for inchikey in inchikeys
        ]

        self._idx += 1
        return spectra, similarity_matrix

    def __len__(self):
        return len(self.nlet_indices)


    def __str__(self):
        return (
            f"SpectrumNletGenerator with "
            f"{len(self.nlet_indices)} N-lets available "
            f"(N={self.n_let_size})"
        )


    def get_inchikey_counts(self) -> Counter:
        inchikey_counts = Counter()

        for indices in self.nlet_indices:
            inchikey_counts.update(
                str(self.inchikeys[i]) for i in indices
            )

        return inchikey_counts

    def get_scores(self):
        return [
            self._compute_similarity_matrix(indices).tolist()
            for indices in self.nlet_indices
        ]


    def get_scores_per_inchikey(self):
        inchikey_scores = defaultdict(list)

        for indices in self.nlet_indices:
            similarity_matrix = self._compute_similarity_matrix(indices)

            for i, index in enumerate(indices):
                inchikey = str(self.inchikeys[index])

                scores = [
                    float(similarity_matrix[i, j])
                    for j in range(len(indices))
                    if i != j
                ]

                inchikey_scores[inchikey].extend(scores)

        return dict(inchikey_scores)


    def save_as_json(self, file_name):
        data_for_json = []

        for indices in self.nlet_indices:
            inchikeys = [str(self.inchikeys[i]) for i in indices]
            similarity_matrix = self._compute_similarity_matrix(indices)

            data_for_json.append(
                (inchikeys, similarity_matrix.tolist())
            )

        with open(file_name, "w", encoding="utf-8") as f:
            json.dump(data_for_json, f)

    def _compute_similarity_matrix(self, indices):
        nlet_fingerprints = self.fingerprints[indices]

        return np.asarray(
            compute_fingerprint_similarity_matrix(
                nlet_fingerprints,
                nlet_fingerprints,
                fingerprint_type=self.fingerprint_type,
            ),
            dtype=np.float32,
        )

    def _get_spectrum_with_inchikey(
        self,
        inchikey: str,
        random_number_generator,
    ) -> Spectrum:
        """
        Get a random spectrum matching the supplied InChIKey14.
        """
        indices = self.spectra_by_inchikey.get(inchikey)

        if not indices:
            raise ValueError(f"No matching inchikey found: {inchikey}")

        return self.spectra[
            random_number_generator.choice(indices)
        ]