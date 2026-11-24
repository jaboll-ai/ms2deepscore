"""Data generator for training an N-let MS2DeepScore model."""

import numpy as np
from torch import tensor, float32

from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore
from ms2deepscore.tensorize_spectra import tensorize_spectra
from ms2deepscore.train_new_model.data_augmentation import data_augmentation

from SpectrumNletGenerator import SpectrumNletGenerator


class NletTrainingBatchGenerator:
    """Generate augmented and tensorized batches of selected spectral N-lets.

    N-let selection is handled by SpectrumNletGenerator.
    Each training batch contains spectra, metadata, and full N×N
    molecular similarity target matrices.
    """

    def __init__(
        self,
        spectrum_nlet_generator: SpectrumNletGenerator,
        settings: SettingsMS2Deepscore,
    ):
        self.current_batch_index = 0
        self.model_settings = settings

        # Preserve fixed-set behavior.
        if self.model_settings.use_fixed_set:
            if self.model_settings.shuffle:
                raise ValueError(
                    "The generator cannot run reproducibly when shuffling is on."
                )
            if self.model_settings.random_seed is None:
                self.model_settings.random_seed = 0

        self.rng = np.random.default_rng(self.model_settings.random_seed)
        self.spectrum_nlet_generator = spectrum_nlet_generator
        self.n = spectrum_nlet_generator.n_let_size

        # Estimate the number of unique InChIKeys.
        # Each N-let contributes N compound occurrences.
        nr_of_unique_inchikeys = int(
            len(spectrum_nlet_generator)
            / settings.average_inchikey_sampling_count
            * self.n
        )

        if nr_of_unique_inchikeys < self.model_settings.batch_size:
            raise ValueError(
                "The number of unique inchikeys must be larger than the batch size."
            )

        self.fixed_set = {}

        self.nr_of_batches = int(self.model_settings.num_turns) * int(
            np.ceil(
                nr_of_unique_inchikeys / self.model_settings.batch_size
            )
        )

    def __len__(self):
        return self.nr_of_batches

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_batch_index < self.nr_of_batches:
            batch = self.__getitem__(self.current_batch_index)
            self.current_batch_index += 1
            return batch

        self.current_batch_index = 0
        raise StopIteration

    def _spectrum_nlet_generator(self):
        """Sample one batch of N-lets."""
        for _ in range(self.model_settings.batch_size):
            try:
                spectra, similarity_matrix = next(
                    self.spectrum_nlet_generator
                )
                yield spectra, similarity_matrix

            except StopIteration as exc:
                raise RuntimeError(
                    "The N-let generator is expected to generate infinite N-lets."
                ) from exc

    def __getitem__(self, batch_index: int):
        """Generate one batch, preserving fixed-set and augmentation behavior."""

        if (
            self.model_settings.use_fixed_set
            and batch_index in self.fixed_set
        ):
            return self.fixed_set[batch_index]

        if (
            self.model_settings.random_seed is not None
            and batch_index == 0
        ):
            self.rng = np.random.default_rng(
                self.model_settings.random_seed
            )

        spectrum_nlets = self._spectrum_nlet_generator()

        spectra, metadata, targets = self._tensorize_all(
            spectrum_nlets
        )

        if self.model_settings.use_fixed_set:
            self.fixed_set[batch_index] = (
                spectra,
                metadata,
                targets,
            )
        else:
            # Apply original augmentation to every spectrum.
            batch_size, n, num_bins = spectra.shape

            spectra = data_augmentation(
                spectra.reshape(batch_size * n, num_bins),
                self.model_settings,
                self.rng,
            ).reshape(batch_size, n, num_bins)

        return spectra, metadata, targets

    def _tensorize_all(self, spectrum_nlets):
        """Tensorize N-lets and construct full similarity target matrices."""

        all_spectra = []
        targets = []

        for spectra, similarity_matrix in spectrum_nlets:
            if len(spectra) != self.n:
                raise ValueError("Invalid N-let size.")

            if np.shape(similarity_matrix) != (self.n, self.n):
                raise ValueError("Invalid similarity matrix shape.")

            all_spectra.extend(spectra)
            targets.append(similarity_matrix)

        # Tensorize all spectra in one call.
        binned_spectra, metadata = tensorize_spectra(
            all_spectra,
            self.model_settings,
        )

        batch_size = len(targets)

        # Restore batch and N-let dimensions.
        binned_spectra = binned_spectra.reshape(
            batch_size, self.n, -1
        )
        metadata = metadata.reshape(
            batch_size, self.n, -1
        )

        targets = tensor(
            np.asarray(targets),
            dtype=float32,
        )

        return binned_spectra, metadata, targets