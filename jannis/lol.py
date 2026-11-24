from pathlib import Path
from time import perf_counter

import numpy as np

from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore
from ms2deepscore.wrapper_functions.training_wrapper_functions import (
    split_data_if_necessary,
    load_spectra_in_ionmode,
)

# Import YOUR modified function
from sampling import create_spectrum_nlet_generator


settings = SettingsMS2Deepscore(
    spectrum_file_path="./runs/out.mgf",
    ionisation_mode="both",
    random_seed=42,
    batch_size=32,
    average_inchikey_sampling_count=1,
    fingerprint_type="rdkit_logcount",
    same_prob_bins = [
        (-0.2, 0.25),
        (0.25, 0.5),
        (0.5, 0.75),
        (0.75, 1.0),
    ]
)

split_data_if_necessary(settings)

spectra = load_spectra_in_ionmode(
    settings.training_spectra_file_name,
    settings.ionisation_mode,
)

cache_folder = Path("./runs/test_pair_cache")

from sampling import create_spectrum_nlet_generator

# Adjust the import path to your actual module.

def test_nlet_generator(label, settings):
    print(f"\n========== {label} ==========")

    start = perf_counter()

    generator = create_spectrum_nlet_generator(
        spectra=spectra,
        settings=settings,
        n=3,
        strategy=[1, 1, 0, 0],
        pair_data_folder=cache_folder,
    )

    print(f"Time: {perf_counter() - start:.2f}s")
    print(f"Available N-lets: {len(generator)}")
    print(f"Selection type: {type(generator.selected_inchikey_nlets).__name__}")
    print(f"Index array shape: {generator.nlet_indices.shape}")

    # Confirm that similarity matrices are calculated on demand.
    selected_spectra, similarity_matrix = next(generator)

    print("Similarity matrix:")
    print(similarity_matrix)

    assert len(selected_spectra) == 3
    assert similarity_matrix.shape == (3, 3)
    assert np.allclose(similarity_matrix, similarity_matrix.T)
    assert np.allclose(np.diag(similarity_matrix), 1.0)

    return generator


generator1 = test_nlet_generator("FIRST RUN", settings)
generator2 = test_nlet_generator("SECOND RUN", settings)