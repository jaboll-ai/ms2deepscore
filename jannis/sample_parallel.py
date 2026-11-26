
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from hyperp import *
from ms2deepscore.wrapper_functions.training_wrapper_functions import (
    load_spectra_in_ionmode,
    split_data_if_necessary,
)
from sampling import (
    prepare_candidate_data,
    select_nlets,
    SAMPLING_STRATEGIES,
)


CACHE_FOLDER = Path("./runs/new_sample_cache")

# Populated before forking, inherited by the workers.
shared_candidate_data = None


def run_sampling(strat: str):
    (
        available_pairs_per_bin_matrix,
        _,
        inchikeys14_unique,
        fingerprints,
        candidate_manifest,
    ) = shared_candidate_data

    select_nlets(
        available_pairs_per_bin_matrix=available_pairs_per_bin_matrix,
        inchikeys14_unique=inchikeys14_unique,
        candidate_manifest=candidate_manifest,
        settings=settings,
        n=N,
        strategy=strat,
        pair_data_folder=CACHE_FOLDER,
        fingerprints=fingerprints,
    )

    return strat


if __name__ == "__main__":
    split_data_if_necessary(settings)

    # Load spectra once.
    training_spectra = load_spectra_in_ionmode(
        settings.training_spectra_file_name,
        settings.ionisation_mode,
    )

    # Compute or load candidate data once.
    shared_candidate_data = prepare_candidate_data(
        training_spectra,
        settings,
        CACHE_FOLDER,
    )

    # Spectra are no longer needed for sampling.
    del training_spectra

    # Workers inherit shared_candidate_data through fork.
    ctx = mp.get_context("fork")

    with ProcessPoolExecutor(
        max_workers=3,
        mp_context=ctx,
    ) as executor:

        futures = {
            executor.submit(run_sampling, strat): strat
            for strat in SAMPLING_STRATEGIES
        }

        for future in as_completed(futures):
            print(f"Finished sampling: {future.result()}")
