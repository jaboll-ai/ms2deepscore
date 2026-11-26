from hyperp import *
from ms2deepscore.wrapper_functions.training_wrapper_functions import load_spectra_in_ionmode, split_data_if_necessary
from sampling import create_spectrum_nlet_generator, SAMPLING_STRATEGIES


split_data_if_necessary(settings)

training_spectra = load_spectra_in_ionmode(
    settings.training_spectra_file_name,
    settings.ionisation_mode,
)
# validation_spectra = load_spectra_in_ionmode(
#     settings.validation_spectra_file_name,
#     settings.ionisation_mode,
# )
for strat in SAMPLING_STRATEGIES:
    print(f"Sampling strategy: {strat}")
    # N-let-specific training data.
    spectrum_nlet_generator = create_spectrum_nlet_generator(
        spectra=training_spectra,
        settings=settings,
        n=N,
        strategy=strat,
        pair_data_folder=Path("./runs/new_sample_cache"),
    )