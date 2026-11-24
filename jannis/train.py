from pathlib import Path

from ms2deepscore.SettingsMS2Deepscore import SettingsEmbeddingEvaluator, SettingsMS2Deepscore

from nlet_training_wrapper import train_nlet_ms2deepscore_wrapper

def num2tuple(n: int) -> str:
    names = {
        1: "singlet",
        2: "doublet",
        3: "triplet",
        4: "quadruplet",
        5: "quintuplet",
        6: "sextuplet",
        7: "septuplet",
        8: "octuplet",
        9: "nonuplet",
        10: "decuplet",
        11: "undecuplet",
        12: "duodecuplet",
    }
    return names.get(n, f"{n}-tuple")

RANDOM_SEED = 16
N = 4
STRATEGY = [0, 0, 0, 1, 0, 1, 0, 1, 0, 0]
NUM_EPOCHS = 250
SPECTRUM_FILE = Path("./runs/merged_and_cleaned_libraries_1.mgf")
OUTPUT_DIR = Path(f"./runs/{num2tuple(N)}-e{NUM_EPOCHS}-")
PAIR_CACHE_DIR = Path("./runs/nlet_cache")

settings = SettingsMS2Deepscore(
    # Dataset
    spectrum_file_path=str(SPECTRUM_FILE),
    results_folder="./trained_models",
    # training_spectra_file_name="./trained_models/train_merged_and_cleaned_libraries_1.mgf",
    # validation_spectra_file_name="./trained_models/val_merged_and_cleaned_libraries_1.mgf",
    # test_spectra_file_name="./trained_models/test_merged_and_cleaned_libraries_1.mgf",
    train_test_split_fraction=20,

    # Model architecture
    base_dims=[10000],
    embedding_dim=500,
    ionisation_mode="both",
    activation_function="relu",
    train_binning_layer=False,
    train_binning_layer_group_size=20,
    train_binning_layer_output_per_group=2,
    dropout_rate=0.0,

    # Training
    learning_rate=0.00025,
    epochs=NUM_EPOCHS,
    patience=20,
    loss_function="mse",
    weighting_factor=0,
    batch_size=32,
    num_turns=1,
    shuffle=True,
    use_fixed_set=False,
    random_seed=RANDOM_SEED,

    # Spectrum preprocessing
    min_mz=10,
    max_mz=1000,
    mz_bin_width=0.1,
    intensity_scaling=0.5,

    # Metadata encoders
    additional_metadata=[
        (
            "CategoricalToBinary",
            {
                "metadata_field": "ionmode",
                "entries_becoming_one": "positive",
                "entries_becoming_zero": "negative",
            },
        ),
        (
            "StandardScaler",
            {
                "metadata_field": "precursor_mz",
                "mean": 0,
                "standard_deviation": 1000,
            },
        ),
    ],

    # Compound sampling
    average_inchikey_sampling_count=100,
    max_inchikey_sampling=110,
    max_pairs_per_bin=300,
    same_prob_bins=[
        (0.8, 0.9),
        (0.7, 0.8),
        (0.9, 1.0),
        (0.6, 0.7),
        (0.5, 0.6),
        (0.4, 0.5),
        (0.3, 0.4),
        (0.2, 0.3),
        (0.1, 0.2),
        (-0.01, 0.1),
    ],
    include_diagonal=True,
    max_pair_resampling=10_000_000,

    # Fingerprints
    fingerprint_type="daylight",
    fingerprint_nbits=4096,

    # Validation
    val_spectra_per_inchikey=1,

    # Data augmentation
    augment_removal_max=0.2,
    augment_removal_intensity=0.2,
    augment_intensity=0.2,
    augment_noise_max=10,
    augment_noise_intensity=0.02,
)

embedding_settings = SettingsEmbeddingEvaluator(
    evaluator_distribution_size=1000,
    evaluator_num_filters=48,
    evaluator_depth=3,
    evaluator_kernel_size=20,
    random_seed=RANDOM_SEED,
    mini_batch_size=100,
    batches_per_iteration=1000,
    learning_rate=0.0001,
    num_epochs=5,
)

RUN_NAME = f"{num2tuple(N)}-training-b{settings.batch_size}-ion_{settings.ionisation_mode}"


train_nlet_ms2deepscore_wrapper(
    settings=settings,
    n=N,
    strategy=STRATEGY,
    output_dir=OUTPUT_DIR,
    pair_data_folder=PAIR_CACHE_DIR,
    settings_embedding_evaluator=embedding_settings,
    early_stopping=False,
    run_benchmark=False,
    trackio_project="ms2deepscore-nlet",
    trackio_run_name=RUN_NAME,
)