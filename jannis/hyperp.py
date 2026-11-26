from pathlib import Path

from ms2deepscore.SettingsMS2Deepscore import SettingsEmbeddingEvaluator, SettingsMS2Deepscore

RANDOM_SEED = 16
N = 8
TRACKIO_NAME = "ms2deepscore-nlet-v2"
NUM_EPOCHS = 250
SPECTRUM_FILE = Path("./runs/merged_and_cleaned_libraries_1.mgf")

settings = SettingsMS2Deepscore(
    # Dataset
    spectrum_file_path=str(SPECTRUM_FILE),
    results_folder="./trained_models",
    training_spectra_file_name="./trained_models/train_merged_and_cleaned_libraries_1.mgf",
    validation_spectra_file_name="./trained_models/val_merged_and_cleaned_libraries_1.mgf",
    test_spectra_file_name="./trained_models/test_merged_and_cleaned_libraries_1.mgf",
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
    average_inchikey_sampling_count=50,
    max_inchikey_sampling=220,
    max_pairs_per_bin=300,
    same_prob_bins = [
        (0.9, 1.0),   # 0
        (0.8, 0.9),   # 1
        (0.7, 0.8),   # 2
        (0.6, 0.7),   # 3
        (0.5, 0.6),   # 4
        (0.4, 0.5),   # 5
        (0.3, 0.4),   # 6
        (0.2, 0.3),   # 7
        (0.1, 0.2),   # 8
        (-0.01, 0.1), # 9
    ],
    include_diagonal=True,
    max_pair_resampling=10_000_000,

    # Fingerprints
    fingerprint_type="rdkit_binary",
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