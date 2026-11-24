import os
from pathlib import Path
from typing import Union

import numpy as np

from jannis.model import SpectralNletModel
from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore
from ms2deepscore.models.SiameseSpectralModel import train
from ms2deepscore.train_new_model import TrainingBatchGenerator
from ms2deepscore.train_new_model.inchikey_pair_selection import create_spectrum_pair_generator
from ms2deepscore.train_new_model.inchikey_pair_selection_cross_ionmode import create_data_generator_across_ionmodes, select_compound_pairs_wrapper_across_ionmode
from ms2deepscore.train_new_model.pair_data_persistence import _jsonable, prepare_pair_data_folder, spectra_structure_signature
from ms2deepscore.utils import split_by_ionmode
from ms2deepscore.validation_loss_calculation.ValidationLossCalculator import ValidationLossCalculator
from ms2deepscore.wrapper_functions.training_wrapper_functions import (
    load_spectra_in_ionmode,
    split_data_if_necessary,
)

NLET_DATA_FORMAT_VERSION = 1
spectrum_file = "./20241003_enamdisc_neg_ms2.mgf"
settings = SettingsMS2Deepscore(
    spectrum_file_path=spectrum_file,
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
            {"metadata_field": "precursor_mz", "mean": 0, "standard_deviation": 1000},
        ),
    ],
    ionisation_mode="negative",
    random_seed=42,
    epochs=5,
)

split_data_if_necessary(settings)

training_spectra = load_spectra_in_ionmode(
    settings.training_spectra_file_name, settings.ionisation_mode
)

validation_spectra = load_spectra_in_ionmode(
    settings.validation_spectra_file_name, settings.ionisation_mode
)

def build_nlet_data_manifest(
    *,
    kind: str,
    settings,
    spectra_signature_1: str,
    spectra_signature_2: str | None = None,
    n_let_size: int = 3,
) -> dict:
    """Build the manifest describing exactly which persisted N-let data is valid."""
    candidate_settings = {
        "fingerprint_type": settings.fingerprint_type,
        "fingerprint_nbits": settings.fingerprint_nbits,
        "max_pairs_per_bin": settings.max_pairs_per_bin,
        "same_prob_bins": np.asarray(settings.same_prob_bins).tolist(),
        "include_diagonal": settings.include_diagonal if kind == "within_set" else False,
        "random_seed": settings.random_seed,
        "n_let_size": n_let_size,
    }
    balancing_settings = {
        "average_inchikey_sampling_count": settings.average_inchikey_sampling_count,
        "max_inchikey_sampling": settings.max_inchikey_sampling,
        "max_pair_resampling": settings.max_pair_resampling,
    }

    manifest = {
        "format_version": NLET_DATA_FORMAT_VERSION,
        "kind": kind,
        "spectra_signature_1": spectra_signature_1,
        "candidate_settings": candidate_settings,
        "balancing_settings": balancing_settings,
    }
    if spectra_signature_2 is not None:
        manifest["spectra_signature_2"] = spectra_signature_2
    return _jsonable(manifest)

def create_data_generator_across_ionmodes(
    training_spectra,
    settings: SettingsMS2Deepscore,
    pair_data_folder: Union[str, Path, None] = None,
    n_let_size: int = 3,
) -> TrainingBatchGenerator:
    """Create balanced positive, negative and cross-ionmode training generators.

    When ``pair_data_folder`` is given, explicit subfolders ``positive``,
    ``negative`` and ``positive_negative`` are used for persisted pair data.
    """
    pos_spectra, neg_spectra = split_by_ionmode(training_spectra)

    base_folder = Path(pair_data_folder) if pair_data_folder is not None else None
    if base_folder is not None:
        # A top-level manifest makes the folder layout explicit and prevents accidentally
        # reusing a folder that was previously prepared for single/within-set training.
        layout_manifest = build_nlet_data_manifest(
            kind="balanced_across_ionmodes",
            settings=settings,
            spectra_signature_1=spectra_structure_signature(training_spectra),
            n_let_size=n_let_size,
        )
        prepare_pair_data_folder(base_folder, layout_manifest)

    pos_folder = base_folder / "positive" if base_folder is not None else None
    neg_folder = base_folder / "negative" if base_folder is not None else None
    cross_folder = base_folder / "positive_negative" if base_folder is not None else None

    pos_spectrum_pair_generator = create_spectrum_pair_generator(
        pos_spectra, settings=settings, pair_data_folder=pos_folder
    )
    neg_spectrum_pair_generator = create_spectrum_pair_generator(
        neg_spectra, settings=settings, pair_data_folder=neg_folder
    )
    pos_neg_spectrum_pair_generator = select_compound_pairs_wrapper_across_ionmode(
        pos_spectra, neg_spectra, settings, pair_data_folder=cross_folder
    )

    spectrum_pair_generator = CombinedSpectrumGenerator(
        [pos_spectrum_pair_generator, neg_spectrum_pair_generator, pos_neg_spectrum_pair_generator]
    )

    train_generator = TrainingBatchGenerator(
        spectrum_pair_generator=spectrum_pair_generator, settings=settings
    )
    return train_generator

def train_ms2ds_model(
        training_spectra,
        validation_spectra,
        results_folder,
        settings: SettingsMS2Deepscore,
        pair_data_folder: Union[str, Path, None] = None,
        n_let_size: int =3,
        ):
    """Full workflow to train a MS2DeepScore model.

    ``pair_data_folder`` is an explicit persistence/reuse location for the expensive
    training-pair preparation. Reusing the same folder skips pair preparation only
    when its manifest matches the current structural data and pair-selection settings.
    """
    # Make folder and save settings
    os.makedirs(results_folder, exist_ok=True)
    settings.save_to_file(os.path.join(results_folder, "settings.json"))
    if settings.balanced_sampling_across_ionmodes:
        train_generator = create_data_generator_across_ionmodes(
            training_spectra, settings=settings, pair_data_folder=pair_data_folder,
            n_let_size=n_let_size,
        )
    else:
        spectrum_pair_generator = create_spectrum_pair_generator(
            training_spectra, settings=settings, pair_data_folder=pair_data_folder,
            n_let_size=n_let_size
        )
        train_generator = TrainingBatchGenerator(spectrum_pair_generator=spectrum_pair_generator, settings=settings)
    # Create a validation loss calculator
    validation_loss_calculator = ValidationLossCalculator(validation_spectra,
                                                          settings=settings)

    model = SpectralNletModel(settings=settings)

    output_model_file_name = os.path.join(results_folder, settings.model_file_name)

    history = train(model,
                    train_generator,
                    num_epochs=settings.epochs,
                    learning_rate=settings.learning_rate,
                    validation_loss_calculator=validation_loss_calculator,
                    patience=settings.patience,
                    loss_function=settings.loss_function,
                    checkpoint_filename=output_model_file_name, lambda_l1=0, lambda_l2=0)
    model.export_to_onnx(results_folder)

    return model, history

model, history = train_ms2ds_model(
    training_spectra, validation_spectra, settings.model_directory_name, settings
)
