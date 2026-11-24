
"""Wrapper functions for training and evaluating N-let MS2DeepScore models."""

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch

from ms2deepscore.SettingsMS2Deepscore import (
    SettingsMS2Deepscore,
    SettingsEmbeddingEvaluator,
)
from ms2deepscore.benchmarking.CalculateScoresBetweenAllIonmodes import (
    CalculateScoresBetweenAllIonmodes,
)
from ms2deepscore.models import EmbeddingEvaluationModel
from ms2deepscore.models.load_model import load_model
from ms2deepscore.train_new_model.train_ms2deepscore import (
    plot_history,
    save_history,
)
from ms2deepscore.validation_loss_calculation.ValidationLossCalculator import (
    ValidationLossCalculator,
)
from ms2deepscore.wrapper_functions.plotting_wrapper_functions import (
    create_plots_between_ionmodes,
)
from ms2deepscore.wrapper_functions.training_wrapper_functions import (
    load_spectra_in_ionmode,
    split_data_if_necessary,
)

from NletSpectralModel import NletSpectralModel, train_nlet
from NletTrainingBatchGenerator import NletTrainingBatchGenerator
from sampling import create_spectrum_nlet_generator


def train_nlet_ms2deepscore_wrapper(
    settings: SettingsMS2Deepscore,
    n: int,
    strategy: list[int],
    output_dir: Union[str, Path],
    pair_data_folder: Optional[Union[str, Path]] = None,
    settings_embedding_evaluator: Optional[SettingsEmbeddingEvaluator] = None,
    run_benchmark: bool = False,
    early_stopping: bool = True,
    trackio_project: str = "ms2deepscore-nlet",
    trackio_run_name: Optional[str] = None,
):
    """Split data, train an N-let model, and optionally run benchmarking.

    Reuses the original MS2DeepScore split, validation, history,
    benchmarking, and embedding-evaluation functionality.

    Trackio logging is handled exclusively by train_nlet().
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save settings.
    settings.save_to_file(str(output_dir / "settings.json"))

    # Reproducibility.
    if settings.random_seed is not None:
        np.random.seed(settings.random_seed)
        torch.manual_seed(settings.random_seed)
        torch.cuda.manual_seed_all(settings.random_seed)

    # Split and load data, as in the original wrapper.
    split_data_if_necessary(settings)

    training_spectra = load_spectra_in_ionmode(
        settings.training_spectra_file_name,
        settings.ionisation_mode,
    )
    validation_spectra = load_spectra_in_ionmode(
        settings.validation_spectra_file_name,
        settings.ionisation_mode,
    )

    # N-let-specific training data.
    spectrum_nlet_generator = create_spectrum_nlet_generator(
        spectra=training_spectra,
        settings=settings,
        n=n,
        strategy=strategy,
        pair_data_folder=pair_data_folder,
    )

    train_generator = NletTrainingBatchGenerator(
        spectrum_nlet_generator=spectrum_nlet_generator,
        settings=settings,
    )

    # Original validation calculator.
    validation_loss_calculator = ValidationLossCalculator(
        validation_spectra,
        settings=settings,
    )

    # N-let model.
    model = NletSpectralModel(settings=settings)

    best_model_file = output_dir / "best_nlet_model.pt"
    final_model_file = output_dir / "nlet_model.pt"

    # Train: Trackio is initialized, logged, and finished inside train_nlet.
    history = train_nlet(
        model=model,
        data_generator=train_generator,
        num_epochs=settings.epochs,
        learning_rate=settings.learning_rate,
        validation_loss_calculator=validation_loss_calculator,
        early_stopping=early_stopping,
        patience=settings.patience,
        checkpoint_filename=str(best_model_file),
        loss_function=settings.loss_function,
        weighting_factor=0,
        monitor_rmse=True,
        trackio_project=trackio_project,
        trackio_run_name=trackio_run_name,
    )

    # Save final model separately from best validation checkpoint.
    model.save(final_model_file)

    # Preserve original JSON history and history plot.
    save_history(
        str(output_dir / "history.json"),
        history,
    )

    plot_history(
        history["losses"],
        history["val_losses"],
        str(output_dir / "history.png"),
    )

    # Preserve the NPZ format from our previous training script.
    np.savez(
        output_dir / "training_history.npz",
        losses=np.asarray(history["losses"], dtype=np.float64),
        val_losses=np.asarray(history["val_losses"], dtype=np.float64),
        rmse=np.asarray(history["rmse"], dtype=np.float64),
        val_rmse=np.asarray(history["val_rmse"], dtype=np.float64),
    )

    # Original ion-mode benchmarking.
    # Requires the MS2DeepScore checkpoint loader to support NletSpectralModel.
    if run_benchmark:
        if settings.ionisation_mode == "both":
            positive_validation_spectra = load_spectra_in_ionmode(
                settings.validation_spectra_file_name, "positive"
            )
            negative_validation_spectra = load_spectra_in_ionmode(
                settings.validation_spectra_file_name, "negative"
            )

            if positive_validation_spectra and negative_validation_spectra:
                scores_between_all_ionmodes = CalculateScoresBetweenAllIonmodes(
                    model_file_name=str(best_model_file),
                    positive_validation_spectra=positive_validation_spectra,
                    negative_validation_spectra=negative_validation_spectra,
                    fingerprint_type=settings.fingerprint_type,
                    fingerprint_nbits=settings.fingerprint_nbits,
                )

                create_plots_between_ionmodes(
                    scores_between_all_ionmodes,
                    results_folder=str(output_dir / "benchmarking_results"),
                    nr_of_bins=50,
                )
            else:
                print("Skipping cross-ion-mode benchmarking: one ion mode is missing.")
        else:
            print(
                f"Skipping cross-ion-mode benchmarking: "
                f"ionisation_mode='{settings.ionisation_mode}'."
            )

    # Original optional embedding evaluator.
    if settings_embedding_evaluator is not None:
        best_model = load_model(best_model_file)

        embedding_evaluator = EmbeddingEvaluationModel(
            settings_embedding_evaluator
        )

        embedding_evaluator.train_evaluator(
            ms2ds_model=best_model,
            training_spectra=training_spectra,
            validation_spectra=validation_spectra,
        )

        embedding_evaluator.save(
            str(output_dir / settings_embedding_evaluator.model_file_name)
        )

    return str(output_dir)
