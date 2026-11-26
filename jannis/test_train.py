import os
from pathlib import Path
from time import perf_counter
from typing import Optional, Union

import numpy as np
from torch.optim import AdamW
from tqdm import tqdm
import trackio

from ms2deepscore.SettingsMS2Deepscore import SettingsEmbeddingEvaluator, SettingsMS2Deepscore
from ms2deepscore.benchmarking.CalculateScoresBetweenAllIonmodes import CalculateScoresBetweenAllIonmodes
from ms2deepscore.models.EmbeddingEvaluatorModel import EmbeddingEvaluationModel
from ms2deepscore.models.SiameseSpectralModel import SiameseSpectralModel
from ms2deepscore.models.helper_functions import initialize_device, l1_regularization, l2_regularization
from ms2deepscore.models.loss_functions import LOSS_FUNCTIONS, rmse_loss
from ms2deepscore.train_new_model.TrainingBatchGenerator import TrainingBatchGenerator
from ms2deepscore.train_new_model.inchikey_pair_selection import create_spectrum_pair_generator
from ms2deepscore.train_new_model.inchikey_pair_selection_cross_ionmode import create_data_generator_across_ionmodes
from ms2deepscore.train_new_model.train_ms2deepscore import plot_history, save_history
from ms2deepscore.validation_loss_calculation.ValidationLossCalculator import ValidationLossCalculator
from ms2deepscore.wrapper_functions.plotting_wrapper_functions import create_plots_between_ionmodes
from ms2deepscore.wrapper_functions.training_wrapper_functions import load_spectra_in_ionmode, split_data_if_necessary, train_ms2deepscore_wrapper
from torch.utils.tensorboard import SummaryWriter

from hyperp import *

def train(
    model: SiameseSpectralModel,
    data_generator,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    validation_loss_calculator=None,
    early_stopping=True,
    patience: int = 10,
    checkpoint_filename: Optional[str] = None,
    loss_function="MSE",
    weighting_factor=0,
    monitor_rmse: bool = True,
    collect_all_targets: bool = False,
    lambda_l1: float = 0,
    lambda_l2: float = 0,
    progress_bar: bool = True,
    log_dir: str = "./runs",
    trackio_project: str = "ms2deepscore-nlet",
    trackio_run_name: Optional[str] = None,
):
    """Train the original Siamese model with TensorBoard and Trackio logging."""

    device = initialize_device()
    model.to(device)

    if loss_function.lower() not in LOSS_FUNCTIONS:
        raise ValueError(
            f"Unknown loss function. Must be one of: {LOSS_FUNCTIONS.keys()}"
        )

    criterion = LOSS_FUNCTIONS[loss_function.lower()]

    optimizer = AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        foreach=False,
        fused=False,
    )

    # Preserve original TensorBoard logging.
    if checkpoint_filename:
        writer = SummaryWriter(
            log_dir=os.path.join(
                log_dir,
                checkpoint_filename.split(".")[0],
            )
        )
    else:
        writer = SummaryWriter(log_dir=log_dir)

    history = {
        "losses": [],
        "val_losses": [],
        "rmse": [],
        "val_rmse": [],
        "collection_targets": [],
    }

    min_val_loss = np.inf
    epochs_no_improve = 0

    try:
        # Initialize Trackio.
        trackio.init(
            project=trackio_project,
            name=trackio_run_name,
            config={
                "model": "SiameseSpectralModel",
                "n": 2,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "weight_decay": weight_decay,
                "loss_function": loss_function,
                "weighting_factor": weighting_factor,
                "patience": patience,
                "lambda_l1": lambda_l1,
                "lambda_l2": lambda_l2,
            },
        )

        training_start = perf_counter()

        for epoch in range(num_epochs):
            epoch_start = perf_counter()

            model.train(True)

            with tqdm(
                data_generator,
                unit="batch",
                mininterval=0,
                disable=(not progress_bar),
            ) as training:

                training.set_description(f"Epoch {epoch}")

                batch_losses = []
                batch_rmse = []

                for spectra_1, spectra_2, meta_1, meta_2, targets in training:
                    if collect_all_targets:
                        history["collection_targets"].extend(targets)

                    optimizer.zero_grad()

                    # Original forward pass.
                    outputs = model(
                        spectra_1.to(device),
                        spectra_2.to(device),
                        meta_1.to(device),
                        meta_2.to(device),
                    )

                    # Original loss function.
                    loss = criterion(
                        outputs,
                        targets.to(device),
                        weighting_factor=weighting_factor,
                    )

                    if lambda_l1 > 0 or lambda_l2 > 0:
                        loss += l1_regularization(
                            model, lambda_l1
                        ) + l2_regularization(
                            model, lambda_l2
                        )

                    loss_value = loss.detach().item()
                    batch_losses.append(loss_value)

                    if monitor_rmse:
                        batch_rmse.append(
                            rmse_loss(
                                outputs,
                                targets.to(device),
                            ).cpu().detach().numpy()
                        )

                    # Original backward pass.
                    loss.backward()
                    optimizer.step()

                    # Preserve original tqdm output.
                    training.set_postfix(
                        loss=loss_value,
                        rmse=np.mean(batch_rmse),
                    )

            epoch_loss = np.mean(batch_losses)
            epoch_rmse = (
                np.mean(batch_rmse) if monitor_rmse else None
            )

            history["losses"].append(epoch_loss)
            history["rmse"].append(epoch_rmse)

            # Preserve TensorBoard.
            writer.add_scalar("Loss/train", epoch_loss, epoch)

            # Trackio training metrics.
            metrics = {
                "epoch": epoch + 1,
                "train/loss": float(epoch_loss),
            }

            if monitor_rmse:
                metrics["train/rmse"] = float(epoch_rmse)

            # Original validation.
            if validation_loss_calculator is not None:
                val_losses, _ = (
                    validation_loss_calculator.compute_binned_validation_loss(
                        model,
                        loss_types=(loss_function, "rmse"),
                    )
                )

                val_loss = val_losses[loss_function]
                val_rmse = val_losses["rmse"]

                history["val_losses"].append(val_loss)
                history["val_rmse"].append(val_rmse)

                # Preserve TensorBoard.
                writer.add_scalar("Loss/validation", val_loss, epoch)
                writer.add_scalar("RMSE/validation", val_rmse, epoch)

                # Trackio validation metrics.
                metrics["validation/loss"] = float(val_loss)
                metrics["validation/rmse"] = float(val_rmse)

                # Original checkpoint selection.
                if val_loss < min_val_loss:
                    if checkpoint_filename:
                        print("Saving checkpoint model.")
                        model.save(checkpoint_filename)

                    epochs_no_improve = 0
                    min_val_loss = val_loss
                else:
                    epochs_no_improve += 1

            # Trackio timing metrics.
            metrics["time/epoch_seconds"] = (
                perf_counter() - epoch_start
            )

            metrics["time/total_minutes"] = (
                perf_counter() - training_start
            ) / 60

            # Log the completed epoch.
            trackio.log(metrics, step=epoch + 1)

            # Preserve original console output.
            print(
                f"Epoch [{epoch + 1}/{num_epochs}], "
                f"Loss: {np.mean(batch_losses):.4f}"
            )

            if validation_loss_calculator is not None:
                print(
                    f"Validation Loss: {val_loss:.4f} "
                    f"(RMSE: {val_losses['rmse']:.4f})."
                )

            # Original early-stopping condition.
            if (
                validation_loss_calculator is not None
                and early_stopping
                and epochs_no_improve >= patience
            ):
                print("Early stopping!")
                break

        return history

    finally:
        writer.close()
        trackio.finish()

def train_ms2ds_model(
    training_spectra,
    validation_spectra,
    results_folder,
    settings: SettingsMS2Deepscore,
    pair_data_folder: Union[str, Path, None] = None,
    trackio_project: str = "ms2deepscore-nlet",
    trackio_run_name: Optional[str] = None,
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
            training_spectra,
            settings=settings,
            pair_data_folder=pair_data_folder,
        )
    else:
        spectrum_pair_generator = create_spectrum_pair_generator(
            training_spectra,
            settings=settings,
            pair_data_folder=pair_data_folder,
        )
        train_generator = TrainingBatchGenerator(
            spectrum_pair_generator=spectrum_pair_generator,
            settings=settings,
        )

    # Create a validation loss calculator
    validation_loss_calculator = ValidationLossCalculator(
        validation_spectra,
        settings=settings,
    )

    model = SiameseSpectralModel(settings=settings)

    output_model_file_name = os.path.join(
        results_folder,
        settings.model_file_name,
    )

    history = train(
        model,
        train_generator,
        num_epochs=settings.epochs,
        learning_rate=settings.learning_rate,
        validation_loss_calculator=validation_loss_calculator,
        patience=settings.patience,
        loss_function=settings.loss_function,
        checkpoint_filename=output_model_file_name,
        lambda_l1=0,
        lambda_l2=0,
        trackio_project=trackio_project,
        trackio_run_name=trackio_run_name,
    )

    model.export_to_onnx(results_folder)

    return model, history

def train_ms2deepscore_wrapper(
    settings: SettingsMS2Deepscore,
    settings_embedding_evaluator: SettingsEmbeddingEvaluator = None,
    pair_data_folder: Optional[Union[str, Path]] = None,
    trackio_project: str = "ms2deepscore-nlet",
    trackio_run_name: Optional[str] = None,
):
    """Splits data, trains a ms2deepscore model, and does benchmarking.

    If the data split was already done, the data split will be reused.

    spectra_file_path:
        The path to the spectra that should be used for training.
        (it will be split in train, val and test)
    settings:
        An object with the MS2Deepscore model settings.
    validation_split_fraction:
        The fraction of the inchikeys that will be used for validation and test.
    pair_data_folder:
        Explicit folder for storing/reusing expensive training-pair preparation.
        On the first run an empty/new folder is populated. On later runs the same
        folder is reused only if its manifest matches the current structural data
        and pair-selection settings; otherwise a ValueError is raised.
    """
    split_data_if_necessary(settings)

    training_spectra = load_spectra_in_ionmode(
        settings.training_spectra_file_name,
        settings.ionisation_mode,
    )
    validation_spectra = load_spectra_in_ionmode(
        settings.validation_spectra_file_name,
        settings.ionisation_mode,
    )

    # Train model
    ms2ds_model, history = train_ms2ds_model(
        training_spectra,
        validation_spectra,
        settings.model_directory_name,
        settings,
        pair_data_folder=pair_data_folder,
        trackio_project=trackio_project,
        trackio_run_name=trackio_run_name,
    )

    ms2ds_history_plot_file_name = os.path.join(
        settings.model_directory_name,
        settings.history_plot_file_name,
    )
    plot_history(
        history["losses"],
        history["val_losses"],
        ms2ds_history_plot_file_name,
    )
    save_history(
        os.path.join(settings.model_directory_name, "history.json"),
        history,
    )

    scores_between_all_ionmodes = CalculateScoresBetweenAllIonmodes(
        os.path.join(settings.model_directory_name, settings.model_file_name),
        load_spectra_in_ionmode(
            settings.validation_spectra_file_name, "positive"
        ),
        load_spectra_in_ionmode(
            settings.validation_spectra_file_name, "negative"
        ),
        settings.fingerprint_type,
        settings.fingerprint_nbits,
    )

    # Create performance plots for validation spectra
    create_plots_between_ionmodes(
        scores_between_all_ionmodes,
        results_folder=os.path.join(
            settings.model_directory_name,
            "benchmarking_results",
        ),
        nr_of_bins=50,
    )

    if settings_embedding_evaluator:
        model = EmbeddingEvaluationModel(settings_embedding_evaluator)
        model.train_evaluator(
            ms2ds_model=ms2ds_model,
            training_spectra=training_spectra,
            validation_spectra=validation_spectra,
        )
        model.save(
            os.path.join(
                settings.model_directory_name,
                settings_embedding_evaluator.model_file_name,
            )
        )

    return settings.model_directory_name

train_ms2deepscore_wrapper(
    settings,
    embedding_settings,
    Path("./runs/pair_cache"),
    TRACKIO_NAME,
    "siamese-BASELINE"
)