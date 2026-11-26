from time import perf_counter
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
import trackio

from torch.optim import AdamW

from ms2deepscore.models.SiameseSpectralModel import SiameseSpectralModel
from ms2deepscore.models.helper_functions import (
    initialize_device,
    l1_regularization,
    l2_regularization,
)
from ms2deepscore.models.loss_functions import LOSS_FUNCTIONS, rmse_loss


class NletSpectralModel(SiameseSpectralModel):
    """Spectral model that encodes N spectra using a shared encoder.

    Input tensors have shape (batch_size, N, features).
    Returns embeddings of shape (batch_size, N, embedding_dim).

    N can vary between batches.
    """
    def forward(self, spectra_tensors, metadata_tensors):
        batch_size, n, _ = spectra_tensors.shape

        # Flatten N-let dimension into batch dimension.
        spectra_tensors = spectra_tensors.reshape(batch_size * n, -1)
        metadata_tensors = metadata_tensors.reshape(batch_size * n, -1)

        # Pass all spectra through the same encoder.
        embeddings = self.encoder(
            spectra_tensors,
            metadata_tensors,
        )

        # Restore N-let dimension.
        return embeddings.reshape(batch_size, n, -1)


def train_nlet(
    model: NletSpectralModel,
    data_generator,
    num_epochs: int,
    learning_rate: float,
    weight_decay: float = 0.0,
    validation_loss_calculator=None,
    early_stopping: bool = True,
    patience: int = 10,
    checkpoint_filename: Optional[str] = None,
    loss_function: str = "MSE",
    weighting_factor: float = 0,
    monitor_rmse: bool = True,
    collect_all_targets: bool = False,
    lambda_l1: float = 0,
    lambda_l2: float = 0,
    trackio_project: str = "ms2deepscore-nlet",
    trackio_run_name: Optional[str] = None,
):
    """Train an N-let spectral model using pairwise cosine similarities.

    Each training batch contains:
        spectra:  (batch_size, N, number_of_bins)
        metadata: (batch_size, N, metadata_features)
        targets:  (batch_size, N, N)

    The model produces embeddings of shape (batch_size, N, embedding_dim).

    Training compares cosine similarities with molecular Tanimoto
    similarities for all unique pairs in each N-let.

    Only the upper triangle of the similarity matrix is used,
    excluding the diagonal.

    Training and validation metrics are logged using Trackio.
    """

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

    # Initialize Trackio.
    trackio.init(
        project=trackio_project,
        name=trackio_run_name,
        config={
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
        # Ten Tanimoto bins, ordered from 0 to 1.
        similarity_edges = np.linspace(0.0, 1.0, 11)

        # Counts accumulated throughout this training run.
        cumulative_pair_counts = np.zeros(10, dtype=np.int64)
        training_start = perf_counter()
        for epoch in range(num_epochs):
            epoch_start = perf_counter()
            model.train(True)

            batch_losses = []
            batch_rmse = []

            for spectra, metadata, targets in data_generator:
                spectra = spectra.to(device)
                metadata = metadata.to(device)
                targets = targets.to(device)
                # The anchor is always at index 0.
                # Remove it to retain only partner–partner similarities.
                partner_targets = targets[:, 1:, 1:]

                # Extract unique pairs, excluding the diagonal.
                i, j = torch.triu_indices(
                    partner_targets.shape[1],
                    partner_targets.shape[2],
                    offset=1,
                    device=device,
                )

                indirect_similarities = partner_targets[:, i, j]

                # Count the similarities in ten Tanimoto bins.
                batch_counts = torch.histc(
                    indirect_similarities.float(),
                    bins=10,
                    min=0.0,
                    max=1.0,
                )

                cumulative_pair_counts += (
                    batch_counts.cpu().numpy().astype(np.int64)
                )

                optimizer.zero_grad()

                # Forward pass: (batch_size, N, embedding_dim).
                embeddings = model(spectra, metadata)

                # Normalize embeddings for cosine similarity.
                normalized_embeddings = F.normalize(
                    embeddings,
                    p=2,
                    dim=-1,
                    eps=1e-6,
                )

                # Full cosine similarity matrix: (batch_size, N, N).
                cosine_matrix = normalized_embeddings @ normalized_embeddings.transpose(1, 2)

                # Select unique pairs, excluding self-similarity.
                n = embeddings.shape[1]

                pair_indices = torch.triu_indices(n, n, offset=1, device=device)

                outputs = cosine_matrix[:, pair_indices[0], pair_indices[1]]

                pair_targets = targets[:, pair_indices[0], pair_indices[1]]

                # Optionally collect target values.
                if collect_all_targets:
                    history["collection_targets"].extend(
                        pair_targets.detach().cpu().flatten().tolist()
                    )

                # Original MS2DeepScore loss function.
                loss = criterion(
                    outputs,
                    pair_targets,
                    weighting_factor=weighting_factor,
                )

                # Preserve regularization.
                if lambda_l1 > 0 or lambda_l2 > 0:
                    loss += l1_regularization(model, lambda_l1) + l2_regularization(
                        model, lambda_l2
                    )

                loss_value = loss.detach().item()
                batch_losses.append(loss_value)

                if monitor_rmse:
                    batch_rmse.append(
                        rmse_loss(
                            outputs.detach(),
                            pair_targets,
                        ).item()
                    )

                # Backward pass and optimization.
                loss.backward()
                optimizer.step()

            if not batch_losses:
                raise ValueError("Training generator produced no batches.")

            epoch_loss = float(np.mean(batch_losses))
            epoch_rmse = float(np.mean(batch_rmse)) if monitor_rmse else None

            history["losses"].append(epoch_loss)
            history["rmse"].append(epoch_rmse)

            # Trackio epoch metrics.
            metrics = {
                "train/loss": epoch_loss,
                "epoch": epoch + 1,
            }

            if monitor_rmse:
                metrics["train/rmse"] = epoch_rmse

            # Preserve existing validation interface.
            if validation_loss_calculator is not None:
                model.eval()

                with torch.no_grad():
                    val_losses, _ = (
                        validation_loss_calculator.compute_binned_validation_loss(
                            model,
                            loss_types=(loss_function, "rmse"),
                        )
                    )

                val_loss = float(val_losses[loss_function])
                val_rmse = float(val_losses["rmse"])

                history["val_losses"].append(val_loss)
                history["val_rmse"].append(val_rmse)

                metrics["validation/loss"] = val_loss
                metrics["validation/rmse"] = val_rmse

                if val_loss < min_val_loss:
                    if checkpoint_filename:
                        model.save(checkpoint_filename)

                    epochs_no_improve = 0
                    min_val_loss = val_loss
                else:
                    epochs_no_improve += 1

            metrics["time/epoch_seconds"] = perf_counter() - epoch_start
            metrics["time/total_minutes"] = (perf_counter() - training_start) / 60
            # Cumulative distribution of indirect Tanimoto similarities.
            metrics["sampling/indirect_tanimoto"] = trackio.Histogram(
                np_histogram=(
                    cumulative_pair_counts.copy(),
                    similarity_edges,
                )
            )

            # Log percentages as ordinary scalar metrics too.
            total = cumulative_pair_counts.sum()

            if total > 0:
                for bin_idx in range(10):
                    metrics[f"sampling/bin_{bin_idx}_pct"] = (
                        100.0 * cumulative_pair_counts[bin_idx] / total
                    )

            trackio.log(metrics, step=epoch + 1)
            if (
                validation_loss_calculator is not None
                and early_stopping
                and epochs_no_improve >= patience
            ):
                print(f"Early stopping at epoch {epoch + 1}.")
                break

        return history

    finally:
        trackio.finish()
