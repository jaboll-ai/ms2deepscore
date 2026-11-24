from ms2deepscore.models.SiameseSpectralModel import SiameseSpectralModel


class SpectralNletModel(SiameseSpectralModel):

    def forward(self, spectra_tensors, metadata_tensors):
        batch_size, n_let_size, n_peaks = spectra_tensors.shape
        n_metadata = metadata_tensors.shape[-1]

        flat_peaks = spectra_tensors.reshape(
            batch_size * n_let_size, n_peaks
        )
        flat_metadata = metadata_tensors.reshape(
            batch_size * n_let_size, n_metadata
        )

        embeddings = self.encoder(flat_peaks, flat_metadata)

        return embeddings.reshape(batch_size, n_let_size, -1)