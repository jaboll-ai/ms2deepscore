from ms2deepscore.SettingsMS2Deepscore import SettingsMS2Deepscore, SettingsEmbeddingEvaluator
from ms2deepscore.wrapper_functions.training_wrapper_functions import train_ms2deepscore_wrapper

spectrum_file = "./20241003_enamdisc_neg_ms2.mgf"
# The settins below use default training settings and use precursor mz and ionmode as additional metadata input.
# Have a look in the SettingsMS2Deepscore class to check other hyperparameters.
settings = SettingsMS2Deepscore(
    spectrum_file_path=spectrum_file,
    additional_metadata=[("CategoricalToBinary", {"metadata_field": "ionmode",
                                                  "entries_becoming_one": "positive",
                                                  "entries_becoming_zero": "negative"}),
                         ("StandardScaler", {"metadata_field": "precursor_mz",
                                             "mean": 0, "standard_deviation": 1000})],
    ionisation_mode="both")

train_ms2deepscore_wrapper(
    settings,
    SettingsEmbeddingEvaluator() # this results in also training the embedding evaluator. Leave as None if you don't want to train this.
)