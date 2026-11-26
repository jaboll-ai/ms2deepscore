from pathlib import Path

from nlet_training_wrapper import train_nlet_ms2deepscore_wrapper
from hyperp import *

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


strategies = {
    # 1. Balanced across similarity ranges
    "BALANCED": [0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    # 2. Cluster + background
    "CLUSTER_BACKGROUND": [1, 2, 0, 0, 0, 0, 0, 0, 1, 3],
    # 3. Low-similarity only (NOT truly random)
    "LOW_SIMILARITY": [0, 0, 0, 0, 0, 0, 0, 0, 3, 4],
    # 4. Diverse / mostly dissimilar
    "DIVERSE": [0, 0, 0, 0, 0, 0, 1, 1, 2, 3],
}

PAIR_CACHE_DIR = Path("./runs/nlet_cache")



for strat_name, strat in strategies.items():
    RUN_NAME = f"{num2tuple(N)}-s_{strat_name}-e{NUM_EPOCHS}-b{settings.batch_size}-ion_{settings.ionisation_mode}"
    OUTPUT_DIR = Path(f"./runs/{RUN_NAME}")

    try:
        train_nlet_ms2deepscore_wrapper(
            settings=settings,
            n=N,
            strategy=strat,
            output_dir=OUTPUT_DIR,
            pair_data_folder=PAIR_CACHE_DIR,
            settings_embedding_evaluator=embedding_settings,
            early_stopping=True,
            run_benchmark=True,
            trackio_project=TRACKIO_NAME,
            trackio_run_name=RUN_NAME,
        )
    except Exception as e:
        print(f"{strat_name} failed: {e}")
        continue