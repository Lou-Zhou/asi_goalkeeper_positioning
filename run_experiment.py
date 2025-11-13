from functools import partial
from pathlib import Path
from typing import Callable

import json
import hydra
from omegaconf import DictConfig, OmegaConf

from sdm import path_data


@hydra.main(version_base="1.2", config_path="config/", config_name="config.yaml")
def main(config: DictConfig):
    # Imports should be nested inside @hydra.main to optimize tab completion
    from unxpass import utils
    from unxpass.components.base import UnxpassComponent
    from unxpass.config import logger
    from unxpass.datasets import PassesDataset

    utils.extras(config)

    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    if config.get("seed"):
        utils.set_seeds(config.seed)

    logger.info("Instantiating training dataset")
    dataset_train: Callable = partial(
        PassesDataset,
        # subset of one game of code to test, should be changed when actually running
        path="/home/lz80/rdf/sp161/shared/asi_gk_pos/gk_pos_model/test",
    )

    logger.info(f"Instantiating model component <{config.component._target_}>")
    component: UnxpassComponent = hydra.utils.instantiate(
        config.component, _convert_="partial"
    )

    train_cfg = OmegaConf.to_object(config.get("train_cfg", DictConfig({})))
    utils.instantiate_callbacks(train_cfg)
    utils.instantiate_loggers(train_cfg)

    logger.info("⌛ Starting training!")
    result = component.train(
        dataset_train,
        optimized_metric=config.get("optimized_metric"),
        **train_cfg,
    )
    logger.info("✅ Finished training.")

    # ---- Make sure we return a scalar for Optuna ----
    optimized_metric = config.get("optimized_metric")
    if isinstance(result, dict):
        score = float(result[optimized_metric])
    else:
        score = float(result)

    # ---- Save params + score in this run directory ----
    run_summary = {
        "optimized_metric": optimized_metric,
        "score": score,
        "component": OmegaConf.to_container(config.component, resolve=True),
        "train_cfg": OmegaConf.to_container(config.get("train_cfg", {}), resolve=True),
    }

    # JSON version
    Path("run_summary.json").write_text(json.dumps(run_summary, indent=2))

    # Optional: YAML version (nice for Hydra reuse)
    summary_yaml = OmegaConf.to_yaml(OmegaConf.create(run_summary))
    Path("run_summary.yaml").write_text(summary_yaml)

    return score



if __name__ == "__main__":
    main()
"""
python3 run_experiment.py \
  experiment="pass_selection/soccermap" \
  hparams_search="soccermap_optuna" 


python3 run_experiment.py \
  experiment="pass_value/soccermap" \
  hparams_search="soccermap_optuna" 
"""