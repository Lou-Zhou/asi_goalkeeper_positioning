#convert checkpoint to mlflow model
import pandas as pd
from unxpass.components.withSpeeds import pass_success_speeds, pass_selection_speeds, pass_value_speeds
from unxpass.components.utils import log_model, load_model
from unxpass.components.withSpeeds.pass_selection_speeds import PytorchSoccerMapModel, SoccerMapComponent
from unxpass.components.withSpeeds.pass_success_speeds import PytorchSoccerMapModel, SoccerMapComponent
from unxpass.components.withSpeeds.pass_value_speeds import PytorchSoccerMapModel, SoccerMapComponent
import torch
import mlflow
import sys

def main():
    model_checkpoint_path = "/home/lz80/asi_goalkeeper_positioning/stores/model/363320086480977760/7762def8fdf64884af3101dcdfc331cd/artifacts/restored_model_checkpoint/epoch_054.ckpt"
    loaded_model = pass_selection_speeds.PytorchSoccerMapModel.load_from_checkpoint(model_checkpoint_path)

    model = SoccerMapComponent(model=loaded_model)
    mlflow.set_experiment("pass_value/soccermap")
    with mlflow.start_run() as run:
        # Log the model
        mlflow.pytorch.log_model(model.model, "model")

        # Retrieve the run ID
        run_id = run.info.run_id
        print(f"Pass Seletion Model saved with run_id: {run_id}")
main()
