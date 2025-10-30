"""Implements the pass selection component."""
import json
import math
from typing import Any, Dict, List, Optional

import hydra
import mlflow
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import xgboost as xgb
from rich.progress import track
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from unxpass.components.soccermap import SoccerMap, pixel
from unxpass.config import logger as log

from unxpass.components.base import UnxpassComponent, UnxPassPytorchComponent, UnxPassXGBoostComponent


class PassSelectionComponent(UnxpassComponent):
    """The pass selection component.

    From any given game situation where a player controls the ball, the model
    estimates the most likely destination of a potential pass.
    """

    component_name = "pass_selection_custom"

    def _get_metrics(self, y, y_hat):
        return {
            "log_loss": log_loss(y, y_hat, labels=[0, 1]),
            "brier": brier_score_loss(y, y_hat),
        }


class PytorchSoccerMapModel(pl.LightningModule):
    """A pass selection model based on the SoccerMap architecture."""

    def __init__(
        self,
        lr: float = 1e-5,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = SoccerMap(in_channels=13)
        self.softmax = nn.Softmax(2)

        # loss function
        self.criterion = torch.nn.BCELoss()

    def forward(self, x: torch.Tensor):
        x = self.model(x)
        x = self.softmax(x.view(*x.size()[:2], -1)).view_as(x)
        return x

    def step(self, batch: Any):
        x, mask, y = batch

        #print(torch.nonzero(torch.isnan(x)))
        surface = self.forward(x)

        y_hat = pixel(surface, mask)
        loss = self.criterion(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def predict_step(self, batch: Any, batch_idx: int):
        x, _, _ = batch
        surface = self(x)
        return surface

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr)


class ToSoccerMapTensor:
    """Convert inputs to a spatial representation.

    Parameters
    ----------
    dim : tuple(int), default=(68, 104)
        The dimensions of the pitch in the spatial representation.
        The original pitch dimensions are 105x68, but even numbers are easier
        to work with.
    """

    def __init__(self, dim=(68, 104)):
        assert len(dim) == 2
        self.y_bins, self.x_bins = dim

    def _get_cell_indexes(self, x, y):
        x_bin = np.clip(x / 105 * self.x_bins, 0, self.x_bins - 1).astype(np.uint8)
        y_bin = np.clip(y / 68 * self.y_bins, 0, self.y_bins - 1).astype(np.uint8)
        return x_bin, y_bin

    def __call__(self, sample):
        
        ball_x, ball_y = (sample['ball_freeze_frame_x'], sample['ball_freeze_frame_y'])
        ball_x_velo, ball_y_velo = (sample['ball_freeze_frame_x_velo'], sample['ball_freeze_frame_y_velo'])
        frame= pd.DataFrame.from_records(sample['player_freeze_frame'])
        

        gk_x, gk_y = (sample['gk_frame_x'], sample['gk_frame_y'])


        # Location of the ball
        ball_coo = np.array([[ball_x, ball_y]])
        # Location of the goal
        goal_coo = np.array([[105, 34]])
        
        # Locations of the passing player's teammates
        players_att_coo = frame.loc[frame.in_possession, ["x", "y"]].values.reshape(
            -1, 2
        )
        # Locations and speed vector of the defending players
        players_def_coo = frame.loc[~frame.in_possession & ~frame.opp_gk, ["x", "y"]].values.reshape(-1, 2)

        
        # Output
        matrix = np.zeros((13, self.y_bins, self.x_bins))#3D, 14 x 68 x 104

        # CH 1: Locations of attacking team
        x_bin_att, y_bin_att = self._get_cell_indexes(
            players_att_coo[:, 0],
            players_att_coo[:, 1],
        )
        matrix[0, y_bin_att, x_bin_att] = 1

        # CH 2: Locations of defending team
        x_bin_def, y_bin_def = self._get_cell_indexes(
            players_def_coo[:, 0],
            players_def_coo[:, 1],
        )
        matrix[1, y_bin_def, x_bin_def] = 1

        # CH 3: Distance to ball
        yy, xx = np.ogrid[0.5 : self.y_bins, 0.5 : self.x_bins]

        x0_ball, y0_ball = self._get_cell_indexes(ball_coo[:, 0], ball_coo[:, 1])
        matrix[2, :, :] = np.sqrt((xx - x0_ball) ** 2 + (yy - y0_ball) ** 2)

        # CH 4: Distance to goal
        x0_goal, y0_goal = self._get_cell_indexes(goal_coo[:, 0], goal_coo[:, 1])
        matrix[3, :, :] = np.sqrt((xx - x0_goal) ** 2 + (yy - y0_goal) ** 2)

        # CH 5: Cosine of the angle between the ball and goal
        coords = np.dstack(np.meshgrid(xx, yy))
        goal_coo_bin = np.concatenate((x0_goal, y0_goal))
        ball_coo_bin = np.concatenate((x0_ball, y0_ball))
        a = goal_coo_bin - coords
        b = ball_coo_bin - coords
        matrix[4, :, :] = np.clip(
            np.sum(a * b, axis=2) / (np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2)), -1, 1
        )

        # CH 6: Sine of the angle between the ball and goal
        # sin = np.cross(a,b) / (np.linalg.norm(a, axis=2) * np.linalg.norm(b, axis=2))
        matrix[5, :, :] = np.sqrt(1 - matrix[4, :, :] ** 2)  # This is much faster

        # CH 7: Angle (in radians) to the goal location
        matrix[6, :, :] = np.abs(
            np.arctan((y0_goal - coords[:, :, 1]) / (x0_goal - coords[:, :, 0]))
        )

        # CH 8-9: Ball speed 
        matrix[7, y0_ball, x0_ball] = ball_x_velo
        matrix[8, y0_ball, x0_ball] = ball_y_velo

        # Get velocities
        players_att_vx = frame.loc[frame.in_possession, "x_velo"].values  # shape: (n_att_players,)
        players_att_vy = frame.loc[frame.in_possession, "y_velo"].values  # shape: (n_att_players,)

        # Similarly for defending players, using the correct condition (assuming defenders are non-teammates)
        players_def_vx = frame.loc[~frame.in_possession & ~frame.opp_gk, "x_velo"].values  # shape: (n_def_players,)
        players_def_vy = frame.loc[~frame.in_possession & ~frame.opp_gk, "y_velo"].values
        # Channels 10 and 11, attacking team velocities 
        matrix[9, y_bin_att, x_bin_att] = players_att_vx
        matrix[10, y_bin_att, x_bin_att] = players_att_vy

        #print(players_att_vx)
        #print(players_att_vy)

        # Channel 12 & 13: Defending player velocities (x and y)
        matrix[11, y_bin_def, x_bin_def] = players_def_vx
        matrix[12, y_bin_def, x_bin_def] = players_def_vy
        #print(players_def_vx)
        #print(players_def_vy)
        mask = np.zeros((1, self.y_bins, self.x_bins))
        end_gk_coo = np.array([[gk_x, gk_y]])
        if np.isnan(end_gk_coo).any():
            raise ValueError("End coordinates not known.")
        x0_gk_end, y0_gk_end = self._get_cell_indexes(end_gk_coo[:, 0], end_gk_coo[:, 1])

        mask[0, y0_gk_end, x0_gk_end] = 1

        return (
            torch.from_numpy(matrix).float(),
            torch.from_numpy(mask).float(),
            torch.tensor([1]).float(),
        )


class SoccerMapComponent(PassSelectionComponent, UnxPassPytorchComponent):
    """A SoccerMap deep-learning model."""

    def __init__(self, model: PytorchSoccerMapModel):
        super().__init__(
            model=model,
            features={
                "player_freeze_frame": ["player_freeze_frame"],
                "ball_freeze_frame": ["ball_freeze_frame_x", "ball_freeze_frame_y", 
                                      "ball_freeze_frame_x_velo", "ball_freeze_frame_y_velo"],
                "gk_frame": ['gk_frame_x', 'gk_frame_y'],
            },
            label=["success"],  # just a dummy lalel
            transform=ToSoccerMapTensor(dim=(68, 104)),
        )

    def test(self, dataset, batch_size=1, num_workers=0, pin_memory=False, **test_cfg) -> Dict:
        # Load dataset
        data = self.initialize_dataset(dataset)
        dataloader = DataLoader(
            data,
            shuffle=False,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        # Switch to test mode
        torch.set_grad_enabled(False)
        self.model.eval()

        # Apply model on test set
        all_preds, all_targets = [], []
        for batch in track(dataloader):
            x, mask, _ = batch
            surface = self.model(x)
            for i in range(x.shape[0]):
                teammate_locations = torch.nonzero(x[i, 0, :, :])
                if len(teammate_locations) > 0:
                    p_teammate_selection = surface[
                        i, 0, teammate_locations[:, 0], teammate_locations[:, 1]
                    ]
                    selected_teammate = torch.argmin(
                        torch.cdist(torch.nonzero(mask[i, 0]).float(), teammate_locations.float())
                    )
                    all_targets.append(
                        (torch.argmax(p_teammate_selection) == selected_teammate).item()
                    )
                else:
                    all_targets.append(True)
            y_hat = pixel(surface, mask)
            all_preds.append(y_hat)
        all_preds = torch.cat(all_preds, dim=0).detach().numpy()[:, 0]

        # Compute metrics
        y = np.ones(len(all_preds))
        return {
            "log_loss": log_loss(y, all_preds, labels=[0, 1]),
            "brier": brier_score_loss(y, all_preds),
            "acc": sum(all_targets) / len(all_targets),
        }
