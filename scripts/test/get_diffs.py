from functools import partial
from unxpass.components.withSpeeds import pass_selection_speeds
from unxpass.datasets import PassesDataset

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from features.generate_value_features import compute_gk_feats

import pandas as pd
import numpy as np
import mlflow
import xgboost as xgb
from tqdm import tqdm

PITCH_LENGTH = 105
PITCH_WIDTH = 68
GOAL_WIDTH = 7.32
GOAL_CENTER_Y = PITCH_WIDTH / 2.0
HALF_GOAL = GOAL_WIDTH / 2.0
GOAL_X = PITCH_LENGTH

LEFT_POST_Y = GOAL_CENTER_Y - HALF_GOAL
RIGHT_POST_Y = GOAL_CENTER_Y + HALF_GOAL
def generate_gk_position_surface(idx,
    features_df, ball_ff, player_ff,
    model,
):
    """
    Generate a 2D surface of predicted positioning value over possible GK locations
    for a single play (ball_ff + player_ff fixed).

    Output is a FULL-PITCH grid:
      - values inside the penalty box = model predictions
      - values everywhere else = 0

    Returns:
      xs_full: 1D array of x grid points for full pitch
      ys_full: 1D array of y grid points for full pitch
      full_surface: 2D array of predictions (len(ys_full), len(xs_full))
      feats_df: DataFrame of feature rows used for predictions (only box points)
    """
    xs_full = np.arange(0.0, PITCH_LENGTH - 2 + 1e-6, 1)
    ys_full = np.arange(0.0, PITCH_WIDTH - 1 + 1e-6, 1)
    full_surface = np.zeros((len(ys_full), len(xs_full)), dtype=float)

    goal_center_y = PITCH_WIDTH / 2.0
    box_depth = 18
    box_width = 40.3
    half_box_width = box_width / 2.0

    y_min = goal_center_y - half_box_width
    y_max = goal_center_y + half_box_width


    x_max = PITCH_LENGTH
    x_min = PITCH_LENGTH - box_depth

    mask_x = (xs_full >= x_min) & (xs_full <= x_max)
    mask_y = (ys_full >= y_min) & (ys_full <= y_max)

    grid_points = []
    positions = [] 
    for j, gx in enumerate(xs_full):
        if not mask_x[j]:
            continue
        for i, gy in enumerate(ys_full):
            if not mask_y[i]:
                continue
            
            gk_ff = {
                "gk_frame_x": gx,
                "gk_frame_y": gy,
                "gk_frame_x_velo": 0.0,
                "gk_frame_y_velo": 0.0,
                "in_possession": False,
                "opp_gk": True,
                "player_id": -1,
            }
            feats = dict(features_df.loc[idx])
            
            gk_feats = compute_gk_feats(ball_ff.loc[idx], player_ff.loc[idx], gk_ff)
            feats.update(gk_feats)
            grid_points.append(feats)
            positions.append((i, j))

    if not grid_points:
        return xs_full, ys_full, full_surface, pd.DataFrame()

    feats_df = pd.DataFrame(grid_points).drop(columns=['match_id', 'frame'])
    feats
    preds = model.predict(feats_df)

    for (i, j), val in zip(positions, preds):
        full_surface[i, j] = val

    return full_surface

def main():

    val_model = xgb.XGBRegressor()
    val_model.load_model('/home/lz80/asi_goalkeeper_positioning/stores/model/value_model.model')


    model_pass_selection = pass_selection_speeds.SoccerMapComponent(
        model=mlflow.pytorch.load_model(
            'runs:/ceea16e4a0254542bccf1f953f91381b/model', map_location='cpu'
        )
    )

    test = "/home/lz80/rdf/sp161/shared/asi_gk_pos/gk_pos_model/illustration"

    dataset_test = partial(PassesDataset, path=test)

    sel_surfaces = model_pass_selection.predict_surface(dataset_test)

    value_features = pd.read_csv("/home/lz80/asi_goalkeeper_positioning/stores/value_features.csv").drop(columns = ['scores_xg'])

    gks = pd.read_parquet(f"{test}/x_gk_frame.parquet")

    feats_path = "/home/lz80/rdf/sp161/shared/asi_gk_pos/gk_pos_model/feats"
    ball_ff = pd.read_parquet(f"{feats_path}/x_ball_freeze_frame.parquet")
    player_ff = pd.read_parquet(f"{feats_path}/x_player_freeze_frame.parquet")
    value_features.index = ball_ff.index
    rows = []

    for idx in tqdm(gks.index):
        row = gks.loc[idx]
        gk_id = row['player_id']
        gk_x, gk_y = row[['gk_frame_x', 'gk_frame_y']]

        gk_x_bin = np.clip(gk_x, 0, 103).astype(np.uint8)
        gk_y_bin = np.clip(gk_y, 0, 68).astype(np.uint8)

        sel_surface = sel_surfaces[idx[0]][idx[1]]
        
        val_surface = generate_gk_position_surface(idx, value_features, ball_ff, player_ff, val_model)
        best = np.min(val_surface[val_surface > 0])
        expected = np.sum(sel_surface * val_surface)
        actual = val_surface[gk_y_bin, gk_x_bin]
        diff = expected - actual #lower is worse for the gk

        rows.append({
            "match_id": idx[0],
            "frame": idx[1],
            "gk_id": gk_id,
            "expected": expected,
            "actual": actual,
            "diff": diff,
            "optimal" : best
        })

    df = pd.DataFrame(rows)

    df = df.set_index(["match_id", "frame"]).sort_index()

    df.to_csv("/home/lz80/asi_goalkeeper_positioning/stores/frame_results_po.csv")

if __name__ == "__main__":
    main()