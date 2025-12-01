"""Generates all visualizations for a model on our test set, two plots side by side"""
from functools import partial
from sdm import visualize_features as vf
from unxpass.datasets import PassesDataset
from unxpass.components.withSpeeds import pass_selection_speeds
import mlflow
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb

from get_diffs import generate_gk_position_surface

OG_PATH = "/home/lz80/rdf/sp161/shared/asi_gk_pos/gk_pos_model/"
TEST = f"{OG_PATH}/poi"
OUTPUT_PATH = "/home/lz80/asi_goalkeeper_positioning/stores/visualizations/"
OUT_PDF = f"{OUTPUT_PATH}/selection_value_surfaces_poi.pdf"

dataset_test = partial(PassesDataset, path=TEST)

model_pass_selection = pass_selection_speeds.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        "runs:/ceea16e4a0254542bccf1f953f91381b/model",
        map_location="cpu",
    )
)

val_model = xgb.XGBRegressor()
val_model.load_model('/home/lz80/asi_goalkeeper_positioning/stores/model/value_model.model')

surfaces_sel = model_pass_selection.predict_surface(dataset_test)

ball_ff = pd.read_parquet(f"{TEST}/x_ball_freeze_frame.parquet")
ff = pd.read_parquet(f"{TEST}/x_player_freeze_frame.parquet")
value_features = pd.read_csv("/home/lz80/asi_goalkeeper_positioning/stores/value_features.csv").drop(columns = ['scores_xg'])
indices = list(ff.index)

with PdfPages(OUT_PDF) as pdf:
    for idx in tqdm(indices, desc="Rendering plots"):
        game, frame = idx
        surf_sel = surfaces_sel[game][frame]
        feats = value_features[(value_features['match_id'] == game) & (value_features['frame'] == frame)]
        feats.index = pd.MultiIndex.from_tuples(
            [idx], 
            names=["match_id", "frame"]
        ) #this is really dumb.
        surf_val = generate_gk_position_surface(idx, feats, ball_ff, ff, val_model)

        fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

        vf.plot_from_features(
            idx, ff, ball_ff, surface=surf_sel, ax=axes[0], log = True
        )
        axes[0].set_title("Pass Selection Surface")

        vf.plot_from_features(
            idx, ff, ball_ff, surface=surf_val, ax=axes[1], log = False, show_max= True
        )
        axes[1].set_title("Pass Value Surface")

        fig.suptitle(f"Game {game}, Frame {frame}", fontsize=14)
        pdf.savefig(fig, dpi=150, bbox_inches="tight")
        plt.close(fig)
