"""Generates all visualizations for a model on our test set"""
from functools import partial
from sdm import visualize_features as vf
from unxpass.datasets import PassesDataset
from unxpass.components.withSpeeds import pass_selection_speeds, pass_value_speeds
import mlflow
from tqdm import tqdm
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import matplotlib.pyplot as plt
#
#git remote set-url origin https://YOUR_TOKEN_HERE@github.com/your_user/your_project.git



OG_PATH = "/home/lz80/rdf/sp161/shared/asi_gk_pos/gk_pos_model/"
TEST = f"{OG_PATH}/test"
OUTPUT_PATH = "/home/lz80/asi_goalkeeper_positioning/stores/visualizations/"
OUT_PDF = f"{OUTPUT_PATH}/value_surfaces.pdf"
dataset_test = partial(PassesDataset, path=TEST)

model_pass_value = pass_value_speeds.SoccerMapComponent(
    model=mlflow.pytorch.load_model(
        'runs:/5d9a6f3b0d64447588577d748c8b4edf/model', map_location='cpu'
    )
)

surfaces = model_pass_value.predict_surface(dataset_test)

ball_ff = pd.read_parquet(f"{TEST}/x_ball_freeze_frame.parquet")
ff = pd.read_parquet(f"{TEST}/x_player_freeze_frame.parquet")
indices = list(ff.index)
with PdfPages(OUT_PDF) as pdf:
    for k, idx in enumerate(tqdm(indices, desc="Rendering plots")):
        game, frame = idx
        surface = surfaces[game][frame]
        fig = vf.plot_from_features(idx, ff, ball_ff, surface=surface)
        fig.suptitle(f"Game {game}, Frame {frame}")
        pdf.savefig(fig, dpi=150, bbox_inches="tight")
        plt.close(fig)
