"""Visualizes events from parquet feature data"""
from typing import Dict, Tuple, List
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import numpy as np
def plot_from_features(idx : Tuple[str, int], ff: Dict[int, List[int]],
                       ball_ff : Dict[int, List[int]], surface : np.ndarray | None = None) -> None:
    """ 
    Plots an event from the raw feature set

    Parameters
    ------
        idx: Tuple[str, int]
            The index of the event
        ff: Dict[int, List[int]]
            The player freeze frame of the event
        ball_ff: Dict[int, List[int]]
            The ball freeze frame of the event
    """
    pitch = Pitch(pitch_type= "custom", pitch_length = 105, pitch_width = 68)
    fig, ax = pitch.draw()
    ff_event = ff.loc[idx]
    ball_ff_event = ball_ff.loc[idx]

    player_ff = ff_event['player_freeze_frame']
    ball_x = ball_ff_event['ball_freeze_frame_x']
    ball_y = ball_ff_event['ball_freeze_frame_y']

    ball_x_velo = ball_ff_event['ball_freeze_frame_x_velo']
    ball_y_velo = ball_ff_event['ball_freeze_frame_y_velo']

    offense = [player for player in player_ff if player['in_possession']]
    defense = [player for player in player_ff
               if not player['in_possession'] and not player['opp_gk']]

    offense_x = [player['x'] for player in offense]
    offense_y = [player['y'] for player in offense]

    offense_x_end = [player['x'] + player['x_velo'] for player in offense]
    offense_y_end = [player['y'] + player['y_velo'] for player in offense]

    defense_x = [player['x'] for player in defense]
    defense_y = [player['y'] for player in defense]

    defense_x_end = [player['x'] + player['x_velo'] for player in defense]
    defense_y_end= [player['y'] + player['y_velo'] for player in defense]

    gk_x = [player['x'] for player in player_ff if player['opp_gk']][0]
    gk_y = [player['y'] for player in player_ff if player['opp_gk']][0]

    gk_x_end = [player['x'] + player['x_velo'] for player in player_ff if player['opp_gk']][0]
    gk_y_end = [player['y'] + player['y_velo'] for player in player_ff if player['opp_gk']][0]

    pitch.scatter(offense_x, offense_y, ax=ax, marker = 'o',
                edgecolors ='blue', facecolors = 'none', label='Offense', s=50)
    pitch.scatter(defense_x, defense_y, ax=ax, marker = 'x',
                c='red', label='Defense', s=50)

    pitch.scatter(ball_x, ball_y, ax=ax, c='black', label='Ball', s=50)

    pitch.scatter(gk_x, gk_y, ax = ax, marker = 'o',
                edgecolors ='red', facecolors = 'none', label='Goalkeeper', s=50)

    pitch.arrows(offense_x, offense_y, offense_x_end, offense_y_end,
                width = 2, headwidth = 5, color = 'blue', ax = ax)

    pitch.arrows(defense_x, defense_y, defense_x_end, defense_y_end,
                width = 2, headwidth = 5, color = 'red', ax = ax)

    pitch.arrows(ball_x, ball_y, ball_x + ball_x_velo,
                 ball_y + ball_y_velo, width = 2, headwidth = 5, color = 'black', ax = ax)

    pitch.arrows(gk_x, gk_y, gk_x_end, gk_y_end,
                width = 2, headwidth = 5, color = 'red', ax = ax)

    if surface is not None:
        plt_settings = {"interpolation": "bilinear"}
        surface_kwargs={**plt_settings, "vmin": None, "vmax": None, "cmap": "Greens"}
        ax.imshow(surface, extent=[0.0, 105.0, 0.0, 68.0], origin="lower", **surface_kwargs)
    ax.legend()
    fig.tight_layout()
    return fig
    plt.show()
