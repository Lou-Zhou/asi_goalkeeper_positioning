"""Visualization functions for SkillCorner tracking data"""

from ast import literal_eval

from typing import Optional, List, Dict, Callable, Iterable
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch

from matplotlib.patches import Polygon
import matplotlib.animation as animation
from matplotlib.axes import Axes


def get_players(game: int, metadata :  pd.DataFrame,
                eval_func : Callable[[str],
                        Iterable] = literal_eval) -> tuple[list[int], list[int]]:
    """
    Cleans the players list from metadata and returns home and away player ids

    Args:
        game: int
            The game id
        metadata: pd.DataFrame
            The metadata dataframe
        eval_func : function

    Returns:
        home_players: List[int]
            List of home player ids
        away_players: List[int]
            List of away player ids
    """
    game_meta = metadata[metadata['id'] == game].iloc[0]
    players = eval_func(game_meta['players'])
    home_players = [(p['id'],  p['player_role']['id'])for p in players
                    if p['team_id'] == eval_func(game_meta['home_team'])['id']]
    away_players = [(p['id'], p['player_role']['id']) for p in players
                    if p['team_id'] == eval_func(game_meta['away_team'])['id']]
    return home_players, away_players


def plot_objects(pitch : Pitch, ax : Axes, tracking : pd.DataFrame,
                 frame : int, home_players : Optional[List[int]] = None) -> Dict[str, str | float]:
    """
    Plots the players, ball, and camera view on the pitch for a given frame.

    Args:
        pitch: Pitch
            A mplsoccer Pitch object
        ax: Axes 
            A matplotlib Axes object
        tracking: pd.DataFrame
            The tracking dataframe
        frame: int
            The frame number to plot
        home_players : List[int], Optional
            List of home player ids. Defaults to None.
    Returns:
        dict: metadata about the frame (frame number, period, 
        timestamp, possession player id,
        possession group)
    """
    frame_data = tracking[tracking['frame'] == frame].iloc[0]
    image_corners = literal_eval(frame_data['image_corners_projection'])
    verts = [ #building camera view polygon
        (image_corners['x_top_left'], image_corners['y_top_left']),
        (image_corners['x_top_right'], image_corners['y_top_right']),
        (image_corners['x_bottom_right'], image_corners['y_bottom_right']),
        (image_corners['x_bottom_left'], image_corners['y_bottom_left'])
    ]
    #poly = Polygon(verts, closed=True, fill=True, linewidth=2,
    #               alpha= 0.6, color = "gray", label='Camera View')
    #ax.add_patch(poly)
    players = literal_eval(frame_data['player_data'])
    ball = literal_eval(frame_data['ball_data'])
    possession = literal_eval(frame_data['possession'])
    posession_group = possession['group']
    possession_player_id = possession['player_id']
    #if home_players is given, color players based on team
    if home_players is not None:
        home_vals = [p for p in players if
                     any(p['player_id'] == player[0] for player in home_players)]
        away_vals = [p for p in players if
                     not any(p['player_id'] == player[0] for player in home_players)]
        home_player_x = [p['x'] for p in home_vals]
        home_player_y = [p['y'] for p in home_vals]
        away_player_x = [p['x'] for p in away_vals]
        away_player_y = [p['y'] for p in away_vals]

        if posession_group == 'away team':
            pitch.scatter(home_player_x, home_player_y,
                          ax=ax, marker = 'x', c='red', label='Home', s=50)
            pitch.scatter(away_player_x, away_player_y,
                          ax=ax, marker = 'o', edgecolors ='blue',
                          facecolors = 'none', label='Away', s=50)

        else:
            pitch.scatter(home_player_x, home_player_y, ax=ax, marker = 'o',
                         edgecolors ='blue', facecolors = 'none', label='Home', s=50)
            pitch.scatter(away_player_x, away_player_y, ax=ax, marker = 'x',
                        c='red', label='Away', s=50)
    else:
        player_x = [p['x'] for p in players]
        player_y = [p['y'] for p in players]
        pitch.scatter(player_x, player_y, ax=ax, c='blue', s=50)
    #highlight possession player
    if possession_player_id is not None:
        poss_player = [p for p in players if p['player_id'] == possession_player_id][0]
        pitch.scatter(poss_player['x'], poss_player['y'], ax=ax,
                    edgecolors='green',facecolors = 'none', s=50, label='Possession')

    pitch.scatter(ball['x'], ball['y'], ax=ax, c='black', label='Ball', s=50)
    return {
        'frame': frame,
        'period': frame_data['period'],
        'timestamp': frame_data['timestamp'],
        'possession_player_id': possession_player_id,
        'possession_group': posession_group
    }

def plot_gamestate(tracking : pd.DataFrame, frame : int,
                   home_players: Optional[list[int]] = None, title : Optional[str] = None) -> None:
    """
    Plots a single gamestate from tracking data

    Parameters:
        tracking (pd.DataFrame): tracking dataframe
        frame (int): frame number to plot
        home_players (list, optional): list of home player ids. Defaults to None.

    Returns:
        Plot of the single gamestate
    """
    pitch = Pitch(pitch_type='skillcorner', pitch_length = 105, pitch_width = 68)
    _, ax = pitch.draw()
    meta = plot_objects(pitch, ax, tracking, frame, home_players)
    #
    if title is not None:
        plt.suptitle(title, y= 1.05, fontsize = 14)
    else:
        plt.suptitle(f'Game State at Frame {frame}', y=1.05, fontsize=14)
        plt.title(f'''Time {meta["timestamp"]}s, period {meta["period"]},
              Possession: {meta["possession_group"]}, {meta["possession_player_id"]}''',
              fontsize=10)
    plt.legend()
    plt.show()

def animate_gamestate(tracking : pd.DataFrame, frames : List[int],
                      home_players: Optional[List[int]] = None) -> animation.FuncAnimation:
    """
    Creates an animation of the gamestate over a series of frames
    Args:
        tracking: pd.DataFrame
            The tracking dataframe
        frames: List[int]
            List of frame numbers to animate
        home_player: List[int]
            Optional list of home player ids. Defaults to None.
    Returns:
        Animation of the gamestate over the specified frames
    """

    pitch = Pitch(pitch_type='skillcorner', pitch_length = 105, pitch_width = 68)
    fig, ax = pitch.draw()
    def update(i):
        frame = frames[i]
        ax.clear()
        pitch.draw(ax=ax)
        meta = plot_objects(pitch, ax, tracking, frame, home_players)
        plt.suptitle(f'''Time {meta["timestamp"]}s, period {meta["period"]},
                    Possession: {meta["possession_group"]}, 
                    {meta["possession_player_id"]} 
                    \n Game State at Frame {frame}''', fontsize=14)
        plt.legend()
    ani = animation.FuncAnimation(fig, update, len(frames), interval=20, blit=False)
    return ani
