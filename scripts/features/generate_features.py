"""
Generates Features for both models from raw skillcorner tracking data
"""
from pathlib import Path
from typing import Dict, Tuple, Union, List, Literal
import sys
import warnings
from ast import literal_eval

from tqdm import tqdm
from sdm import visualize_skillcorner as vsc

import pandas as pd
import numpy as np

sys.path.append(str(Path("..").resolve() / ".."))

warnings.simplefilter('ignore', pd.errors.DtypeWarning)

def get_threat(row: pd.Series, xthreat : np.array) -> np.float64:
    """
    Gets the xT of a player possession

    Parameters
    ----------
        row: pd.Series
            The row defining a player possession
        xthreat: np.array
            A numpy array defining the xthreat surface
    
    Returns
    -------
        np.float
            The threat at the start location of the player possession
    """
    start_x = int((row['x_start'] + 52.5) / 8.75)
    start_y = int((row['y_start'] + 34) / 8.5)
    clipped_x = np.clip(start_x, 0, 11)
    clipped_y = np.clip(start_y, 0, 7)
    return xthreat[clipped_y, clipped_x]

def get_freeze_frame(frame_id : int, tracking : pd.DataFrame,
                     player_map : Dict[int, List[int]], team : int,
                     receiver : int, switch : int) -> Tuple[Dict[str, str | float]]:
    """
    Builds both the ball and player freeze frames at a specific frame

    Parameters
    ----------
        frame_id: int
            The frame of interest
        tracking: pd.DataFrame
            The tracking dataframe
        player_map: Dict[int, int]
            A dictionary mapping the team to 
        team: int
            The team_id in possession
        receiver: int
            The player_id of the receiver of the ball possession
        switch: int
            A boolean-like integer(1 or -1) that determines whether or not the 
            coordinates need to be flipped to ensure left-to-right 
    
    Returns
    -------
        A tuple of dictionaries which describe the 
        freeze frame for both the ball and the players
    """
    tracking_frame = tracking[tracking['frame'] == frame_id].iloc[0]
    ball_data = literal_eval(tracking_frame['ball_data'])
    player_data = literal_eval(tracking_frame['player_data'])
    for player in player_data:
        if switch:
            player['x'] = player['x'] * -1
        player['in_possession'] = any(player['player_id'] == poss_player[0]
                                      for poss_player in player_map[team])
        player['opp_gk'] = any((player['player_id'] == poss_player[0]) and (0 == poss_player[1])
                               for poss_player in player_map[team])
        player['receiver'] = any(player['player_id'] == receiver
                                for _ in player_map[team])
    if switch:
        ball_data['x'] = ball_data['x'] * -1
    return player_data, ball_data

def get_player_speeds(ref_ff : Dict[str, Union[str | float]],
                      new_ff : Dict[str, Union[str | float]],
                      frame_diff : int, flip : int = 1) -> Dict[str, str | float]:
    """
    Adds the player velocities in reference to a reference freeze frame to the new freeze frame

    Parameters
    ----------
        ref_ff: Dict[str, Union[str, float]]
            The reference freeze frame to calculate speed
        new_ff: Dict[str, Union[str, float]]
            The freeze frame of interest
        frame_diff: int
            The number of frames between ref_ff and new_ff
        flip: int
            Boolean integer describing whether to flip
            Set to -1 if ref_ff happens after new_ff
    
    Returns
    -------
        Mutates new_ff so that player velocities are added
    """
    for player in new_ff:
        old_player = [old_frame for old_frame in ref_ff
                      if old_frame['player_id'] == player['player_id']]
        if len(old_player) == 0:
            player['x_speed'] = None
            player['y_speed'] = None
            continue
        new_loc = [player['x'], player['y']]
        old_player = old_player[0]
        ref_loc = [old_player['x'], old_player['y']]
        player['x_velo'] = (new_loc[0] - ref_loc[0]) / (frame_diff * 0.1) * flip
        player['y_velo'] = (new_loc[1] - ref_loc[1]) / (frame_diff * 0.1) * flip
    return new_ff

def get_ball_speeds(ref_ff : Dict[str, Union[str | float]],
                    new_ff : Dict[str, Union[str | float]],
                    frame_diff : int, flip : int = 1) -> Dict[str, str | float]:
    """
    Adds the ball velocities in reference to a reference freeze frame to the new freeze frame

    Parameters
    ----------
        ref_ff: Dict[str, Union[str, float]]
            The reference freeze frame to calculate speed
        new_ff: Dict[str, Union[str, float]]
            The freeze frame of interest
        frame_diff: int
            The number of frames between ref_ff and new_ff
        flip: int
            Boolean integer describing whether to flip
            Set to -1 if ref_ff happens after new_ff
    Returns
    -------
        Mutates new_ff so that ball velocities are added
    """
    new_ff['x_velo'] = (new_ff['x'] - ref_ff['x']) / (frame_diff * 0.1) * flip
    new_ff['y_velo'] = (new_ff['y'] - ref_ff['y']) / (frame_diff * 0.1) * flip
    return new_ff

def get_player_map(match_id : int, metadata : pd.Series) -> Dict[int, List[int]]:
    """
    Builds map between teams and players

    Parameters
    ----------
        match_id : int
            The match_id
        metadata : pd.Series
            A dataframe describing the metadata of all games
    Returns
    -------
        Dict[Int, List[Int]]
            A dictionary mapping the team_id to a list of player_ids
    """
    match_meta = metadata[metadata['id'] == match_id].iloc[0]
    team_list = vsc.get_players(match_id, metadata)
    return {literal_eval(match_meta[f'{team}_team'])['id'] : team_list
            for team, team_list in zip(['home', 'away'], team_list)}

def find_invalid_frames(prev_frame : int, pres_frame : int, tracking : pd.DataFrame) -> int:
    """Finds the number of invalid frames in a frame range"""
    tracking_frames = tracking[tracking['frame'].isin(range(int(prev_frame), int(pres_frame) + 1))]
    tracking_frames['ball_data'] = tracking_frames['ball_data'].apply(literal_eval)
    return find_missing_frames(tracking_frames) + find_outofbound_frames(tracking_frames)

def find_outofbound_frames(tracking_frames : pd.DataFrame) -> int:
    """Finds the number of frames where the ball is out of bounds""" 
    tracking_frames['tracking_x'] = tracking_frames['ball_data'].apply(lambda x: x['x'])
    tracking_frames['tracking_y'] = tracking_frames['ball_data'].apply(lambda x: x['y'])

    oob_mask = ((tracking_frames['tracking_x'] > 52.5) | (tracking_frames['tracking_x'] < -52.5) |
                (tracking_frames['tracking_y'] > 34) | (tracking_frames['tracking_y'] < -34))
    return tracking_frames[oob_mask].shape[0]

def find_missing_frames(tracking_frames: pd.DataFrame) -> int:
    """Finds the number of frames where there is no tracking data found"""
    num_missing = tracking_frames[tracking_frames['player_data'] == '[]'].shape[0]
    return num_missing


def generate_feats(match_id : int, xthreat : np.array) -> pd.DataFrame:
    """Generates the feature set given the match"""
    event_data = pd.read_csv(
        f"/home/lz80/rdf/sp161/shared/asi_gk_pos/data/event/{match_id}_events.csv")
    metadata = pd.read_csv(
        "/home/lz80/rdf/sp161/shared/asi_gk_pos/data/matches_meta.csv")
    tracking = pd.read_csv(
        f"/home/lz80/rdf/sp161/shared/asi_gk_pos/data/tracking/{match_id}_tracking.csv")

    possessions = event_data[event_data['event_type'] == "player_possession"].copy()
    possessions['prev_frame'] = possessions['frame_end'].shift(1)

    possessions['correct_orient'] = possessions['attacking_side'] == 'left_to_right'
    possessions['xT'] = possessions.apply(lambda x: get_threat(x, xthreat), axis=1)

    possessions = possessions[~possessions['start_type'].isin(
        ['free_kick_reception', 'throw_in_reception',
        'goal_kick_reception', 'corner_reception'])]
    threatening = possessions[(possessions['xT'] >= .04)]
    threatening = threatening[threatening['prev_frame'] != threatening['frame_end']]
    player_map = get_player_map(match_id, metadata)

    threatening['invalid_count'] = threatening.apply(lambda x:
            find_invalid_frames(x['prev_frame'], x['frame_start'], tracking), axis = 1)
    no_missing = threatening[threatening['invalid_count'] == 0].copy()
    no_missing[['player_freeze_frame', 'ball_freeze_frame']] = no_missing.apply(
        lambda x: get_freeze_frame(x['prev_frame'], tracking, player_map,
            x['team_id'],x['player_id'],  not x['correct_orient']), axis = 1, result_type="expand")

    no_missing[['player_freeze_frame_5forward', 'ball_freeze_frame_5forward']] = no_missing.apply(
        lambda x: get_freeze_frame(x['prev_frame'] + 5, tracking, player_map,
            x['team_id'], x['player_id'], not x['correct_orient']), axis = 1, result_type="expand")

    no_missing[['player_freeze_frame_reception', 'ball_freeze_frame_reception']] = no_missing.apply(
        lambda x: get_freeze_frame(x['frame_start'], tracking, player_map,
            x['team_id'],x['player_id'], not x['correct_orient']), axis = 1, result_type = "expand")

    no_missing['player_freeze_frame'] = no_missing.apply(
        lambda x: get_player_speeds(x['player_freeze_frame_5forward'],
            x['player_freeze_frame'], 5, flip = -1), axis = 1)

    no_missing['ball_freeze_frame'] = no_missing.apply(
        lambda x: get_ball_speeds(x['ball_freeze_frame_5forward'],
            x['ball_freeze_frame'], 5, flip = -1), axis = 1)

    return no_missing[['event_id', 'match_id','player_freeze_frame',
                    'player_freeze_frame_reception', 'ball_freeze_frame',
                    'ball_freeze_frame_reception']]

def build_parquet(df : pd.DataFrame, filepath : str) -> Literal[True]:
    """
    Builds a parquet of features from a dataframe of features

    Parameters
    ----------
        df : pd.DataFrame
            The dataframe of features
        filepath : str
            A string describing the filepath to save the parquets
    Returns
    -------
        Literal[True]
            True if the parquets were able to be saved
    """
    dict_feats = ['ball_freeze_frame', 'ball_freeze_frame_reception']
    idx = pd.MultiIndex.from_tuples(
                list(zip(df['event_id'], df['match_id'])),
                names=["event_id", "match_id"]
            )

    for feat in dict_feats:
        path = Path(filepath / f"x_{feat}.parquet")
        feat_df = pd.json_normalize(df[feat])
        feat_df.rename(columns= {'x': f'{feat}_x', 'y': f'{feat}_y',
                                 'x_velo': f'{feat}_x_velo', 'y_velo': f'{feat}_y_velo'},
                                 inplace = True)
        feat_df.index = idx
        feat_df.to_parquet(path)

    non_dict_feats = ['player_freeze_frame', 'player_freeze_frame_reception']
    for feat in non_dict_feats:
        path = Path(filepath / f"x_{feat}.parquet")
        feat_df = df[feat]
        feat_df.index = idx
        feat_df.to_frame().to_parquet(path)
    success_dummy = pd.Series(0, idx, name = "success")

    success_dummy.to_frame().to_parquet(Path(filepath / "y_success.parquet"))
    return True

def main():
    """Main method to write all SkillCorner Data"""
    filepath = "../../../rdf/sp161/shared/asi_gk_pos/"
    xthreat = pd.read_json("https://karun.in/blog/data/open_xt_12x8_v1.json").values
    feature_path = Path(filepath) / "gk_pos_model" / "feats"

    metadata = pd.read_csv(Path(filepath) / "data" / "matches_meta.csv")

    game_ids = metadata['id']
    feature_dfs = []
    for game_id in tqdm(game_ids[0:1]):
        feature_df = generate_feats(game_id, xthreat)
        feature_dfs.append(feature_df)

    feature_dfs = pd.concat(feature_dfs)
    build_parquet(feature_dfs, feature_path)

if __name__ == "__main__":
    main()
