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
            The row defining a player possession and tracking
        xthreat: np.array
            A numpy array defining the xthreat surface
    
    Returns
    -------
        np.float
            The threat at the start location of the player possession
    """
    ball_frame = literal_eval(row['ball_data'])
    if ball_frame['x'] is None or ball_frame['y'] is None:
        return 0
    start_x = int((ball_frame['x'] + 52.5) / 8.75)
    start_y = int((ball_frame['y'] + 34) / 8.5)
    if not row['correct_orient']:
        start_x = start_x * -1
    clipped_x = np.clip(start_x, 0, 11)
    clipped_y = np.clip(start_y, 0, 7)
    return xthreat[clipped_y, clipped_x]

def get_freeze_frame(ball_dict: str, player_dict: str,
                     player_map : Dict[int, List[int]], team : int,
                     switch : int) -> Tuple[Dict[str, str | float]]:
    """
    Builds both the ball and player freeze frames at a specific frame

    Parameters
    ----------
        ball_dict: str  
            A dictionary in raw string form describing the ball data at a specific frame
        player_dict: str
            A dictionary in raw string form describing the player data at a specific frame
        player_map: Dict[int, int]
            A dictionary mapping the team to 
        team: int
            The team_id in possession
        switch: int
            A boolean-like integer(1 or -1) that determines whether or not the 
            coordinates need to be flipped to ensure left-to-right 
    
    Returns
    -------
        A tuple of dictionaries which describe the 
        freeze frame for both the ball and the players
    """
    ball_data = literal_eval(ball_dict)
    player_data = literal_eval(player_dict)
    other_team = (set(player_map) - {team}).pop()
    for player in player_data:
        player['x'] += 52.5
        player['y'] += 34
        if switch:
            player['x'] = 105 - player['x']
        player['in_possession'] = any(player['player_id'] == poss_player[0]
                                      for poss_player in player_map[team])
        player['opp_gk'] = any((player['player_id'] == poss_player[0]) and (0 == poss_player[1])
                               for poss_player in player_map[other_team])

    ball_data['x'] += 52.5
    ball_data['y'] += 34
    if switch:
        ball_data['x'] = 105 - ball_data['x']
    return player_data, ball_data

def get_specific_player(ff : Dict[str, Union[str | float]], col_name : str):
    """ Gets a specific player from a freeze frame"""
    player_list = [player for player in ff if player[col_name]]
    if len(player_list) == 0:
        raise KeyError("No Such Player Found in Freeze Frame")
    return player_list[0]

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
            player['x_velo'] = 0
            player['y_velo'] = 0 #just say he isn't moving if we don't have the data
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

def check_na(row : pd.Series, na_frames : set) -> pd.DataFrame:
    """
    Row level function determining if the frame and the corresponding event contains NAs between
    Parameters
    ----------
        row : pd.Series
            A row describing a frame of interest
        na_frames : set
            A set of frames which correspond to NA values
    Returns
    -------
        A new value describing whether or not there is an NA between the frame 
        and its corresponding event

    """
    frame_range = range(int(row['event_frame']), int(row['frame'] + 1))
    intersection = set(frame_range).intersection(na_frames)
    return len(intersection)



def get_match_metadata(match_id : int, metadata : pd.DataFrame) -> Dict[int, List[int]]:
    """
    Builds map between teams and players

    Parameters
    ----------
        match_id : int
            The match_id
        metadata : pd.DataFrame
            A dataframe describing the metadata of all games
    Returns
    -------
        Dict[Int, List[Int]]
            A dictionary mapping the team_id to a list of player_ids,
        Dict[Int, Int]
            A dictionary determining the home and away team
        pd.DataFrame
            A dataframe describing the attacking direction of each team
    """
    match_meta = metadata[metadata['id'] == match_id].iloc[0]
    team_list = vsc.get_players(match_id, metadata)
    team_dict = {
        f'{team} team' : literal_eval(match_meta[f'{team}_team'])['id'] for team in ['home', 'away']
    }
    home_sides = literal_eval(match_meta['home_team_side'])
    away_sides = []
    for side in home_sides:
        if side == "left_to_right":
            away_sides.append("right_to_left")
        else:
            away_sides.append("left_to_right")
    home_sides = [(s, i + 1, 'home team') for i, s in enumerate(home_sides)]
    away_sides = [(s, i + 1, 'away team') for i, s in enumerate(away_sides)]
    combined_sides = home_sides + away_sides
    team_sides = pd.DataFrame(combined_sides, columns = ["direction", "period", "team"])
    team_sides['team'] = team_sides['team'].map(team_dict)
    return {literal_eval(match_meta[f'{team}_team'])['id'] : team_list
            for team, team_list in zip(['home', 'away'], team_list)}, team_dict, team_sides

def check_out_of_bounds(row : pd.Series):
    """
    Checks if a frame has the ball out of bounds
    """
    ball_frame = literal_eval(row['ball_data'])
    if ball_frame['x'] is None or ball_frame['y'] is None:
        return True
    return (abs(ball_frame['x']) >= 52.5) | (abs(ball_frame['y']) >= 34)

def generate_feats(match_id : int, xthreat : np.array, wyscout_data : pd.DataFrame) -> pd.DataFrame:
    """Generates the feature set given the match"""

    metadata = pd.read_csv(
        "/home/lz80/rdf/sp161/shared/asi_gk_pos/data/matches_meta.csv")
    tracking = pd.read_csv(
        f"/home/lz80/rdf/sp161/shared/asi_gk_pos/data/tracking/{match_id}_tracking.csv")
    
    wyscout_game = wyscout_data[wyscout_data['sk_match_id'] == match_id]
    
    wyscout_game = wyscout_game.sort_values(by = "event_frame")
    missing_data_frames = set(tracking[tracking['player_data'] == '[]']['frame'])
    tracking['out_of_bounds'] = tracking.apply(check_out_of_bounds, axis = 1)
    out_of_bounds = set(tracking[tracking['out_of_bounds']]['frame'])
    missing_data_frames = missing_data_frames.union(out_of_bounds)
    tracking = tracking[tracking['player_data'] != '[]'].copy()
    tracking = tracking.sort_values('frame')
    tracking = pd.merge_asof(tracking, wyscout_game,
                        left_on = ["frame"], right_on = ["event_frame"], direction='backward',
                        suffixes = ("_sk", "ws"))
    tracking = tracking[abs(tracking['frame'] - tracking['event_frame']) < 100] 
    #sanity checking, if more than 10 seconds from event, drop
    tracking['5_frames_behind'] = tracking['frame'] - 5
    tracking = pd.merge(tracking, tracking, left_on = "5_frames_behind", right_on = "frame",
                        suffixes = ("", "_5behind"))
    
    player_map, team_map, direction_df = get_match_metadata(match_id, metadata)

    tracking['tracking_poss_team'] = tracking['possession'].apply(lambda x: literal_eval(x)['group'])

    mask = tracking['tracking_poss_team'].notna()
    tracking['poss_team_id'] = tracking['sk_team_id']
    tracking.loc[mask, 'poss_team_id'] = tracking.loc[mask, 'tracking_poss_team'].map(team_map)

    tracking = pd.merge(tracking, direction_df, left_on = ["period_sk", "sk_team_id"], right_on = ['period', 'team'])

    tracking['correct_orient'] = tracking['direction'] == 'left_to_right'
    
    flip_mask = tracking['sk_team_id'] != tracking['poss_team_id']
    tracking.loc[flip_mask, 'correct_orient'] = ~tracking.loc[flip_mask, 'correct_orient']
    if tracking.shape[0] == 0:
        return pd.DataFrame()
    tracking['xT'] = tracking.apply(lambda row: get_threat(row, xthreat), axis=1)

    threatening = tracking[(tracking['xT'] >= .05)].copy()
    threatening['has_na'] = threatening.apply(lambda row: check_na(row,
                                                missing_data_frames), axis = 1)
    #removing bad frames:
    threatening = threatening[threatening['has_na'] < 3] #tolerate at most na frames between event and frame

    event_stop_mask =  (threatening['type_primary'].isin(
        ["offside", "own_goal", "penalty", "infraction"]))
    
    keep_non_goal_or_na = (~threatening['shot_is_goal']) | (threatening['shot_is_goal'].isna())
    threatening = threatening[keep_non_goal_or_na & (~event_stop_mask)]

    threatening[['player_freeze_frame_5behind', 'ball_freeze_frame_5behind']] = threatening.apply(
        lambda x: get_freeze_frame(x['ball_data_5behind'], x['player_data_5behind'], player_map,
            x['poss_team_id'], not x['correct_orient']), axis = 1, result_type="expand")

    threatening[['player_freeze_frame', 'ball_freeze_frame']] = threatening.apply(
        lambda x: get_freeze_frame(x['ball_data'], x['player_data'],  player_map,
            x['poss_team_id'], not x['correct_orient']), axis = 1, result_type = "expand")

    threatening['player_freeze_frame'] = threatening.apply(
        lambda x: get_player_speeds(x['player_freeze_frame_5behind'],
            x['player_freeze_frame'], 5, flip = 1), axis = 1)

    threatening['ball_freeze_frame'] = threatening.apply(
        lambda x: get_ball_speeds(x['ball_freeze_frame_5behind'],
            x['ball_freeze_frame'], 5, flip = 1), axis = 1)

    threatening['gk_frame'] = threatening['player_freeze_frame'].apply(lambda x:
                get_specific_player(x, "opp_gk"))
    shots = wyscout_game[~pd.isna(wyscout_game['shot_xg'])]

    threatening['scores_xg'] = threatening.apply(lambda x: get_xg(x['frame'], x['poss_team_id'], shots),
                                          axis = 1)

    return threatening[['sk_match_id', 'frame', 'player_freeze_frame', 'ball_freeze_frame',
                        'gk_frame', 'scores_xg']]

def get_xg(event_frame : int, team_id : int, shots: pd.DataFrame) -> float:
    """
    Gets the complement of the probability that the defending team does not allow a goal from a sequence
    10 seconds after the event

    Parameters
    ----------
        event_frame : int
            The frame of the event
        team_id : int
            The attacking team
        shots : pd.DataFrame
            A dataframe describing all shots in our data
    Returns
    -------
        float
            A float describing this complement
    """
    total_xg = shots[shots['event_frame'].isin(range(event_frame, event_frame + 101))
                      & (shots['sk_team_id'] == team_id)]
    if total_xg.shape[0] > 0:
        return 1 - (1 - total_xg['shot_xg']).prod()
    return 0

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
    dict_feats = ['ball_freeze_frame', 'gk_frame']
    idx = pd.MultiIndex.from_tuples(
                list(zip(df['sk_match_id'], df['frame'])),
                names=["match_id", "frame"]
            )

    for feat in dict_feats:
        path = Path(filepath / f"x_{feat}.parquet")
        feat_df = pd.json_normalize(df[feat])
        feat_df.rename(columns= {'x': f'{feat}_x', 'y': f'{feat}_y',
                                 'x_velo': f'{feat}_x_velo', 'y_velo': f'{feat}_y_velo'},
                                 inplace = True)
        feat_df.index = idx
        feat_df.to_parquet(path)

    non_dict_feats = ['player_freeze_frame']
    for feat in non_dict_feats:
        path = Path(filepath / f"x_{feat}.parquet")
        feat_df = df[feat]
        feat_df.index = idx
        feat_df.to_frame().to_parquet(path)

    #predictors
    success_dummy = pd.Series(True, idx, name = "success")
    path = Path(filepath / f"y_scores_xg.parquet")
    lab_df = df['scores_xg']
    lab_df.index = idx
    lab_df.to_frame().to_parquet(path)



    success_dummy.to_frame().to_parquet(Path(filepath / "y_success.parquet"))
    return True

def main():
    """Main method to write all SkillCorner Data"""
    filepath = "../../../rdf/sp161/shared/asi_gk_pos/"
    xthreat = pd.read_json("https://karun.in/blog/data/open_xt_12x8_v1.json").values
    feature_path = Path(filepath) / "gk_pos_model" / "feats"
    wyscout = pd.read_parquet(
        Path(filepath) / "data" / "wyscout_events_2026.parquet"
    )

    wyscout = wyscout.rename(columns= {"frame" : "event_frame"})
    wyscout['event_frame'] = wyscout['event_frame'].astype(int)
    #wyscout = wyscout[wyscout['is_matched']]
    metadata = pd.read_csv(Path(filepath) / "data" / "matches_meta.csv")

    game_ids = metadata['id']
    feature_dfs = []
    for game_id in tqdm(game_ids):
        feature_df = generate_feats(game_id, xthreat, wyscout)
        if feature_df.shape[0] > 0:
            feature_dfs.append(feature_df)

    feature_dfs = pd.concat(feature_dfs)
    build_parquet(feature_dfs, feature_path)

if __name__ == "__main__":
    main()
