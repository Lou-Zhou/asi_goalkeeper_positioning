#Generates features from hawkeye data
from unxpass import load_xml
import pandas as pd
import numpy as np
import json
import os
from tqdm import tqdm
from unxpass.databases import SQLiteDatabase
import traceback
from collections import defaultdict
def getGksTM(game_id, teams = False):
    """
    Gets Goalkeepers and Team mapping for a game
    """
    lineups = f"../../../../rdf/sp161/shared/soccer-decision-making/WomensEuro/womens_euro_receipts/lineups/{game_id}.json"
    lineup_df = pd.read_json(lineups, convert_dates = False)
    team_1 = lineup_df['team_id'].loc[0]
    team_2 = lineup_df['team_id'].loc[1]
    team_1_dict = lineup_df['lineup'].loc[0]
    team_2_dict = lineup_df['lineup'].loc[1]
    #print(lineup_df)
    team_1_lineup = [player_dict['player_id'] for player_dict in team_1_dict]
    team_2_lineup = [player_dict['player_id'] for player_dict in team_2_dict]
    team_map = {team_1 : team_1_lineup, team_2 : team_2_lineup}#building a map of team id to player ids
    player_to_team = {player_id: team_id for team_id, players in team_map.items() for player_id in players} #mapping players to teams
    pos_dict = {player['player_id']: player['positions'][0]['position'] for player in team_1_dict if len(player['positions']) > 0}
    team_2_pos_dict = {player['player_id']: player['positions'][0]['position'] for player in team_2_dict if len(player['positions']) > 0}
    pos_dict.update(team_2_pos_dict)
    goalkeepers = [key for (key,value) in pos_dict.items() if value == "Goalkeeper"]
    if teams:
        return player_to_team, goalkeepers, [team_1, team_2]
    return player_to_team, goalkeepers#gets set of goalkeepers too
from timeit import default_timer as timer

def he_ball_speed(sb_action_id, frame_idx, frame_back, frame_forward, game, trackingdf, sequences):
    """
    Gets features derived from the ball for a hawkeye frame
    """
    dummy_set = {"start_x":-10000, 
        "start_y": -10000,
        "speed_x": 0,
        "speed_y": 0,
        "end_x": 0,#don't need end locations since we aren't training, just evaluating
        "end_y":0}
    sequence_df = sequences[sequences["id"] == sb_action_id].iloc[0] 
    period = sequence_df["period"]
    time = sequence_df["BallReceipt"]
    if pd.isna(time):
        return dummy_set#ballreceipt can be empty
    time = time + ((int(frame_idx)) * .04)
    start_time = time - (2 * frame_back * .04)#goes framesback * 2 frames(10 in this case) back
    end_time = time #+ (frame_forward * .04)
    times = trackingdf[(trackingdf["elapsed"] >= start_time) & (trackingdf["elapsed"] <= end_time) & (trackingdf['period'] == period)]
    times = times.sort_values(by = ["elapsed"])
    middle = len(times) // 2
    if len(times) == 0:
        return dummy_set
    event_time = times[times['elapsed'] == min(times['elapsed'].unique(), key=lambda x:abs(x-time))].iloc[0]['position'].strip("[]").split(", ")
    #print(event_time == times.iloc[middle]['position'].strip("[]").split(", "))
    #event_time = times.iloc[middle]['position']
    #in the "[a,b]" form in string form, need to manually convert - probably a library that does this but whatevs

    #end_time = 
    #may have overlapping values due to added time
    pre_ball_pos = times.iloc[0]['position']
    post_ball_pos = times.iloc[-1]['position']
    event_x = float(event_time[0]) + 105/2
    event_y = 68 - (float(event_time[1]) + 68/2)
    time_elapsed = .04 * (frame_back + frame_forward)
    pre_ball_pos = pre_ball_pos.strip("[]").split(", ")
    post_ball_pos = post_ball_pos.strip("[]").split(", ")
    speedx = (float(post_ball_pos[0]) - float(pre_ball_pos[0])) / time_elapsed
    speedy = (float(post_ball_pos[1]) - float(pre_ball_pos[1])) / time_elapsed
    return {"start_x":event_x, 
        "start_y": event_y,
        "speed_x": speedx,
        "speed_y": speedy,
        "end_x": 0,
        "end_y":0}

def he_speed_dict(sb_action_id, frame_idx, frame_back, frame_forward, game, trackingdf, sequences, gkslist, ball = False, balldf = None):
    """
    Gets features derived from player data and combines with ball data if ball is True
    """
    
    output = []
    if ball:
        ball_dict = he_ball_speed(sb_action_id, frame_idx, frame_back, frame_forward, game, balldf, sequences)
    else:
        ball_dict = None
    
    dummy_set =  {
            "teammate": False,
            "x":-10000,
            "y":-10000,
            "player": None,
            "actor": False,
            "goalkeeper":False,
            "x_velo": -10000,
            "y_velo": -10000
        }
    sequence_df = sequences[sequences["id"] == sb_action_id].iloc[0] 
    time = sequence_df["BallReceipt"]
    if pd.isna(time):
        if ball:
            return [dummy_set],  ball_dict
        return [dummy_set]#ballreceipt can be empty
    
    team = sequence_df["possession_team_id"]
    period = sequence_df['period']
    time = time + ((int(frame_idx)) * .04)
    start_time = time - (frame_back * .04)#need to account for time before and after half
    end_time = time + (frame_forward * .04)
    times = trackingdf.loc[(trackingdf["elapsed"] >= start_time) & (trackingdf["elapsed"] <= end_time) & (trackingdf['period'] == period), "elapsed"].unique()
    times.sort()
    #may have overlapping values due to added time
    start_time = times[0]
    end_time = times[-1]
    
    #check edge case of x frames back causes to go back to prev period or next period
    #print(start['period'].iloc[0]
    
    middle = int((frame_back + frame_forward)/2)
    event_time = times[middle]
    actor = sequence_df["player_id"]
    current_tracking = clean_he_frame_df(trackingdf[(trackingdf["elapsed"].values == event_time) & (trackingdf['period'] == period)], team)
    start_tracking = clean_he_frame_df(trackingdf[(trackingdf["elapsed"].values == start_time)  & (trackingdf['period'] == period)], team)
    end_tracking = clean_he_frame_df(trackingdf[(trackingdf["elapsed"].values == end_time)  & (trackingdf['period'] == period)], team)
    time_elapsed = (frame_back + frame_forward) * .04
    
    start_tracking = start_tracking.set_index("statsbombid")
    end_tracking = end_tracking.set_index("statsbombid")
    current_tracking = current_tracking.set_index("statsbombid")

    gks_set = set(gkslist)

    for player in start_tracking.index:
        try:
            isActor = (actor == player)
            goalkeeper = (player in gks_set)

            start_x = start_tracking.at[player, "x"]
            start_y = start_tracking.at[player, "y"]
            end_x = end_tracking.at[player, "x"]
            end_y = end_tracking.at[player, "y"]
            x_velo = (end_x - start_x) / time_elapsed
            y_velo = (end_y - start_y) / time_elapsed
            x_loc = current_tracking.at[player, "x"]
            y_loc = current_tracking.at[player, "y"]
            isTeammate = end_tracking.at[player, "isTeammate"]

            speed_dict = {
                "teammate": isTeammate,
                "goalkeeper": goalkeeper,
                "x": x_loc,
                "y": y_loc,
                "actor": isActor,
                "player": player,
                "x_velo": x_velo,
                "y_velo": y_velo
            }
            output.append(speed_dict)
        except KeyError:
            continue  # Skip player if missing from any DataFrame
    return getFlip(output, ball_dict)

def getFlip(freezeframe, secondary_frame = None):
    """
    Determine if a flip is needed based on gk position
    """
    for player in freezeframe:
        if player['teammate'] and player['goalkeeper']:
            attack_gk_x = player['x']
        if not player['teammate'] and player['goalkeeper']:
            defend_gk_x = player['x']
    if attack_gk_x > defend_gk_x: #attacking gk is on wrong side of field, need to flip
        for player in freezeframe:
            player['x'] = 105 - player['x']
            player['x_velo'] = -1 * player['x_velo']
        if secondary_frame:
            secondary_frame["end_x"] = 105 - secondary_frame['end_x']
            secondary_frame["start_x"] = 105 - secondary_frame['start_x']
            secondary_frame["speed_x"] = -1 * secondary_frame["speed_x"]
    if secondary_frame is not None:
        return freezeframe, secondary_frame
    return freezeframe

def convert_Hawkeye(coords):
    """
    Convert hawkeye coords to statsbomb coords
    """
    x, y = coords
    x = float(x)
    y = float(y)
    x = (x + 105/2)
    y = 68 - (y + 34)
    return [x, y]

def clean_he_frame_df(df, team):
    start = timer()
    """
    Cleans the hawkeye feature data — optimized
    """
    df = df.copy()

    # Boolean mask for teammates
    df["isTeammate"] = df["team"] == int(team)

    # Get GK X-position for teammate
    gk_pos = df.loc[df["isGk"] & df["isTeammate"], "position"].iloc[0]
    gk_x = float(gk_pos.strip("[]").split(", ")[0])
    needFlip = gk_x > 0

    # Vectorized position cleaning
    # Remove brackets and split into x/y strings
    pos_strs = df["position"].str.strip("[]").str.split(", ")

    # Convert list of strings to list of tuples, then apply convert_Hawkeye
    converted_positions = [convert_Hawkeye([float(x), float(y)]) for x, y in pos_strs]

    # Extract x and y from converted positions
    df["x"] = [pos[0] for pos in converted_positions]
    df["y"] = [pos[1] for pos in converted_positions]
    df["position"] = converted_positions
    end = timer()
    #print(end - start)
    return df


#Game specific
def getHeGameSpeed(game_file, uefa_map, hawkeye_to_sb, skeleton, db, framesback, framesforward, sequences, ball = False):
    """
    Gets hawkeye features for an entire game
    """
    #game_file = "2032219_Portugal_Switzerland.csv"
    game = game_file.split(".")[0]
    
    tracking_path = f"../../../../rdf/sp161/shared/soccer-decision-making/Hawkeye/raw_data/tracking_csvs/{game_file}"
    
    tracking = pd.read_csv(tracking_path)#.sort_values(by = ["elapsed"])
    tracking['statsbombid'] = tracking['uefaId'].astype(int).map(uefa_map)
#3835338, action 7218
    statsbomb_id = hawkeye_to_sb[game]

    game_mask = skeleton.get_level_values(0) == str(statsbomb_id)
    game_skeleton = skeleton[game_mask]
    team_dict, gks = getGksTM(statsbomb_id)
    tracking['team'] = tracking['statsbombid'].map(team_dict)
    tracking['isGk'] = tracking['role'] == "Goalkeeper"
    speed_df = pd.DataFrame(index = game_skeleton)
    action_df = db.actions(game_id = int(statsbomb_id))
    action_map = pd.Series(action_df['original_event_id'].values, index=action_df.index).to_dict()
    player_speeds = pd.DataFrame(index = game_skeleton)
    if ball:
        ball_tracking_path = f"/../../../../rdf/sp161/shared/soccer-decision-making/Hawkeye/raw_data/tracking_ball_csvs/{game_file}"
        ball_df = pd.read_csv(ball_tracking_path)
        ball_speeds = pd.DataFrame(index = game_skeleton)
        ball_starts = pd.DataFrame(index = game_skeleton)
    player_speeds["freeze_frame_360_a0"] = np.nan
    player_speeds["freeze_frame_360_a0"] = player_speeds["freeze_frame_360_a0"].astype(object) #enforce some consistency
    for game_id, action_id in tqdm(game_skeleton, leave = False):
        #action_id = 7597
        #pbar.set_description(f"Processing game {game_id}, action {action_id}")
        action_sb_id = action_map.get((game_id, action_id))
        if len(action_sb_id.split("-")) == 5:#if non-interesting event
            sb_action_id = action_sb_id
            frame_idx = 0
        else:
            sb_action_id = action_sb_id.rsplit("-", 1)[0]#if interesting event(denoted by dash)
            frame_idx = action_sb_id.rsplit("-", 1)[1]
        
           
        try:
            if ball:
                speed_dict, ball_dict = he_speed_dict(sb_action_id, frame_idx, framesback, framesforward, game, tracking, sequences, gks, ball, ball_df)
                
                ball_starts.at[(game_id, action_id), "start_x_a0"] = ball_dict["start_x"]
                ball_starts.at[(game_id, action_id), "start_y_a0"] = ball_dict["start_y"]
                ball_speeds.at[(game_id, action_id), "speedx_a02"] = ball_dict["speed_x"]
                ball_speeds.at[(game_id, action_id), "speedy_a02"] = ball_dict["speed_y"]
            else:
                speed_dict = he_speed_dict(sb_action_id, frame_idx, framesback, framesforward, game, tracking, sequences, gks, ball)
            player_speeds.at[(game_id, action_id), "freeze_frame_360_a0"] = speed_dict
        except Exception as e:
            res = dict((v,k) for k,v in hawkeye_to_sb.items())
            print(f"Error processing game {game_id}, {res[int(game_id)]}, action {action_id}: {traceback.format_exc()}")
            speed_dict = {}
        end = timer()
        print(end - start)
    if ball:
        return player_speeds, ball_starts, ball_speeds
    return player_speeds

def getHeSpeed(tracking_folder, skeleton_path, dbpath, framesback, framesforward, ball = False):
    """
    Function which generates for all hawkeye data from a db implementation(originally in un-xpass)
    """
    player_speeds = []
    ball_starts = []
    ball_speeds = []
    sequences = pd.read_csv("../../../../rdf/sp161/shared/soccer-decision-making/steffen/sequences_new.csv")
    timeelapsed = {
    1:0,
    2:45 * 60,
    3: 90 * 60,
    4: 105 * 60
    }
    sequences["BallReceipt"] = sequences["period"].map(timeelapsed) + sequences["BallReceipt"]#minute adjustment
    with open("../../../../rdf/sp161/shared/soccer-decision-making/hawkeye_to_sb.json", 'r') as file:
        hawkeye_to_sb = json.load(file)
    skeleton = pd.read_parquet(skeleton_path).index
    hawkeye_db = SQLiteDatabase(dbpath)
    framesback = 5
    framesforward = 5
    alltracking = [file for file in os.listdir(tracking_folder) if file.endswith(".csv")]
    uefa_map = pd.read_csv("../../../../rdf/sp161/shared/soccer-decision-making/steffen/player_ids_matched.csv")
    uefa_map = pd.Series(uefa_map["sb_player_id"].values,index=uefa_map["uefa_player_id"]).to_dict()
    for game_file in tqdm(alltracking):
    #for game_file in test_track:
        if ball:
            all_dfs = getHeGameSpeed(game_file, uefa_map, hawkeye_to_sb, skeleton, hawkeye_db, framesback, framesforward, sequences, ball)
            player_speeds.append(all_dfs[0])
            ball_starts.append(all_dfs[1])
            ball_speeds.append(all_dfs[2])
        else:
            player_speeds.append(getHeGameSpeed(game_file, uefa_map, hawkeye_to_sb, skeleton, hawkeye_db, framesback, framesforward, sequences, ball))
    player_speed_df = pd.concat(player_speeds)
    if ball:
        ball_start_df = pd.concat(ball_starts)
        ball_speeds_df = pd.concat(ball_speeds)
        return player_speed_df, ball_start_df, ball_speeds_df
    #return speed_df
    return player_speed_df
#sequences

def generate_Hawkeye_From_Features(output_dir, frame_forward = 5, frame_back = 5, ball = False, frame_idxs = [0]):
    """
    Generates features independent of converted statsbomb data, completely from hawkeye data and sequences
    """
    uefa_map = pd.read_csv("../../../../rdf/sp161/shared/soccer-decision-making/steffen/player_ids_matched.csv")
    uefa_map = pd.Series(uefa_map["sb_player_id"].values,index=uefa_map["uefa_player_id"]).to_dict()
    
    with open("../../../../rdf/sp161/shared/soccer-decision-making/hawkeye_to_sb.json", 'r') as file:
        hawkeye_to_sb = json.load(file)
    sb_to_hawkeye = dict((v,k) for k,v in hawkeye_to_sb.items())
    minute_adjustment = {
    1: 0,
    2: 45 * 60,
    3: 90 * 60,
    4: 105 * 60
    }
    sequences = pd.read_csv("../../../../rdf/sp161/shared/soccer-decision-making/steffen/sequence_filtered.csv", delimiter = ";")
    sequences = sequences.rename(columns = {"Half":"period"})
    sequences["hawkeye_game_id"] = sequences["match_id"].map(sb_to_hawkeye)
    sequences["BallReceipt"] = sequences["period"].map(minute_adjustment) + sequences["BallReceipt"]
    sequences["Start"] = sequences["period"].map(minute_adjustment) + sequences["Start"]
    frame_path = f"{output_dir}/x_freeze_frame_360.parquet"
    frame_dfs = []
    if ball:
        ball_start_output = f"{output_dir}/x_startlocation.parquet"
        ball_speed_output = f"{output_dir}/x_speed.parquet"
        ball_end_output = f"{output_dir}/x_endlocation.parquet"
        ball_start_dfs = []
        ball_speed_dfs = []
        ball_end_dfs = []
    
    for game in tqdm(sequences['hawkeye_game_id'].unique()):
        if ball:
            player_speeds, ball_starts, ball_speeds, ball_end = hawkeyeFeaturesGame(game, sequences, hawkeye_to_sb, uefa_map, frame_idxs, frame_back, frame_forward, ball)
            ball_start_dfs.append(ball_starts)
            ball_speed_dfs.append(ball_speeds)
            ball_end_dfs.append(ball_end)
        else:
            player_speeds = hawkeyeFeaturesGame(game, sequences, hawkeye_to_sb, uefa_map, frame_idxs, frame_back, frame_forward, ball)
        frame_dfs.append(player_speeds)
    if ball:
        combined_ball_start = pd.concat(ball_start_dfs)
        combined_ball_speed = pd.concat(ball_speed_dfs)
        combined_ball_end = pd.concat(ball_end_dfs)

        combined_ball_start.to_parquet(ball_start_output)
        combined_ball_speed.to_parquet(ball_speed_output)
        combined_ball_end.to_parquet(ball_end_output)
    combined_frame_dfs = pd.concat(frame_dfs)
    combined_frame_dfs.to_parquet(frame_path)

def hawkeyeFeaturesGame(game, sequences, hawkeye_to_sb, uefa_map, frame_idxs, frame_back = 5, frame_forward = 5, ball = False):
    """
    Gets hawkeye features from scratch for each game
    """
    sequence_games = sequences[sequences['hawkeye_game_id'] == game].copy()
    multiIndex = []
    for frame_idx in frame_idxs:
        for idx, row in sequence_games.iterrows():
            if frame_idx < 0:
                index = f"{row['index']}-n{abs(int(frame_idx))}"
            else:
                index = f"{row['index']}-{int(frame_idx)}"
            match = row['match_id']
            multiIndex.append((match, index))
    multiindex = pd.MultiIndex.from_tuples(multiIndex, names = ['game_id', 'action_id'])
    player_speeds = pd.DataFrame(index = multiindex)
    player_speeds["freeze_frame_360_a0"] = np.nan
    player_speeds["freeze_frame_360_a0"] = player_speeds["freeze_frame_360_a0"].astype(object)
    if ball:
        ball_starts = pd.DataFrame(index = multiindex)
        ball_speeds = pd.DataFrame(index = multiindex)
        ball_end = pd.DataFrame(index = multiindex)
        ball_tracking_path = f"../../../../rdf/sp161/shared/soccer-decision-making/Hawkeye/raw_data/tracking_ball_csvs/{game}.csv"
        ball_df = pd.read_csv(ball_tracking_path)
    tracking_path = f"../../../../rdf/sp161/shared/soccer-decision-making/Hawkeye/raw_data/tracking_csvs/{game}.csv"
    statsbombid = hawkeye_to_sb[game]
    team_dict, gkslist, teams = getGksTM(statsbombid, True)
    trackingdf = pd.read_csv(tracking_path)
    
    trackingdf['statsbombid'] = trackingdf['uefaId'].astype(int).map(uefa_map)
    trackingdf['team'] = trackingdf['statsbombid'].map(team_dict)
    
    if teams[0] not in trackingdf['team'].values and teams[1] not in trackingdf['team'].values:
        raise Exception("No Team Found")
    elif teams[0] not in trackingdf['team'].values:
        trackingdf['team'] = trackingdf['team'].fillna(teams[0])
    elif teams[1] not in trackingdf['team'].values:
        trackingdf['team'] = trackingdf['team'].fillna(teams[1])
    
    trackingdf['isGk'] = trackingdf['role'] == "Goalkeeper" 
    
    for frame_idx in tqdm(frame_idxs):
        for idx, row in tqdm(sequence_games.iterrows(), leave = False):
            sb_action_id = row['id']
            if frame_idx < 0:
                action_id = f"{row['index']}-n{abs(int(frame_idx))}"
            else:
                action_id = f"{row['index']}-{int(frame_idx)}"
            
            game_id = int(row['match_id'])
            if ball:
                try:
                    speed_dict, ball_dict = he_speed_dict(sb_action_id, frame_idx, frame_back, frame_forward, game, trackingdf, sequences, gkslist, ball, ball_df)
                except Exception as e:
                    print(f"Error processing game {game_id}, action {action_id}: {traceback.format_exc()}")
                    speed_dict = {}
                    ball_dict = defaultdict(int)
                
                ball_starts.at[(game_id, action_id), "start_x_a0"] = ball_dict["start_x"]
                ball_starts.at[(game_id, action_id), "start_y_a0"] = ball_dict["start_y"]
                ball_speeds.at[(game_id, action_id), "speedx_a02"] = ball_dict["speed_x"]
                ball_speeds.at[(game_id, action_id), "speedy_a02"] = ball_dict["speed_y"]
                ball_end.at[(game_id, action_id), "end_x_a0"] = 60
                ball_end.at[(game_id, action_id), "end_y_a0"] = 40#dummy to center
            else:
                speed_dict = he_speed_dict(sb_action_id, frame_idx, frame_back, frame_forward, game, trackingdf, sequences, gkslist, ball)
            player_speeds.at[(game_id, action_id), "freeze_frame_360_a0"] = speed_dict
    
    if ball:
        return player_speeds, ball_starts, ball_speeds, ball_end
    return player_speeds
def getDummyLabels(output_dir, dummy_idxs):
    """
    Generates dummy labels for hawkeye data
    """
    concedes_xg = f"{output_dir}/y_concedes_xg.parquet"
    concedes = f"{output_dir}/y_concedes.parquet"
    scores_xg = f"{output_dir}/y_scores_xg.parquet"
    scores = f"{output_dir}/y_scores.parquet"
    success = f"{output_dir}/y_success.parquet"
    c_xg = pd.DataFrame(index = dummy_idxs)
    c = pd.DataFrame(index = dummy_idxs)
    s_xg = pd.DataFrame(index = dummy_idxs)
    s = pd.DataFrame(index = dummy_idxs)
    suc = pd.DataFrame(index = dummy_idxs)
    for idx in dummy_idxs:
        c_xg.at[idx, "concedes_xg"] = 0
        c.at[idx, "concedes"] = False
        s_xg.at[idx, "scores_xg"] = 0
        s.at[idx, "scores"] = False
        suc.at[idx, "success"] = True
    c_xg.to_parquet(concedes_xg)
    c.to_parquet(concedes)
    s_xg.to_parquet(scores_xg)
    s.to_parquet(scores)
    suc.to_parquet(success)
def main(hawkeye, hawkeyeRaw, ball):
    if hawkeye:
        #generate from pregenerated statsbomb features - legacy code
        dbpath = "../../../../rdf/sp161/shared/soccer-decision-making/hawkeye_all.sql"
        tracking_folder = "../../../../rdf/sp161/shared/soccer-decision-making/Hawkeye/raw_data/tracking_csvs"
        skeleton_path = "../../../../rdf/sp161/shared/soccer-decision-making/HawkEye_Features_2/x_endlocation.parquet"
        output_path = "../../../../rdf/sp161/shared/soccer-decision-making/HawkEye_Features_2/x_speed_frame_360.parquet"
        #player_speed_df.loc[skeleton], ball_start_df.loc[skeleton], ball_speeds_df.loc[skeleton]
        if ball:
            all_dfs = getHeSpeed(tracking_folder, skeleton_path, dbpath, 5, 5, ball)
            ball_start_output = "../../../../rdf/sp161/shared/soccer-decision-making/HawkEye_Features_2/x_startlocation_3.parquet"
            ball_speed_output = "../../../../rdf/sp161/shared/soccer-decision-making/HawkEye_Features_2/x_speed_3.parquet"
            speeddf = all_dfs[0]
            all_dfs[1].to_parquet(ball_start_output)
            all_dfs[2].to_parquet(ball_speed_output)
        #print(speeddf)
        else:
            speeddf = getHeSpeed(tracking_folder, skeleton_path, dbpath, 5, 5, ball)
        speeddf.to_parquet(output_path)
    if hawkeyeRaw:
        #generate completely from hawkeye data - recommend this
        #/home/lz80/rdf/sp161/shared/soccer-decision-making/steffen/sequence_filtered.csv
        output_dir = "../../../../rdf/sp161/shared/soccer-decision-making/Hawkeye/Hawkeye_Features/sequences_tenSecPrior"
        sequence_games = pd.read_csv("../../../../rdf/sp161/shared/soccer-decision-making/steffen/sequence_filtered.csv", delimiter = ";")
        
        generate_Hawkeye_From_Features(output_dir, ball = ball, frame_idxs = range(-250,1))
        dummy_idxs = pd.read_parquet(f"{output_dir}/x_startlocation.parquet").index
        getDummyLabels(output_dir, dummy_idxs)
#main(buli, hawkeye, hawkeye_raw, ball)
if __name__ == '__main__':  main(False, True, True)