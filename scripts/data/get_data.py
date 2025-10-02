"""Gets data from the skillcorner API"""
import io
from pathlib import Path

import config
from skillcorner.client import SkillcornerClient
import pandas as pd

from tqdm import tqdm

SC_USERNAME = config.sc_username
SC_PASSWORD = config.sc_password
current_file_directory = Path(__file__).parent

rdf_path = Path("../../../rdf/sp161/shared/asi_gk_pos/data")

client = SkillcornerClient(username = SC_USERNAME, password = SC_PASSWORD)
matches_meta = []
matches = [match['id'] for match in client.list_matches()]
errors = []
for match in tqdm(matches):
    try:
        match_tracking = client.get_match_tracking_data(match_id = match)
        match_meta = client.get_match(match_id = match)
        match_events = client.get_dynamic_events(match_id = match)

        matches_meta.append(match_meta)
        match_tracking = pd.DataFrame(match_tracking)
        match_events = pd.read_csv(io.BytesIO(match_events))

        match_tracking.to_csv(rdf_path / "tracking" / f"{match}_tracking.csv", index = False)
        match_events.to_csv(rdf_path / "event" / f"{match}_events.csv", index = False)
    except (IOError, pd.errors.ParserError, KeyError, TypeError) as e:
        print(f"Error for match {match}: {e}")
        errors.append(match)
pd.DataFrame(matches_meta).to_csv(rdf_path / "matches_meta.csv", index = False)
