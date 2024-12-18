import argparse
import pandas as pd

from utils.data_loader import STATS_COLUMNS

parser = argparse.ArgumentParser()
parser.add_argument("--load_path", default="../data", type=str, help="Path to load data.")
parser.add_argument("--store_path", default="../featured_data", type=str, help="Path to store data.")
args = parser.parse_args()

def load_scaled_data(load_path: str) -> pd.DataFrame:
    return pd.read_csv(f"{load_path}/games.csv")

def get_team_data(data):
    pattern = '|'.join([f'^HOME_{stat}|^AWAY_{stat}' for stat in STATS_COLUMNS])
    pattern += '|HOME_TEAM_WIN'
    return data.filter(regex=pattern)

def measure_significance(store_path: str, load_path: str) -> None:
    data = load_scaled_data(load_path)
    data = get_team_data(data)

    correlation_matrix = data.corr()
    home_team_win_corr = correlation_matrix['HOME_TEAM_WIN'].drop('HOME_TEAM_WIN')
    home_team_win_corr = home_team_win_corr.abs().sort_values(ascending=False)
    print(home_team_win_corr.head(16))





if __name__ == '__main__':
    args: argparse.Namespace = parser.parse_args([] if "__file__" not in globals() else None)
    measure_significance(**vars(args))