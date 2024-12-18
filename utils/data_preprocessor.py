import argparse
import pandas as pd

from collections import defaultdict
from utils.data_loader import STATS_COLUMNS
from utils.data_preprocessor_classes import Team

parser = argparse.ArgumentParser()
parser.add_argument("--load_path", default="../data", type=str, help="Path to load data.")
parser.add_argument("--store_path", default="../featured_data", type=str, help="Path to store featured data.")
parser.add_argument("--last_n_games", default=5, type=int, help="Number of last games to use for prediction.")
parser.add_argument("--longest_rest", default=4, type=int, help="Number of rest days to become completely fit")
args = parser.parse_args()


def load_data(load_path: str) -> (pd.DataFrame, pd.DataFrame):
    players: pd.DataFrame = pd.read_csv(f"{load_path}/players.csv")
    games: pd.DataFrame = pd.read_csv(f"{load_path}/games.csv")
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE']).dt.date
    games = games.sort_values(by='GAME_ID', ascending=True)
    return games, players


def prepare_season_data(season: int, games: pd.DataFrame, players: pd.DataFrame, args: argparse.Namespace):
    teams: defaultdict[int, Team] = defaultdict(lambda: Team(args))
    feature_data: pd.DataFrame = pd.DataFrame()
    meta_columns = ['SEASON_YEAR', 'GAME_ID', 'HOME_TEAM_WIN']
    home_columns = [f'HOME_{col}' for col in STATS_COLUMNS]
    away_columns = [f'AWAY_{col}' for col in STATS_COLUMNS]

    def get_team_stats(game: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        return pd.DataFrame(game[columns]).T.reset_index(drop=True)

    for _, game in games[games['SEASON_YEAR'] == season].iterrows():
        if teams[game['HOME_TEAM_ID']].ready_to_predict() and teams[game['AWAY_TEAM_ID']].ready_to_predict():
            merged_data = pd.concat([
                pd.DataFrame([game[meta_columns].values.flatten()], columns=meta_columns),
                teams[game['HOME_TEAM_ID']].get_data(is_home_team=True),
                teams[game['AWAY_TEAM_ID']].get_data(is_home_team=False)
            ], axis=1)
            feature_data = pd.concat([feature_data, merged_data], axis=0, ignore_index=True)

        # get game data
        home_team = get_team_stats(game, home_columns).rename(columns={f'HOME_{col}': col for col in STATS_COLUMNS})
        away_team = get_team_stats(game, away_columns).rename(columns={f'AWAY_{col}': col for col in STATS_COLUMNS})
        home_players = players[(players['GAME_ID'] == game['GAME_ID']) & (players['TEAM_ID'] == game['HOME_TEAM_ID'])]
        away_players = players[(players['GAME_ID'] == game['GAME_ID']) & (players['TEAM_ID'] == game['AWAY_TEAM_ID'])]

        # update values
        teams[game['HOME_TEAM_ID']].add_game(game['GAME_DATE'], int(game['HOME_TEAM_WIN'] == 1), home_team, home_players)
        teams[game['AWAY_TEAM_ID']].add_game(game['GAME_DATE'], int(game['HOME_TEAM_WIN'] == 0), away_team, away_players)

    return feature_data


def prepare_data(args: argparse.Namespace) -> None:
    games, players = load_data(args.load_path)
    data = pd.DataFrame()

    for season in games['SEASON_YEAR'].unique():
        season_data = prepare_season_data(season, games, players, args)
        data = pd.concat([data, season_data], ignore_index=True)
        print(f"Season {season} prepared")

    data.to_csv(f"{args.store_path}/data.csv", index=False)


if __name__ == '__main__':
    args: argparse.Namespace = parser.parse_args([] if "__file__" not in globals() else None)
    prepare_data(args)