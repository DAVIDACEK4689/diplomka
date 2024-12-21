import argparse
from collections import defaultdict

import pandas as pd

from utils.data_preprocessor_classes import Team

parser = argparse.ArgumentParser()
parser.add_argument("--load_path", default="../data", type=str, help="Path to load data.")
parser.add_argument("--store_path", default="../featured_data", type=str, help="Path to store featured data.")
parser.add_argument("--last_n_games", default=5, type=int, help="Number of last n games to use for prediction.")
parser.add_argument("--max_rest", default=4, type=int, help="Number of rest days to become completely fit")
parser.add_argument("--average", default="ema", type=str, choices=["mean", "ema", "ema_mean"])
parser.add_argument("--separate_home_away", default=False, type=bool, help="Separate home and away games.")
args = parser.parse_args()

GENERAL_COLUMNS = ['SEASON_YEAR', 'GAME_DATE', 'GAME_ID', 'HOME_TEAM_WIN', 'HOME_TEAM_ODDS', 'AWAY_TEAM_ODDS']


def load_data(load_path: str) -> (pd.DataFrame, pd.DataFrame):
    players: pd.DataFrame = pd.read_csv(f"{load_path}/players.csv")
    games: pd.DataFrame = pd.read_csv(f"{load_path}/games.csv")
    games['GAME_DATE'] = pd.to_datetime(games['GAME_DATE']).dt.date
    games = games.sort_values(by='GAME_DATE', ascending=True)
    return games, players


def get_game_data(game: pd.DataFrame, teams: defaultdict[int, Team], home_team_id: int, away_team_id: int) -> pd.DataFrame:
    game_info = game[['SEASON_YEAR', 'GAME_DATE', 'GAME_ID', 'HOME_TEAM_WIN', 'HOME_TEAM_ODDS', 'AWAY_TEAM_ODDS']]
    game_info = pd.DataFrame(game_info).T.reset_index(drop=True)

    home_team_stats = teams[home_team_id].get_stats_before_game(game['GAME_DATE'], teams, is_home_team=True)
    away_team_stats = teams[away_team_id].get_stats_before_game(game['GAME_DATE'], teams, is_home_team=False)
    return pd.concat([game_info, home_team_stats, away_team_stats], axis=1)


def prepare_season_data(season: int, games: pd.DataFrame, players: pd.DataFrame, args: argparse.Namespace):
    teams: defaultdict[int, Team] = defaultdict(lambda: Team(args))
    feature_data: pd.DataFrame = pd.DataFrame()

    for _, game in games[games['SEASON_YEAR'] == season].iterrows():
        home_team_id = game['HOME_TEAM_ID']
        away_team_id = game['AWAY_TEAM_ID']

        if teams[home_team_id].ready_to_predict() and teams[away_team_id].ready_to_predict():
            feature_data = pd.concat([feature_data, get_game_data(game, teams, home_team_id, away_team_id)], axis=0)

        # get game data
        home_players = players[(players['GAME_ID'] == game['GAME_ID']) & (players['TEAM_ID'] == game['HOME_TEAM_ID'])]
        away_players = players[(players['GAME_ID'] == game['GAME_ID']) & (players['TEAM_ID'] == game['AWAY_TEAM_ID'])]

        # update values
        teams[game['HOME_TEAM_ID']].add_game(pd.DataFrame(game).T, home_players, is_home_team=True)
        teams[game['AWAY_TEAM_ID']].add_game(pd.DataFrame(game).T, away_players, is_home_team=False)

    return feature_data


def prepare_data(args: argparse.Namespace) -> None:
    games, players = load_data(args.load_path)
    data = pd.DataFrame()

    for season in games['SEASON_YEAR'].unique()[:4]:
        season_data = prepare_season_data(season, games, players, args)
        data = pd.concat([data, season_data])
        print(f"Season {season} prepared")

    data.to_csv(f"{args.store_path}/data.csv", index=False)


if __name__ == '__main__':
    args: argparse.Namespace = parser.parse_args([] if "__file__" not in globals() else None)
    prepare_data(args)