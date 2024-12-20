import argparse
from typing import Union, Callable

import pandas as pd
import nba_api.stats.endpoints as nba

from utils.data_preprocessor_classes import GameEntity, Team

parser = argparse.ArgumentParser()
parser.add_argument("--season_count", default=10, type=int, help="Number of seasons.")
parser.add_argument("--last_year", default=2019, type=int, help="Last year to use (in 2020 was COVID-19).")
parser.add_argument("--odds_load_path", default="../data", type=str, help="Path to load odds data.")
parser.add_argument("--store_path", default="../data", type=str, help="Path to store data.")
args = parser.parse_args()

"""
Method to load games data from NBA regular seasons for years 2010 to 2019
"""
def load_data(season_count: int, last_year: int, store_path: str, odds_load_path: str) -> None:
    # load games
    print("\nLoading teams data...")
    games = load_from_endpoint(season_count, last_year, nba.TeamGameLogs)
    games = prepare_games(games)
    games = add_odds(games, load_odds(season_count, last_year, odds_load_path))
    games.to_csv(f"{store_path}/games.csv", index=False)

    # load players
    print("\nLoading players data...")
    players = load_from_endpoint(season_count, last_year, nba.PlayerGameLogs)
    players = prepare_players(players)
    players.to_csv(f"{store_path}/players.csv", index=False)


def load_from_endpoint(season_count: int, last_year: int, endpoint: Callable[..., Union[nba.TeamGameLogs, nba.PlayerGameLogs]]) -> pd.DataFrame:
    seasons = pd.DataFrame()
    for year in range(last_year - season_count, last_year):
        season = endpoint(season_nullable=f"{year}-{str(year + 1)[2:]}")
        season = season.get_data_frames()[0]
        seasons = pd.concat([seasons, season])
        print(f"Season {year + 1} loaded, matches: {len(season['GAME_ID'].unique())}")

    seasons['GAME_DATE'] = pd.to_datetime(seasons['GAME_DATE']).dt.date
    return seasons


def load_odds(season_count: int, last_year: int, odds_load_path: str) -> pd.DataFrame:
    rename1 = {'date': 'GAME_DATE', 'score': 'HOME_PTS', 'opponentScore': 'AWAY_PTS'}
    rename2 = {'moneyLine': 'HOME_TEAM_ODDS', 'opponentMoneyLine': 'AWAY_TEAM_ODDS'}
    rename3 = {'team': 'HOME_TEAM_NAME', 'opponent': 'AWAY_TEAM_NAME'}
    rename_columns = {**rename1, **rename2, **rename3}

    odds = pd.read_csv(f"{odds_load_path}/odds.csv")
    odds = odds.rename(columns=rename_columns)
    odds = odds[odds['season'].between(last_year - season_count + 1, last_year)]
    odds['GAME_DATE'] = pd.to_datetime(odds['GAME_DATE']).dt.date
    return odds[rename_columns.values()]


def get_team_id_mapping(merged: pd.DataFrame) -> pd.DataFrame:
    merged['IS_DUPLICATE'] = merged.duplicated(subset=['GAME_DATE', 'HOME_PTS', 'AWAY_PTS'], keep=False)
    unique_matches = merged[~merged['IS_DUPLICATE']].copy()
    return unique_matches[['HOME_TEAM_ID', 'HOME_TEAM_NAME']].drop_duplicates()


def convert_odd_to_decimal(odd: int) -> float:
    if odd > 0:
        return round(odd / 100 + 1, 2)
    else:
        return round(100 / abs(odd) + 1, 2)


def convert_odds_to_decimal(odds: pd.DataFrame) -> pd.DataFrame:
    odds['HOME_TEAM_ODDS'] = odds['HOME_TEAM_ODDS'].apply(convert_odd_to_decimal)
    odds['AWAY_TEAM_ODDS'] = odds['AWAY_TEAM_ODDS'].apply(convert_odd_to_decimal)
    return odds


def add_odds(season: pd.DataFrame, odds: pd.DataFrame) -> pd.DataFrame:
    merged = pd.merge(season, odds, on=['GAME_DATE', 'HOME_PTS', 'AWAY_PTS'], how='left', indicator=True)
    if (merged['_merge'] == 'left_only').any():
        raise Exception("There are rows in dataframe season that are not in dataframe odds")

    team_id_mapping = get_team_id_mapping(merged)
    odds = pd.merge(odds, team_id_mapping, on='HOME_TEAM_NAME', how='left')
    odds = convert_odds_to_decimal(odds)
    return pd.merge(season, odds, on=['GAME_DATE', 'HOME_TEAM_ID', 'HOME_PTS', 'AWAY_PTS'], how='left')

def replace_fgm_by_fg2m(data: pd.DataFrame) -> pd.DataFrame:
    data['FGM'] -= data['FG3M']
    data['FGA'] -= data['FG3A']
    data['FG_PCT'] = data['FGM'] / data['FGA']
    data['FG_PCT'] = data['FG_PCT'].fillna(0.0)
    return data.rename(columns={'FGM': 'FG2M', 'FGA': 'FG2A', 'FG_PCT': 'FG2_PCT'})


def update_games_data(games: pd.DataFrame) -> pd.DataFrame:
    replace_fgm_by_fg2m(games)
    games = games.map(lambda x: round(x, 2) if isinstance(x, float) else x)
    games['WL'] = games['WL'].apply(lambda x: 1 if x == 'W' else 0)
    games['SEASON_YEAR'] = games['SEASON_YEAR'].apply(lambda x: int(x[:4]) + 1)
    return games


def prepare_games(games: pd.DataFrame) -> pd.DataFrame:
    games = update_games_data(games)
    home_games = games[games['MATCHUP'].str.contains('vs')].copy()
    away_games = games[~games['MATCHUP'].str.contains('vs')].copy()

    # select columns
    home_games = home_games[['SEASON_YEAR', 'GAME_DATE', 'GAME_ID', 'TEAM_ID', 'WL'] + GameEntity.STATS_COLUMNS]
    away_games = away_games[['GAME_ID', 'TEAM_ID', 'WL'] + GameEntity.STATS_COLUMNS]

    # rename columns
    home_games = home_games.rename(columns={'TEAM_ID': 'HOME_TEAM_ID', 'WL': 'HOME_TEAM_WIN'})
    away_games = away_games.rename(columns={'TEAM_ID': 'AWAY_TEAM_ID', 'WL': 'AWAY_TEAM_WIN'})
    home_games = home_games.rename(columns=dict(zip(GameEntity.STATS_COLUMNS, Team.HOME_STATS_COLUMNS)))
    away_games = away_games.rename(columns=dict(zip(GameEntity.STATS_COLUMNS, Team.AWAY_STATS_COLUMNS)))

    return pd.merge(home_games, away_games, on='GAME_ID')


def update_players_data(players) -> pd.DataFrame:
    replace_fgm_by_fg2m(players)
    players = players.map(lambda x: round(x, 2) if isinstance(x, float) else x)
    players['SEASON_YEAR'] = players['SEASON_YEAR'].apply(lambda x: int(x[:4]) + 1)
    return players


def prepare_players(players: pd.DataFrame) -> pd.DataFrame:
    selected_columns = ['SEASON_YEAR', 'GAME_DATE', 'PLAYER_ID', 'TEAM_ID', 'GAME_ID', 'MIN'] + GameEntity.STATS_COLUMNS
    players = players[selected_columns].copy()
    players = update_players_data(players)
    return players


if __name__ == '__main__':
    args: argparse.Namespace = parser.parse_args([] if "__file__" not in globals() else None)
    load_data(**vars(args))
