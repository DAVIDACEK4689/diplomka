import argparse
import time
from typing import Union

import pandas as pd
from datetime import date
from collections import defaultdict

class SimpleAverage:
    def __init__(self, last_n_games: int) -> None:
        self.__last_n_games = last_n_games
        self.__values = []

    def add_value(self, value: float) -> None:
        self.__values.append(value)

class ExpectedMovingAverage:
    def __init__(self, last_n_games: int) -> None:
        self.__alpha = 2 / (last_n_games + 1)

    def update(self, old_ema: Union[float, pd.DataFrame], new_value: Union[float, pd.DataFrame]) -> Union[float, pd.DataFrame]:
        """
        Compute the exponential moving average (EMA).

        Parameters:
        - old_ema: Can be a float or a pd.DataFrame representing the previous EMA.
        - new_value: Can be a float or a pd.DataFrame representing the new value.

        Returns:
        - The updated EMA, either as a float or pd.DataFrame, matching the input type.
        """
        if old_ema is None:
            return new_value
        return self.__alpha * new_value + (1 - self.__alpha) * old_ema


def rename_columns(data: pd.DataFrame, prefix: str, entity: str) -> pd.DataFrame:
    """Helper function to rename columns with appropriate prefix and entity."""
    return data.rename(columns={col: f"{prefix}{entity}_{col}" for col in data.columns})


class Team:
    def __init__(self, args: argparse.Namespace) -> None:
        self.__data: pd.DataFrame() = pd.DataFrame()
        self.__players: defaultdict[int, Player] = defaultdict(Player)
        self.__last_game_date: date = date.min
        self.__args: argparse.Namespace = args
        self.__average: Union[SimpleAverage, ExpectedMovingAverage] = args.average

    def __count_rest(self, game_date: date) -> float:
        return min(1.0, (game_date - self.__last_game_date).days / self.__args.longest_rest)

    def ready_to_predict(self):
        return len(self.__data) >= self.__args.last_n_games

    def add_game(self, game_date: time, result: int, game_data: pd.DataFrame, players_data: pd.DataFrame) -> None:
        self.__data = pd.concat([self.__data, game_data], ignore_index=True)
        self.__data.append(game_data.assign(RESULT=result), ignore_index=True)
        self.__add_players_stats(players_data)
        self.__last_game_date = game_date

    def __add_players_stats(self, game_players: pd.DataFrame) -> None:
        for _, player_data in game_players.iterrows():
            self.__players[player_data['PLAYER_ID']].add_game(player_data, self.__average)

        for player in self.__players.values():
            if not player.has_played():
                player.add_empty_game()
            player.reset_played_status()

    def get_data(self, is_home_team: bool) -> pd.DataFrame:
        players_data = self.__get_players_data()
        game_data = self.__data.copy()
        game_data[['EMA', 'REST']] = self.__game_rest, self.__game_ema
        game_data = game_data.map(lambda x: round(x, 2))

        prefix = 'HOME_' if is_home_team else 'AWAY_'
        game_data = rename_columns(game_data, prefix, 'TEAM')
        players_data = rename_columns(players_data, prefix, 'PLAYERS')
        return pd.concat([game_data, players_data], axis=1)

    def __get_players_data(self):
        players_data = list(player.get_data() for player in self.__players.values())
        players_data = pd.concat(players_data, axis=0, ignore_index=True)
        players_data = players_data.sort_values(by='MIN', ascending=False)
        assert len(players_data) >= 8, "Not enough players"

        top_8_players = players_data.head(8)
        top_8_dict = top_8_players.to_dict('list')
        return pd.DataFrame({key: [value] for key, value in top_8_dict.items()})

class Player:
    def __init__(self) -> None:
        self.__has_played: bool = False
        self.__data: pd.DataFrame = pd.DataFrame()

    def has_played(self):
        return self.__has_played

    def add_empty_game(self, ema: ExpectedMovingAverage):
        self.__data = pd.concat(self.__data, pd.DataFrame(), ignore_index=True)

    def reset_played_status(self):
        self.__has_played = False

    def add_game(self, player_data: pd.Series, ema: ExpectedMovingAverage) -> None:
        player_data = player_data.drop(['SEASON_YEAR', 'PLAYER_ID', 'TEAM_ID', 'GAME_ID'])
        self.__data = ema.update(self.__data, pd.DataFrame(player_data).T.reset_index(drop=True))
        self.__has_played = True

    def get_data(self) -> pd.DataFrame:
        data = self.__data.copy()
        data = data.map(lambda x: round(x, 2))
        return data
