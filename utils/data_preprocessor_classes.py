import argparse
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import date

import pandas as pd
from overrides import overrides


def add_prefix(data: pd.DataFrame, prefix: str, columns: list[str]) -> pd.DataFrame:
    """Helper function to rename columns with appropriate prefix."""
    return data.rename(columns={col: f"{prefix}_{col}" for col in columns})


class GameEntity(ABC):
    STATS_COLUMNS: list[str] = ['FGM', 'FGA', 'FG_PCT', 'FG3M', 'FG3A', 'FG3_PCT', 'FTM', 'FTA', 'FT_PCT',
                                'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'BLKA', 'PF', 'PFD', 'PTS']

    def __init__(self, args: argparse.Namespace) -> None:
        self._games: pd.DataFrame = pd.DataFrame()
        self._home_games: pd.DataFrame = pd.DataFrame()
        self._away_games: pd.DataFrame = pd.DataFrame()
        self._args: argparse.Namespace = args

    def add_data(self, game: pd.DataFrame, is_home_team: bool, **kwargs) -> None:
        data = pd.concat([
            self.__add_base_stats(game),
            self._add_advanced_stats(game, is_home_team, **kwargs)
        ], axis=1)

        # Rename columns
        data = add_prefix(data, self.__get_entity_name(), self.STATS_COLUMNS)
        games_attr = '_home_games' if is_home_team else '_away_games'

        # Set data
        setattr(self, "_games", pd.concat([getattr(self, "_games"), data], axis=0))
        setattr(self, games_attr, pd.concat([getattr(self, games_attr), data], axis=0))

    def __get_entity_name(self) -> str:
        return self.__class__.__name__.upper()

    def __add_base_stats(self, game: pd.DataFrame) -> pd.DataFrame:
        return game[self.STATS_COLUMNS]

    def get_data(self, is_home_team: bool, **kwargs) -> pd.DataFrame:
        # Concatenate data
        games = self._get_games(is_home_team)
        return pd.concat([
            self.__get_base_stats(games),
            self._get_advanced_stats(games, is_home_team, **kwargs)
        ], axis=1)

    def _compute_average(self, games: pd.DataFrame) -> pd.DataFrame:
        last_n_games = games.tail(self._args.last_n_games)
        result = None

        if self._args.average == 'mean':
            result = last_n_games.mean()
        elif self._args.average == 'ema':
            result = last_n_games.ewm(com=self._args.last_n_games).mean().iloc[-1]
        elif self._args.average == 'ema_mean':
            result = last_n_games.ewm(com=self._args.last_n_games).mean().mean()

        data = pd.DataFrame(result).T.reset_index(drop=True)
        return data.map(lambda x: round(x, 2))


    def __get_base_stats(self, games: pd.DataFrame) -> pd.DataFrame:
        return self._compute_average(games[[f'{self.__get_entity_name()}_{col}' for col in self.STATS_COLUMNS]])

    def _get_games(self, is_home_team: bool) -> pd.DataFrame:
        game_attribute = '_games'
        if self._args.separate_home_away:
            game_attribute = '_home_games' if is_home_team else '_away_games'

        # Return data
        return getattr(self, game_attribute).copy()

    @abstractmethod
    def _add_advanced_stats(self, game: pd.DataFrame, is_home_team: bool, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def _get_advanced_stats(self, games: pd.DataFrame, is_home_team: bool, **kwargs) -> pd.DataFrame:
        raise NotImplementedError


class Team(GameEntity):
    HOME_STATS_COLUMNS: list[str] = [f'HOME_{col}' for col in GameEntity.STATS_COLUMNS]
    AWAY_STATS_COLUMNS: list[str] = [f'AWAY_{col}' for col in GameEntity.STATS_COLUMNS]

    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.__rest: float = 1.0
        self.__last_game_date: date = date.min
        self.__alpha: float = 2 / (self._args.last_n_games + 1)
        self.__players: defaultdict[int, Player] = defaultdict(lambda: Player(args))

    def __update_rest(self, game_date: date) -> None:
        time_diff = (game_date - self.__last_game_date).days
        old_rest = self.__rest
        new_rest = min(1.0, time_diff / self._args.max_rest)

        self.__last_game_date = game_date
        rest = self.__alpha * new_rest + (1 - self.__alpha) * old_rest
        self.__rest = round(rest, 2)

    def __add_players_data(self, players: pd.DataFrame, is_home_team: bool) -> None:
        for _, player_data in players.iterrows():
            self.__players[player_data['PLAYER_ID']].add_data(pd.DataFrame(player_data).T, is_home_team)

        for player in self.__players.values():
            player.notify_about_game()

    def __get_players_data(self, is_home_team: bool):
        players_data = list(player.get_data(is_home_team) for player in self.__players.values())
        players_data = pd.concat(players_data, axis=0)
        players_data = players_data.sort_values(by='PLAYER_MIN', ascending=False)
        assert len(players_data) >= 8, "Not enough players"

        top_8_players = players_data.head(8)
        top_8_dict = top_8_players.to_dict('list')
        return pd.DataFrame({key: [value] for key, value in top_8_dict.items()})

    def ready_to_predict(self):
        if not self._args.separate_home_away:
            return len(self._games) >= self._args.last_n_games
        return len(self._home_games) >= self._args.last_n_games and len(self._away_games) >= self._args.last_n_games

    def add_game(self, game: pd.DataFrame, players: pd.DataFrame, is_home_team: bool) -> None:
        if is_home_team:
            game = game.rename(columns=dict(zip(self.HOME_STATS_COLUMNS, GameEntity.STATS_COLUMNS)))
            self.add_data(game, is_home_team)
        else:
            game = game.rename(columns=dict(zip(self.AWAY_STATS_COLUMNS, GameEntity.STATS_COLUMNS)))
            self.add_data(game, is_home_team)

        # Add players data
        self.__update_rest(game['GAME_DATE'].iloc[-1])
        self.__add_players_data(players, is_home_team)

    def get_stats_before_game(self, game_date: date, teams: defaultdict[int, 'Team'], is_home_team: bool) -> pd.DataFrame:
        if not self.ready_to_predict():
            raise Exception("Not enough games recorded to make predictions")

        team_data = self.get_data(is_home_team, teams=teams, game_date=game_date)
        players_data = self.__get_players_data(is_home_team)
        data = pd.concat([team_data, players_data], axis=1)
        return add_prefix(data, "HOME" if is_home_team else "AWAY", data.columns)

    @overrides
    def _get_advanced_stats(self, games: pd.DataFrame, is_home_team: bool, **kwargs) -> pd.DataFrame:
        game_date = kwargs['game_date']
        teams = kwargs['teams']

        win_score, lose_score = self.__get_win_lose_score(games, teams, is_home_team)

        data = pd.DataFrame({
            'TEAM_REST': self.__rest,
            'TEAM_WIN_SCORE': win_score,
            'TEAM_LOSE_SCORE': lose_score
        }, index=[0])
        return data.map(lambda x: round(x, 2))

    def get_win_percentage(self, is_home_team: bool) -> float:
        games = self._get_games(is_home_team)
        last_n_games = games.tail(self._args.last_n_games)
        return last_n_games['WIN'].mean()

    def __get_win_lose_score(self, games: pd.DataFrame, teams: defaultdict[int, 'Team'], is_home_team: bool) -> tuple[int, int]:
        last_games = games.tail(self._args.last_n_games)
        win_score, lose_score = [], []

        for _, row in last_games.iterrows():
            opponent = row['OPPONENT_ID']
            result = row['WIN']

            # Get the opponent's win and loss percentages
            opponent_win_percentage = teams[opponent].get_win_percentage(not is_home_team)
            opponent_loss_percentage = 1 - opponent_win_percentage

            # Append the calculated scores
            win_score.append(opponent_win_percentage * result)
            lose_score.append(opponent_loss_percentage * (1 - result))

        return sum(win_score), sum(lose_score)

    @overrides
    def _add_advanced_stats(self, game: pd.DataFrame, is_home_team: bool, **kwargs) -> pd.DataFrame:
        if is_home_team:
            result_column = 'HOME_TEAM_WIN'
            opponent_column = 'AWAY_TEAM_ID'
        else:
            result_column = 'AWAY_TEAM_WIN'
            opponent_column = 'HOME_TEAM_ID'

        # Filter and rename columns
        game = game[['GAME_DATE', result_column, opponent_column]]
        return game.rename(columns={opponent_column: 'OPPONENT_ID', result_column: 'WIN'})


class Player(GameEntity):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__(args)
        self.__minutes: pd.DataFrame = pd.DataFrame(columns=['PLAYER_MIN'])
        self.__has_played: bool = False

    def notify_about_game(self) -> None:
        if not self.__has_played:
            self.__minutes.loc[len(self.__minutes)] = 0.0

        # reset for the next game
        self.__has_played = False

    @overrides
    def get_data(self, is_home_team: bool, **kwargs) -> pd.DataFrame:
        # Player surely played at least one game. If the player has not
        # played any home games, return away games and vice versa
        if self._get_games(is_home_team).empty:
            is_home_team = not is_home_team

        return super().get_data(is_home_team)

    @overrides
    def _get_advanced_stats(self, games: pd.DataFrame, is_home_team: bool, **kwargs) -> pd.DataFrame:
        return super()._compute_average(self.__minutes)

    @overrides
    def _add_advanced_stats(self, game: pd.DataFrame, is_home_team: bool, **kwargs) -> pd.DataFrame:
        self.__minutes.loc[len(self.__minutes)] = game['MIN'].iloc[-1]
        self.__has_played = True
        return pd.DataFrame()