import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class NBADataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        self.teams_data, self.players_data, self.labels, self.mapping = prepare_data(data)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        team_data = torch.tensor(self.teams_data[idx], dtype=torch.float32)
        player_data = torch.tensor(self.players_data[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return (team_data, player_data), label

    def get_mapping(self):
        return self.mapping


def prepare_data(data: pd.DataFrame) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
    labels = data['HOME_TEAM_WIN'].to_numpy()
    mapping = data['GAME_ID'].to_numpy()
    data = data.drop(['SEASON_YEAR', 'GAME_ID', 'HOME_TEAM_WIN'], axis=1)
    players_features = get_players_features(data)

    teams_data = data.drop(players_features, axis=1).to_numpy()
    players_data = get_players_data(data[players_features])
    return teams_data, players_data, labels, mapping


def get_players_features(data: pd.DataFrame) -> list:
    scaled_features: list[str] = []
    for column_name in data.columns:
        if isinstance(data[column_name].iloc[0], list):
            scaled_features.append(column_name)
    return scaled_features


def get_players_data(players: pd.DataFrame) -> np.ndarray:
    home_players_data = players.loc[:, players.columns.str.startswith('HOME_')].reset_index(drop=True)
    away_players_data = players.loc[:, players.columns.str.startswith('AWAY_')].reset_index(drop=True)
    assert home_players_data.shape == away_players_data.shape

    home_players_data = prepare_players_data(home_players_data)
    away_players_data = prepare_players_data(away_players_data)
    return np.concatenate((home_players_data, away_players_data), axis=1)


def prepare_players_data(players_data: pd.DataFrame) -> np.ndarray:
    num_games, num_players, num_features = players_data.shape[0], 8, players_data.shape[1]
    array = np.zeros((num_games, num_players, num_features), dtype=np.float32)

    for idx, row in players_data.iterrows():
        player_data = np.zeros((8, num_features), dtype=np.float32)
        for feature_idx in range(num_features):
            player_data[:, feature_idx] = row.iloc[feature_idx]
        array[idx] = player_data

    return array


