import argparse
import ast
import numpy as np
import pandas as pd
import torch
import torchmetrics
from torch.utils.data import DataLoader

from NBADataset import NBADataset
from model.NBAModel import NBAModel

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
parser.add_argument("--epochs", default=250, type=int, help="Number of epochs.")
parser.add_argument("--epochs_warmup", default=50, type=int, help="Number of warmup epochs.")
parser.add_argument("--seed", default=42, type=int, help="Random seed.")
parser.add_argument("--lr", default=1e-5, type=float, help="Learning rate.")
parser.add_argument("--learning_rate_decay", default=True, action="store_true", help="Decay learning rate")
args = parser.parse_args()

def load_data(path: str, test_seasons: int = 1) -> (NBADataset, NBADataset, NBADataset):
    data = pd.read_csv(path)
    data = data.map(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    data = data.sort_values('GAME_ID', ascending=True)

    max_season = data['SEASON_YEAR'].max()
    train_data = data[data['SEASON_YEAR'] <= max_season - 2 * test_seasons]
    temp_data = data[data['SEASON_YEAR'] > max_season - 2 * test_seasons]

    dev_data = temp_data[temp_data['SEASON_YEAR'] <= max_season - test_seasons]
    test_data = temp_data[temp_data['SEASON_YEAR'] > max_season - test_seasons]
    return NBADataset(train_data), NBADataset(dev_data), NBADataset(test_data)

def collate_fn(batch):
    """
    Custom collate function to preserve the tuple structure (team_data, player_data).
    """
    team_data = torch.stack([item[0][0] for item in batch])
    player_data = torch.stack([item[0][1] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return (team_data, player_data), labels


class CosineWithWarmup(torch.optim.lr_scheduler.LambdaLR):
    def __init__(self, optimizer, args, train):
        self._warmup_steps = args.epochs_warmup * len(train)
        self._decay_steps = args.learning_rate_decay and (args.epochs - args.epochs_warmup) * len(train)
        super().__init__(optimizer, self.current_learning_rate)

    def current_learning_rate(self, step):
        assert step >= 0 and (not self._decay_steps or step <= self._warmup_steps + self._decay_steps)
        return step / self._warmup_steps if step < self._warmup_steps else \
            0.5 * (1 + np.cos(np.pi * (step - self._warmup_steps) / self._decay_steps)) if self._decay_steps else 1


def main(args):
    train, dev, test = load_data("../featured_data/scaled_data.csv")
    team_features, players_count, players_features = train.teams_data.shape[1], train.players_data.shape[1], train.players_data.shape[2]

    torch.manual_seed(args.seed)
    train = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev = DataLoader(dev, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test = DataLoader(test, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = NBAModel(team_features, players_count, players_features)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    schedule = CosineWithWarmup(optimizer, args, train)
    model.configure(
        optimizer=torch.optim.Adam(model.parameters(), lr=args.lr),
        schedule=schedule,
        loss=torch.nn.BCELoss(),
        metrics={"accuracy": torchmetrics.Accuracy(task="binary")},
    )
    model.fit(train, dev=dev, epochs=args.epochs)


if __name__ == '__main__':
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
