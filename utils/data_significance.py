import argparse
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument("--load_path", default="../featured_data", type=str, help="Path to load data.")
parser.add_argument("--store_path", default="../featured_data", type=str, help="Path to store data.")
args = parser.parse_args()

def load_scaled_data(load_path: str) -> pd.DataFrame:
    return pd.read_csv(f"{load_path}/data.csv")

def get_team_data(data):
    return data.filter(regex='^HOME_TEAM|^AWAY_TEAM')


from collections import Counter

def count_rest_influence(data):
    # Filter the necessary columns
    data = data[['HOME_TEAM_REST', 'HOME_TEAM_WIN']]

    # Initialize counters for wins and total games
    win_counter = Counter()
    total_counter = Counter()

    # Count wins and total games for each rest value
    for _, row in data.iterrows():
        rest = row['HOME_TEAM_REST']
        win = row['HOME_TEAM_WIN']
        win_counter[rest] += win
        total_counter[rest] += 1

    # Compute average win rate for each rest value
    avg_win_rate = {
        rest: win_counter[rest] / total_counter[rest]
        for rest in total_counter
    }

    # Print the average win rates
    print("Average Win Rates by HOME_TEAM_REST:")
    for rest, win_rate in sorted(avg_win_rate.items()):
        print(f"Rest {rest}: {win_rate:.2f}")

    return avg_win_rate





def measure_significance(store_path: str, load_path: str) -> None:
    data = load_scaled_data(load_path)
    data = get_team_data(data)


    count_rest_influence(data)
    return

    correlation_matrix = data.corr()
    home_team_win_corr = correlation_matrix['HOME_TEAM_WIN'].drop('HOME_TEAM_WIN')
    home_team_win_corr = home_team_win_corr.abs().sort_values(ascending=False)
    print(home_team_win_corr.head(20))

    X = data.drop(['HOME_TEAM_WIN'], axis=1)
    y = data['HOME_TEAM_WIN']
    features = X.columns
    rf = RandomForestRegressor()
    rf.fit(X.to_numpy(), y.to_numpy())

    importances = rf.feature_importances_
    sorted_features = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)

    for feature, importance in sorted_features:
        print(f'Feature: {feature}, has an importance of {importance}')

    print(accuracy_score(y, rf.predict(X.to_numpy()).round()))


if __name__ == '__main__':
    args: argparse.Namespace = parser.parse_args([] if "__file__" not in globals() else None)
    measure_significance(**vars(args))