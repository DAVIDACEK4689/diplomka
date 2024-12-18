import argparse
import ast
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--load_path", default="../featured_data", type=str, help="Path to load data.")
parser.add_argument("--store_path", default="../featured_data", type=str, help="Path to store data.")
args = parser.parse_args()

def load_featured_data(load_path: str) -> pd.DataFrame:
    data = pd.read_csv(f"{load_path}/data.csv")
    data = data.map(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    return data

def get_scaled_features(data: pd.DataFrame) -> list:
    scaled_features: list[str] = []
    for column_name in data.columns:
        if column_name.startswith('HOME_') and isinstance(data[column_name].iloc[0], list):
            scaled_features.append(column_name[len('HOME_'):])

    return scaled_features

def scale_data(store_path: str, load_path: str) -> None:
    data = load_featured_data(load_path)
    scaled_features = get_scaled_features(data)

    for feature in scaled_features:
        home_data = data[f"HOME_{feature}"].copy()
        away_data = data[f"AWAY_{feature}"].copy()
        merged = home_data.combine(away_data, lambda x, y: x + y)

        for idx, row in merged.items():
            max_value = max(row)
            scaled_value = [round(item / max_value, 2) for item in row]
            home_data[idx] = scaled_value[0:8]
            away_data[idx] = scaled_value[8:16]

        data[f"HOME_{feature}"] = home_data
        data[f"AWAY_{feature}"] = away_data

    data.to_csv(f"{store_path}/scaled_data.csv", index=False)

if __name__ == '__main__':
    args: argparse.Namespace = parser.parse_args([] if "__file__" not in globals() else None)
    scale_data(**vars(args))
