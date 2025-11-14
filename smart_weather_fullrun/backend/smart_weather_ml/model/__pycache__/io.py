import pandas as pd

def get_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)
