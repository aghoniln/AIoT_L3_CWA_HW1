"""Download source archives and reproduce the bundled teaching datasets."""
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent


def download(url):
    r = requests.get(url, timeout=(10, 120))
    r.raise_for_status()
    if not r.content.startswith(b"PK"):
        raise ValueError("Kaggle 未回傳 ZIP；請至來源頁手動下載壓縮檔。")
    return zipfile.ZipFile(io.BytesIO(r.content))


def ensure_data():
    folder = ROOT / "data"
    folder.mkdir(exist_ok=True)
    flood = folder / "flood_train_sample.csv"
    cars = folder / "CarPrice_Assignment.csv"
    if not flood.exists():
        with download("https://www.kaggle.com/api/v1/datasets/download/naiyakhalid/flood-prediction-dataset") as z:
            with z.open("train.csv") as f:
                original = pd.read_csv(f)
            sample = original.sample(50000, random_state=42).sort_values("id").reset_index(drop=True)
            sample.to_csv(flood, index=False)
    if not cars.exists():
        with download("https://www.kaggle.com/api/v1/datasets/download/hellbuoy/car-price-prediction") as z:
            cars.write_bytes(z.read("CarPrice_Assignment.csv"))
    return folder


if __name__ == "__main__":
    ensure_data()
    print("資料集已就緒。")
