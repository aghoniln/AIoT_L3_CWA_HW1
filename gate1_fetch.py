import json
import sys
from fetch_cwa_data import fetch_cwa_dataset
from config import CWA_DATASET_ID

def run_gate1():
    """Gate 1: API Data Ingestion - Fetch CWA weather JSON and save to gate1_output.json."""
    print("Executing Gate 1: API Data Ingestion...")
    data = fetch_cwa_dataset(CWA_DATASET_ID)
    if data:
        with open("gate1_output.json", "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print("PASS: Output saved to gate1_output.json")
        return True
    else:
        print("FAIL: API Data Ingestion failed.")
        return False

if __name__ == "__main__":
    run_gate1()
