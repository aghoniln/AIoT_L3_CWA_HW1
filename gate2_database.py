from fetch_cwa_data import fetch_and_store_weather_data
from db_manager import init_db

def run_gate2():
    """Gate 2: Database Storage - Populate SQLite database with UNIQUE constraint and UPSERT logic."""
    print("Executing Gate 2: Database Storage...")
    init_db()
    success = fetch_and_store_weather_data()
    if success:
        print("PASS: Forecast records loaded into SQLite (data.db)")
        return True
    else:
        print("FAIL: Database Storage failed.")
        return False

if __name__ == "__main__":
    run_gate2()
