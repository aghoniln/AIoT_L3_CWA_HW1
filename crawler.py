import os
from fetch_cwa_data import fetch_and_store_weather_data

def run_crawler():
    """Gate 5: Automated Crawler - Periodically or on-demand sync with CWA Open Data API."""
    print("Executing Automated Crawler & Deployment Sync...")
    success = fetch_and_store_weather_data()
    if success:
        print("PASS: Weather data crawler sync complete.")
        return True
    else:
        print("FAIL: Crawler sync failed.")
        return False

if __name__ == "__main__":
    run_crawler()
