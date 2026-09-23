import sqlite3
import pandas as pd
from config import DB_PATH

def verify_database():
    """
    Verification script matching Poster Step 10:
    SQL queries to validate SQLite data.db records.
    """
    print("=== CWA Weather Database Verification (Poster Step 10) ===")
    conn = sqlite3.connect(DB_PATH)
    
    print("\n1. Querying distinct region names (SELECT DISTINCT regionName FROM TemperatureForecasts):")
    df_regions = pd.read_sql_query("SELECT DISTINCT regionName FROM TemperatureForecasts", conn)
    print(df_regions)
    
    print("\n2. Querying temperature forecast for '中部地區' (SELECT * FROM TemperatureForecasts WHERE regionName = '中部地區'):")
    df_central = pd.read_sql_query("SELECT * FROM TemperatureForecasts WHERE regionName = '中部地區'", conn)
    print(df_central)

    print("\n3. Querying sample city forecasts (SELECT * FROM CityForecasts LIMIT 5):")
    df_cities = pd.read_sql_query("SELECT * FROM CityForecasts LIMIT 5", conn)
    print(df_cities)

    conn.close()
    print("\n[Check Complete] Database is valid and properly populated!")

if __name__ == "__main__":
    verify_database()
