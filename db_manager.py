import os
import shutil
import sqlite3
import pandas as pd
from datetime import datetime
from config import DB_PATH, DATA_DIR_DB_PATH

def get_connection(path=DB_PATH):
    """Create and return a SQLite database connection."""
    # Ensure directory exists
    dir_name = os.path.dirname(path)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def sync_data_dir_db():
    """Sync data.db to data/data.db for multi-location compatibility."""
    try:
        data_dir = os.path.dirname(DATA_DIR_DB_PATH)
        os.makedirs(data_dir, exist_ok=True)
        if os.path.exists(DB_PATH):
            shutil.copy2(DB_PATH, DATA_DIR_DB_PATH)
    except Exception as e:
        print(f"[Warning] DB Sync exception: {e}")

def init_db():
    """
    Initialize SQLite database tables according to poster steps 8 & 9.
    Tables:
      - TemperatureForecasts (Regional temperature forecast)
      - CityForecasts (City/County temperature forecast)
    """
    for path in [DB_PATH, DATA_DIR_DB_PATH]:
        conn = get_connection(path)
        cursor = conn.cursor()
        
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS TemperatureForecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            regionName TEXT NOT NULL,
            dataDate TEXT NOT NULL,
            mint REAL NOT NULL,
            maxt REAL NOT NULL,
            wx TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(regionName, dataDate) ON CONFLICT REPLACE
        );
        """)

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS CityForecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cityName TEXT NOT NULL,
            regionName TEXT NOT NULL,
            dataDate TEXT NOT NULL,
            mint REAL NOT NULL,
            maxt REAL NOT NULL,
            wx TEXT,
            latitude REAL,
            longitude REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(cityName, dataDate) ON CONFLICT REPLACE
        );
        """)
        
        conn.commit()
        conn.close()

def save_regional_forecasts(df_regional):
    """Insert or replace regional forecasts into TemperatureForecasts."""
    if df_regional is None or df_regional.empty:
        return
    
    for path in [DB_PATH, DATA_DIR_DB_PATH]:
        conn = get_connection(path)
        cursor = conn.cursor()
        
        for _, row in df_regional.iterrows():
            cursor.execute("""
            INSERT INTO TemperatureForecasts (regionName, dataDate, mint, maxt, wx)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(regionName, dataDate) DO UPDATE SET
                mint = excluded.mint,
                maxt = excluded.maxt,
                wx = excluded.wx,
                created_at = CURRENT_TIMESTAMP
            """, (row["regionName"], row["dataDate"], row["mint"], row["maxt"], row.get("wx", "")))
            
        conn.commit()
        conn.close()

def save_city_forecasts(df_city):
    """Insert or replace city forecasts into CityForecasts."""
    if df_city is None or df_city.empty:
        return

    for path in [DB_PATH, DATA_DIR_DB_PATH]:
        conn = get_connection(path)
        cursor = conn.cursor()

        for _, row in df_city.iterrows():
            cursor.execute("""
            INSERT INTO CityForecasts (cityName, regionName, dataDate, mint, maxt, wx, latitude, longitude)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cityName, dataDate) DO UPDATE SET
                regionName = excluded.regionName,
                mint = excluded.mint,
                maxt = excluded.maxt,
                wx = excluded.wx,
                latitude = excluded.latitude,
                longitude = excluded.longitude,
                created_at = CURRENT_TIMESTAMP
            """, (
                row["cityName"], row["regionName"], row["dataDate"],
                row["mint"], row["maxt"], row.get("wx", ""),
                row.get("latitude", 0.0), row.get("longitude", 0.0)
            ))

        conn.commit()
        conn.close()

def get_distinct_regions():
    """Get list of distinct regions (Poster Step 10)."""
    conn = get_connection()
    df = pd.read_sql_query("SELECT DISTINCT regionName FROM TemperatureForecasts ORDER BY regionName", conn)
    conn.close()
    return df["regionName"].tolist() if not df.empty else []

def query_regional_forecasts(region_name=None):
    """Query regional temperature forecasts from TemperatureForecasts (Poster Step 10 & 12)."""
    conn = get_connection()
    if region_name and region_name != "全區" and region_name != "全部地區":
        sql = "SELECT * FROM TemperatureForecasts WHERE regionName = ? ORDER BY dataDate"
        df = pd.read_sql_query(sql, conn, params=(region_name,))
    else:
        sql = "SELECT * FROM TemperatureForecasts ORDER BY regionName, dataDate"
        df = pd.read_sql_query(sql, conn)
    conn.close()
    return df

def query_city_forecasts(city_name=None, region_name=None):
    """Query city level temperature forecasts."""
    conn = get_connection()
    if city_name:
        sql = "SELECT * FROM CityForecasts WHERE cityName = ? ORDER BY dataDate"
        df = pd.read_sql_query(sql, conn, params=(city_name,))
    elif region_name and region_name != "全區" and region_name != "全部地區":
        sql = "SELECT * FROM CityForecasts WHERE regionName = ? ORDER BY cityName, dataDate"
        df = pd.read_sql_query(sql, conn, params=(region_name,))
    else:
        sql = "SELECT * FROM CityForecasts ORDER BY cityName, dataDate"
        df = pd.read_sql_query(sql, conn)
    conn.close()
    return df
