import requests
import json
import urllib3
import pandas as pd
from datetime import datetime
from config import CWA_API_KEY, CWA_DATASET_ID, CWA_36H_DATASET_ID, COUNTY_TO_REGION, CITY_COORDINATES
from db_manager import init_db, save_city_forecasts, save_regional_forecasts, cleanup_outdated_forecasts

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def fetch_cwa_dataset(dataset_id=CWA_DATASET_ID):
    """Fetch raw JSON dataset from CWA Open Data API."""
    url = f"https://opendata.cwa.gov.tw/api/v1/rest/datastore/{dataset_id}?Authorization={CWA_API_KEY}"
    try:
        response = requests.get(url, verify=False, timeout=15)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"[Error] API returned status code {response.status_code}: {response.text[:200]}")
            return None
    except Exception as e:
        print(f"[Exception] Failed to fetch CWA data: {e}")
        return None

def safe_float(val, default=0.0):
    """Safely convert API string values to float, handling '-' or empty strings."""
    try:
        if val is None:
            return default
        s = str(val).strip()
        if not s or s == "-":
            return default
        return float(s)
    except Exception:
        return default

def parse_f_d0047_091(data):
    """
    Parse F-D0047-091 (臺灣各縣市未來1週天氣預報).
    Extracts date-wise MinT, MaxT, Wx, PoP, UVI for each location.
    """
    if not data or "records" not in data or "Locations" not in data["records"]:
        return pd.DataFrame(), pd.DataFrame()

    city_records = []
    locations = data["records"]["Locations"][0]["Location"]

    for loc in locations:
        city_name = loc.get("LocationName")
        region_name = COUNTY_TO_REGION.get(city_name, "其他地區")
        lat, lon = CITY_COORDINATES.get(city_name, (23.8, 121.0))

        # Extract elements: 最低溫度, 最高溫度, 天氣現象, 12小時降雨機率, 紫外線指數
        mint_dict = {}
        maxt_dict = {}
        wx_dict = {}
        pop_dict = {}
        uvi_dict = {}

        for elem in loc.get("WeatherElement", []):
            elem_name = elem.get("ElementName")
            times = elem.get("Time", [])

            for t in times:
                start_time_str = t.get("StartTime", "")
                if not start_time_str:
                    continue
                # Extract date part YYYY-MM-DD
                date_str = start_time_str.split("T")[0]
                values = t.get("ElementValue", [{}])
                val_obj = values[0] if values else {}

                if elem_name == "最低溫度":
                    mint_val = safe_float(val_obj.get("MinTemperature"), 20.0)
                    if date_str not in mint_dict or mint_val < mint_dict[date_str]:
                        mint_dict[date_str] = mint_val
                elif elem_name == "最高溫度":
                    maxt_val = safe_float(val_obj.get("MaxTemperature"), 30.0)
                    if date_str not in maxt_dict or maxt_val > maxt_dict[date_str]:
                        maxt_dict[date_str] = maxt_val
                elif elem_name == "天氣現象":
                    wx_val = val_obj.get("Weather", "")
                    if wx_val and date_str not in wx_dict:
                        wx_dict[date_str] = wx_val
                elif elem_name == "12小時降雨機率":
                    pop_val = safe_float(val_obj.get("ProbabilityOfPrecipitation"), 0.0)
                    if date_str not in pop_dict or pop_val > pop_dict[date_str]:
                        pop_dict[date_str] = pop_val
                elif elem_name == "紫外線指數":
                    uvi_val = safe_float(val_obj.get("UVIndex"), 0.0)
                    if date_str not in uvi_dict or uvi_val > uvi_dict[date_str]:
                        uvi_dict[date_str] = uvi_val

        # Combine into city records
        all_dates = sorted(list(set(list(mint_dict.keys()) + list(maxt_dict.keys()))))
        for d in all_dates:
            mint = mint_dict.get(d, 20.0)
            maxt = maxt_dict.get(d, 30.0)
            wx = wx_dict.get(d, "多雲")
            pop = pop_dict.get(d, 0.0)
            uvi = uvi_dict.get(d, 0.0)
            city_records.append({
                "cityName": city_name,
                "regionName": region_name,
                "dataDate": d,
                "mint": round(mint, 1),
                "maxt": round(maxt, 1),
                "wx": wx,
                "pop": round(pop, 1),
                "uvi": round(uvi, 1),
                "latitude": lat,
                "longitude": lon
            })

    df_city = pd.DataFrame(city_records)

    # Compute Regional Aggregates (Poster Step 6 & 7)
    df_regional = pd.DataFrame()
    if not df_city.empty:
        regional_grouped = df_city.groupby(["regionName", "dataDate"]).agg(
            mint=("mint", "min"),
            maxt=("maxt", "max"),
            wx=("wx", "first"),
            pop=("pop", "max"),
            uvi=("uvi", "max")
        ).reset_index()
        df_regional = regional_grouped

    return df_city, df_regional

def parse_f_c0032_001(data):
    """
    Fallback parser for F-C0032-001 (今明36小時天氣預報).
    """
    if not data or "records" not in data or "location" not in data["records"]:
        return pd.DataFrame(), pd.DataFrame()

    city_records = []
    locations = data["records"]["location"]

    for loc in locations:
        city_name = loc.get("locationName")
        region_name = COUNTY_TO_REGION.get(city_name, "其他地區")
        lat, lon = CITY_COORDINATES.get(city_name, (23.8, 121.0))

        mint_dict = {}
        maxt_dict = {}
        wx_dict = {}
        pop_dict = {}

        for elem in loc.get("weatherElement", []):
            elem_name = elem.get("elementName")
            for t in elem.get("time", []):
                start_time_str = t.get("startTime", "")
                if not start_time_str:
                    continue
                date_str = start_time_str.split(" ")[0]
                param = t.get("parameter", {})

                if elem_name == "MinT":
                    mint_val = safe_float(param.get("parameterName"), 20.0)
                    mint_dict[date_str] = min(mint_dict.get(date_str, 999.0), mint_val)
                elif elem_name == "MaxT":
                    maxt_val = safe_float(param.get("parameterName"), 30.0)
                    maxt_dict[date_str] = max(maxt_dict.get(date_str, -999.0), maxt_val)
                elif elem_name == "Wx":
                    wx_val = param.get("parameterName", "多雲")
                    if date_str not in wx_dict:
                        wx_dict[date_str] = wx_val
                elif elem_name == "PoP":
                    pop_val = safe_float(param.get("parameterName"), 0.0)
                    pop_dict[date_str] = max(pop_dict.get(date_str, 0.0), pop_val)

        all_dates = sorted(list(set(list(mint_dict.keys()) + list(maxt_dict.keys()))))
        for d in all_dates:
            city_records.append({
                "cityName": city_name,
                "regionName": region_name,
                "dataDate": d,
                "mint": round(mint_dict.get(d, 20.0), 1),
                "maxt": round(maxt_dict.get(d, 30.0), 1),
                "wx": wx_dict.get(d, "多雲"),
                "pop": round(pop_dict.get(d, 0.0), 1),
                "uvi": 0.0,
                "latitude": lat,
                "longitude": lon
            })

    df_city = pd.DataFrame(city_records)
    df_regional = pd.DataFrame()
    if not df_city.empty:
        df_regional = df_city.groupby(["regionName", "dataDate"]).agg(
            mint=("mint", "min"),
            maxt=("maxt", "max"),
            wx=("wx", "first"),
            pop=("pop", "max"),
            uvi=("uvi", "max")
        ).reset_index()

    return df_city, df_regional

def fetch_and_store_weather_data():
    """Fetch weather data from CWA API and store in SQLite data.db."""
    print("Initializing database...")
    init_db()

    print(f"Fetching CWA dataset {CWA_DATASET_ID}...")
    raw_data = fetch_cwa_dataset(CWA_DATASET_ID)
    df_city, df_regional = parse_f_d0047_091(raw_data)

    if df_city.empty:
        print(f"Fallback to CWA dataset {CWA_36H_DATASET_ID}...")
        raw_data_36h = fetch_cwa_dataset(CWA_36H_DATASET_ID)
        df_city, df_regional = parse_f_c0032_001(raw_data_36h)

    if not df_city.empty:
        print("Cleaning up outdated past forecasts...")
        cleanup_outdated_forecasts()
        print(f"Saving {len(df_city)} city records and {len(df_regional)} regional records to SQLite...")
        save_city_forecasts(df_city)
        save_regional_forecasts(df_regional)
        print("Data successfully stored in data.db!")
        return True
    else:
        print("[Warning] Could not parse weather forecast data.")
        return False

if __name__ == "__main__":
    fetch_and_store_weather_data()
