import os

# Load .env if present
env_path = os.path.join(os.path.dirname(__file__), ".env")
if os.path.exists(env_path):
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

# CWA Open Data API Configuration
CWA_API_KEY = os.environ.get("CWA_API_KEY", "CWA-009C2096-4019-4163-9C59-587ED1DD7B1E")
CWA_DATASET_ID = os.environ.get("CWA_DATASET_ID", "F-D0047-091")  # 臺灣各縣市未來1週天氣預報
CWA_36H_DATASET_ID = "F-C0032-001"  # 今明36小時天氣預報

# Database Configuration
DB_PATH = os.path.join(os.path.dirname(__file__), "data.db")
DATA_DIR_DB_PATH = os.path.join(os.path.dirname(__file__), "data", "data.db")

# Region to County Mapping
REGION_MAPPING = {
    "北部地區": ["臺北市", "新北市", "基隆市", "桃園市", "新竹市", "新竹縣", "宜蘭縣"],
    "中部地區": ["臺中市", "苗栗縣", "彰化縣", "南投縣", "雲林縣"],
    "南部地區": ["高雄市", "臺南市", "嘉義市", "嘉義縣", "屏東縣"],
    "東部地區": ["花蓮縣", "臺東縣"],
    "離島地區": ["澎湖縣", "金門縣", "連江縣"]
}

COUNTY_TO_REGION = {}
for region, counties in REGION_MAPPING.items():
    for county in counties:
        COUNTY_TO_REGION[county] = region

# Coordinates for Taiwan Cities/Counties (Lat, Lon)
CITY_COORDINATES = {
    "臺北市": (25.0330, 121.5654),
    "新北市": (24.9157, 121.6739),
    "基隆市": (25.1283, 121.7419),
    "桃園市": (24.9936, 121.3010),
    "新竹市": (24.8138, 120.9675),
    "新竹縣": (24.8387, 121.0177),
    "苗栗縣": (24.5602, 120.8214),
    "臺中市": (24.1477, 120.6736),
    "彰化縣": (24.0518, 120.5161),
    "南投縣": (23.9037, 120.6859),
    "雲林縣": (23.7092, 120.4313),
    "嘉義市": (23.4801, 120.4491),
    "嘉義縣": (23.4588, 120.5740),
    "臺南市": (22.9997, 120.2270),
    "高雄市": (22.6273, 120.3014),
    "屏東縣": (22.5519, 120.5487),
    "宜蘭縣": (24.7570, 121.7530),
    "花蓮縣": (23.9872, 121.6016),
    "臺東縣": (22.7583, 121.1444),
    "澎湖縣": (23.5711, 119.5793),
    "金門縣": (24.4493, 118.3766),
    "連江縣": (26.1505, 119.9499)
}

REGION_CENTER_COORDINATES = {
    "北部地區": (25.0000, 121.5000),
    "中部地區": (24.1000, 120.6500),
    "南部地區": (22.9000, 120.3000),
    "東部地區": (23.5000, 121.4000),
    "離島地區": (23.6000, 119.6000)
}
