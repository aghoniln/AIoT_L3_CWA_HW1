from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATASETS = {
    "洪水機率預測": {
        "file": "flood_train_sample.csv", "target": "FloodProbability", "exclude": ["id"],
        "source": "https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset",
        "description": "以 20 個環境與管理因素預測洪水機率。從 Kaggle 壓縮檔的 train.csv 固定種子 42 抽樣 50,000 筆，排除 id。原檔 1,117,957 筆；採樣僅為降低互動運算成本。合成資料，只作課堂方法展示。",
        "business": "探索環境風險因素與洪水機率的關係，建立可解釋的課堂回歸模型。此合成資料模型不適合作為真實災害預警依據。",
        "unit": "機率", "license_note": "請依 Kaggle 來源頁的最新授權與使用條款使用。",
    },
    "汽車價格預測": {
        "file": "CarPrice_Assignment.csv", "target": "price", "exclude": ["car_ID", "CarName"],
        "source": "https://www.kaggle.com/datasets/hellbuoy/car-price-prediction",
        "description": "以引擎、尺寸、油耗及車身類型等 23 個原始特徵預測價格；排除 car_ID 與高基數完整車名 CarName。類別編碼後欄數可能超過 39，原始特徵數另行呈現。",
        "business": "研究車輛規格與價格的關係，作為歷史樣本的估價練習。205 筆樣本較少，不代表目前市場行情。",
        "unit": "資料集價格單位", "license_note": "請依 Kaggle 來源頁的最新授權與使用條款使用。",
    },
}

FLOOD_ZH = dict(zip([
    "MonsoonIntensity", "TopographyDrainage", "RiverManagement", "Deforestation", "Urbanization",
    "ClimateChange", "DamsQuality", "Siltation", "AgriculturalPractices", "Encroachments",
    "IneffectiveDisasterPreparedness", "DrainageSystems", "CoastalVulnerability", "Landslides",
    "Watersheds", "DeterioratingInfrastructure", "PopulationScore", "WetlandLoss",
    "InadequatePlanning", "PoliticalFactors", "FloodProbability"
], ["季風強度", "地形排水", "河川管理", "森林砍伐", "都市化", "氣候變遷", "水壩品質", "淤積",
    "農業活動", "侵占", "防災準備不足", "排水系統", "海岸脆弱性", "山崩", "集水區",
    "基礎設施劣化", "人口評分", "濕地流失", "規劃不足", "政治因素", "洪水機率（目標）"]))

CAR_ZH = dict(zip([
    "symboling", "fueltype", "aspiration", "doornumber", "carbody", "drivewheel", "enginelocation",
    "wheelbase", "carlength", "carwidth", "carheight", "curbweight", "enginetype", "cylindernumber",
    "enginesize", "fuelsystem", "boreratio", "stroke", "compressionratio", "horsepower", "peakrpm",
    "citympg", "highwaympg", "price", "car_ID", "CarName"
], ["保險風險評級", "燃料類型", "進氣方式", "車門數", "車身類型", "驅動方式", "引擎位置",
    "軸距", "車長", "車寬", "車高", "整備重量", "引擎類型", "汽缸數", "引擎排氣量",
    "供油系統", "缸徑", "衝程", "壓縮比", "馬力", "峰值轉速", "市區油耗 mpg", "高速油耗 mpg",
    "價格（目標）", "車輛編號（排除）", "完整車名（排除）"]))

FIELD_ZH = {**FLOOD_ZH, **CAR_ZH}
