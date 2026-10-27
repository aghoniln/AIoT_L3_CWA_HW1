# HW2 多元線性回歸實驗報告

本文件由實際執行產生；種子 42，80/20 切分，訓練集五折交叉驗證。

## 洪水機率預測

來源：https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset

以 20 個環境與管理因素預測洪水機率。從 Kaggle 壓縮檔的 train.csv 固定種子 42 抽樣 50,000 筆，排除 id。原檔 1,117,957 筆；採樣僅為降低互動運算成本。合成資料，只作課堂方法展示。

### Business Understanding

探索環境風險因素與洪水機率的關係，建立可解釋的課堂回歸模型。此合成資料模型不適合作為真實災害預警依據。

### Data Understanding

50,000 筆、20 個有效原始輸入特徵；目標 FloodProbability。缺失儲存格 0。

### Data Preparation

排除 ['id']；先移除目標缺失與完全重複觀測，再切分為訓練 40,000 筆、測試 10,000 筆。補值、編碼、標準化及 RFE 在各訓練折內擬合。

### Modeling

使用 OLS，比較全特徵與 Group RFE。類別變數的 dummy 欄位視為同一原始特徵群組。

| 模型 | 原始特徵數 | CV RMSE | CV RMSE 標準差 | CV R² |
|---|---:|---:|---:|---:|
| 全特徵 OLS | 20 | 0.0200581 | 0.000214032 | 0.845382 |
| Group RFE：14 特徵 | 14 | 0.0342036 | 0.000417593 | 0.550471 |
| Group RFE：16 特徵 | 16 | 0.0304015 | 0.000277924 | 0.644861 |
| Group RFE：18 特徵 | 18 | 0.0258853 | 0.000250355 | 0.742535 |

最終模型：**全特徵 OLS**，依訓練 CV RMSE 決定，未用測試集調參。

RFE 保留：TopographyDrainage、RiverManagement、Deforestation、Urbanization、ClimateChange、DamsQuality、Siltation、AgriculturalPractices、Encroachments、IneffectiveDisasterPreparedness、DrainageSystems、CoastalVulnerability、Landslides、Watersheds、PopulationScore、WetlandLoss、InadequatePlanning、PoliticalFactors

### Evaluation

| 模型 | MAE | MSE | RMSE | R² | 訓練調整後 R² | 測試 PI 覆蓋率 |
|---|---:|---:|---:|---:|---:|---:|
| 全特徵 OLS | 0.0157534 | 0.000396979 | 0.0199243 | 0.848246 | 0.845553 | 94.53% |
| Group RFE：18 特徵 | 0.0207085 | 0.000659529 | 0.0256813 | 0.747879 | 0.746608 | 95.79% |

平均值基準 RMSE：0.05115。

模型診斷：`{"Jarque_Bera_p": 0.0, "Breusch_Pagan_p": 0.0, "design_rank": 21, "design_columns": 21, "residual_df": 39979, "condition_number": 1.1474363101241798, "dependent_encoded_columns_removed": 0, "independence": "須依資料取得方式判斷；隨機列順序無法證明獨立性。"}`

圖表：`flood/prediction_interval.html`；完整結果：`flood/predictions.csv`。

95% PI 描述個別新觀測的不確定性，CI 描述平均反應的不確定性。這些是條件於已選模型的傳統 OLS 區間，未納入特徵選擇不確定性。JB/BP 與殘差圖用於檢查假設，違反假設時名目覆蓋率可能不成立。

### Deployment

以 Streamlit 提供資料預覽、特徵選擇、模型比較、圖表、新樣本及批次預測，並可下載結果。網頁實際部署網址須以成功部署後的網址為準。

## 汽車價格預測

來源：https://www.kaggle.com/datasets/hellbuoy/car-price-prediction

以引擎、尺寸、油耗及車身類型等 23 個原始特徵預測價格；排除 car_ID 與高基數完整車名 CarName。類別編碼後欄數可能超過 39，原始特徵數另行呈現。

### Business Understanding

研究車輛規格與價格的關係，作為歷史樣本的估價練習。205 筆樣本較少，不代表目前市場行情。

### Data Understanding

205 筆、23 個有效原始輸入特徵；目標 price。缺失儲存格 0。

### Data Preparation

排除 ['car_ID', 'CarName']；先移除目標缺失與完全重複觀測，再切分為訓練 163 筆、測試 41 筆。補值、編碼、標準化及 RFE 在各訓練折內擬合。

### Modeling

使用 OLS，比較全特徵與 Group RFE。類別變數的 dummy 欄位視為同一原始特徵群組。

| 模型 | 原始特徵數 | CV RMSE | CV RMSE 標準差 | CV R² |
|---|---:|---:|---:|---:|
| 全特徵 OLS | 23 | 3016.17 | 969.16 | 0.812094 |
| Group RFE：14 特徵 | 14 | 3282.01 | 1273.28 | 0.782861 |
| Group RFE：16 特徵 | 16 | 3077.52 | 1155.68 | 0.805641 |
| Group RFE：18 特徵 | 18 | 3047.8 | 1102.75 | 0.810144 |
| Group RFE：21 特徵 | 21 | 3024.28 | 996.684 | 0.811919 |

最終模型：**全特徵 OLS**，依訓練 CV RMSE 決定，未用測試集調參。

RFE 保留：symboling、fueltype、aspiration、doornumber、carbody、drivewheel、enginelocation、wheelbase、carlength、carwidth、curbweight、enginetype、cylindernumber、enginesize、fuelsystem、boreratio、stroke、compressionratio、peakrpm、citympg、highwaympg

### Evaluation

| 模型 | MAE | MSE | RMSE | R² | 訓練調整後 R² | 測試 PI 覆蓋率 |
|---|---:|---:|---:|---:|---:|---:|
| 全特徵 OLS | 2251.87 | 9.45268e+06 | 3074.52 | 0.884096 | 0.934431 | 90.24% |
| Group RFE：21 特徵 | 2274.09 | 9.49069e+06 | 3080.7 | 0.88363 | 0.935424 | 87.80% |

平均值基準 RMSE：9234.91。

模型診斷：`{"Jarque_Bera_p": 0.05024971935316906, "Breusch_Pagan_p": 0.00015990162856286725, "design_rank": 40, "design_columns": 40, "residual_df": 123, "condition_number": 59.26094023784187, "dependent_encoded_columns_removed": 2, "independence": "須依資料取得方式判斷；隨機列順序無法證明獨立性。"}`

圖表：`cars/prediction_interval.html`；完整結果：`cars/predictions.csv`。

95% PI 描述個別新觀測的不確定性，CI 描述平均反應的不確定性。這些是條件於已選模型的傳統 OLS 區間，未納入特徵選擇不確定性。JB/BP 與殘差圖用於檢查假設，違反假設時名目覆蓋率可能不成立。

### Deployment

以 Streamlit 提供資料預覽、特徵選擇、模型比較、圖表、新樣本及批次預測，並可下載結果。網頁實際部署網址須以成功部署後的網址為準。

## 結論與限制

特徵較少不保證比較好，應以訓練交叉驗證決定；測試集只負責最後評估。洪水資料是合成資料，表現不能外推到真實災害；汽車樣本僅 205 筆，且部分特徵共線、類別稀少。模型係數為關聯而非因果，外插時誤差可能增加。

### GPT 輔助內容

本專案使用 ChatGPT 協助理解作業要求、規劃 CRISP-DM 分析流程、撰寫與除錯 Python / Streamlit 程式、設計特徵選擇與交叉驗證、製作圖表及整理說明文字。模型指標由程式實際讀取 Kaggle CSV 並訓練後計算，未以文字生成或預設數值代替實驗結果。

學生應實際執行程式、確認資料來源與輸出、理解結果限制並負責最後繳交。NotebookLM 的研究摘要與完整聊天紀錄 PDF 為另外的學習證據，不可用 GPT 文字冒充 NotebookLM 輸出。

### 對話 PDF

請在 ChatGPT 中將此次作業的完整相關對話透過瀏覽器列印、pdfCrowd 或其他可用方法匯出 PDF，確認程式碼與圖表沒有被截斷。此專案附有 AI 使用揭露，尚未包含完整聊天紀錄的 PDF 匯出。


### NotebookLM 摘要｜待完成實際研究

**狀態：尚未透過 NotebookLM 完成研究，以下是操作步驟，不是 NotebookLM 摘要。**

1. 開啟 NotebookLM，建立「HW2 多元線性回歸」筆記本。
2. 加入與本專案同主題的 Kaggle 解題文章或 Notebook，並加入 scikit-learn 與 statsmodels 官方文件。
3. 請 NotebookLM 比較資料清理、特徵選擇、交叉驗證、回歸假設與預測區間的做法。
4. 檢查其引述來源，取得至少 100 字摘要後，將本段替換成實際摘要，保留來源連結與研究日期。
5. 說明你的程式採用哪些做法、與網路解法有何差異。

可直接使用的研究提示詞：

> 請只依據我加入的來源，研究洪水機率或汽車價格的多元線性回歸解法。比較資料準備、特徵選擇、模型評估與不確定性呈現，特別指出資料洩漏與測試集使用上的問題。請以繁體中文寫 150～250 字摘要，並逐項標示來源；不要將未實作的方法當成我的實驗結果。

建議來源：

- https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset
- https://www.kaggle.com/datasets/hellbuoy/car-price-prediction
- https://scikit-learn.org/stable/common_pitfalls.html
- https://scikit-learn.org/stable/modules/feature_selection.html
- https://www.statsmodels.org/stable/examples/notebooks/generated/ols.html
