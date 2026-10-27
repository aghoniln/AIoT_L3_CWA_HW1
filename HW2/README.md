# HW2｜多元線性回歸研究室

以 Python、Streamlit、scikit-learn Pipeline 與 statsmodels 完成兩種回歸問題，包含 CRISP-DM、Group RFE 特徵選擇、五折交叉驗證、測試集評估與 95% 信賴／預測區間。可上傳 CSV 或輸入公開 HTTPS CSV 網址，也能輸入新樣本與批次預測。

## 已發布線上版

https://hw2-regression-lab-aghoniln.freshplum4098.chatgpt.site

目前為公開網站，任何人取得網址即可瀏覽，不需要登入 ChatGPT，可直接提供給老師。此伴隨版內建題目顯示實際 Python 實驗，自訂 CSV 則以瀏覽器重新訓練。

## Windows 11 快速開始

建議 Python 3.11 或 3.12。在本資料夾開啟 PowerShell：

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

瀏覽器開啟終端機顯示的本機網址。左側選擇題目，再按「開始分析與訓練」。若 Python 版本不同，把 `py -3.12` 改為已安裝的版本。

ZIP 內已附 CSV；GitHub 版本可在首次啟動時自動下載，也可以先執行：

```powershell
.\.venv\Scripts\python.exe setup_data.py
.\.venv\Scripts\python.exe run_analysis.py
```

`run_analysis.py` 會重建 `reports/RESULTS.md`、模型摘要、CSV 與可離線開啟的互動預測圖。

## 兩份資料集與特徵數

| 題目 | Kaggle 來源 | 使用資料 | 有效原始輸入特徵 | 目標 |
|---|---|---|---:|---|
| 洪水機率 | https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset | 壓縮檔 train.csv，固定種子 42 抽 50,000 筆 | 20（不含 id） | FloodProbability |
| 汽車價格 | https://www.kaggle.com/datasets/hellbuoy/car-price-prediction | CarPrice_Assignment.csv，205 筆 | 23（不含 car_ID、CarName） | price |

原始資料集授權與使用規定以 Kaggle 來源頁為準。汽車資料字典附於 ZIP 的 `data/car_data_dictionary.xlsx`。模型使用的是原始欄位數；One-hot 編碼後欄數另外呈現。不能把 ID、目標或編碼後欄數當成原始特徵數。

洪水上游 train.csv 為 1,117,957 筆，取樣只為降低互動運算成本。`data/flood_sampling.json` 記錄方法與 SHA-256。沒有使用同壓縮檔的 flood.csv：該檔的目標可幾乎精確由輸入加總推得，不適合拿滿分表現展示一般化能力。

## 分析方法

1. 檢查資料與欄位；移除缺失目標及完全重複觀測，避免完全重複列跨越訓練／測試集。
2. 預設 80/20 隨機切分、種子 42。時間資料另可採依日期切分與 TimeSeriesSplit；群組／重複個體資料需另採群組切分，不宜直接使用隨機模式。
3. 在每個訓練折內擬合補值、One-hot 編碼與標準化。Group RFE 以同一原始欄位所有編碼欄的最大絕對標準化係數作組重要性，每次重新擬合並移除最弱組。
4. 只用訓練集五折 CV RMSE 選擇保留數與最終模型。測試集不參與特徵選擇與模型選擇。
5. 使用訓練集 pivoted QR 移除編碼後的固定／完全相依欄位；保留獨立基底，避免完全共線造成不唯一係數。
6. 在訓練資料擬合 OLS，在保留測試資料計算 MAE、MSE、RMSE、R²；附訓練調整後 R²與平均值基準。
7. 透過 `get_prediction().summary_frame()` 計算平均反應 CI 與個別觀測 PI。測試集實際 PI 覆蓋率與平均寬度另列，並不保證等於名目 95%。

報告與網頁包括殘差圖、Q–Q、Jarque–Bera、Breusch–Pagan、VIF 與矩陣秩。獨立性需依資料取得方式判斷，隨機列序無法證明獨立性。係數與 p 值只供探索；區間條件於已選模型，未納入特徵選擇的不確定性。異變異等假設不符時，傳統 OLS 區間可能覆蓋不足。洪水合成資料不適用真實防災；汽車價格也不代表目前行情。

## 網頁與加分功能

- 兩種回歸題目下拉選單。
- CSV 上傳與公開 HTTPS 網址，支援選擇目標與原始特徵。
- 非 14～39 個特徵時可進入示範模式，但會標示不符合主要作業資料條件。
- 資料理解、特徵選擇、模型比較、測試評估、含區間預測圖。
- 新資料表單、批次 CSV 預測與結果下載。
- CRISP-DM 六個階段與 AI 使用揭露。
- 預設網路 CSV 上限 20 MB／100,000 筆，僅接受公開 HTTPS。大型資料請先取樣。

教師提供的 50_Startups.csv 可用來驗證自訂網址功能；不應當作本作業符合特徵數條件的主要資料集。

## GitHub 與 Streamlit Community Cloud

建議將本次 GitHub 版本置於原 HW1 儲存庫的新分支 `hw2-multiple-linear-regression`、子資料夾 `HW2/`，保留原 HW1 根目錄。部署時請設定：

- Repository：`aghoniln/AIoT_L3_CWA_HW1`
- Branch：`hw2-multiple-linear-regression`
- Main file path：`HW2/app.py`
- Python：3.12

Streamlit Community Cloud 首次啟動會從 Kaggle 下載所需資料；若來源暫時不可用，可手動將 ZIP 內 data/ 的兩份 CSV 上傳至該分支的 HW2/data/。

也可在自己的其他 Python 主機執行 `python -m streamlit run app.py`。公開網址請以成功部署的實際網址為準，不要繳交 localhost。

## HW1 延伸

ZIP 中 `hw1/` 保留原氣象專案，可由左側切換。原程式寫死的 API 金鑰已在副本移除，改由 `CWA_API_KEY` 環境變數提供；沒有金鑰不影響 HW2。原上傳檔案未被覆寫。

若未附 hw1/，HW1 切換會讀取父層原有程式，因此沿用既有專案；HW2 模組不依賴 CWA API。

## 測試

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
```

檢查保留測試集、測試標籤不影響特徵選擇、區間順序、新資料格式、未知類別、無效網址與時間切分。兩個 Kaggle 題目另有實際訓練與 Streamlit AppTest 驗證。

## AI 與尚需人工完成的學習證據

- GPT 輔助揭露：`reports/AI_DISCLOSURE.md`。
- NotebookLM：依 `reports/NOTEBOOKLM_GUIDE.md` 實際研究，補入至少 100 字摘要及來源；目前不冒充已完成。
- 完整相關 ChatGPT 對話需由使用者匯出 PDF；此 ZIP 不含帳號內完整對話匯出。
- 實驗報告：`reports/RESULTS.md`，由實際訓練產生。

## 官方方法參考

- https://scikit-learn.org/stable/common_pitfalls.html
- https://scikit-learn.org/stable/modules/feature_selection.html
- https://www.statsmodels.org/stable/examples/notebooks/generated/ols.html

## 原始程式位置

本專案位於 `hw2-multiple-linear-regression` 分支的 `HW2/` 資料夾。原 HW1 保留於儲存庫根目錄，HW2 可獨立執行。

瀏覽器版原始碼位於 `web/`；在該資料夾執行 `python -m http.server 8000` 後開啟本機網址即可檢視。
