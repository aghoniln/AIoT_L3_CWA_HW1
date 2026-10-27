from __future__ import annotations

import hashlib
import json
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import streamlit as st
from threadpoolctl import threadpool_limits

from datasets import ROOT, DATASETS, FIELD_ZH
from regression_core import read_csv_bytes, fetch_csv, train_analysis, predict_new


@st.cache_data
def load_builtin(filename):
    if not (ROOT / "data" / filename).exists():
        from setup_data import ensure_data
        ensure_data()
    return read_csv_bytes((ROOT / "data" / filename).read_bytes())


def signature(df, settings):
    h = hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    h.update(json.dumps(settings, sort_keys=True, ensure_ascii=False).encode())
    return h.hexdigest()


def clean_json(value):
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None
    if isinstance(value, np.integer):
        return int(value)
    return value


def diagnostics_view(r):
    st.markdown("**回歸假設與診斷**")
    d = r["diagnostics"]
    for key, val in d.items():
        st.write(f"{key}：{val:.5g}" if isinstance(val, float) else f"{key}：{val}")
    st.caption("JB 檢查殘差常態性；BP 檢查等變異性。小 p 值表示有偏離證據，不代表模型完全不能預測。VIF 較大表示共線性較強。")
    st.dataframe(r["vif"], hide_index=True, width="stretch")
    if d["design_rank"] < d["design_columns"]:
        st.warning("設計矩陣有線性相依，係數無法唯一識別；請減少重複或共線特徵。區間與係數解讀需審慎。")
    st.info("95% 區間為條件於已選模型的傳統 OLS 區間，依賴線性、等變異與誤差分布等假設，未納入特徵選擇不確定性。係數 p 值僅供探索，不能當作選擇後的確證性檢定。")


def prediction_plots(r):
    p = r["predictions"]
    shown = p.sample(min(1500, len(p)), random_state=42)
    fig = px.scatter(shown, x="actual", y="prediction", opacity=0.6,
                     labels={"actual": "實際值", "prediction": "預測值"}, title="測試集：實際值 vs 預測值")
    low, high = float(p[["actual", "prediction"]].min().min()), float(p[["actual", "prediction"]].max().max())
    fig.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines", name="理想預測", line=dict(dash="dash", color="#e7a65b")))
    st.plotly_chart(fig, width="stretch")
    # Sort solely for display, never reinterpret sample order as time.
    q = p.sort_values("prediction").iloc[np.linspace(0, len(p) - 1, min(200, len(p)), dtype=int)]
    x = np.arange(len(q))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=q.pi_upper, mode="lines", line=dict(width=0), showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=q.pi_lower, mode="lines", fill="tonexty", line=dict(width=0),
                             fillcolor="rgba(35,156,165,0.18)", name="95% 預測區間"))
    fig.add_trace(go.Scatter(x=x, y=q.prediction, mode="lines", name="預測值", line=dict(color="#239ca5")))
    fig.add_trace(go.Scatter(x=x, y=q.actual, mode="markers", name="實際值", marker=dict(color="#e7a65b", size=5)))
    fig.update_layout(title="測試集預測與 95% 預測區間", xaxis_title="依預測值排序的展示樣本（非時間）", yaxis_title=r["target"])
    st.plotly_chart(fig, width="stretch")
    st.caption("最多展示 200 筆；完整測試集與區間可下載。CI 是平均反應的信賴區間，PI 是個別新觀測的預測區間。")
    a, b = st.columns(2)
    with a:
        fig = px.scatter(shown, x="prediction", y="residual", opacity=0.5, title="測試集殘差 vs 預測值")
        fig.add_hline(y=0, line_dash="dash")
        st.plotly_chart(fig, width="stretch")
    with b:
        os, observed = stats.probplot(np.asarray(r["ols"].resid), dist="norm", fit=False)
        fig = go.Figure(go.Scatter(x=os, y=observed, mode="markers", marker=dict(size=4)))
        fig.update_layout(title="訓練集殘差 Q–Q 圖", xaxis_title="常態理論分位數", yaxis_title="殘差分位數")
        st.plotly_chart(fig, width="stretch")


def render():
    st.set_page_config(page_title="HW2｜多元線性回歸研究室", page_icon="📈", layout="wide")
    st.markdown("""<style>
    .stApp {background:#f6f8fb} h1,h2,h3 {color:#102a43}
    [data-testid='stMetric'] {background:white;border:1px solid #dde5ed;padding:16px;border-radius:12px}
    .hero {padding:30px;border-radius:18px;background:#102a43;color:#fff;margin-bottom:20px}
    .hero h1 {color:#fff;margin:8px 0}.hero p {color:#cfdae7;margin-bottom:0}
    </style><div class="hero"><small>HW2 · CRISP-DM · INTERPRETABLE REGRESSION</small>
    <h1>多元線性回歸研究室</h1><p>從資料理解到新資料預測，讓每一次預測都有依據。</p></div>""", unsafe_allow_html=True)
    problem = st.sidebar.selectbox("選擇預測問題", list(DATASETS) + ["自訂 CSV 資料集"])
    if problem in DATASETS:
        meta = DATASETS[problem]
        try:
            with st.spinner("準備資料集（首次啟動可能需要下載）…"):
                df = load_builtin(meta["file"])
        except Exception as e:
            st.error(f"無法準備資料：{e}。請執行 python setup_data.py，或使用自訂 CSV 上傳。")
            return
        st.caption(meta["description"])
        st.markdown(f"資料來源：[Kaggle 原始頁面]({meta['source']})")
        default_target = meta["target"]
        excludes = meta["exclude"]
    else:
        meta = {"business": "請描述預測目標、使用情境及模型限制。", "source": "使用者自行提供"}
        method = st.sidebar.radio("資料輸入方式", ["上傳 CSV", "輸入 CSV 網址"])
        if method == "上傳 CSV":
            up = st.sidebar.file_uploader("上傳 UTF-8 / Big5 CSV", type="csv", key="dataset_upload")
            if up is None:
                st.info("請上傳資料集，再選擇要預測的數值欄位。主要作業資料需有 14～39 個原始輸入特徵。")
                return
            try:
                df = read_csv_bytes(up.getvalue())
            except Exception as e:
                st.error(str(e)); return
        else:
            url = st.sidebar.text_input("公開 HTTPS CSV 網址", placeholder="https://raw.githubusercontent.com/…/data.csv")
            if st.sidebar.button("載入網址資料"):
                try:
                    st.session_state["url_data"] = (url, fetch_csv(url))
                except Exception as e:
                    st.error(f"載入失敗：{e}")
            cached = st.session_state.get("url_data")
            if not cached or cached[0] != url:
                st.info("輸入直接下載 CSV 的公開網址，再按載入。Kaggle 資料頁網址不能直接當成 CSV。")
                return
            df = cached[1]
        excludes = [c for c in df if c.lower() in ("id", "car_id", "index") or c.lower().startswith("unnamed:")]
        numeric = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric:
            st.error("資料中沒有可作為回歸目標的數值欄位。"); return
        default_target = numeric[-1]
    nums = df.select_dtypes(include=np.number).columns.tolist()
    target = st.sidebar.selectbox("預測目標 y", nums, index=nums.index(default_target), key=f"target_{problem}")
    available = [c for c in df if c != target]
    features = st.sidebar.multiselect("輸入特徵 X（原始欄位）", available,
                                      default=[c for c in available if c not in excludes], key=f"features_{problem}_{target}")
    if not features:
        st.warning("請選擇輸入特徵。"); return
    meets = 14 <= len(features) <= 39
    st.sidebar.caption(f"已選 {len(features)} 個原始特徵｜{'符合主要作業條件' if meets else '示範模式，未符合主要作業特徵數條件'}")
    test_size = st.sidebar.slider("測試集比例", 0.15, 0.35, 0.20, 0.05)
    seed = int(st.sidebar.number_input("隨機種子", min_value=0, max_value=100000, value=42))
    split_mode = st.sidebar.selectbox("切分方式", ["random", "time"], format_func=lambda x: "隨機切分" if x == "random" else "依日期切分（保留未來資料）")
    order_column = st.sidebar.selectbox("日期排序欄位", df.columns) if split_mode == "time" else None
    st.sidebar.caption("同一人的重複觀測或群組資料，請先依群組分割，不宜直接隨機切分。")
    defaults = sorted({k for k in [14, 16, 18, len(features) - 2] if 1 <= k < len(features)})
    if not defaults and len(features) > 1:
        defaults = [len(features) - 1]
    ks = st.sidebar.multiselect("RFE 保留原始特徵數（五折 CV 比較）", list(range(1, len(features))), default=defaults, key=f"ks_{len(features)}")
    settings = dict(target=target, features=features, test_size=test_size, seed=seed,
                    split_mode=split_mode, order_column=order_column, k_values=ks)
    sig = signature(df, settings)
    train = st.sidebar.button("開始分析與訓練", type="primary", width="stretch")
    if train:
        try:
            bar = st.progress(0, text="正在準備資料…")
            with threadpool_limits(limits=1):
                with warnings.catch_warnings(record=True):
                    warnings.simplefilter("always")
                    r = train_analysis(df, **settings, progress=lambda v, msg: bar.progress(v, text=msg))
            st.session_state["analysis"] = (sig, r)
            bar.empty()
        except Exception as e:
            st.error(f"分析未完成：{e}")
    saved = st.session_state.get("analysis")
    r = saved[1] if saved and saved[0] == sig else None
    data_tab, model_tab, eval_tab, predict_tab, crisp_tab, ai_tab = st.tabs([
        "01 資料理解", "02 特徵與模型", "03 評估與區間", "04 新資料預測", "05 CRISP-DM", "06 AI 與繳交"])
    with data_tab:
        cols = st.columns(4)
        for c, label, value in zip(cols, ["資料筆數", "原始輸入特徵", "目標欄位", "缺失儲存格"],
                                  [f"{len(df):,}", len(features), target, int(df[features + [target]].isna().sum().sum())]):
            c.metric(label, value)
        if not meets:
            st.warning("此資料可用來測試自訂載入功能，但主要作業資料須有 14～39 個原始輸入特徵；不含目標與 ID。")
        st.dataframe(df.head(100), width="stretch")
        with st.expander("欄位字典與統計資料"):
            dictionary = pd.DataFrame([{"欄位": c, "中文說明": FIELD_ZH.get(c, "使用者提供，請補充欄位語意"),
                                        "型態": str(df[c].dtype), "缺失數": int(df[c].isna().sum())} for c in df])
            st.dataframe(dictionary, width="stretch", hide_index=True)
            st.dataframe(df.describe(include="all").T, width="stretch")
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.histogram(df, x=target, nbins=40, title="目標值分布"), width="stretch")
        with c2:
            corr = df[features + [target]].select_dtypes(include=np.number).corr()
            st.plotly_chart(px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                                      title="數值特徵相關係數（僅供描述）"), width="stretch")
        st.caption("全資料相關圖只供資料描述；實際特徵選擇與所有補值、編碼、標準化均在訓練資料內完成。")
    with model_tab:
        if r is None:
            st.info("按左側「開始分析與訓練」。變更資料或設定後，必須重新訓練。");
        else:
            st.success(f"最終模型：{r['chosen_model']}｜由訓練集交叉驗證 RMSE 決定")
            st.dataframe(r["cv"].drop(columns="k"), hide_index=True, width="stretch")
            st.write(f"訓練 {r['train_rows']:,} 筆；測試 {r['test_rows']:,} 筆；清理：{r['cleaning']}")
            st.markdown("**Group RFE 保留的原始特徵**")
            st.write("、".join(r["rfe_selected_features"]))
            st.markdown("**最終模型保留的原始特徵**")
            st.write("、".join(r["selected_features"]))
            with st.expander("RFE 移除順序與選擇方法"):
                st.write("反覆重新補值、編碼與標準化，再將同一原始欄位的 dummy 係數分為一組。以組內最大絕對標準化係數作為重要性，每次移除最小的一組。")
                st.dataframe(pd.DataFrame(r["rfe_history"]), hide_index=True)
            coef = r["coefficients"].query("term != 'Intercept'")
            shown = coef.reindex(coef.coefficient.abs().sort_values().index).tail(25)
            st.plotly_chart(px.bar(shown, x="coefficient", y="term", orientation="h", title="最終模型標準化係數（最多 25 項）"), width="stretch")
            st.dataframe(r["coefficients"], hide_index=True, width="stretch")
            st.caption("數值與 dummy 欄位均使用訓練集均值與標準差標準化；係數描述條件關聯，不代表因果。")
    with eval_tab:
        if r is None:
            st.info("訓練後會顯示測試集評估與 95% 區間。")
        else:
            st.dataframe(r["metrics"], hide_index=True, width="stretch")
            st.caption("MAE / MSE / RMSE 越小越好；R² 越大越好且可能為負。train_adjusted_R2 是訓練集 OLS 調整後 R²，不是測試集指標。PI_coverage 是測試實測覆蓋率，不能保證等於 95%。")
            st.write("僅預測訓練集平均值的基準：", r["mean_baseline"])
            prediction_plots(r)
            diagnostics_view(r)
            st.download_button("下載完整測試集預測與區間 CSV", r["predictions"].to_csv(index=False).encode("utf-8-sig"), "test_predictions.csv", "text/csv")
            summary = clean_json({"chosen_model": r["chosen_model"], "cv": r["cv"].to_dict("records"),
                                  "metrics": r["metrics"].to_dict("records"), "settings": settings,
                                  "diagnostics": r["diagnostics"], "source": meta["source"]})
            st.download_button("下載模型摘要 JSON", json.dumps(summary, ensure_ascii=False, indent=2), "model_summary.json", "application/json")
    with predict_tab:
        if r is None:
            st.info("完成訓練後即可輸入新資料。")
        else:
            st.write("輸入一筆新樣本；數值預設為訓練集的中位數，類別預設為眾數。")
            values = {}
            with st.form("new_observation"):
                columns = st.columns(3)
                for i, col in enumerate(features):
                    with columns[i % 3]:
                        label = f"{FIELD_ZH.get(col, col)} · {col}" if col in FIELD_ZH else col
                        if pd.api.types.is_numeric_dtype(r["X_train"][col]):
                            values[col] = st.number_input(label, value=float(r["X_train"][col].median()), key=f"new_{sig}_{col}")
                        else:
                            choices = r["X_train"][col].dropna().astype(str).value_counts().index.tolist()
                            values[col] = st.selectbox(label, choices, key=f"new_{sig}_{col}")
                submit = st.form_submit_button("預測這筆資料")
            if submit:
                try:
                    out = predict_new(r, pd.DataFrame([values]))
                    row = out.iloc[0]
                    st.metric("預測值", f"{row.prediction:,.5g}")
                    st.write(f"95% 平均反應信賴區間：[{row.ci_lower:,.5g}, {row.ci_upper:,.5g}]")
                    st.write(f"95% 個別觀測預測區間：[{row.pi_lower:,.5g}, {row.pi_upper:,.5g}]")
                    st.caption("未對模型輸出截斷；線性模型可能產生超出合理範圍的機率或價格。超出訓練範圍屬外插，結果較不可靠。")
                except Exception as e:
                    st.error(str(e))
            st.divider()
            up = st.file_uploader("批次預測 CSV（需包含所有原始輸入欄位）", type="csv", key="predict_upload")
            if up:
                try:
                    new = read_csv_bytes(up.getvalue())
                    out = predict_new(r, new)
                    st.dataframe(out.head(100), width="stretch")
                    st.download_button("下載批次預測與區間", out.to_csv(index=False).encode("utf-8-sig"), "new_predictions.csv", "text/csv")
                except Exception as e:
                    st.error(str(e))
    with crisp_tab:
        stages = [
            ("1. Business Understanding｜問題理解", meta["business"]),
            ("2. Data Understanding｜資料理解", f"來源：{meta['source']}。目前 {len(df):,} 筆，選擇 {len(features)} 個原始輸入特徵，目標為 {target}。查看缺失值、類別與相關性；欄位編碼後數量另列。"),
            ("3. Data Preparation｜資料準備", "移除目標缺失與完全重複觀測；排除 ID、名稱等不適合欄位。先切分資料，補值、One-hot 編碼、標準化與 Group RFE 均在各訓練折內重新擬合。"),
            ("4. Modeling｜建模", "建立全特徵 OLS 與 Group RFE OLS；以五折訓練集 CV RMSE 比較候選特徵數，預先鎖定最終模型。以 scikit-learn Pipeline 處理資料，statsmodels OLS 計算區間。"),
            ("5. Evaluation｜評估", "在保留的測試集計算 MAE、MSE、RMSE、R²、PI 覆蓋率與寬度；比較平均值基準。以殘差圖、Q–Q、BP、JB 與 VIF 檢查假設並說明限制。"),
            ("6. Deployment｜部署", "Streamlit 網頁提供兩種問題下拉選單、CSV 檔案與網址、自訂欄位、新樣本輸入、批次預測與結果下載；README 提供 GitHub 與 Streamlit Community Cloud 部署步驟。"),
        ]
        for title, body in stages:
            st.subheader(title); st.write(body)
    with ai_tab:
        st.markdown((ROOT / "reports" / "AI_DISCLOSURE.md").read_text())
        st.markdown((ROOT / "reports" / "NOTEBOOKLM_GUIDE.md").read_text())
        st.markdown("**繳交清單**：GitHub 網址、可操作網頁網址、來源與欄位說明、模型比較、含區間的預測圖、CRISP-DM 六階段；另保留對話 PDF 與實際 NotebookLM 摘要。")
