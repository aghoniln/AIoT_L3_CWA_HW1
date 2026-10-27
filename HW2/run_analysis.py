"""Generate reproducible experiment outputs and a Chinese Markdown report."""
import json
import hashlib
import platform
import warnings
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from threadpoolctl import threadpool_limits

from datasets import ROOT, DATASETS
from regression_core import read_csv_bytes, train_analysis
from regression_ui import clean_json


def main():
    summaries = []
    report = ["# HW2 多元線性回歸實驗報告", "", "本文件由實際執行產生；種子 42，80/20 切分，訓練集五折交叉驗證。", ""]
    for name, meta in DATASETS.items():
        content = (ROOT / "data" / meta["file"]).read_bytes()
        df = read_csv_bytes(content)
        features = [c for c in df if c != meta["target"] and c not in meta["exclude"]]
        print(f"開始：{name}，{len(df)} 筆，{len(features)} 個原始特徵", flush=True)
        with threadpool_limits(1), warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            r = train_analysis(df, meta["target"], features,
                               progress=lambda v, s: print(s, flush=True))
        slug = "flood" if "洪水" in name else "cars"
        dest = ROOT / "reports" / slug
        dest.mkdir(exist_ok=True)
        for key in ["cv", "metrics", "predictions", "coefficients", "vif"]:
            r[key].to_csv(dest / f"{key}.csv", index=False, encoding="utf-8-sig")
        r["X_train"].head(5).to_csv(dest / "example_new_samples.csv", index=False, encoding="utf-8-sig")
        summary = clean_json({"problem": name, "source": meta["source"], "data_file": meta["file"],
                              "sha256": hashlib.sha256(content).hexdigest(), "rows": len(df),
                              "raw_features": features, "target": meta["target"],
                              "train_rows": r["train_rows"], "test_rows": r["test_rows"],
                              "seed": 42, "test_size": 0.2, "chosen_model": r["chosen_model"],
                              "cv": r["cv"].to_dict("records"), "metrics": r["metrics"].to_dict("records"),
                              "selected_features": r["selected_features"],
                              "rfe_selected_features": r["rfe_selected_features"],
                              "diagnostics": r["diagnostics"], "cleaning": r["cleaning"],
                              "mean_baseline": r["mean_baseline"]})
        (dest / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        p = r["predictions"].sort_values("prediction")
        q = p.iloc[np.linspace(0, len(p)-1, min(200,len(p)), dtype=int)]
        fig = go.Figure()
        x = list(range(len(q)))
        fig.add_trace(go.Scatter(x=x,y=q.pi_upper,mode="lines",line=dict(width=0),showlegend=False))
        fig.add_trace(go.Scatter(x=x,y=q.pi_lower,mode="lines",line=dict(width=0),fill="tonexty",name="95% 預測區間"))
        fig.add_trace(go.Scatter(x=x,y=q.prediction,mode="lines",name="預測值"))
        fig.add_trace(go.Scatter(x=x,y=q.actual,mode="markers",name="實際值"))
        fig.update_layout(title=name+"：測試集預測與區間",xaxis_title="依預測值排序的展示樣本（非時間）")
        fig.write_html(dest / "prediction_interval.html", include_plotlyjs=True)
        summaries.append(summary)
        report += [f"## {name}", "", f"來源：{meta['source']}", "", meta["description"], "",
                   "### Business Understanding", "", meta["business"], "",
                   "### Data Understanding", "", f"{len(df):,} 筆、{len(features)} 個有效原始輸入特徵；目標 {meta['target']}。缺失儲存格 {int(df.isna().sum().sum())}。", "",
                   "### Data Preparation", "", f"排除 {meta['exclude']}；先移除目標缺失與完全重複觀測，再切分為訓練 {r['train_rows']:,} 筆、測試 {r['test_rows']:,} 筆。補值、編碼、標準化及 RFE 在各訓練折內擬合。", "",
                   "### Modeling", "", "使用 OLS，比較全特徵與 Group RFE。類別變數的 dummy 欄位視為同一原始特徵群組。", "",
                   "| 模型 | 原始特徵數 | CV RMSE | CV RMSE 標準差 | CV R² |", "|---|---:|---:|---:|---:|"]
        for a in summary["cv"]:
            report.append(f"| {a['model']} | {a['raw_features']} | {a['CV_RMSE']:.6g} | {a['CV_RMSE_SD']:.6g} | {a['CV_R2']:.6g} |")
        report += ["", f"最終模型：**{r['chosen_model']}**，依訓練 CV RMSE 決定，未用測試集調參。", "",
                   "RFE 保留："+"、".join(r["rfe_selected_features"]), "",
                   "### Evaluation", "", "| 模型 | MAE | MSE | RMSE | R² | 訓練調整後 R² | 測試 PI 覆蓋率 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for a in summary["metrics"]:
            report.append(f"| {a['model']} | {a['MAE']:.6g} | {a['MSE']:.6g} | {a['RMSE']:.6g} | {a['R2']:.6g} | {a['train_adjusted_R2']:.6g} | {a['PI_coverage']:.2%} |")
        report += ["", f"平均值基準 RMSE：{r['mean_baseline']['RMSE']:.6g}。", "",
                   "模型診斷：`"+json.dumps(summary["diagnostics"],ensure_ascii=False)+"`", "",
                   "圖表：`"+slug+"/prediction_interval.html`；完整結果：`"+slug+"/predictions.csv`。", "",
                   "95% PI 描述個別新觀測的不確定性，CI 描述平均反應的不確定性。這些是條件於已選模型的傳統 OLS 區間，未納入特徵選擇不確定性。JB/BP 與殘差圖用於檢查假設，違反假設時名目覆蓋率可能不成立。", "",
                   "### Deployment", "", "以 Streamlit 提供資料預覽、特徵選擇、模型比較、圖表、新樣本及批次預測，並可下載結果。網頁實際部署網址須以成功部署後的網址為準。", ""]
    report += ["## 結論與限制", "", "特徵較少不保證比較好，應以訓練交叉驗證決定；測試集只負責最後評估。洪水資料是合成資料，表現不能外推到真實災害；汽車樣本僅 205 筆，且部分特徵共線、類別稀少。模型係數為關聯而非因果，外插時誤差可能增加。", "",
               (ROOT / "reports/AI_DISCLOSURE.md").read_text(), "", (ROOT / "reports/NOTEBOOKLM_GUIDE.md").read_text()]
    (ROOT / "reports/RESULTS.md").write_text("\n".join(report), encoding="utf-8")
    (ROOT / "reports/all_results.json").write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding="utf-8")
    env = {"python": platform.python_version()}
    for name in ["numpy","pandas","scikit-learn","statsmodels","scipy","streamlit","plotly"]:
        env[name] = version(name)
    (ROOT / "reports/environment.json").write_text(json.dumps(env,indent=2),encoding="utf-8")
    print("實驗與報告已完成。",flush=True)


if __name__ == "__main__":
    main()
