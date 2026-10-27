"""Leakage-safe regression, raw-column group RFE, OLS intervals and diagnostics."""
from __future__ import annotations

import io
import ipaddress
import socket
from urllib.parse import urlparse, urljoin
import warnings

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from scipy import stats
from scipy.linalg import qr
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

MAX_BYTES = 20 * 1024 * 1024
MAX_ROWS = 100_000


def read_csv_bytes(content: bytes) -> pd.DataFrame:
    if len(content) > MAX_BYTES:
        raise ValueError("CSV 大於 20 MB，請先取樣或縮小檔案。")
    if not content.strip():
        raise ValueError("CSV 檔案是空的。")
    if content.lstrip().lower().startswith((b"<!doctype html", b"<html")):
        raise ValueError("網址回傳網頁，請改用直接下載 CSV 的網址。")
    for encoding in ("utf-8-sig", "cp950"):
        try:
            df = pd.read_csv(io.BytesIO(content), encoding=encoding, nrows=MAX_ROWS + 1)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise ValueError("無法辨識編碼，請另存成 UTF-8 CSV。")
    if len(df) > MAX_ROWS:
        raise ValueError("超過 100,000 筆，請先取樣後再上傳。")
    if df.empty or len(df.columns) < 2:
        raise ValueError("需要至少兩欄、且含資料列的 CSV。")
    df.columns = df.columns.astype(str).str.strip()
    if df.columns.duplicated().any():
        raise ValueError("欄位名稱不可重複。")
    return df.replace([np.inf, -np.inf], np.nan)


def _check_public_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("請使用不含帳號密碼的 HTTPS CSV 網址。")
    if parsed.port not in (None, 443):
        raise ValueError("只支援 HTTPS 標準連接埠。")
    # These public, provider-owned raw-content hosts also work behind proxies
    # that resolve upstream DNS remotely. Redirects are still validated again.
    if parsed.hostname.lower() in {"raw.githubusercontent.com", "gist.githubusercontent.com"}:
        return
    try:
        addresses = socket.getaddrinfo(parsed.hostname, 443, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ValueError("無法解析網址。") from e
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise ValueError("只允許公開網路的 CSV 網址。")


def fetch_csv(url: str) -> pd.DataFrame:
    """Bounded public HTTPS reads; revalidate every redirect, no local file URLs."""
    for _ in range(5):
        _check_public_url(url)
        with requests.get(url, timeout=(10, 30), stream=True, allow_redirects=False) as r:
            if r.status_code in (301, 302, 303, 307, 308):
                url = urljoin(url, r.headers.get("Location", ""))
                continue
            r.raise_for_status()
            data = bytearray()
            for chunk in r.iter_content(64 * 1024):
                data.extend(chunk)
                if len(data) > MAX_BYTES:
                    raise ValueError("CSV 超過 20 MB。")
            return read_csv_bytes(bytes(data))
    raise ValueError("網址轉址次數過多。")


def prepare_xy(df: pd.DataFrame, target: str, features: list[str]):
    if not features or target in features:
        raise ValueError("請選擇輸入特徵，且不可包含目標欄位。")
    if not set(features + [target]).issubset(df.columns):
        raise ValueError("選擇的欄位不存在。")
    if len(features) > 39:
        raise ValueError("此作業版本最多支援 39 個原始輸入特徵。")
    y = pd.to_numeric(df[target], errors="coerce").replace([np.inf, -np.inf], np.nan)
    keep = y.notna()
    # Exact duplicate observations must not appear on both sides of the split.
    selected = df.loc[keep, features].copy().replace([np.inf, -np.inf], np.nan)
    combined = pd.concat([selected.reset_index(drop=True),
                          y.loc[keep].reset_index(drop=True)], axis=1)
    unique = ~combined.duplicated()
    removed_duplicates = int((~unique).sum())
    X = selected.reset_index(drop=True).loc[unique].reset_index(drop=True)
    y = y.loc[keep].reset_index(drop=True).loc[unique].astype(float).reset_index(drop=True)
    for col in X.select_dtypes(exclude=np.number):
        X[col] = X[col].map(lambda v: str(v) if pd.notna(v) else np.nan)
        if X[col].nunique() > 60:
            raise ValueError(f"{col} 的類別超過 60 種，請排除名稱、ID 或自由文字欄位。")
    if len(X) < max(40, 3 * (len(features) + 1)):
        raise ValueError("有效資料不足：至少需 40 筆，且需超過 3 ×（原始特徵數 + 1）。")
    if y.nunique() < 3:
        raise ValueError("目標值種類太少，請選擇適合連續數值回歸的目標。")
    return X, y, {"missing_target_rows": int((~keep).sum()),
                  "duplicate_rows_removed": removed_duplicates}


def make_preprocessor():
    nums = Pipeline([("impute", SimpleImputer(strategy="median", keep_empty_features=True))])
    cats = Pipeline([("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                     ("encode", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))])
    return ColumnTransformer([
        ("num", nums, make_column_selector(dtype_include=np.number)),
        ("cat", cats, make_column_selector(dtype_exclude=np.number)),
    ], verbose_feature_names_out=True)


def encoded_groups(prep) -> list[str]:
    groups = []
    for name, transformer, cols in prep.transformers_:
        if name == "num":
            groups.extend(cols)
        elif name == "cat" and len(cols):
            encoder = transformer.named_steps["encode"]
            for col, categories in zip(cols, encoder.categories_):
                groups.extend([col] * max(0, len(categories) - 1))
    return groups


class GroupRFE(TransformerMixin, BaseEstimator):
    """Eliminate raw columns, grouping all dummy columns of a categorical feature.

    The preprocessing and coefficient ranking are refitted inside every CV fold.
    Importance is maximum absolute standardized coefficient within each group.
    """
    def __init__(self, n_features_to_select=14):
        self.n_features_to_select = n_features_to_select

    def fit(self, X, y):
        selected = list(X.columns)
        self.history_ = []
        while len(selected) > self.n_features_to_select:
            prep = make_preprocessor()
            z = prep.fit_transform(X[selected])
            z = StandardScaler().fit_transform(z)
            model = LinearRegression().fit(z, y)
            importance = dict.fromkeys(selected, 0.0)
            for col, coef in zip(encoded_groups(prep), np.abs(model.coef_)):
                importance[col] = max(importance[col], float(coef))
            removed = min(selected, key=lambda c: importance[c])
            self.history_.append({"removed": removed, "importance": importance[removed]})
            selected.remove(removed)
        self.selected_features_ = selected
        return self

    def transform(self, X):
        return X[self.selected_features_].copy()


def make_model(k=None):
    steps = []
    if k is not None:
        steps.append(("select", GroupRFE(k)))
    steps += [("prep", make_preprocessor()), ("scale", StandardScaler()),
              ("independent", IndependentColumns()),
              ("model", LinearRegression())]
    return Pipeline(steps)


def design_matrix(pipeline, X):
    if "select" in pipeline.named_steps:
        X = pipeline.named_steps["select"].transform(X)
    a = pipeline.named_steps["prep"].transform(X)
    a = pipeline.named_steps["scale"].transform(a)
    a = pipeline.named_steps["independent"].transform(a)
    return np.column_stack([np.ones(len(a)), a])


class IndependentColumns(TransformerMixin, BaseEstimator):
    """Training-only pivoted QR removes constants and exact linear dependencies."""
    def fit(self, X, y=None):
        _, r, pivot = qr(np.asarray(X), mode="economic", pivoting=True)
        diagonal = np.abs(np.diag(r))
        threshold = max(X.shape) * np.finfo(float).eps * (diagonal.max() if len(diagonal) else 0)
        rank = int((diagonal > max(threshold, 1e-10)).sum())
        if rank == 0:
            raise ValueError("沒有可用的非固定輸入欄位。")
        self.support_ = np.sort(pivot[:rank])
        self.removed_count_ = X.shape[1] - rank
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]


def interval_frame(ols, design, alpha=0.05):
    p = ols.get_prediction(design).summary_frame(alpha=alpha)
    return p.rename(columns={"mean": "prediction", "mean_ci_lower": "ci_lower",
                             "mean_ci_upper": "ci_upper", "obs_ci_lower": "pi_lower",
                             "obs_ci_upper": "pi_upper"})


def evaluate(y, pred):
    return {"MAE": float(mean_absolute_error(y, pred)),
            "MSE": float(mean_squared_error(y, pred)),
            "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "R2": float(r2_score(y, pred))}


def train_analysis(df, target, features, test_size=0.2, seed=42, k_values=None,
                   split_mode="random", order_column=None, progress=None):
    source = df.copy()
    if split_mode == "time":
        if not order_column or order_column not in df:
            raise ValueError("時間切分需要排序欄位。")
        order = pd.to_datetime(source[order_column], errors="coerce")
        if order.isna().any():
            raise ValueError("時間欄位有無法解析的值，請先清理。")
        source = source.assign(__sort=order).sort_values("__sort", kind="stable").drop(columns="__sort")
        if order_column in features:
            raise ValueError("日期排序欄位請勿直接當作輸入特徵。")
    X, y, cleaning = prepare_xy(source, target, features)
    tr, te = train_test_split(np.arange(len(X)), test_size=test_size, random_state=seed,
                             shuffle=split_mode == "random")
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
    empty = Xtr.columns[Xtr.isna().all()].tolist()
    if empty:
        raise ValueError("訓練資料中有全空欄位，請排除：" + ", ".join(empty))
    if split_mode == "time":
        from sklearn.model_selection import TimeSeriesSplit
        cv = TimeSeriesSplit(n_splits=5)
    else:
        cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    n = len(features)
    if k_values is None:
        k_values = sorted(set(k for k in [14, 16, 18, n - 2] if 1 <= k < n))
        if not k_values and n > 1:
            k_values = [max(1, n - 1)]
    k_values = sorted(set(int(k) for k in k_values if 1 <= int(k) < n))
    if not k_values:
        raise ValueError("特徵選擇至少需要兩個輸入特徵。")
    candidates = [("全特徵 OLS", None)] + [(f"Group RFE：{k} 特徵", k) for k in k_values]
    cv_rows = []
    for i, (label, k) in enumerate(candidates):
        if progress:
            progress(i / len(candidates), f"交叉驗證：{label}")
        scores = cross_validate(make_model(k), Xtr, ytr, cv=cv,
                                scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"},
                                n_jobs=1, error_score="raise")
        cv_rows.append({"model": label, "raw_features": n if k is None else k,
                        "CV_RMSE": float(-scores["test_rmse"].mean()),
                        "CV_RMSE_SD": float(scores["test_rmse"].std(ddof=1)),
                        "CV_R2": float(scores["test_r2"].mean()), "k": k})
    best_rfe = min(cv_rows[1:], key=lambda r: r["CV_RMSE"])
    # Lock final model from training CV, before inspecting test metrics.
    chosen = min([cv_rows[0], best_rfe], key=lambda r: r["CV_RMSE"])
    fitted, test_rows, predicted = {}, [], {}
    ols_models = {}
    for row in (cv_rows[0], best_rfe):
        k = None if row["model"] == "全特徵 OLS" else int(row["k"])
        pipeline = make_model(k).fit(Xtr, ytr)
        dtr, dte = design_matrix(pipeline, Xtr), design_matrix(pipeline, Xte)
        rank = np.linalg.matrix_rank(dtr)
        if len(Xtr) - rank < 5:
            raise ValueError("編碼後變數太多、殘差自由度不足。請減少特徵或類別數。")
        ols = sm.OLS(ytr.to_numpy(), dtr).fit()
        frame = interval_frame(ols, dte).reset_index(drop=True)
        frame.insert(0, "actual", yte.to_numpy())
        frame["residual"] = frame["actual"] - frame["prediction"]
        metrics = evaluate(yte, frame["prediction"])
        metrics.update({"model": row["model"], "raw_features": row["raw_features"],
                        "encoded_features": dtr.shape[1] - 1,
                        "train_adjusted_R2": float(ols.rsquared_adj),
                        "PI_coverage": float(((frame.actual >= frame.pi_lower) &
                                              (frame.actual <= frame.pi_upper)).mean()),
                        "PI_mean_width": float((frame.pi_upper - frame.pi_lower).mean())})
        test_rows.append(metrics)
        fitted[row["model"]] = pipeline
        ols_models[row["model"]] = ols
        predicted[row["model"]] = frame
    label = chosen["model"]
    pipeline, ols = fitted[label], ols_models[label]
    selected = (pipeline.named_steps["select"].selected_features_
                if "select" in pipeline.named_steps else list(features))
    rfe_label = best_rfe["model"]
    rfe_selected = fitted[rfe_label].named_steps["select"].selected_features_
    dtr = design_matrix(pipeline, Xtr)
    rank = int(np.linalg.matrix_rank(dtr))
    encoded_names = pipeline.named_steps["prep"].get_feature_names_out()
    names = ["Intercept"] + list(encoded_names[pipeline.named_steps["independent"].support_])
    coef = pd.DataFrame({"term": names, "coefficient": np.asarray(ols.params),
                         "p_value_exploratory": np.asarray(ols.pvalues)})
    jb = stats.jarque_bera(ols.resid)
    try:
        bp = het_breuschpagan(ols.resid, dtr)
        bp_p = float(bp[1])
    except (ValueError, AssertionError):
        bp_p = float("nan")
    vif = []
    for i in range(1, dtr.shape[1]):
        if np.std(dtr[:, i]) < 1e-12:
            v = np.inf
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                v = variance_inflation_factor(dtr, i)
        vif.append({"term": names[i], "VIF": float(v)})
    diagnostics = {"Jarque_Bera_p": float(jb.pvalue), "Breusch_Pagan_p": bp_p,
                   "design_rank": rank, "design_columns": dtr.shape[1],
                   "residual_df": int(ols.df_resid), "condition_number": float(ols.condition_number),
                   "dependent_encoded_columns_removed": pipeline.named_steps["independent"].removed_count_,
                   "independence": "須依資料取得方式判斷；隨機列順序無法證明獨立性。"}
    if split_mode == "time":
        from statsmodels.stats.stattools import durbin_watson
        diagnostics["Durbin_Watson"] = float(durbin_watson(ols.resid))
    mean_pred = np.repeat(ytr.mean(), len(yte))
    result = {"pipeline": pipeline, "ols": ols, "chosen_model": label,
              "selected_features": selected, "rfe_selected_features": rfe_selected,
              "rfe_history": fitted[rfe_label].named_steps["select"].history_,
              "cv": pd.DataFrame(cv_rows), "metrics": pd.DataFrame(test_rows),
              "predictions": predicted[label], "all_predictions": predicted,
              "coefficients": coef, "vif": pd.DataFrame(vif), "diagnostics": diagnostics,
              "X_train": Xtr, "y_train": ytr, "train_indices": tr, "test_indices": te,
              "target": target, "features": features, "train_rows": len(tr), "test_rows": len(te),
              "cleaning": cleaning, "seed": seed, "test_size": test_size,
              "split_mode": split_mode, "mean_baseline": evaluate(yte, mean_pred)}
    if progress:
        progress(1.0, "分析完成")
    return result


def predict_new(result, frame):
    missing = set(result["features"]) - set(frame.columns)
    if missing:
        raise ValueError("缺少輸入欄位：" + ", ".join(sorted(missing)))
    X = frame[result["features"]].copy().replace([np.inf, -np.inf], np.nan)
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(result["X_train"][col]):
            X[col] = pd.to_numeric(X[col], errors="coerce")
        else:
            X[col] = X[col].map(lambda v: str(v) if pd.notna(v) else np.nan)
            known = set(result["X_train"][col].dropna())
            unknown = set(X[col].dropna()) - known
            if unknown:
                raise ValueError(f"{col} 包含訓練時未見的類別：{sorted(unknown)[:5]}。請使用已知類別。")
    d = design_matrix(result["pipeline"], X)
    out = interval_frame(result["ols"], d).reset_index(drop=True)
    return pd.concat([frame.reset_index(drop=True), out], axis=1)
