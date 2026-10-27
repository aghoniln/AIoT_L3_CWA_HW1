import io
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from regression_core import (train_analysis, predict_new, prepare_xy,
                             read_csv_bytes, _check_public_url)


def sample(n=150):
    rng=np.random.default_rng(42)
    d=pd.DataFrame(rng.normal(size=(n,5)),columns=list("abcde"))
    d["kind"]=np.where(d.a>0,"gas","diesel")
    d["y"]=3*d.a-2*d.b+rng.normal(0,0.3,n)
    return d


def test_intervals_predictions_and_holdout():
    d=sample()
    r=train_analysis(d,"y",list(d.columns[:-1]),k_values=[3])
    p=r["predictions"]
    assert not set(r["train_indices"]) & set(r["test_indices"])
    assert (p.pi_lower <= p.ci_lower).all()
    assert (p.pi_upper >= p.ci_upper).all()
    assert (p.ci_lower <= p.prediction).all()
    assert (p.ci_upper >= p.prediction).all()
    out=predict_new(r,r["X_train"].head(2))
    assert len(out)==2 and np.isfinite(out.prediction).all()
    assert len(r["rfe_selected_features"])==3


def test_test_labels_do_not_influence_selection_or_preprocessing():
    d=sample()
    features=list(d.columns[:-1])
    r1=train_analysis(d,"y",features,k_values=[3,4])
    changed=d.copy()
    changed.loc[r1["test_indices"],"y"] += 10000
    r2=train_analysis(changed,"y",features,k_values=[3,4])
    assert r1["chosen_model"]==r2["chosen_model"]
    assert r1["rfe_selected_features"]==r2["rfe_selected_features"]
    np.testing.assert_allclose(r1["cv"].CV_RMSE,r2["cv"].CV_RMSE)
    np.testing.assert_allclose(r1["predictions"].prediction,r2["predictions"].prediction)


def test_missing_targets_duplicates_and_bad_input():
    d=sample()
    d.loc[0,"y"]=np.nan
    d=pd.concat([d,d.iloc[[1]]],ignore_index=True)
    X,y,info=prepare_xy(d,"y",list("abcde"))
    assert info=={"missing_target_rows":1,"duplicate_rows_removed":1}
    assert len(X)==149
    with pytest.raises(ValueError): prepare_xy(d,"y",["y"])
    with pytest.raises(ValueError): read_csv_bytes(b"<html>not csv</html>")
    for url in ["file:///etc/passwd","http://example.com/x.csv","https://127.0.0.1/a.csv"]:
        with pytest.raises(ValueError): _check_public_url(url)


def test_new_categories_and_missing_columns_fail_clearly():
    d=sample()
    r=train_analysis(d,"y",list(d.columns[:-1]),k_values=[3])
    with pytest.raises(ValueError, match="缺少"):
        predict_new(r,pd.DataFrame({"a":[1]}))
    x=r["X_train"].head(1).copy();x["kind"]="unknown"
    with pytest.raises(ValueError, match="未見"):
        predict_new(r,x)


def test_chronological_split_and_cv():
    d=sample(180)
    d["date"]=pd.date_range("2025-01-01",periods=len(d)).astype(str)
    r=train_analysis(d,"y",list("abcde"),k_values=[3],split_mode="time",order_column="date")
    assert r["train_indices"].max()<r["test_indices"].min()
    assert "Durbin_Watson" in r["diagnostics"]


def test_exact_dependencies_removed_and_reserved_name_preserved():
    d=sample(180)
    d['__target__']=2*d.a
    d['constant']=1
    r=train_analysis(d,'y',list('abcde')+['__target__','constant'],k_values=[4])
    diag=r['diagnostics']
    assert diag['design_rank']==diag['design_columns']
    assert '__target__' in r['features']
