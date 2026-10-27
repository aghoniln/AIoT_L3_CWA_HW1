"""Export exact Python experiments for the browser companion (no fabricated scores)."""
import json
import shutil
import warnings
from pathlib import Path
import numpy as np
from scipy.stats import t
from threadpoolctl import threadpool_limits
from datasets import ROOT, DATASETS, FIELD_ZH
from regression_core import read_csv_bytes, train_analysis
from regression_ui import clean_json


def main(destination):
    dest=Path(destination);(dest/'data').mkdir(parents=True,exist_ok=True)
    for name,meta in DATASETS.items():
        df=read_csv_bytes((ROOT/'data'/meta['file']).read_bytes())
        features=[c for c in df if c!=meta['target'] and c not in meta['exclude']]
        with threadpool_limits(1),warnings.catch_warnings(record=True):
            warnings.simplefilter('always')
            r=train_analysis(df,meta['target'],features)
        p=r['pipeline'];prep=p.named_steps['prep'];schema=[]
        for kind,trans,cols in prep.transformers_:
            if kind=='num':
                for col,median in zip(cols,trans.named_steps['impute'].statistics_):
                    schema.append({'column':col,'kind':'number','median':float(median)})
            elif kind=='cat' and len(cols):
                for col,cats in zip(cols,trans.named_steps['encode'].categories_):
                    schema.append({'column':col,'kind':'category','categories':list(cats)})
        defaults={}
        for col in features:
            v=r['X_train'][col]
            defaults[col]=float(v.median()) if np.issubdtype(v.dtype,np.number) else v.mode().iloc[0]
        frame=r['predictions'].sort_values('prediction')
        frame=frame.iloc[np.linspace(0,len(frame)-1,min(200,len(frame)),dtype=int)]
        summary=json.loads((ROOT/'reports'/('flood' if '洪水' in name else 'cars')/'summary.json').read_text())
        payload={**summary,'schema':schema,'scaler_mean':p.named_steps['scale'].mean_.tolist(),
                 'scaler_scale':p.named_steps['scale'].scale_.tolist(),
                 'support':p.named_steps['independent'].support_.tolist(),
                 'beta':np.asarray(r['ols'].params).tolist(),
                 'covariance':np.asarray(r['ols'].cov_params()).tolist(),
                 'noise_variance':float(r['ols'].scale),'tcritical':float(t.ppf(.975,r['ols'].df_resid)),
                 'defaults':defaults,'display_predictions':frame.to_dict('records'),
                 'coefficients':r['coefficients'].to_dict('records'),'vif':r['vif'].to_dict('records'),
                 'preview':df.head(20).to_dict('records'),'description':meta['description'],
                 'business':meta['business'],'field_labels':FIELD_ZH,'implementation':'Python：scikit-learn + statsmodels'}
        slug='flood' if '洪水' in name else 'cars'
        (dest/'data'/f'{slug}.json').write_text(json.dumps(clean_json(payload),ensure_ascii=False),encoding='utf-8')
        shutil.copyfile(ROOT/'data'/meta['file'],dest/'data'/f'{slug}.csv')
        r['predictions'].to_csv(dest/'data'/f'{slug}_predictions.csv',index=False,encoding='utf-8-sig')
    shutil.copyfile(ROOT/'reports/RESULTS.md',dest/'data/RESULTS.md')


if __name__=='__main__':
    import sys
    main(sys.argv[1] if len(sys.argv)>1 else str(ROOT/'web'))
