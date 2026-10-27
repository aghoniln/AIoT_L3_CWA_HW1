/* Browser-only OLS. SVD computes coefficients and covariance without normal equations. */
(function(root){
'use strict';
const mean=a=>a.reduce((s,v)=>s+v,0)/a.length;
const missing=v=>v==null||String(v).trim()==='';
const number=v=>missing(v)?NaN:Number(v);
const median=a=>{a=a.slice().sort((x,y)=>x-y);let n=a.length;return n%2?a[n>>1]:(a[n/2-1]+a[n/2])/2;};
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
function rng(seed=42){return()=>{seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
function shuffle(a,seed=42){let r=rng(seed),b=a.slice();for(let i=b.length-1;i>0;i--){let j=Math.floor(r()*(i+1));[b[i],b[j]]=[b[j],b[i]];}return b;}
function makeSchema(rows,features){return features.map(column=>{let vals=rows.map(r=>r[column]).filter(v=>!missing(v));if(!vals.length)throw Error(column+' 在訓練資料中全空，請排除此欄。');
 if(vals.every(v=>Number.isFinite(Number(v))))return{column,kind:'number',median:median(vals.map(Number))};
 let categories=[...new Set(vals.map(String))];if(rows.some(r=>missing(r[column])))categories.push('Missing');categories=[...new Set(categories)].sort();if(categories.length>40)throw Error(column+' 類別過多，請排除 ID 或名稱欄。');return{column,kind:'category',categories};});}
function encode(row,schema){let values=[];for(let s of schema){let v=row[s.column];if(s.kind==='number')values.push(Number.isFinite(number(v))?Number(v):s.median);else{v=missing(v)?'Missing':String(v);for(let c of s.categories.slice(1))values.push(v===c?1:0);}}return values;}
function names(schema){return schema.flatMap(s=>s.kind==='number'?[s.column]:s.categories.slice(1).map(c=>s.column+'='+c));}
function groups(schema){return schema.flatMap(s=>s.kind==='number'?[s.column]:s.categories.slice(1).map(()=>s.column));}
function fit(rows,features,target){
 const schema=makeSchema(rows,features),encoded=rows.map(r=>encode(r,schema)),p=encoded[0].length;
 if(p>160)throw Error('類別編碼後超過 160 欄，請減少類別或特徵。');
 let means=Array(p).fill(0),scales=Array(p).fill(0);for(let a of encoded)for(let j=0;j<p;j++)means[j]+=a[j]/rows.length;
 for(let a of encoded)for(let j=0;j<p;j++)scales[j]+=(a[j]-means[j])**2/rows.length;
 scales=scales.map(v=>Math.sqrt(v)||1);
 let D=encoded.map(a=>[1,...a.map((v,j)=>(v-means[j])/scales[j])]),y=rows.map(r=>Number(r[target]));
 let svd=new root.mlMatrix.SingularValueDecomposition(new root.mlMatrix.Matrix(D),{autoTranspose:true});
 let beta=svd.solve(root.mlMatrix.Matrix.columnVector(y)).to1DArray();
 let singular=svd.diagonal,V=svd.rightSingularVectors,eps=Math.max(rows.length,p+1)*Number.EPSILON*singular[0];
 let active=singular.map(v=>v>Math.max(eps,1e-10)),rank=active.filter(Boolean).length,df=rows.length-rank;
 if(df<5)throw Error('殘差自由度不足，請增加樣本或減少特徵。');
 let resid=D.map((a,i)=>y[i]-dot(a,beta)),sse=resid.reduce((s,v)=>s+v*v,0),variance=sse/df;
 let covariance=Array.from({length:p+1},()=>Array(p+1).fill(0));
 for(let i=0;i<=p;i++)for(let j=0;j<=p;j++)for(let k=0;k<singular.length;k++)if(active[k])covariance[i][j]+=variance*V.get(i,k)*V.get(j,k)/(singular[k]**2);
 const tcritical=root.jStat.studentt.inv(.975,df),avg=mean(y),sst=y.reduce((s,v)=>s+(v-avg)**2,0);
 return{schema,scaler_mean:means,scaler_scale:scales,support:Array.from({length:p},(_,i)=>i),beta,covariance,noise_variance:variance,tcritical,features,df,rank,trainR2:1-sse/sst,trainAdjR2:1-(sse/df)/(sst/(rows.length-1)),groups:groups(schema),names:['Intercept',...names(schema)]};
}
function predict(model,row,strict=false){if(strict)for(let col of (model.raw_features||model.features||[]))if(!(col in row))throw Error('缺少原始輸入欄位：'+col);if(strict)for(let s of model.schema){if(!(s.column in row))throw Error('缺少欄位：'+s.column);if(s.kind==='category'&&!missing(row[s.column])&&!s.categories.includes(String(row[s.column])))throw Error(s.column+' 含訓練未見類別。');}
 let a=encode(row,model.schema).map((v,j)=>(v-model.scaler_mean[j])/model.scaler_scale[j]);let d=[1,...model.support.map(i=>a[i])];
 let prediction=dot(d,model.beta),v=0;for(let i=0;i<d.length;i++)for(let j=0;j<d.length;j++)v+=d[i]*model.covariance[i][j]*d[j];v=Math.max(0,v);
 let ci=model.tcritical*Math.sqrt(v),pi=model.tcritical*Math.sqrt(v+model.noise_variance);
 return{prediction,ci_lower:prediction-ci,ci_upper:prediction+ci,pi_lower:prediction-pi,pi_upper:prediction+pi};}
function rfe(rows,features,target,k){let selected=features.slice();while(selected.length>k){let m=fit(rows,selected,target),scores=Object.fromEntries(selected.map(c=>[c,0]));m.groups.forEach((g,i)=>scores[g]=Math.max(scores[g],Math.abs(m.beta[i+1])));let remove=selected.reduce((a,c)=>scores[c]<scores[a]?c:a,selected[0]);selected=selected.filter(c=>c!==remove);}return selected;}
function metrics(rows,target,model){let p=rows.map(r=>({...predict(model,r),actual:Number(r[target])}));let avg=mean(p.map(r=>r.actual)),mae=0,mse=0,sst=0,coverage=0,width=0;for(let r of p){r.residual=r.actual-r.prediction;mae+=Math.abs(r.residual);mse+=r.residual**2;sst+=(r.actual-avg)**2;coverage+=(r.actual>=r.pi_lower&&r.actual<=r.pi_upper)?1:0;width+=r.pi_upper-r.pi_lower;}return{metrics:{MAE:mae/p.length,MSE:mse/p.length,RMSE:Math.sqrt(mse/p.length),R2:1-mse/sst,train_adjusted_R2:model.trainAdjR2,PI_coverage:coverage/p.length,PI_mean_width:width/p.length},predictions:p};}
function run(data,target,excluded,k,onprogress=()=>{}){
 let features=Object.keys(data[0]).filter(c=>c!==target&&!excluded.includes(c));if(features.length<2||features.length>39)throw Error('請保留 2～39 個輸入特徵；主要作業需 14～39 個。');
 let seen=new Set();data=data.filter(r=>{if(!Number.isFinite(number(r[target])))return false;let key=JSON.stringify([...features.map(c=>r[c]),r[target]]);if(seen.has(key))return false;seen.add(key);return true;});
 let original=data.length;data=shuffle(data,42).slice(0,3000);if(data.length<Math.max(50,3*(features.length+1)))throw Error('有效樣本不足，請增加資料或減少特徵。');
 if(new Set(data.map(r=>number(r[target]))).size<3)throw Error('目標值種類不足，請選擇連續數值目標。');
 let ntest=Math.ceil(data.length*.2),test=data.slice(0,ntest),train=data.slice(ntest);k=Math.min(features.length-1,Math.max(1,Math.floor(k)));
 let candidates=[{model:'全特徵 OLS',k:null},{model:'Group RFE：'+k+' 特徵',k}],cv=[];
 for(let candidate of candidates){let scores=[];for(let fold=0;fold<5;fold++){onprogress(candidate.model+'：交叉驗證 '+(fold+1)+'/5');let tr=train.filter((_,i)=>i%5!==fold),va=train.filter((_,i)=>i%5===fold);let cols=candidate.k?rfe(tr,features,target,candidate.k):features;let m=fit(tr,cols,target);scores.push(metrics(va,target,m).metrics);}
 let rmses=scores.map(s=>s.RMSE),avg=mean(rmses);cv.push({model:candidate.model,k:candidate.k,raw_features:candidate.k||features.length,CV_RMSE:avg,CV_RMSE_SD:Math.sqrt(rmses.reduce((s,v)=>s+(v-avg)**2,0)/4),CV_R2:mean(scores.map(s=>s.R2))});}
 let chosen=cv.slice().sort((a,b)=>a.CV_RMSE-b.CV_RMSE)[0],testmetrics=[],chosenFit,chosenPred,selected;
 let rfeCols=rfe(train,features,target,k);
 for(let c of candidates){let cols=c.k?rfeCols:features,m=fit(train,cols,target),e=metrics(test,target,m);testmetrics.push({...e.metrics,model:c.model,raw_features:cols.length,encoded_features:m.beta.length-1});if(c.model===chosen.model){chosenFit=m;chosenPred=e.predictions;selected=cols;}}
 let defaults={};for(let c of features){let vals=train.map(r=>r[c]).filter(v=>!missing(v));if(vals.every(v=>Number.isFinite(Number(v))))defaults[c]=median(vals.map(Number));else defaults[c]=vals[0];}
 let sorted=chosenPred.slice().sort((a,b)=>a.prediction-b.prediction),display=sorted.filter((_,i)=>i%Math.max(1,Math.ceil(sorted.length/200))===0).slice(0,200);
 return{...chosenFit,problem:'自訂 CSV 回歸',source:'使用者提供的 CSV',description:`清理後 ${original.toLocaleString()} 筆，固定種子抽樣最多 3,000 筆進行瀏覽器分析。80/20 隨機切分；完整資料可使用 Python 版。`,business:'由使用者定義預測目標與使用情境。',rows:data.length,raw_features:features,target,train_rows:train.length,test_rows:test.length,chosen_model:chosen.model,cv,metrics:testmetrics,selected_features:selected,rfe_selected_features:rfeCols,defaults,display_predictions:display,all_predictions:chosenPred,preview:data.slice(0,20),coefficients:chosenFit.names.map((term,i)=>({term,coefficient:chosenFit.beta[i]})),vif:[],diagnostics:{implementation:'JavaScript / ml-matrix SVD',design_rank:chosenFit.rank,design_columns:chosenFit.beta.length,residual_df:chosenFit.df,independence:'須依資料取得方式判斷；未提供 BP/JB/VIF，完整診斷請用 Python 版。'},implementation:'瀏覽器：ml-matrix + jStat'};
}
root.RegressionEngine={run,predict,fit,metrics,shuffle};
})(typeof globalThis!=='undefined'?globalThis:this);
