importScripts('vendor/matrix.js','vendor/jstat.js','engine.js');
onmessage=e=>{try{let d=e.data;let result=RegressionEngine.run(d.rows,d.target,d.excluded,d.k,message=>postMessage({progress:message}));postMessage({result});}catch(e){postMessage({error:e.message});}};
