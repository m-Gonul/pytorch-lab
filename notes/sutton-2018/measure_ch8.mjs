// Actual full-budget Dyna / changing maze / PS / RTDP experiments.
// node notes/sutton-2018/measure_ch8.mjs
// --verify writes only small independent-Python comparison fixtures.
import {writeFileSync} from 'node:fs';
import {fileURLToPath} from 'node:url';
import {meanRuns,dyna} from '../../site/js/labs/dynasource.js';
import {prioritized,rtdp} from '../../site/js/labs/sweepfocus.js';
const target=fileURLToPath(new URL('.',import.meta.url));
if(process.argv.includes('--verify')){
 const rows=[];for(const seed of [1,17])for(const n of [0,5,50])rows.push({seed,n,...dyna({seed,n,E:3})});
 writeFileSync(target+'measurements_ch8_dyna_fixtures.json',JSON.stringify(rows));
}else{
 const out={static:[],changing:[],priority:[],rtdp:rtdp({E:200,seed:1})};
 for(const n of [0,5,50])out.static.push({n,...meanRuns({n,E:50,seed:1,alpha:.1,eps:.1},30)});
 out.snapshots=[0,50].map(n=>({n,...dyna({n,E:2,seed:1,snapshot:true})}));
 for(const kind of [1,2])for(const variant of [0,1,2]){
  const o={kind,variant,T:kind===1?3000:6000,change:kind===1?1000:3000,n:50,alpha:1,eps:.1,kappa:.001,seed:1};out.changing.push({parameters:o,runs:30,...meanRuns(o,30)});
 }
 const measure=(scale,method,theta)=>{let results=[];for(let seed=1;seed<=5;seed++){const z=prioritized({scale,method,seed,n:5,alpha:.5,theta,maxSteps:2000000});delete z.Q;results.push(z);}return {scale,method,runs:5,results};};
 out.priority_threshold_experiment=[];
 for(let scale=0;scale<=7;scale++)out.priority.push(measure(scale,0,1e-4));
 for(let scale=0;scale<=7;scale++)out.priority_threshold_experiment.push(measure(scale,1,1e-4));
 for(let scale=0;scale<=7;scale++)out.priority.push({...measure(scale,1,1e-12),theta:1e-12});
 writeFileSync(target+'measurements_ch8_js.json',JSON.stringify(out));
}
