// Actual JS runs plus independent NumPy model comparisons.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import YAML from 'yaml';
import {rng} from '../../site/js/labs/plot.js';
import {defaults,evaluate} from '../../site/js/labs/runtime.js';
import mctd from '../../site/js/labs/mctd.js';
import batch from '../../site/js/labs/batchmctd.js';
import cliff from '../../site/js/labs/cliffwalk.js';
import bias from '../../site/js/labs/maxbias.js';
import {windModel,trainWind,greedyTrace,evaluateWind} from '../../site/js/labs/windytd.js';
const here=path.dirname(fileURLToPath(import.meta.url)),root=path.resolve(here,'../..'),P=JSON.parse(fs.readFileSync(process.argv[2]||path.join(here,'measurements_ch6_python.json'))),out={presets:{},wind:{},numpy_js_maxdiff:{}};
for(const [module,id,spec] of [['m11-4-mc-td','l1',mctd],['m11-4-mc-td','l2',batch],['m11-5-sarsa-q','l1',cliff],['m11-5-sarsa-q','l2',bias]]){
 const meta=YAML.parse(fs.readFileSync(path.join(root,'content/modules',module,'module.yaml'),'utf8')),lab=meta.labs.find(x=>x.id===id);out.presets[`${module}/${id}`]=[];
 for(const preset of lab.presets){const {seed=1,...params}=preset.params||{},z=evaluate(spec,{...defaults(spec),...params},seed);if(!z.out)throw Error(JSON.stringify(z.issues));out.presets[`${module}/${id}`].push({name:preset.name,params,seed,stats:z.out.stats,note:z.out.note,plots:z.out.plots});}
}
for(const actions of[4,8,9])for(const stochastic of[false,true]){const key=`${actions}/${Number(stochastic)}`,m=windModel(actions,stochastic),p=P.wind[key],error=Math.max(...m.V.map((v,i)=>Math.abs(v-p.V[i])));if(error>1e-9)throw Error('wind model mismatch');out.numpy_js_maxdiff[key]=error;out.wind[key]={model:m};}
for(const q of[{steps:8000,actions:4,stochastic:false,alpha:.5,epsilon:.1,seed:6},{steps:8000,actions:8,stochastic:false,alpha:.5,epsilon:.1,seed:1},{steps:8000,actions:9,stochastic:false,alpha:.5,epsilon:.1,seed:1},{steps:100000,actions:8,stochastic:true,alpha:.1,epsilon:.1,seed:1},{steps:8000,actions:4,stochastic:false,alpha:.5,epsilon:.1,seed:1}]){const d=trainWind(q.steps,q.actions,q.stochastic,q.alpha,q.epsilon,q.seed),trace=greedyTrace(d.policy,1,q.stochastic),assessment=evaluateWind(d.policy,1000,1,q.stochastic);out.wind[JSON.stringify(q)]={params:q,training:d,trace,assessment};}
fs.writeFileSync(process.argv[3]||path.join(here,'measurements_ch6_js.json'),JSON.stringify(out,null,2)+'\n');console.log('wind maximum differences',out.numpy_js_maxdiff);console.log('source max-bias',out.presets['m11-5-sarsa-q/l2'].at(-1).stats);
