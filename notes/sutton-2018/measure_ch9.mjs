// Actual browser arithmetic + raw moment matrices for independent NumPy solving.
import {writeFileSync,readFileSync} from 'node:fs';
import {walk,smUpdate,dot,interestRun} from '../../site/js/fa-core.js';
import {rng} from '../../site/js/labs/plot.js';
import {defaults,evaluate} from '../../site/js/labs/runtime.js';
import YAML from 'yaml';
const fixtures=[];
for(const seed of [1,7])for(const episodes of [200,1000]){
 const K=10,eps=.125,R=rng(seed),A=Array.from({length:K},(_,i)=>Array.from({length:K},(_,j)=>eps*+(i===j))),b=Array(K).fill(0);
 let B=Array.from({length:K},(_,i)=>Array.from({length:K},(_,j)=>+(i===j)/eps)),steps=0;
 for(let e=0;e<episodes;e++)for(const[s,reward,next]of walk(R)){
  const i=Math.floor((s-1)/100),j=next===null?-1:Math.floor((next-1)/100),x=Array.from({length:K},(_,k)=>+(k===i)),z=Array.from({length:K},(_,k)=>+(k===j));
  const u=smUpdate(B,x,z,1);if(!u.ok)throw Error('unexpected singular aggregate update');B=u.inv;b[i]+=reward;for(let k=0;k<K;k++)A[i][k]+=x[k]-z[k];steps++;
 }
 fixtures.push({seed,episodes,epsilon:eps,gamma:1,steps,A,b,inverse:B,w:B.map(r=>dot(r,b))});
}
const m=YAML.parse(readFileSync(new URL('../../content/modules/m11-6-dqn/module.yaml',import.meta.url),'utf8')),presets=[];
for(const L of m.labs.slice(2)){const spec=(await import(new URL(`../../site/js/labs/${L.lab}.js`,import.meta.url))).default;for(const p of L.presets){const{seed=1,...params}=p.params,result=evaluate(spec,{...defaults(spec),...params},seed);if(!result.out)throw Error(JSON.stringify(result.issues));presets.push({lab:L.lab,name:p.name,params:p.params,stats:result.out.stats});}}
const interest=[];for(const n of [1,2,4])for(const mode of [0,1,2]){const {w,M}=interestRun(n,mode,10000,.2);interest.push({n,mode,w,M});}
writeFileSync(new URL('./measurements_ch9_js.json',import.meta.url),JSON.stringify({LSTD_fixtures:fixtures,interest,presets},null,2)+'\n');
console.log(`${fixtures.length} LSTD matrices, ${interest.length} interest cases, ${presets.length} presets measured.`);
