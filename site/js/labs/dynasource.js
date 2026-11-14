// Sutton–Barto Ch8: source Dyna uses uniform seen S, then uniform tried A.
import {fmt, range, rng} from './plot.js';
export function maze(kind=0, changed=false, rs=1, cs=1) {
  const H=6*rs,W=9*cs,goal=W-1,start=kind===0?2*rs*W:5*W+3;
  const walls=new Set();
  for(let r=0;r<H;r++)for(let c=0;c<W;c++){
    const y=Math.floor(r/rs),x=Math.floor(c/cs);
    const blocked=kind===0?[[1,2],[2,2],[3,2],[4,5],[0,7],[1,7],[2,7]].some(([i,j])=>y===i&&x===j)
      :y===3&&(kind===1?(changed?x>=1:x<=7):(x>=1&&(changed?x<=7:true)));
    if(blocked)walls.add(r*W+c);
  }
  const step=(s,a)=>{if(s===goal)return [goal,0,true];let r=Math.floor(s/W)+[-1,1,0,0][a],c=s%W+[0,0,-1,1][a];let z=s;if(r>=0&&r<H&&c>=0&&c<W&&!walls.has(r*W+c))z=r*W+c;return[z,z===goal?1:0,z===goal];};
  const distance=new Array(H*W).fill(Infinity);distance[goal]=0;let queue=[goal];
  // Undirected grid: reverse BFS gives all finite shortest-path references.
  for(let h=0;h<queue.length;h++){let s=queue[h];for(let a=0;a<4;a++){const r=Math.floor(s/W)+[-1,1,0,0][a],c=s%W+[0,0,-1,1][a];if(r>=0&&r<H&&c>=0&&c<W&&!walls.has(r*W+c)){const z=r*W+c;if(distance[z]===Infinity){distance[z]=distance[s]+1;queue.push(z);}}}}
  return {H,W,start,goal,walls,step,distance,states:H*W-walls.size,shortest:distance[start]};
}
export function greedy(Q,s,random,bonus=null){let best=-Infinity,choices=[];for(let a=0;a<4;a++){const v=Q[4*s+a]+(bonus?bonus(a):0);if(v>best){best=v;choices=[a];}else if(v===best)choices.push(a);}return choices[Math.floor(random()*choices.length)];}
const maxQ=(Q,s)=>Math.max(Q[4*s],Q[4*s+1],Q[4*s+2],Q[4*s+3]);
export function greedyLength(Q,m){let s=m.start;const seen=new Set();for(let k=1;k<=m.H*m.W;k++){if(seen.has(s))return Infinity;seen.add(s);let a=0;for(let j=1;j<4;j++)if(Q[4*s+j]>Q[4*s+a])a=j;let[z,,done]=m.step(s,a);s=z;if(done)return k;}return Infinity;}
export function dyna({n=5,alpha=.1,eps=.1,E=50,kind=0,variant=0,kappa=.001,seed=1,T=6000,change=3000,snapshot=false}={}){
  let m=maze(kind),Q=new Float64Array(m.H*m.W*4),model=new Map(),actions=new Map(),states=[],last=new Map();
  const behaviour=rng(seed),planning=rng(seed+100000),lengths=[],cum=[],traces=[];let s=m.start,total=0,episodeSteps=0,steps=0,updates=0,failures=0;
  const put=(s,a,v)=>{if(!actions.has(s)){actions.set(s,[]);states.push(s);}if(!model.has(4*s+a))actions.get(s).push(a);model.set(4*s+a,v);};
  const update=(s,a,r,z,done)=>{Q[4*s+a]+=alpha*(r+(done?0:.95*maxQ(Q,z))-Q[4*s+a]);updates++;};
  // Static episodes are complete; 100000 is an explicitly reported failure guard.
  const limit=kind===0?100000*E:T;
  while(steps<limit&&(kind!==0||lengths.length<E)){
    if(kind&&steps===change)m=maze(kind,true);
    const time=steps+1;let a;
    if(variant===2)a=greedy(Q,s,behaviour,a=>kappa*Math.sqrt(time-(last.get(4*s+a)??0)));
    else a=behaviour()<eps?Math.floor(behaviour()*4):greedy(Q,s,behaviour);
    let[z,reward,done]=m.step(s,a);update(s,a,reward,z,done);
    if(variant===1&&!actions.has(s))for(let j=0;j<4;j++)put(s,j,[s,0,false]);
    put(s,a,[z,reward,done]);last.set(4*s+a,time);
    for(let j=0;j<n;j++){
      const ps=states[Math.floor(planning()*states.length)],aa=actions.get(ps),pa=aa[Math.floor(planning()*aa.length)];
      let[pz,pr,pd]=model.get(4*ps+pa);if(variant===1)pr+=kappa*Math.sqrt(time-(last.get(4*ps+pa)??0));update(ps,pa,pr,pz,pd);
    }
    steps++;episodeSteps++;total+=reward;if(kind)cum.push(total);
    if(snapshot&&lengths.length===1)traces.push({s,Q:Array.from(Q)});
    s=done?m.start:z;
    if(done||episodeSteps===100000){lengths.push(episodeSteps);if(!done)failures++;episodeSteps=0;s=m.start;}
  }
  return {lengths,cum,Q:Array.from(Q),steps,updates,failures,trace:snapshot?traces[Math.floor(traces.length/2)]:null,greedy:greedyLength(Q,m)};
}
export function meanRuns(options,R=30){const runs=range(R).map(i=>dyna({...options,seed:(options.seed??1)+i}));const key=options.kind?'cum':'lengths',size=runs[0][key].length;
  const mean=range(size).map(i=>runs.reduce((v,r)=>v+r[key][i]/R,0)),se=range(size).map(i=>R>1?Math.sqrt(runs.reduce((v,r)=>v+(r[key][i]-mean[i])**2,0)/(R*(R-1))):0);
  return {mean,se,steps:runs.reduce((s,r)=>s+r.steps,0),updates:runs.reduce((s,r)=>s+r.updates,0),failures:runs.reduce((s,r)=>s+r.failures,0),first:runs.map(r=>r.lengths[0]),lastGreedy:runs.map(r=>Number.isFinite(r.greedy)?r.greedy:null)};
}
export default {
 params:[{id:'mode',label:'Ortam',options:['Statik: Şekil8.2','Engel: Şekil8.4','Kısayol: Şekil8.5'],value:0},{id:'n',label:'Planlama adımı',min:0,max:50,step:5,value:5,int:true},{id:'alpha',label:'α',min:.05,max:1,step:.05,value:.1},{id:'eps',label:'ε',min:0,max:.3,step:.01,value:.1},{id:'R',label:'Bağımsız koşu',min:1,max:30,value:5,int:true},{id:'kappa',label:'κ',min:0,max:.01,step:.001,value:.001,on:{mode:[1,2]}}],seed:true,
 run(q,random){const seed=1+Math.floor(random()*1000000),kind=q.mode;let series=[],statistics=[];
  const variants=kind?[0,1,2]:[0,5,50];
  for(let i=0;i<variants.length;i++){
   const variant=variants[i],o={n:kind?q.n:variant,alpha:q.alpha,eps:q.eps,kind,variant:kind?variant:0,kappa:q.kappa,seed,E:50,T:kind===1?3000:6000,change:kind===1?1000:3000};
   const r=meanRuns(o,q.R),x=range(r.mean.length).filter(j=>!kind||j%50===0||j===r.mean.length-1);
   const name=kind?['Dyna-Q','Dyna-Q+','Yalnız seçim bonusu'][i]:`n = ${variant}`;
   series.push({name,x:x.map(j=>j+1),y:x.map(j=>r.mean[j]),c:i});statistics.push({label:kind?`${name}: değişim sonrası ödül`:`${name}: bölüm1 / 2 / 50`,value:kind?fmt(r.mean.at(-1)-r.mean[o.change-1],4):[r.mean[0],r.mean[1],r.mean[49]].map(v=>fmt(v,4)).join(' / ')});
   statistics.push({label:`${name}: gerçek adım / Q güncellemesi`,value:`${r.steps} / ${r.updates}`});if(!kind&&variant===q.n)statistics.push({label:'Seçili n: ikinci bölüm ortalaması',value:`n=${q.n}: ${fmt(r.mean[1],4)}`});if(r.failures)statistics.push({label:'100000 adımda kesilen bölüm',value:r.failures});
  }
  // Selected n appears as a fourth curve only if different from the source trio.
  if(!kind&&![0,5,50].includes(q.n)){const r=meanRuns({n:q.n,alpha:q.alpha,eps:q.eps,E:50,seed},q.R);series.push({name:`seçili n=${q.n}`,x:range(50).map(i=>i+1),y:r.mean,c:3});}
  return {plots:[{title:kind?'Gerçek ödül: bonus ödüle eklenmez':'Tam bölümler; eşlenmiş davranış tohumları',xlabel:kind?'gerçek etkileşim':'bölüm',ylabel:kind?'birikimli gerçek ödül':'adım',wide:true,ylog:!kind,marks:kind?[{x:kind===1?1000:3000,label:'ortam değişir'}]:[],series}],stats:statistics,note:`${q.R} koşu; α=${q.alpha}, ε=${q.eps}. Statik deney 50 tam bölüm, γ=.95, Q=0; 0/5/50 planlama kaynak ayarlarıdır. Önce görülen durum, sonra o durumdaki denenmiş eylem eşit olasılıkla seçilir. Davranış ve planlama üreteçleri ayrıdır; ilk bölüm aynı tohumda aynıdır. Dyna-Q+ denenmemiş eylemleri de modele alır ve yalnız planlama ödülüne κ√τ ekler. Alıştırma8.4'ün üçüncü yöntemi ε kullanmadan Q+κ√τ ile davranır; güncellemesi bonussuzdur. Değişen ortamın n, κ ve koşu sayıları kendi açık seçimimizdir; sonuçlar bu koşulara aittir.`};
 }
};
