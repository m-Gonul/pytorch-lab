import {fmt,range,rng} from './plot.js';
import {maze,greedy,greedyLength} from './dynasource.js';
class Queue{
 constructor(){this.heap=[];this.active=new Map();}
 put(k,p){if((this.active.get(k)??-1)>=p)return;this.active.set(k,p);const h=this.heap;let i=h.length;h.push([k,p]);while(i){let j=(i-1)>>1;if(h[j][1]>=p)break;h[i]=h[j];i=j;}h[i]=[k,p];}
 pop(){const h=this.heap;while(h.length){const out=h[0],last=h.pop();if(h.length){let i=0;while(2*i+1<h.length){let j=2*i+1;if(j+1<h.length&&h[j+1][1]>h[j][1])j++;if(h[j][1]<=last[1])break;h[i]=h[j];i=j;}h[i]=last;}if(this.active.get(out[0])===out[1]){this.active.delete(out[0]);return out[0];}}return null;}
}
export function prioritized({scale=0,n=5,alpha=.5,eps=.1,theta=1e-4,seed=1,maxSteps=2000000,method=1}={}){
 const rs=2**Math.ceil(scale/2),cs=2**Math.floor(scale/2),m=maze(0,false,rs,cs),Q=new Float64Array(4*m.H*m.W),model=new Map(),pred=new Map(),queue=new Queue(),random=rng(seed),plan=rng(seed+100000),actions=new Map(),states=[];
 const max=s=>Math.max(...Q.subarray(4*s,4*s+4)),target=([z,r,done])=>r+(done?0:.95*max(z));let updates=0,s=m.start,real=0,found=false;
 for(;real<maxSteps;){const a=random()<eps?Math.floor(random()*4):greedy(Q,s,random),k=4*s+a,[z,r,done]=m.step(s,a);real++;
  if(!model.has(k)){if(!actions.has(s)){states.push(s);actions.set(s,[]);}actions.get(s).push(a);}
  model.set(k,[z,r,done]);if(!pred.has(z))pred.set(z,new Set());pred.get(z).add(k);
  if(method){const p=Math.abs(target(model.get(k))-Q[k]);if(p>theta)queue.put(k,p);}
  else{Q[k]+=alpha*(target(model.get(k))-Q[k]);updates++;}
  for(let j=0;j<n;j++){
   let pk;
   if(method){pk=queue.pop();if(pk===null)break;}
   else{const ps=states[Math.floor(plan()*states.length)],aa=actions.get(ps);pk=4*ps+aa[Math.floor(plan()*aa.length)];}
   Q[pk]+=alpha*(target(model.get(pk))-Q[pk]);updates++;
   if(method)for(const bk of pred.get(Math.floor(pk/4))??[]){const p=Math.abs(target(model.get(bk))-Q[bk]);if(p>theta)queue.put(bk,p);}
  }
  // A fixed-first tie greedy policy is checked at every real interaction.
  if(greedyLength(Q,m)===m.shortest){found=true;break;}s=done?m.start:z;
 }
 return {states:m.states,shortest:m.shortest,updates,real,found,Q:Array.from(Q)};
}
export function rtdp({E=100,seed=1}={}){
 const m=maze(),random=rng(seed),V=new Float64Array(m.H*m.W),counts=new Uint32Array(V.length),history=[],lengths=[];
 const target=(s,a)=>{const[z]=m.step(s,a);return -1+(z===m.goal?0:V[z]);};
 for(let ep=0;ep<E;ep++){let s=m.start,k=0;while(s!==m.goal&&k<100000){const qs=range(4).map(a=>target(s,a)),best=Math.max(...qs),aa=range(4).filter(a=>qs[a]===best),a=aa[Math.floor(random()*aa.length)];V[s]=best;counts[s]++;s=m.step(s,a)[0];k++;}lengths.push(k);history.push(-V[m.start]);}
 const relevant=range(V.length).filter(s=>!m.walls.has(s)&&Number.isFinite(m.distance[s])&&m.distance[s]+distanceFromStart(m,s)===m.shortest);
 return {V:Array.from(V),counts:Array.from(counts),history,lengths,relevant,updates:Array.from(counts).reduce((a,b)=>a+b,0),shortest:m.shortest};
}
function distanceFromStart(m,goal){const d=new Array(m.H*m.W).fill(Infinity),q=[m.start];d[m.start]=0;for(let i=0;i<q.length;i++){const s=q[i];if(s===goal)return d[s];for(let a=0;a<4;a++){const[z]=m.step(s,a);if(d[z]===Infinity){d[z]=d[s]+1;q.push(z);}}}return Infinity;}
export default{
 params:[{id:'mode',label:'Güncelleme dağılımı',options:['Öncelikli süpürme / Dyna','RTDP: iyimser en kısa yol'],value:0},{id:'scale',label:'Izgara ölçeği (47×2^k)',min:0,max:4,value:1,int:true,on:{mode:[0]}},{id:'n',label:'En çok kuyruk yedeği n',min:1,max:20,value:5,int:true,on:{mode:[0]}},{id:'alpha',label:'α',min:.1,max:1,step:.1,value:.5,on:{mode:[0]}},{id:'E',label:'RTDP bölüm sayısı',min:10,max:200,value:100,int:true,on:{mode:[1]}}],seed:true,
 run(q,random){const seed=1+Math.floor(random()*1000000);
  if(q.mode){const r=rtdp({E:q.E,seed});return {plots:[{title:'RTDP: yalnız ziyaret edilen durumları güncelle',xlabel:'bölüm',ylabel:'adım / başlangıç maliyeti',wide:true,series:[{name:'gerçek bölüm uzunluğu',x:range(q.E).map(i=>i+1),y:r.lengths,c:0},{name:'−V(S)',x:range(q.E).map(i=>i+1),y:r.history,c:1},{name:'BFS en kısa14',x:[1,q.E],y:[r.shortest,r.shortest],c:2,dash:true}]}],stats:[{label:'RTDP güncellemesi',value:r.updates},{label:'Hiç güncellenmeyen geçilebilir durum',value:r.counts.filter((c,s)=>!maze().walls.has(s)&&c===0).length},{label:'Bazı optimal yollardaki durum sayısı',value:r.relevant.length}],note:'Kendi 47 durumlu deterministik örneğimiz: γ=1, her hareket −1, hedef0, V başlangıç0; iyimser üst sınır. Önce açgözlü eylem (rastgele eşitlik), sonra ziyaret edilen durumun beklenen max yedeği. Hedefin sayacı0 kalır. Bu grid, kaynağın haritası tam verilmemiş 9115 durumlu yarış pistini yeniden üretme iddiası taşımaz. Sonlu bölüm sayısı tek başına yakınsama sertifikası değildir.'};}
  const runs=[0,1].map(method=>prioritized({scale:q.scale,n:q.n,alpha:q.alpha,seed,method,maxSteps:1000000}));
  return {plots:[{title:'Optimal başlangıç yoluna kadar gerçek Q güncellemeleri',xlabel:'yöntem',ylabel:'güncelleme',wide:true,xlim:[-.6,1.6],xticks:[[0,'Dyna-Q'],[1,'PS']],series:[{name:'Dyna-Q',x:[0,1],y:[runs[0].updates,null],kind:'bar',bw:.25,c:0},{name:'Öncelikli süpürme',x:[0,1],y:[null,runs[1].updates],kind:'bar',bw:.25,c:1}]}],stats:runs.flatMap((r,i)=>[{label:`${i?'PS':'Dyna'}: gerçek adım / güncelleme`,value:`${r.real} / ${r.updates}`},{label:`${i?'PS':'Dyna'}: optimal yol bulundu`,value:r.found?'evet':'hayır; sınırda kesildi'}]).concat([{label:'Durum / en kısa adım',value:`${runs[0].states} / ${runs[0].shortest}`}]),note:`${runs[0].states} geçilebilir durum; n=${q.n}, α=${q.alpha}. Öncelikli süpürme gerçek geçişte modeli ve kuyruğu günceller; Q'ya ayrıca doğrudan yedek yapmaz. Her gerçek adımda en çok n kuyruk yedeği; yinelenen çiftin büyük önceliği korunur. Dyna'nın doğrudan yedeği de maliyete dahildir. Durdurma: ilk eylemi eşitlikte seçen başlangıç politikası BFS kadar kısa olduğunda. Bütün durumların değer yakınsaması ölçülmüyor. Bir milyon etkileşim sınırında başarısızlık ayrıca gösterilir. Tek tohum sonucu genel üstünlük garantisi değildir.`};
 }
};
