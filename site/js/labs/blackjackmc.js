import {rng,fmt} from './plot.js';
const prob=[1/13,1/13,1/13,1/13,1/13,1/13,1/13,1/13,1/13,4/13];
export const index=(t,u,d)=>Number(u)*100+(t-12)*10+d-1;
export function add(t,u,c){if(c===1){if(t+11<=21){t+=11;u=true;}else t++;}else t+=c;if(t>21&&u){t-=10;u=false;}return [t,u];}
const draw=R=>Math.min(10,1+Math.floor(R()*13));
export function episode(R,policy,start=null,initial=null){
 let t,u,d,dt,du;const states=[],actions=[];
 if(start){[t,u,d]=start;[dt,du]=add(d===1?11:d,d===1,draw(R));}
 else {const c1=draw(R),c2=draw(R);[t,u]=add(...add(0,false,c1),c2);d=draw(R);[dt,du]=add(d===1?11:d,d===1,draw(R));if(t===21)return {states,actions,reward:dt===21?0:1};while(t<12)[t,u]=add(t,u,draw(R));}
 while(true){const s=index(t,u,d),a=initial??policy(s,t,u,d);initial=null;states.push(s);actions.push(a);if(a===0)break;[t,u]=add(t,u,draw(R));if(t>21)return {states,actions,reward:-1};}
 while(dt<17)[dt,du]=add(dt,du,draw(R));return {states,actions,reward:dt>21||t>dt?1:t===dt?0:-1};
}
let referenceCache;
export function reference(){if(referenceCache)return referenceCache;const memo=new Map();
 function dealer(t,u){const key=`${t}/${u}`;if(memo.has(key))return memo.get(key);const out=Array(6).fill(0);if(t>=17)out[Math.min(5,t-17)]=1;else for(let c=1;c<=10;c++){const a=dealer(...add(t,u,c));for(let j=0;j<6;j++)out[j]+=prob[c-1]*a[j];}memo.set(key,out);return out;}
 const D=Array.from({length:10},(_,d)=>{const p=Array(6).fill(0);for(let c=1;c<=10;c++){const z=dealer(...add(d===0?11:d+1,d===0,c));for(let j=0;j<6;j++)p[j]+=prob[c-1]*z[j];}return p;});
 const stand=Array.from({length:200},(_,s)=>{const t=12+Math.floor(s%100/10),d=s%10;return D[d].reduce((v,p,j)=>v+p*(j===5||j+17<t?1:j+17>t?-1:0),0);});
 function hit(V,s){const t=12+Math.floor(s%100/10),u=s>=100,d=s%10+1;let out=0;for(let c=1;c<=10;c++){const[n,w]=add(t,u,c);out+=prob[c-1]*(n>21?-1:V[index(n,w,d)]);}return out;}
 function solve(opt){let V=Array(200).fill(0),Q;for(let k=0;k<100;k++){Q=V.map((_,s)=>[stand[s],hit(V,s)]);const W=Q.map((q,s)=>opt?Math.max(...q):(12+Math.floor(s%100/10)>=20?q[0]:q[1]));const e=Math.max(...W.map((v,s)=>Math.abs(v-V[s])));V=W;if(e<1e-14)break;}Q=V.map((_,s)=>[stand[s],hit(V,s)]);return {V,Q,policy:Q.map(q=>Number(q[1]>q[0]))};}
 referenceCache={prediction:solve(false),optimal:solve(true),dealer:D};return referenceCache;
}
const memo=new Map();
export function simulate(n,seed=1,control=false){const key=`${n}/${seed}/${control}`;if(memo.has(key))return memo.get(key);const R=rng(seed),C=Array(200).fill(0),S=Array(200).fill(0),Q=Array.from({length:200},()=>[0,0]),counts=Array.from({length:200},()=>[0,0]),policy=Array.from({length:200},(_,s)=>Number(12+Math.floor(s%100/10)<20));let first;
 for(let k=0;k<n;k++){
  const start=control?[12+Math.floor(R()*10),R()<.5,1+Math.floor(R()*10)]:null,a0=control?Number(R()<.5):null,z=episode(R,s=>policy[s],start,a0);
  for(let j=0;j<z.states.length;j++){const s=z.states[j],a=z.actions[j];if(control){const count=++counts[s][a];Q[s][a]+=(z.reward-Q[s][a])/count;policy[s]=Number(Q[s][1]>Q[s][0]);}else{C[s]++;S[s]+=z.reward;}}
  if(k===9999&&!control)first={V:S.map((v,s)=>v/(C[s]||1)),counts:[...C]};
 }
 const d={V:control?Q.map(q=>Math.max(...q)):S.map((v,s)=>v/(C[s]||1)),counts:control?counts.map(q=>q[0]+q[1]):C,Q,policy,first};memo.set(key,d);return d;
}
function surface(V,u,title){const point=(d,t)=>[43+17*d+7*t,245-8*t-70*V[u*100+t*10+d]];let out=`<svg viewBox="0 0 320 365" role="img" aria-label="${title}"><text x="160" y="23" text-anchor="middle" style="font-size:18px;fill:var(--ink)">${title}</text>`;
 for(let axis=0;axis<2;axis++)for(let a=0;a<10;a++){const pts=Array.from({length:10},(_,b)=>point(axis?a:b,axis?b:a).join(',')).join(' ');out+=`<polyline points="${pts}" fill="none" stroke="var(--ink-3)" stroke-width=".8"/>`;}
 out+='<path d="M25,175 V315 M43,323 H196 L259,251" fill="none" stroke="var(--ink)"/>';
 for(const[y,label]of[[175,'+1'],[315,'−1']])out+=`<text x="2" y="${y}" style="font-size:15px;fill:var(--ink)">${label}</text>`;
 return out+'<text x="43" y="341" style="font-size:15px;fill:var(--ink)">A</text><text x="185" y="341" style="font-size:15px;fill:var(--ink)">10</text><text x="275" y="254" style="font-size:15px;fill:var(--ink)">21</text><text x="208" y="326" style="font-size:15px;fill:var(--ink)">12</text><text x="137" y="362" text-anchor="middle" style="font-size:15px;fill:var(--ink)">krupiye açık kartı</text><text transform="translate(264,294) rotate(-48)" text-anchor="middle" style="font-size:13px;fill:var(--ink)">oyuncu toplamı</text></svg>';
}
function policySVG(policy,u,title){let out=`<svg viewBox="0 0 320 365" role="img" aria-label="${title}"><text x="160" y="23" text-anchor="middle" style="font-size:18px;fill:var(--ink)">${title}</text>`;for(let t=12;t<=21;t++){out+=`<text x="30" y="${52+(21-t)*24}" text-anchor="end" style="font-size:16px;fill:var(--ink)">${t}</text>`;for(let d=1;d<=10;d++){const a=policy[index(t,u,d)],x=40+(d-1)*24,y=35+(21-t)*24;out+=`<rect x="${x}" y="${y}" width="24" height="24" fill="${a?'var(--c-a)':'var(--paper)'}" fill-opacity="${a?.3:1}" stroke="var(--ink-3)" stroke-width=".5"/><text x="${x+12}" y="${y+17}" text-anchor="middle" style="font-size:14px;fill:var(--ink)">${a?'Ç':'D'}</text>`;}}
 out+='<text x="30" y="287" text-anchor="end" style="font-size:14px;fill:var(--ink-3)">11</text>';for(let d=1;d<=10;d++)out+=`<text x="${52+(d-1)*24}" y="295" text-anchor="middle" style="font-size:15px;fill:var(--ink)">${d===1?'A':d}</text>`;return out+'<text x="160" y="326" text-anchor="middle" style="font-size:17px;fill:var(--ink)">Ç: çek; D: dur</text><text x="160" y="354" text-anchor="middle" style="font-size:15px;fill:var(--ink)">krupiye açık kartı</text></svg>';
}
let serial=0;
export default {
 custom:true,keys:['mode','n','seed','ref'],
 selfTest(a){const d=reference();a.ok(Math.abs(d.prediction.V[index(13,1,2)]+.277204191825)<1e-10);a.equal(d.optimal.policy[index(18,1,1)],1);a.equal(d.optimal.policy[index(18,1,2)],0);const R=rng(3);for(let k=0;k<1000;k++){const z=episode(R,(s,t)=>Number(t<20));a.equal(new Set(z.states).size,z.states.length);a.ok([-1,0,1].includes(z.reward));}const c=simulate(1000,3);a.equal(c.V.length,200);a.ok(c.V.every(v=>Math.abs(v)<=1));},
 mount(root,{init}){const q={mode:0,n:500000,seed:1,ref:0,...init},id=`bjm-${serial++}`;root.innerHTML=`<label class="lab-p sel" for="${id}-mode">Deney<select id="${id}-mode" data-key="mode"><option value="0">Şekil 5.1: 20/21’de duran politika</option><option value="1">Şekil 5.2: Monte Carlo ES kontrol</option></select></label><label class="lab-p" for="${id}-n">Bölüm sayısı<input id="${id}-n" data-key="n" type="number" min="1000" max="2000000" step="1000"></label><label class="lab-p" for="${id}-seed">Tohum<input id="${id}-seed" data-key="seed" type="number" min="1" max="999" step="1"></label><label class="lab-p sel" for="${id}-ref">Gösterilen değer<select id="${id}-ref" data-key="ref"><option value="0">MC tahmini</option><option value="1">Bağımsız tam model referansı</option></select></label><p data-error role="status"></p><div data-panels style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,270px),1fr));gap:12px"></div><p data-note></p>`;
 function paint(){for(const e of root.querySelectorAll('[data-key]'))e.value=q[e.dataset.key];const bad=!Number.isInteger(q.n)||q.n<1000||q.n>2000000||!Number.isInteger(q.seed)||q.seed<1||q.seed>999||![0,1].includes(q.mode)||![0,1].includes(q.ref);root.querySelector('[data-error]').textContent=bad?'Bölüm 1000–2000000, tohum 1–999 tam sayı; seçimler 0/1 olmalı.':'';if(bad)return;
  const d=simulate(q.n,q.seed,!!q.mode),r=reference()[q.mode?'optimal':'prediction'],V=q.ref?r.V:d.V,policy=q.ref?r.policy:d.policy,panels=[];
  if(!q.mode){const first=q.ref?r.V:d.first?.V||d.V;for(let u=1;u>=0;u--){panels.push(surface(first,u,`${u?'Kullanılabilir as':'As yok'}: ${q.ref?'tam V':Math.min(q.n,10000)}`));panels.push(surface(V,u,`${u?'Kullanılabilir as':'As yok'}: ${q.ref?'tam V':q.n}`));}}
  else for(let u=1;u>=0;u--){panels.push(policySVG(policy,u,`${u?'Kullanılabilir as':'As yok'}: ${q.ref?'π*':'öğrenilen π'}`));panels.push(surface(V,u,`${u?'Kullanılabilir as':'As yok'}: ${q.ref?'V*':'max Q tahmini'}`));}
  root.querySelector('[data-panels]').innerHTML=panels.join('');const rmse=Math.sqrt(d.V.reduce((s,v,i)=>s+(v-r.V[i])**2,0)/200),agree=d.policy.filter((a,s)=>a===r.policy[s]).length/2;
  root.querySelector('[data-note]').textContent=`${q.n} bölüm, tohum ${q.seed}; tam 200 durumun MC/tam-model RMS farkı ${fmt(rmse,5)}.${q.mode?` Politika uyumu %${fmt(agree,4)}. Politika panelinde11 yalnız alt eksen sınırı, veri12–21. Sonlu MC ES tahmini tek başına bütün hücrelerde optimal olduğunu kanıtlamaz; tam model seçimi V*/π* referansını ayrıca gösterir.`:''} 5.1 kaynak ayarı 10000/500000; 5.2 bölüm sayısı kaynakta verilmedi, bizim bütçemiz açık seçilir. ES başlangıçları durum ve ilk eylem üzerinde düzgündür; 21 başlangıcı doğal olmayan karar durumu olarak temsil edilir. Gerçek iki-kart doğal 21 oyununda karar alınmadan +1/0 ile sonlanılır. Tam durum üçlüsü aynı elde tekrar etmediğinden ilk/her ziyaret aynı örnekleri kullanır. Telafi için as 11 veya 1 sayılır; sonsuz deste ve krupiye 17’de durma kuralı korunur. Referans DP yalnız doğrulama içindir. Python ve tarayıcı RNG’leri farklıdır; aynı tohum aynı elleri üretmez.`;
 }
 root.addEventListener('input',e=>{if(e.target.dataset.key){q[e.target.dataset.key]=Number(e.target.value);paint();}});paint();return {set(p){Object.assign(q,p);paint();}};
 }
};
