import { fmt } from "./plot.js";
const moves=[[-1,0],[1,0],[0,1],[0,-1]],glyph=["↑","↓","→","←"];
export function transition(s,a,shift=0){
  if(s===1)return [21,10+shift];
  if(s===3)return [13,5+shift];
  const r=Math.floor(s/5)+moves[a][0],c=s%5+moves[a][1];
  return r<0||r>=5||c<0||c>=5?[s,-1+shift]:[r*5+c,shift];
}
function solve(A,b){
  const M=A.map((row,i)=>[...row,b[i]]),n=b.length;
  for(let i=0;i<n;i++){
    let pivot=i;for(let k=i+1;k<n;k++)if(Math.abs(M[k][i])>Math.abs(M[pivot][i]))pivot=k;
    [M[i],M[pivot]]=[M[pivot],M[i]];
    for(let k=i+1;k<n;k++){const w=M[k][i]/M[i][i];for(let j=i;j<=n;j++)M[k][j]-=w*M[i][j];}
  }
  const x=Array(n).fill(0);
  for(let i=n-1;i>=0;i--)x[i]=(M[i][n]-M[i].slice(i+1,n).reduce((v,m,j)=>v+m*x[i+1+j],0))/M[i][i];
  return x;
}
export function calculate(g,optimal=false,shift=0){
  const q=(V,s,a)=>{const [t,r]=transition(s,a,shift);return r+g*V[t];};
  let V;
  if(!optimal){
    const A=Array.from({length:25},(_,s)=>Array.from({length:25},(_,t)=>Number(s===t))),b=Array(25).fill(0);
    for(let s=0;s<25;s++)for(let a=0;a<4;a++){const[t,r]=transition(s,a,shift);A[s][t]-=g/4;b[s]+=r/4;}
    V=solve(A,b);
  }else{
    V=Array(25).fill(0);
    for(let k=0;k<5000;k++){
      const W=V.map((_,s)=>Math.max(...moves.map((__,a)=>q(V,s,a))));
      const err=Math.max(...W.map((v,s)=>Math.abs(v-V[s])));V=W;if(err<1e-12)break;
    }
  }
  const Q=V.map((_,s)=>moves.map((__,a)=>q(V,s,a)));
  const policy=Q.map(row=>row.map((v,a)=>v>=Math.max(...row)-1e-9?a:-1).filter(a=>a>=0));
  const residual=Math.max(...V.map((v,s)=>Math.abs(v-(optimal?Math.max(...Q[s]):Q[s].reduce((a,b)=>a+b,0)/4))));
  return {V,Q,policy,residual};
}
function svg(title,kind,data,optimal,uid){
  let s=`<svg viewBox="0 0 260 305" role="img" aria-label="${title}"><defs><marker id="${uid}-${kind}-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto"><path d="M0,0 L7,3.5 L0,7 Z" fill="var(--c-a)"/></marker></defs><text x="130" y="22" text-anchor="middle" style="font-size:16px;fill:var(--ink)">${title}</text>`;
  for(let n=0;n<25;n++){
    const x=20+n%5*44,y=40+Math.floor(n/5)*44;
    s+=`<rect x="${x}" y="${y}" width="44" height="44" fill="var(--paper)" stroke="var(--ink-3)" stroke-width="0.6"/>`;
    if(kind===0){const t={1:"A",3:"B",21:"A′",13:"B′"}[n];if(t)s+=`<text x="${x+22}" y="${y+27}" text-anchor="middle" style="font-size:18px;fill:var(--ink)">${t}</text>`;}
    if(kind===1)s+=`<text x="${x+22}" y="${y+27}" text-anchor="middle" style="font-size:16px;fill:var(--ink)">${data.V[n].toFixed(1).replace("-","−")}</text>`;
    if(kind===2)for(const a of optimal?data.policy[n]:[0,1,2,3]){
      const dx=moves[a][1]*11,dy=moves[a][0]*11;
      s+=`<text x="${x+22+dx}" y="${y+28+dy}" text-anchor="middle" style="font-size:18px;fill:var(--ink)">${glyph[a]}</text>`;
    }
  }
  if(kind===0)s+=`<path d="M86,76 L86,225" stroke="var(--c-a)" marker-end="url(#${uid}-${kind}-arrow)" fill="none"/><path d="M174,76 L174,137" stroke="var(--c-a)" marker-end="url(#${uid}-${kind}-arrow)" fill="none"/><text x="91" y="161" style="font-size:14px;fill:var(--ink)">+10</text><text x="180" y="109" style="font-size:14px;fill:var(--ink)">+5</text><text x="130" y="286" text-anchor="middle" style="font-size:14px;fill:var(--ink)">eylemler: ↑ ↓ ← →</text>`;
  return s+'</svg>';
}
let serial=0;
export default {
  custom:true,keys:["g","optimal","shift"],
  selfTest(a){
    const x=calculate(.9),y=calculate(.9,true),zero=calculate(0,true),shift=calculate(.9,false,1);
    a.ok(Math.abs(x.V[1]-8.7892918626)<1e-8);
    a.ok(Math.abs(y.V[1]-10/(1-.9**5))<1e-8);
    a.ok(x.residual<1e-10&&y.residual<1e-10);
    a.equal(zero.V[1],10);a.equal(zero.V[3],5);
    a.ok(shift.V.every((v,i)=>Math.abs(v-x.V[i]-10)<1e-8));
    a.deepEqual(y.policy[1],[0,1,2,3]);
  },
  mount(root,{init}){
    const q={g:.9,optimal:0,shift:0,...init},uid=`sg-${serial++}`;
    root.innerHTML=`<div class="lab-params"><label class="lab-p" for="${uid}-g">İndirim γ <input id="${uid}-g" data-key="g" type="number" min="0" max="0.99" step="0.01"></label><label class="lab-p" for="${uid}-shift">Her ödüle eklenen sabit <input id="${uid}-shift" data-key="shift" type="number" min="-2" max="2" step="0.1"></label><label class="lab-p sel" for="${uid}-optimal">Politika <select id="${uid}-optimal" data-key="optimal"><option value="0">Eşit olasılıklı rastgele</option><option value="1">En iyi politika</option></select></label></div><p data-error role="status"></p><div data-panels style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,230px),1fr));gap:12px"></div><p data-note class="prose"></p>`;
    function paint(){
      for(const e of root.querySelectorAll('[data-key]'))e.value=q[e.dataset.key];
      const error=!(q.g>=0&&q.g<=.99&&q.shift>=-2&&q.shift<=2&&[0,1].includes(q.optimal));
      root.querySelector('[data-error]').textContent=error?"γ 0–0.99, sabit −2–2 aralığında olmalı.":"";if(error)return;
      const d=calculate(q.g,!!q.optimal,q.shift);
      root.querySelector('[data-panels]').innerHTML=[svg('Ortam: A→A′, B→B′',0,d,!!q.optimal,uid),svg(q.optimal?'En iyi V*':'Rastgele politika V',1,d,!!q.optimal,uid),svg(q.optimal?'Bütün en iyi eylemler':'Her eylem 1/4',2,d,!!q.optimal,uid)].join('');
      root.querySelector('[data-note]').textContent=`A değeri ${fmt(d.V[1],6)}, B değeri ${fmt(d.V[3],6)}; Bellman artığı ${fmt(d.residual,3)}. Sınırdan çıkma normalde −1, sıradan geçiş 0; bütün ödüllere eklenen sabit ${fmt(q.shift)}. A/B etiketleri kaynak modelinin +10/+5 taban ödüllerini gösterir. Rastgele politika doğrusal sistemle, en iyi değer büzülmeli yinelemeyle hesaplandı. Kaynak Şekil ${q.optimal?'3.5':'3.2'} ayarı γ=0.9 ve sabit 0. Hücre sayıları kaynak gibi bir ondalığa yuvarlanır; eşit en iyi eylemlerin hepsi gösterilir.`;
    }
    root.addEventListener('input',e=>{if(e.target.dataset.key){q[e.target.dataset.key]=Number(e.target.value);paint();}});
    paint();return {set(p){Object.assign(q,p);paint();}};
  },
};
