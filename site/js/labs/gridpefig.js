import {evaluate} from './gridpe.js';
const moves=[[-1,0],[1,0],[0,-1],[0,1]],arrows=['↑','↓','←','→'];
function next(s,a){const y=Math.floor(s/4)+moves[a][0],x=s%4+moves[a][1];return y<0||y>=4||x<0||x>=4?s:y*4+x;}
export function gridFigure(k,inplace=false){
 const {V}=evaluate(k,1,-1,inplace);
 const policy=V.map((_,s)=>[0,15].includes(s)?[]:moves.flatMap((m,a)=>V[next(s,a)]>=Math.max(...moves.map((n,b)=>V[next(s,b)]))-1e-8?[a]:[]));
 return {V,policy};
}
function rounded(v){if(v===0)return '0.0';const p=10**(1-Math.floor(Math.log10(Math.abs(v))));return (Math.sign(v)*Math.ceil(Math.abs(v)*p-.5-1e-10)/p).toPrecision(2).replace('-','−');}
function panel(d,kind){let out=`<svg viewBox="0 0 290 340" role="img" aria-label="${['Kaynak 4×4 modeli','Rastgele politika değerleri','Değere göre bütün açgözlü oklar'][kind]}"><text x="145" y="24" text-anchor="middle" style="fill:var(--ink);font-size:18px">${['14 durum, tek terminal','Rastgele politika V','Açgözlü politika'][kind]}</text>`;
 for(let s=0;s<16;s++){const x=25+s%4*60,y=48+Math.floor(s/4)*60,t=[0,15].includes(s);out+=`<rect x="${x}" y="${y}" width="60" height="60" fill="${t?'var(--line)':'var(--paper)'}" stroke="var(--ink-3)"/>`;
  if(kind<2)out+=`<text x="${x+30}" y="${y+37}" text-anchor="middle" style="font-size:20px;fill:var(--ink)">${kind===1?rounded(d.V[s]):t?'■':s}</text>`;
  else for(const a of d.policy[s])out+=`<text x="${x+30+15*moves[a][1]}" y="${y+37+15*moves[a][0]}" text-anchor="middle" style="font-size:23px;fill:var(--ink)">${arrows[a]}</text>`;
 }
 return out+`<text x="145" y="321" text-anchor="middle" style="font-size:16px;fill:var(--ink)">${kind===0?'↑ ↓ ← →; her geçiş −1':'γ = 1; terminal değeri 0'}</text></svg>`;
}
let serial=0;
export default {
 custom:true,keys:['k','mode'],
 selfTest(a){const k2=gridFigure(2),lim=gridFigure(400);a.equal(k2.V[1],-1.75);a.ok(Math.abs(lim.V[1]+14)<1e-7);a.ok(Math.abs(lim.V[3]+22)<1e-7);a.equal(rounded(-1.75),'−1.7');a.deepEqual(gridFigure(0).policy[5],[0,1,2,3]);a.deepEqual(gridFigure(3).policy[1],[2]);},
 mount(root,{init}){
  const q={k:3,mode:0,...init},id=`gpf-${serial++}`;
  root.innerHTML=`<div><label class="lab-p" for="${id}-k">Tarama k (400 ≈ kaynak limiti)<input id="${id}-k" type="number" data-key="k" min="0" max="400" step="1"></label><label class="lab-p sel" for="${id}-mode">Güncelleme<select id="${id}-mode" data-key="mode"><option value="0">Eşzamanlı: kaynak Şekil 4.1</option><option value="1">Yerinde: satır sırasıyla</option></select></label></div><p data-error role="status"></p><div data-panels style="display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,240px),1fr));gap:12px"></div><p data-note></p>`;
  function paint(){for(const e of root.querySelectorAll('[data-key]'))e.value=q[e.dataset.key];const bad=!Number.isInteger(q.k)||q.k<0||q.k>400||![0,1].includes(q.mode);root.querySelector('[data-error]').textContent=bad?'k tam sayı ve 0–400; mod 0 veya 1 olmalı.':'';if(bad)return;
   const d=gridFigure(q.k,!!q.mode);root.querySelector('[data-panels]').innerHTML=[0,1,2].map(k=>panel(d,k)).join('');root.querySelector('[data-note]').textContent=`k=${q.k}: V(1)=${d.V[1].toFixed(8)}, V(3)=${d.V[3].toFixed(8)}. Şekil 4.1 kaynak ön ayarları eşzamanlı k=0,1,2,3,10 ve limit. 400 tarama limitin 10⁻⁷ içinde sayısal karşılığıdır. Kaynak gibi iki anlamlı basamak ve tam yarımda sıfıra doğru yuvarlama kullanıldı: −1.75 → −1.7. Terminal tek durumdur, iki köşede gösterilir. Kaynakta k=0 politikası rastgele; açgözlü kümeye giren dört eylemin hepsi gösteriliyor. k≥3 eşzamanlı ayarda bu okların her karışımı en iyi politikadır.`;
  }
  root.addEventListener('input',e=>{if(e.target.dataset.key){q[e.target.dataset.key]=Number(e.target.value);paint();}});paint();return {set(p){Object.assign(q,p);paint();}};
 }
};
