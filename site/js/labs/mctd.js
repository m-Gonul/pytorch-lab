// M11.4 — Sutton & Barto Örnek 6.2: beş durumlu rastgele yürüyüş (A–E, C'den başlanır, her adım sola/sağa 1/2; soldan çıkış ödül 0,
// sağdan çıkış ödül 1, γ = 1). Gerçek değerler 1/6, 2/6, …, 5/6. Rastgele politikanın değerlendirmesi: her ziyaret Monte Carlo
// (bölüm sonu getirisine doğru) ve TD(0) (r + V(s')'ye doğru), sabit α ya da α = 1/n(s). Başlangıç V = 0.5. R bağımsız çalıştırmanın
// ortalama RMS hatası bölüm sayısına göre; ayrıca son çalıştırmanın tahminleri.
import { fmt, range, rng } from "./plot.js";

const NS = 5, TRUE = range(NS).map((i) => (i + 1) / 6);
function episode(r) { const s = [2], rw = []; for (;;) { const c = s[s.length - 1], n = r() < 0.5 ? c - 1 : c + 1; if (n < 0) { rw.push(0); return { s, rw, end: true }; } if (n >= NS) { rw.push(1); return { s, rw, end: true }; } rw.push(0); s.push(n); } }
export function run1(q, r, E) {
  const initial = q.init === 3 ? [...TRUE] : Array(NS).fill(q.init === 1 ? 0 : q.init === 2 ? 1 : .5), Vm = [...initial], Vt = [...initial], snapshots = {0:[...initial]}, nm = Array(NS).fill(0), nt = Array(NS).fill(0), em = [], et = [];
  const rms = (V) => Math.sqrt(V.reduce((a, v, i) => a + (v - TRUE[i]) ** 2, 0) / NS);
  em.push(rms(Vm)); et.push(rms(Vt));
  for (let e = 0; e < E; e++) {
    const { s, rw } = episode(r), T = s.length;
    // TD(0)
    for (let t = 0; t < T; t++) { const st = s[t], nx = t + 1 < T ? Vt[s[t + 1]] : 0; nt[st]++; const a = q.sched === 1 ? 1 / nt[st] : q.at; Vt[st] += a * (rw[t] + nx - Vt[st]); }
    // her ziyaret MC: getiri = bölümün son ödülü (γ = 1, ara ödüller 0)
    const G = rw[T - 1];
    for (let t = 0; t < T; t++) { const st = s[t]; nm[st]++; const a = q.sched === 1 ? 1 / nm[st] : q.am; Vm[st] += a * (G - Vm[st]); }
    em.push(rms(Vm)); et.push(rms(Vt)); if ([1,10,100].includes(e+1)) snapshots[e+1]=[...Vt];
  }
  return { em, et, Vm, Vt, snapshots };
}

export function sourceCurves(q,r){
 const curves=[],xs=range(q.E+1),configs=[...([.01,.02,.03,.04].map(a=>({method:'MC',alpha:a}))),...([.15,.1,.05].map(a=>({method:'TD',alpha:a})))],seeds=Array.from({length:q.R},()=>1+Math.floor(r()*2147483646));
 for(const c of configs){const y=Array(q.E+1).fill(0);for(const seed of seeds){const z=run1({init:q.init,sched:0,am:c.alpha,at:c.alpha},rng(seed),q.E),values=c.method==='MC'?z.em:z.et;for(let t=0;t<=q.E;t++)y[t]+=values[t]/q.R;}curves.push({name:`${c.method} α=${c.alpha}`,x:xs,y,c:[0,0,3,3,2,2,1][curves.length],dash:[false,true,false,true,false,true,false][curves.length]});}
 const single=run1({init:q.init,sched:0,am:.02,at:.1},rng(2),100),snaps=[0,1,10,100];
 return {plots:[{title:'Kaynak sol panel: tek TD koşusu,tohum2',xlabel:'durum',ylabel:'V',side:true,size:{w:320,h:270},ylim:[0,1],xticks:TRUE.map((_,i)=>[i,'ABCDE'[i]]),series:[...snaps.map((n,i)=>({name:`${n} bölüm`,x:range(5),y:single.snapshots[n],c:i,dots:true})),{name:'gerçek değerler',x:range(5),y:TRUE,c:4,dash:true}]},{title:'Kaynak sağ panel: yedi öğrenme eğrisi',xlabel:'bölüm',ylabel:'ortalama RMS',wide:true,size:{w:640,h:290},ylim:[0,q.init===0?.25:.7],series:curves}],stats:[{label:'Son RMS: TD α0.05',value:curves.at(-1).y.at(-1)},{label:'Son RMS: MC α0.02',value:curves[1].y.at(-1)}],note:`Kaynak Örnek6.2,s.125: A…E,başlangıçC,eşit sol/sağ,γ1;sağ terminal+1,diğer ödüller0. Sağda ${q.R} bağımsız tekrar×${q.E} bölüm;MC α0.01/0.02/0.03/0.04,TD α0.15/0.1/0.05. Aynı tekrarın bütün yöntem/α ayarları aynı örnek yolları kullanır;RMS önce beş durumda,sonra tekrarlar üzerinde ortalanır. Soldaki bağımsız tek örnek100 bölüm ve tohum2;standart0.5 başlangıçta ilk bölüm yalnızA değerini0.45 yapar. Diğer başlangıç seçimlerinde bu sayı değişir. Sabitα dalgalanmaları sürdürür;her başlatmada veya bütün α aralıklarında bir yöntem üstün diye çıkarım yapılmaz.`};
}
export function alphaSweep(q,r){const grid=[.001,.002,.005,.01,.02,.03,.04,.05,.075,.1,.15,.2,.3,.5,.75,1],rows=[],seeds=Array.from({length:q.R},()=>1+Math.floor(r()*2147483646));for(const alpha of grid){let mc=0,td=0,lastmc=0,lasttd=0;for(const seed of seeds){const z=run1({init:q.init,sched:0,am:alpha,at:alpha},rng(seed),q.E);mc+=z.em.slice(1).reduce((s,v)=>s+v,0)/q.E/q.R;td+=z.et.slice(1).reduce((s,v)=>s+v,0)/q.E/q.R;lastmc+=z.em.at(-1)/q.R;lasttd+=z.et.at(-1)/q.R;}rows.push({alpha,mc,td,lastmc,lasttd});}const best=key=>rows.reduce((a,b)=>a[key]<=b[key]?a:b);return {plots:[{title:'İlk bölümlerin ortalama RMS hatası',xlabel:'α (log ölçek)',ylabel:'ortalama RMS',xlim:[-3,0],xticks:[[-3,'0.001'],[-2,'0.01'],[-1,'0.1'],[0,'1']],series:[{name:'MC',x:grid.map(Math.log10),y:rows.map(x=>x.mc),c:0,dots:true},{name:'TD',x:grid.map(Math.log10),y:rows.map(x=>x.td),c:2,dots:true}]}],stats:[{label:'Bu ızgarada en iyi MC α / hata',value:`${best('mc').alpha} / ${fmt(best('mc').mc,5)}`},{label:'Bu ızgarada en iyi TD α / hata',value:`${best('td').alpha} / ${fmt(best('td').td,5)}`}],note:`Alıştırma6.4 için kendi16 noktalı taramamız;kitapta bu geniş taramanın sayıları verilmez. ${q.R} tekrar×${q.E} bölüm. Ölçüt1…${q.E} bölümlerinin ortalama RMS değeri;sadece son bölüm hatası değildir. Ortam yolları bütünα/yöntemler arasında eşlenir. Çok küçükα ilk hatayı yavaş siler,büyükα örnekleme dalgalanmasını büyütebilir. En iyi ayar seçilen ufuk ve başlatmaya bağlıdır;sonuç bütünMDP'lerde TD üstünlüğü kanıtı değildir.`};}

export default {
  params: [
    {id:"view",label:"Görünüm",options:["Mevcut MC/TD karşılaştırması","Kaynak Örnek6.2: bütün α eğrileri","Alıştırma6.4: geniş α taraması"],value:0},
    {id:"init",label:"Başlangıç değerleri",options:["Bütün durumlar0.5","Bütün durumlar0","Bütün durumlar1","Tam gerçek değerler1/6…5/6"],value:0},
    { id: "sched", label: "Öğrenme oranı", options: ["sabit α", "α = 1/n(s) (örnek ortalaması)"], value: 0, on:{view:[0]} },
    { id: "am", label: "MC için α", min: 0.005, max: 0.2, step: 0.005, value: 0.02, on: { sched: [0],view:[0] } },
    { id: "at", label: "TD için α", min: 0.005, max: 0.3, step: 0.005, value: 0.1, on: { sched: [0],view:[0] } },
    { id: "E", label: "Bölüm sayısı", min: 10, max: 300, step: 10, value: 100, int: true },
    { id: "R", label: "Ortalaması alınan çalıştırma", min: 1, max: 200, value: 100, int: true },
  ],
  seed: true,
  run(q, r) {
    if(q.view===1)return sourceCurves(q,r);
    if(q.view===2)return alphaSweep(q,r);
    let sm = Array(q.E + 1).fill(0), stt = Array(q.E + 1).fill(0), last = null;
    for (let k = 0; k < q.R; k++) { const o = run1(q, r, q.E); o.em.forEach((v, i) => (sm[i] += v / q.R)); o.et.forEach((v, i) => (stt[i] += v / q.R)); last = o; }
    const xs = range(q.E + 1), P = [`${q.R} çalıştırma ortalaması, ${q.E} bölüm sonunda RMS hata: MC ${fmt(sm[q.E], 3)}, TD ${fmt(stt[q.E], 3)}.`];
    const cross = xs.find((i) => i > 0 && stt[i] < sm[i]);
    if (q.sched === 0) P.push(`Sabit α örnekleme dalgalanmalarını sürdürür;hata tabanı α ve göreve bağlıdır (ör. TD α = ${q.at}).`);
    else P.push("α = 1/n(s) ile her-ziyaret MC getirilerin örnek ortalamasıdır; bölüm içi bağımlılık ve rastgele ziyaret sayısı nedeniyle sonlu veride yanlı olabilir,uygun koşullarda tutarlıdır; TD'de 1/n, ilk bölümlerin yanlı bootstrapping hedeflerini uzun süre ağırlıkta tutar ve yavaşlar.");
    if (cross !== undefined && stt[q.E] < sm[q.E]) P.push(`TD’nin MC’den daha düşük hata verdiği ilk ölçülen bölüm ${cross};son bölümde de TD daha düşük. Bu her aradaki bölümün aynı sıralandığını söylemez.`);
    if (q.sched === 0 && q.at >= 0.15) P.push("Büyük α'da TD hızlı iner ama yüksek bir hata tabanında durur.");
    return {
      plots: [
        { title: "Ortalama RMS hata (gerçek değerlere göre)", xlabel: "bölüm", ylabel: "RMS", side: true, size: { w: 320, h: 240 }, ylim: [0, 0.3], series: [{ name: "MC", x: xs, y: sm, c: 0 }, { name: "TD(0)", x: xs, y: stt, c: 2 }] },
        { title: "Son çalıştırmanın tahminleri", xlabel: "durum", ylabel: "V", side: true, size: { w: 280, h: 240 }, ylim: [0, 1], xticks: range(NS).map((i) => [i, "ABCDE"[i]]),
          series: [{ name: "gerçek", x: range(NS), y: TRUE, c: 4, dash: true }, { name: "MC", x: range(NS), y: last.Vm, c: 0, dots: true }, { name: "TD", x: range(NS), y: last.Vt, c: 2, dots: true }] },
      ],
      stats: [{ label: "RMS (MC / TD)", value: `${fmt(sm[q.E], 3)} / ${fmt(stt[q.E], 3)}` }, { label: "V(C) tahmini (MC / TD), gerçek 0.5", value: `${fmt(last.Vm[2], 3)} / ${fmt(last.Vt[2], 3)}` }],
      note: P.join(" "),
    };
  },
};
