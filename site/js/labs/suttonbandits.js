import { fmt, range } from "./plot.js";

export function configurations(mode, scale) {
  const c = (name, kind, param, init = 0, alpha = null, base = true) => ({ name, kind, param, init, alpha, base });
  if (mode === 0) return [0, .01, .1].map(e => c(`ε = ${fmt(e * scale)}`, "eps", Math.min(1, e * scale)));
  if (mode === 1) return [c(`iyimser Q₀ = ${fmt(5 * scale)}`, "eps", 0, 5 * scale, .1), c("ε = 0.1, Q₀ = 0", "eps", .1, 0, .1)];
  if (mode === 2) return [c(`UCB c = ${fmt(2 * scale)}`, "ucb", 2 * scale), c("ε = 0.1", "eps", .1)];
  if (mode === 3) return [.1, .4].flatMap(a => [true, false].map(base => c(`α = ${fmt(a * scale)}, ${base ? "taban var" : "taban yok"}`, "gradient", a * scale, 0, null, base)));
  if (mode === 5) return [c("örnek ortalaması", "eps", .1), c(`sabit α = ${fmt(.1 * scale)}`, "eps", .1, 0, .1 * scale)];
  const out = [];
  for (const [kind, powers, label] of [["eps", [-7,-6,-5,-4,-3,-2], "ε-açgözlü"], ["gradient", [-5,-4,-3,-2,-1,0,1,2], "gradyan"], ["ucb", [-4,-3,-2,-1,0,1,2], "UCB"], ["optimistic", [-2,-1,0,1,2], "iyimser"]]) {
    for (const power of powers) {
      const v = 2 ** power * scale;
      out.push({ ...c(label, kind === "optimistic" ? "eps" : kind, kind === "optimistic" ? 0 : Math.min(kind === "eps" ? 1 : Infinity, v), kind === "optimistic" ? v : 0, kind === "optimistic" ? .1 : null), x: Math.log2(v), family: label });
    }
  }
  if (mode === 6) for (const power of [-7,-6,-5,-4,-3,-2]) {
    const e = Math.min(1,2 ** power * scale);
    out.push({ ...c("sabit α = 0.1", "eps", e, 0, .1), x: Math.log2(2 ** power * scale), family: "sabit α = 0.1" });
  }
  return out;
}

export function experiment(q, rnd) {
  const configs = configurations(q.mode, q.scale), k = 10, study = q.mode === 4 || q.mode === 6, moving = q.mode >= 5;
  const times = range(q.T).map(i => i + 1).filter(t => t <= 30 || t % Math.max(1,Math.ceil(q.T / 400)) === 0 || t === q.T);
  const idx = new Map(times.map((t,i) => [t,i]));
  const data = configs.map(c => ({ ...c, rewards:new Float64Array(times.length), optimal:new Float64Array(times.length), total:0 }));
  const tail = q.mode === 6 ? Math.floor(q.T / 2) : 0;
  for (let run = 0; run < q.runs; run++) {
    const trueQ = Float64Array.from({length:k}, () => moving ? 0 : rnd.normal() + (q.mode === 3 ? 4 : 0));
    const states = configs.map(c => ({ Q:new Float64Array(k).fill(c.init), N:new Uint32Array(k), H:new Float64Array(k), mean:0, probs:new Float64Array(k) }));
    for (let t = 1; t <= q.T; t++) {
      let best = 0;
      for (let a = 1; a < k; a++) if (trueQ[a] > trueQ[best]) best = a;
      for (let j = 0; j < data.length; j++) {
        const c = data[j], s = states[j]; let a = 0;
        if (c.kind === "gradient") {
          let max = Math.max(...s.H), z = 0;
          for (let b = 0; b < k; b++) { s.probs[b] = Math.exp(s.H[b] - max); z += s.probs[b]; }
          for (let b = 0; b < k; b++) s.probs[b] /= z;
          let u = rnd();
          for (a = 0; a < k-1 && u > s.probs[a]; a++) u -= s.probs[a];
        } else if (c.kind === "eps" && rnd() < c.param) a = Math.floor(rnd() * k);
        else {
          let max = -Infinity, ties = 0;
          for (let b = 0; b < k; b++) {
            const score = c.kind === "ucb" ? (s.N[b] ? s.Q[b] + c.param * Math.sqrt(Math.log(t) / s.N[b]) : Infinity) : s.Q[b];
            if (score > max) { max = score; a = b; ties = 1; }
            else if (score === max && rnd() < 1 / ++ties) a = b;
          }
        }
        const reward = trueQ[a] + rnd.normal();
        if (idx.has(t)) { c.rewards[idx.get(t)] += reward / q.runs; c.optimal[idx.get(t)] += 100 * Number(trueQ[a] === trueQ[best]) / q.runs; }
        if (t > tail) c.total += reward / (q.runs * (q.T-tail));
        if (c.kind === "gradient") {
          s.mean += (reward - s.mean) / t;
          const adv = reward - (c.base ? s.mean : 0);
          for (let b = 0; b < k; b++) s.H[b] += c.param * adv * (Number(a === b) - s.probs[b]);
        } else {
          s.N[a]++; s.Q[a] += (c.alpha ?? 1 / s.N[a]) * (reward - s.Q[a]);
        }
      }
      if (moving) for (let a = 0; a < k; a++) trueQ[a] += .01 * rnd.normal();
    }
  }
  return { data, times, study, tail };
}

export default {
  params: [
    {id:"mode",label:"Kaynak deneyi",options:["Şekil 2.2: ε-açgözlü", "Şekil 2.3: iyimser başlangıç", "Şekil 2.4: UCB", "Şekil 2.5: gradyan ve taban", "Şekil 2.6: parametre taraması", "Alıştırma 2.5: değişen kollar", "Alıştırma 2.11: uzun tarama"],value:0},
    {id:"T",label:"Adım sayısı",min:100,max:200000,step:100,value:1000,int:true,log:true},
    {id:"runs",label:"Bağımsız görev sayısı",min:1,max:2000,step:1,value:40,int:true,log:true},
    {id:"scale",label:"Algoritma parametrelerinin çarpanı",min:.125,max:2,step:.125,value:1},
  ], seed:true,
  validate(q) {
    return q.runs*q.T*configurations(q.mode,q.scale).length*10 > 6e8 ? [{level:"error",msg:"Adım sayısı × Bağımsız görev sayısı: bu tarama çok büyük. Tekrar veya adım sayısını azalt."}] : [];
  },
  run(q,rnd) {
    const {data,times,study,tail}=experiment(q,rnd);
    let plots;
    if(study) {
      const families=[...new Set(data.map(x=>x.family))];
      plots=[{title:q.mode===6?"Son yarıdaki ortalama ödül":"Bütün adımlardaki ortalama ödül",xlabel:"ε / α / c / Q₀ (2 tabanlı log)",ylabel:"ortalama ödül",xticks:[[-7,"1/128"],[-6,"1/64"],[-5,"1/32"],[-4,"1/16"],[-3,"1/8"],[-2,"1/4"],[-1,"1/2"],[0,"1"],[1,"2"],[2,"4"]],
        series:families.map((name,i)=>({name,x:data.filter(c=>c.family===name).map(c=>c.x),y:data.filter(c=>c.family===name).map(c=>c.total),c:i}))}];
    } else {
      const divisions=q.mode===1?5:4;
      const xticks=[[1,"1"],...range(divisions).map(i=>{const x=Math.round(q.T*(i+1)/divisions);return [x,String(x)];})];
      const rewards=data.flatMap(c=>[...c.rewards]);
      const ylim=q.mode<3?[Math.min(0,...rewards),Math.max(1.6,Math.max(...rewards)*1.04)]:undefined;
      plots=[
        {title:"Ortalama ödül",xlabel:"adım",ylabel:"ödül",xticks,ylim,series:data.map((c,i)=>({name:c.name,x:times,y:[...c.rewards],c:i}))},
        {title:"En iyi kolu seçme oranı",xlabel:"adım",ylabel:"yüzde",xticks,ylim:[0,100],series:data.map((c,i)=>({name:c.name,x:times,y:[...c.optimal],c:i}))},
      ];
    }
    const sorted=[...data].sort((a,b)=>b.total-a.total),best=sorted[0];
    return {plots,stats:[{label:"Ölçülen aralık",value:`${tail+1}–${q.T}`},{label:"Bu koşudaki en yüksek ortalama",value:best.total},{label:"Bu koşudaki yöntem",value:best.family||best.name}],
      note:`${q.runs} bağımsız görev, ${q.T} adım, her görevde 10 kol. ${q.mode>=5?"Bütün gerçek değerler 0 ile başlar; her adımda her kola bağımsız N(0, 0.01²) eklenir.":`Gerçek kol ortalamaları N(${q.mode===3?4:0}, 1), ödül gürültüsü N(0, 1).`} Algoritma parametrelerinin çarpanı ${fmt(q.scale)}; kaynak ayarı 1. Bu örneklemde en yüksek ortalama ${best.family||best.name}: ${fmt(best.total,4)}. Bu bir genel üstünlük sonucu değildir. Parametre taramasında her nokta ayrı bir algoritma ayarıdır. Gradyan yönteminde taban, o anki ödül dahil hareketli örnek ortalamasıdır; beklenen güncellemenin (1 − 1/t) çarpanı anlatımda açıklanır. Kaynakta ilk beş şekil için kullanılan görev/adım ayarı 2000/1000; Alıştırma 2.11 için 200000 adım ve son 100000 adım ortalaması seçilmelidir. NumPy ile tarayıcı farklı rastgele üreteçler kullanır; aynı tohum iki dilde aynı yörüngeyi vermez.`,
    };
  },
};
