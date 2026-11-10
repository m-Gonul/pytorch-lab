// M9.1 — İnterpolasyon çekirdekleri: örneklenmiş bir 1-B sinyali (adım h) yeniden kur. En yakın komşu (kutu çekirdek), doğrusal (üçgen),
// Keys kübik (a = −0.5 ya da OpenCV'nin −0.75). Bikübik 2-B interpolasyon bu çekirdeğin ayrılabilir uygulamasıdır. Sinyaller: sinüs,
// basamak (kenar), çırp (frekansı artan). Ölçüt: gerçek sürekli sinyale RMSE ve basamakta aşma (ringing).
import { fmt, range } from "./plot.js";

const SIG = ["sinüs", "basamak (kenar)", "çırp (frekansı artan sinüs)"];
const f = (k, x) => (k === 0 ? Math.sin(2 * Math.PI * x / 8) : k === 1 ? (x > 10.3 ? 1 : 0) : Math.sin(2 * Math.PI * (x * x) / 160));
export function kernel(m, t, a) {
  const u = Math.abs(t);
  if (m === 0) return u < 0.5 || t === -0.5 ? 1 : 0;
  if (m === 1) return u < 1 ? 1 - u : 0;
  return u <= 1 ? (a + 2) * u ** 3 - (a + 3) * u ** 2 + 1 : u < 2 ? a * u ** 3 - 5 * a * u ** 2 + 8 * a * u - 4 * a : 0;
}
export function reconstruct(samples, h, x, m, a) {
  const s = x / h, i0 = Math.floor(s); let v = 0;
  for (let i = i0 - 2; i <= i0 + 2; i++) { const j = Math.max(0, Math.min(samples.length - 1, i)); v += samples[j] * kernel(m, s - i, a); }
  return v;
}
export default {
  params: [
    { id: "sig", label: "Sinyal", options: SIG, value: 1 },
    { id: "m", label: "Yöntem", options: ["en yakın komşu", "doğrusal", "kübik (Keys)"], value: 2 },
    { id: "a", label: "Keys parametresi a", min: -1, max: 0, step: 0.05, value: -0.75, on: { m: [2] } },
    { id: "h", label: "Örnekleme adımı h", min: 0.25, max: 4, step: 0.05, value: 1 },
  ],
  run(q) {
    const X = 20, n = Math.floor(X / q.h) + 1, samples = range(n).map((i) => f(q.sig, i * q.h));
    const xs = range(801).map((i) => (i * X) / 800), tr = xs.map((x) => f(q.sig, x)), rc = xs.map((x) => reconstruct(samples, q.h, x, q.m, q.a));
    const inner = xs.map((x, i) => [x, i]).filter(([x]) => x > 2 && x < X - 2).map(([, i]) => i);
    const rmse = Math.sqrt(inner.reduce((s, i) => s + (rc[i] - tr[i]) ** 2, 0) / inner.length);
    const over = q.sig === 1 ? Math.max(Math.max(...rc) - 1, -Math.min(...rc)) : null;
    const allM = [0, 1, 2].map((m) => { const r2 = xs.map((x) => reconstruct(samples, q.h, x, m, q.a)); return Math.sqrt(inner.reduce((s, i) => s + (r2[i] - tr[i]) ** 2, 0) / inner.length); });
    const P = [`${["En yakın komşu", "Doğrusal", `Kübik (a = ${q.a})`][q.m]}: gerçek sinyale RMSE ${fmt(rmse, 3)} (en yakın ${fmt(allM[0], 3)}, doğrusal ${fmt(allM[1], 3)}, kübik ${fmt(allM[2], 3)}).`];
    if (q.sig === 1) P.push(over > 1e-9 ? `Basamakta aşma ${fmt(over, 3)}: kübik çekirdeğin negatif kanatları kenarın iki yanında hafif "halka" (ringing) üretir; a daha negatif oldukça aşma ve keskinlik artar. Doğrusal ve en yakın komşu hiç aşmaz (çekirdekleri hep ≥ 0).` : "Basamakta aşma yok: çekirdek her yerde ≥ 0 (en yakın komşu, doğrusal).");
    if (q.sig === 2) P.push("Çırpın frekansı sağa doğru artar; adım h, örnekleme kuramının sınırını (periyot başına 2 örnek) aşınca hiçbir interpolasyon gerçek sinyali geri getiremez: kayıp örneklemede olmuştur (aliasing), interpolasyonda değil.");
    P.push(`Bilineer interpolasyon bu doğrusal çekirdeğin iki eksende uygulanmasıdır (2 × 2 komşu); bikübik, kübik çekirdeğin iki eksende uygulanmasıdır (4 × 4 komşu, slaytlardaki "beş kez 1-B" işlemi).`);
    const kx = range(161).map((i) => -2 + i / 40);
    return {
      plots: [
        { title: "Gerçek sinyal, örnekler ve yeniden kurma", xlabel: "x", ylabel: "f", wide: true, size: { w: 620, h: 230 }, series: [{ name: "gerçek", x: xs, y: tr, c: 4, dash: true }, { name: "yeniden kurma", x: xs, y: rc, c: 0 }, { name: "örnekler", x: range(n).map((i) => i * q.h), y: samples, kind: "scatter", c: 1, r: 3 }] },
        { title: "İnterpolasyon çekirdeği", xlabel: "t (örnek aralığı cinsinden)", ylabel: "ağırlık", series: [{ name: "çekirdek", x: kx, y: kx.map((t) => kernel(q.m, t, q.a)), c: 0 }] },
      ],
      stats: [{ label: "RMSE", value: rmse }, { label: "RMSE: en yakın / doğrusal / kübik", value: allM.map((v) => fmt(v, 3)).join(" / ") }, { label: "Basamakta aşma", value: over === null ? "—" : over }],
      note: P.join(" "),
    };
  },
};
