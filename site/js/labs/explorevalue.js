// Kaynak Alıştırma 1.4: keşifli davranışın değeri ve açgözlü devamın değeri.
// Küçük karar ağacının bütün yaprakları tam toplanır; benzetim kullanılmaz.
import { fmt, range } from "./plot.js";

export function probabilities(epsilon, safe) {
  const risky = 1 - epsilon;
  const greedyTarget = (1 - epsilon) * risky + epsilon * safe;
  const preferSafe = safe > risky;
  const behaviorTarget = preferSafe ? (1 - epsilon) * safe + epsilon * risky : greedyTarget;
  return { risky, greedyTarget, behaviorTarget, preferSafe };
}

export default {
  params: [
    { id: "epsilon", label: "Keşif olasılığı ε", min: 0, max: 0.5, step: 0.01, value: 0.3 },
    { id: "safe", label: "Güvenli dalda kazanma olasılığı", min: 0.1, max: 0.99, step: 0.01, value: 0.8 },
  ],
  run(q) {
    const es = range(101).map((i) => i / 200);
    const rows = es.map((e) => probabilities(e, q.safe));
    const v = probabilities(q.epsilon, q.safe);
    return {
      plots: [
        { title: "Dal değerleri: iki farklı devam varsayımı", xlabel: "keşif olasılığı ε", ylabel: "kazanma olasılığı", ylim: [0, 1],
          marks: [{ x: q.epsilon, label: "seçilen ε" }], series: [
            { name: "riskli: devamı açgözlü kabul et", x: es, y: es.map(() => 1), c: 0, dash: true },
            { name: "riskli: keşif dahil", x: es, y: rows.map((x) => x.risky), c: 0 },
            { name: "güvenli", x: es, y: es.map(() => q.safe), c: 1 },
          ] },
        { title: "Kökte de keşif yapıldığında gerçek başarı", xlabel: "keşif olasılığı ε", ylabel: "kazanma olasılığı", ylim: [0, 1],
          marks: [{ x: q.epsilon, label: "seçilen ε" }], series: [
            { name: "açgözlü devam hedefiyle seçim", x: es, y: rows.map((x) => x.greedyTarget), c: 2 },
            { name: "keşifli devam hedefiyle seçim", x: es, y: rows.map((x) => x.behaviorTarget), c: 3 },
          ] },
      ],
      stats: [
        { label: "Riskli dalın gerçek değeri", value: v.risky },
        { label: "Açgözlü hedefle kök başarısı", value: v.greedyTarget },
        { label: "Keşifli hedefle kök başarısı", value: v.behaviorTarget },
      ],
      note: `Riskli dalda doğru hamle kesin kazanç, diğer hamle kesin kayıptır; keşif diğer hamleyi seçer. Gerçek değer 1 − ε = ${fmt(v.risky, 4)}. Güvenli dalın değeri ${fmt(q.safe, 4)}. Keşifli devamı hesaba katan seçim ${v.preferSafe ? "güvenli" : "riskli"} dalı önceleyerek kökte ${fmt(v.behaviorTarget, 4)} başarı verir. Açgözlü devamı varsayan hedef riskli dalı önceleyerek ${fmt(v.greedyTarget, 4)} başarı verir. İkisinde de kökte aynı ε ile keşif yapılır. Sonuçlar ağacın olasılıkları tam toplanarak hesaplandı; XOX eğitiminin ölçümü değildir.`,
    };
  },
};
