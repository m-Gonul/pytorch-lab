import { fmt,range } from "./plot.js";
export function recycling(q){
  const {a,b,rs,rw,g}=q;
  const P=[[[a,1-a],[1,0]],[[1-b,b],[0,1],[1,0]]],R=[[rs,rw],[b*rs-3*(1-b),rw,0]];
  const policies=[];
  for(let h=0;h<2;h++)for(let l=0;l<3;l++){
    const p=P[0][h],z=P[1][l],A=1-g*p[0],B=-g*p[1],C=-g*z[0],D=1-g*z[1],det=A*D-B*C;
    policies.push({h,l,V:[(D*R[0][h]-B*R[1][l])/det,(A*R[1][l]-C*R[0][h])/det]});
  }
  const V=[Math.max(...policies.map(x=>x.V[0])),Math.max(...policies.map(x=>x.V[1]))];
  const Q=P.map((rows,s)=>rows.map((p,k)=>R[s][k]+g*(p[0]*V[0]+p[1]*V[1])));
  return {V,Q,policies};
}
export default {
  params:[
    {id:"a",label:"Yüksek pilde yüksek kalma α",min:0,max:1,step:.05,value:.8},
    {id:"b",label:"Düşük pilde aramayı tamamlama β",min:0,max:1,step:.05,value:.5},
    {id:"rs",label:"Arama beklenen ödülü",min:.1,max:3,step:.1,value:1},
    {id:"rw",label:"Bekleme beklenen ödülü",min:0,max:1,step:.1,value:.2},
    {id:"g",label:"İndirim γ",min:0,max:.99,step:.01,value:.9},
  ],
  validate(q){return q.rs<=q.rw?[{level:"warn",msg:"Arama ödülü kaynakta beklemeden büyüktür; seçilen ayar bu varsayımı sağlamıyor, hesap genel model için sürüyor."}]:[];},
  run(q){
    const {V,Q}=recycling(q),flat=[...Q[0],...Q[1]],lo=Math.min(0,...flat),hi=Math.max(0,...flat),span=hi-lo||1,names=[['ara','bekle'],['ara','bekle','şarj']],best=Q.map((row,s)=>row.flatMap((v,a)=>v>=Math.max(...row)-1e-9?[names[s][a]]:[]));
    return {plots:[{title:"Yüksek/düşük pilde eylem değerleri",xlabel:"durum ve eylem",ylabel:"Q*",xlim:[-.5,4.5],ylim:[lo-(lo<0?.05*span:0),hi+.08*span],xticks:[[0,"h: ara"],[1,"h: bekle"],[2,"l: ara"],[3,"l: bekle"],[4,"l: şarj"]],series:[{name:"en iyi devam ile eylem değeri",x:range(5),y:flat,kind:"bar",c:0}]}],stats:[{label:"V*(yüksek)",value:V[0]},{label:"V*(düşük)",value:V[1]}],note:`Yüksek pilde en iyi: ${best[0].join(' / ')}; düşük pilde: ${best[1].join(' / ')}. h yüksek, l düşük pil. Altı deterministik politika 2×2 doğrusal sistemle ayrı değerlendirildi. Düşük pilde aramanın beklenen ödülü β·r_ara −3(1−β) = ${fmt(q.b*q.rs-3*(1-q.b))}. Kurtarma sonrası pil yüksek; şarja gitmenin ödülü 0. Kaynak parametreleri semboliktir; başlangıçtaki sayısal değerler bu deney için seçildi. Hesap geçiş olasılıkları ve beklenen ödülleri kullanır, kutu sayılarının tam dağılımını varsaymaz.`};
  },
};
