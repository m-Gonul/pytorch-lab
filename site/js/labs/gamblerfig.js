import {range,fmt} from './plot.js';
export function gambler(ph=.4,eps=1e-12){let V=Array(101).fill(0);V[100]=1;const trace={};let k=0;
 for(k=1;k<=100000;k++){let delta=0;for(let s=1;s<100;s++){const old=V[s];let best=0;for(let a=1;a<=Math.min(s,100-s);a++)best=Math.max(best,ph*V[s+a]+(1-ph)*V[s-a]);V[s]=best;delta=Math.max(delta,Math.abs(best-old));}if([1,2,3,32].includes(k))trace[k]=[...V];if(k>=32&&delta<eps)break;}
 const policy=Array(101).fill(0);for(let s=1;s<100;s++){let best=-Infinity;for(let a=1;a<=Math.min(s,100-s);a++){const q=Math.round((ph*V[s+a]+(1-ph)*V[s-a])*1e9)/1e9;if(q>best){best=q;policy[s]=a;}}}
 return {V,policy,trace,sweeps:k};
}
export default {
 params:[{id:'ph',label:'Yazı/kazanma olasılığı p_h',min:.05,max:.95,step:.05,value:.4},{id:'tol',label:'Durma eşiği: 10 üzeri',min:-12,max:-5,step:1,value:-12,int:true}],
 run(q){const d=gambler(q.ph,10**q.tol),x=range(99).map(i=>i+1),series=[1,2,3,32].map((k,i)=>({name:`tarama ${k}`,x,y:d.trace[k].slice(1,100),c:i}));series.push({name:'son V',x,y:d.V.slice(1,100),c:4});
 return {plots:[{title:'Şekil 4.3: getiri = hedefe ulaşma olasılığı',xlabel:'sermaye',ylabel:'V',xlim:[1,99],ylim:[0,1],xticks:[[1,'1'],[25,'25'],[50,'50'],[75,'75'],[99,'99']],series},{title:'Son politika: tam sayı bahis',xlabel:'sermaye',ylabel:'bahis',xlim:[1,99],ylim:[0,51],xticks:[[1,'1'],[25,'25'],[50,'50'],[75,'75'],[99,'99']],series:[{name:'π',x,y:d.policy.slice(1,100),c:0}]}],stats:[{label:'V(50)',value:d.V[50]},{label:'π(50)',value:d.policy[50]},{label:'π(51)',value:d.policy[51]}],note:`${d.sweeps} yerinde tarama (1,2,3,32 görüntülerini korumak için en az 32). p_h=${fmt(q.ph)}, θ=${10**q.tol}. Kazanma +1 ödülü, eşdeğer hesapta V(100)=1 sınır değeriyle temsil edildi; ayrıca yeniden +1 eklenmez. V(0)=0. Bahisler 1…min(s,100−s): 0 bahis sonsuz beklemeyi mümkün kıldığı için sonlanan politika aranırken dışarıda bırakılır. Kaynakta 0 yazmasının bu nüansı anlatımda açıklanır. Eşitlik seçimi mevcut x3 ile aynıdır: Q değerleri 9 ondalığa yuvarlanır, sonra en küçük bahis seçilir. Değerler θ küçüldükçe kararlılaşır; eşit en iyi eylemler yüzünden politika çiziminin tekil olması beklenmez.`};}
};
