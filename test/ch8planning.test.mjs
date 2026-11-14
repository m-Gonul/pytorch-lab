import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {maze,dyna,greedyLength} from '../site/js/labs/dynasource.js';
import {prioritized,rtdp} from '../site/js/labs/sweepfocus.js';
import budget from '../site/js/labs/backupbudget.js';
import trajectory from '../site/js/labs/trajectoryplan.js';
const json=name=>JSON.parse(readFileSync(new URL(`../notes/sutton-2018/${name}`,import.meta.url)));

test('Ch8 Dyna: independent Python full-Q references and paired first episodes',()=>{
 const ref=json('measurements_ch8_reference.json');
 for(const x of ref.dyna){const r=dyna({seed:x.seed,n:x.n,E:3});assert.deepEqual(r.lengths,x.lengths);assert.equal(r.updates,x.updates);assert.ok(Math.max(...r.Q.map((v,i)=>Math.abs(v-x.Q[i])))<1e-12);}
 for(const seed of [1,17]){const first=[0,5,50].map(n=>dyna({seed,n,E:1}).lengths[0]);assert.equal(new Set(first).size,1);}
});
test('Ch8 maze: blocking/shortcut and grid scaling retain explicit geometry',()=>{
 assert.equal(maze().states,47);assert.equal(maze().shortest,14);
 assert.equal(maze(1,false).step(36,0)[0],36);assert.equal(maze(1,true).step(36,0)[0],27);
 assert.equal(maze(2,false).step(44,0)[0],44);assert.equal(maze(2,true).step(44,0)[0],35);
 for(let scale=0;scale<8;scale++){const m=maze(0,false,2**Math.ceil(scale/2),2**Math.floor(scale/2));assert.equal(m.states,47*2**scale);assert.ok(Number.isFinite(m.shortest));}
});
test('Ch8 PS and optimistic RTDP satisfy independent shortest-path references',()=>{
 const m=maze();
 for(const seed of [1,3]){const r=prioritized({seed});assert.ok(r.found);assert.equal(greedyLength(r.Q,m),14);}
 const r=rtdp({E:200,seed:1});assert.equal(r.V[m.goal],0);assert.equal(r.counts[m.goal],0);assert.equal(r.history.at(-1),14);assert.equal(r.relevant.length,18);
 for(let s=0;s<r.V.length;s++)if(!m.walls.has(s)&&Number.isFinite(m.distance[s]))assert.ok(r.V[s]>=-m.distance[s]-1e-12,'optimistic V remains an upper bound');
});
test('Ch8 sample mean: skew can increase variance; full expected result arrives after b work',()=>{
 const base={mode:1,b:100,p:.01,budget:1};const a=budget.run(base),b=budget.run({...base,p:.5});
 const va=a.plots[0].series[0].y.at(-1),vb=b.plots[0].series[0].y.at(-1);assert.ok(Math.abs(va**2-.0099/100)<1e-14);assert.ok(Math.abs(vb**2-.25/100)<1e-14);assert.ok(vb>va);
 assert.equal(budget.run({...base,budget:.5}).stats[2].value.startsWith('hayır'),true);assert.equal(a.stats[2].value.startsWith('evet'),true);
});
test('Ch8 trajectory: full source budget and browser data remain tied to actual measurements',()=>{
 const data=json('measurements_ch8_trajectory.json');assert.equal(data.runs,200);assert.equal(data.cases.reduce((s,c)=>s+c.actual_backups,0),184000000);
 for(const c of data.cases){assert.equal(c.steps_per_method,20*c.S);assert.equal(c.actual_backups,2*200*c.steps_per_method);assert.ok(c.max_policy_Bellman_residual<1.11e-12);const out=trajectory.run({size:c.S===10000?1:0,branch:{1:1,3:2,10:3}[c.b],zoom:0,band:0});for(let j=0;j<2;j++)assert.deepEqual(out.plots[0].series[j].y,c.mean[j]);}
});
