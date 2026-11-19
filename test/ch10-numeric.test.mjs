// Independent saved C++ runs, SciPy 44-state oracle and episode-wide Python arrays.
import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {accessOptimal,accessLearn,averageEvaluate,gainSequence} from '../site/js/average-core.js';
import {mountainRun,sourceTiles,mcStep} from '../site/js/mountain-core.js';
import {rng} from '../site/js/labs/plot.js';
import returnbias from '../site/js/labs/returnbias.js';
const read=name=>JSON.parse(readFileSync(new URL('../notes/sutton-2018/measurements_ch10_'+name+'.json',import.meta.url),'utf8'));
const E=read('exact'),A=read('access'),I=read('independent');
const close=(a,b,tol=1e-10)=>{assert.ok(Number.isFinite(a)&&Math.abs(a-b)<=tol,`${a} ≠ ${b}`);};
test('chapter10: 11-state JS policy iteration equals independent 44-state SciPy model',()=>{for(const z of E.models){const r=accessOptimal(.06,z.epsilon,z.release_new);close(r.rho,z.full44_gain);for(let f=0;f<11;f++){close(r.h[f],z.h_before_priority[f]);for(let a=0;a<2;a++)close(r.D[f][a].reduce((x,y)=>x+y,0),1);assert.deepEqual(r.greedy[f],z.greedy[f]);}}});
test('chapter10: all eight 2m-step SARSA runs match C++ Q, gain, visits and reward',()=>{for(const z of A.cases){const r=accessLearn({steps:2000000,p:.06,alpha:.01,beta:.01,eps:.1,releaseNew:z.release_new},rng(z.seed));close(r.rho,z.rho);close(r.rewardMean,z.reward_mean);r.Q.forEach((q,i)=>close(q,z.Q[i]));assert.deepEqual(r.visits,z.visits);close(r.Q.reduce((a,b)=>a+b,0),r.rho,2e-10);assert.equal(r.visits.reduce((a,b)=>a+b,0),2000000);}});
test('chapter10: n-step Mountain Car ring matches independent full-episode arrays, including budget cuts',()=>{for(const z of I.online_episode_array_fixtures){const r=mountainRun(z.params);assert.deepEqual(r.lengths,z.result.lengths);assert.equal(r.actual_steps,z.result.actual_steps);assert.equal(r.cutoff,z.result.cutoff);assert.equal(r.unfinished_steps,z.result.unfinished_steps);r.w.forEach((v,i)=>close(v,z.result.w[i],2e-10));}assert.deepEqual(mcStep(-1.2,-.07,0),[-1.2,0]);assert.equal(mcStep(.49,.02,2)[0],.5);for(const x of[-1.2,.5])for(const v of[-.07,.07])for(let a=0;a<3;a++){const ids=sourceTiles(x,v,a);assert.equal(new Set(ids).size,8);assert.ok(ids.every(i=>i>=0&&i<2640));}});
test('chapter10: periodic Poisson and Abel limits preserve normalizations',()=>{const z=averageEvaluate([[0,1,0],[0,0,1],[1,0,0]],[0,0,1]);close(z.rho,1/3);z.h.forEach((v,i)=>close(v,i/3));const b=returnbias.run({kind:1,T:24,g:.999,stay:0},rng(1));close(b.plots[1].series[1].y[0],1/(2*1.999));close(b.plots[1].series[0].y[0],.25);const c=returnbias.run({kind:0,T:24,g:.95,stay:0},rng(1));close(c.plots[0].series[0].y.at(-1),0);close(c.plots[1].series[0].y[0],-1/3);});
test('chapter10: gain trace removes initial coefficient only with n normalization',()=>{for(const n of[1,2,8])for(const initial of[-100,10]){const c=gainSequence(n,.1,initial,Array(20).fill(2*n));close(c[2][1],2);close(c[1][1],initial+n*(2-initial));c[2].slice(1).forEach(v=>close(v,2));}});
