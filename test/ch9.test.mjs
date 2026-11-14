import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {dot,smUpdate,emphasis,interestRun,phi,tileIDs,overlap} from '../site/js/fa-core.js';
import E from '../site/js/data/ch9-exact.js';
import W from '../site/js/data/ch9-walk.js';
import kernel from '../site/js/labs/kernelvalue.js';
import {defaults,evaluate} from '../site/js/labs/runtime.js';
const read=name=>JSON.parse(readFileSync(new URL(`../notes/sutton-2018/${name}`,import.meta.url),'utf8'));
const near=(a,b,tol=1e-10)=>assert.ok(Math.abs(a-b)<tol,`${a} ≠ ${b}`);
test('Ch9: exact Bellman/occupancy equations and distinct MC/TD objectives',()=>{
 const sums=a=>{const p=[0];for(const x of a)p.push(p.at(-1)+x);return p;},p=sums(E.V),m=sums(E.mu);let total=0;
 for(let s=1;s<=1000;s++){
  const Pv=(p[s-1]-p[Math.max(0,s-101)]+p[Math.min(1000,s+100)]-p[s])/200;
  near(E.V[s-1],E.reward[s-1]+Pv,1e-12);
  const Pm=(m[s-1]-m[Math.max(0,s-101)]+m[Math.min(1000,s+100)]-m[s])/200;
  near(E.mu[s-1],Pm+(s===500?1/E.expected_episode_length:0),1e-13);total+=E.mu[s-1];
 }
 near(total,1,1e-12);near(E.expected_episode_length,83.1524486824,1e-9);
 for(const g of Object.values(E.groups)){
  for(let i=0;i<g.b.length;i++)near(dot(g.A[i],g.TD_weights),g.b[i],1e-12);
  assert.ok(g.MC_weighted_RMSE<g.TD_weighted_RMSE);
 }
});
test('Ch9: nonsymmetric Sherman–Morrison matches independent 2x2 inverse; singular guard',()=>{
 let A=[[.25,0],[0,.25]],B=[[4,0],[0,4]];
 const transitions=[[[1,0],[0,2]],[[0,2],[2,1]],[[2,1],[0,0]]];
 for(const[x,z]of transitions){const y=x.map((a,j)=>a-.7*z[j]);A=A.map((r,i)=>r.map((a,j)=>a+x[i]*y[j]));const u=smUpdate(B,x,z,.7);assert.ok(u.ok);B=u.inv;const det=A[0][0]*A[1][1]-A[0][1]*A[1][0],exact=[[A[1][1],-A[0][1]],[-A[1][0],A[0][0]]].map(r=>r.map(a=>a/det));for(let i=0;i<2;i++)for(let j=0;j<2;j++)near(B[i][j],exact[i][j]);}
 assert.notEqual(A[0][1],A[1][0]);const bad=smUpdate([[8]],[1],[2.25],.5);assert.equal(bad.ok,false);assert.equal(bad.den,0);
});
test('Ch9: raw targets differ from OLS; correct learned kernel ridge coefficients',()=>{
 const out=evaluate(kernel,{...defaults(kernel),mode:1,lambda:1},1).out;
 const lines=out.plots[0].series;near(lines[0].y.at(-1),.6);near(lines[1].y.at(-1),3);near(lines[2].y.at(-1),1/3);near(lines[3].y.at(-1),.5);
 const beta=[5/6,-1/3];near(dot([3,6],beta),.5);near(dot([1,2],beta),1/6);
});
test('Ch9: feature geometry, hash normalization and interest bootstrap support',()=>{
 for(const kind of [0,1,2]){const x=phi(.4,kind,5,.15),alpha=1/dot(x,x);near(alpha*dot(x,x),1);}
 const t=tileIDs(.4,.5,8,true,.5);assert.equal(new Set(t).size,8);assert.equal(overlap(t,t),8);near(1/8*overlap(t,t),1);
 assert.deepEqual(emphasis(2,2),[1,0,1,0]);assert.deepEqual(emphasis(4,2),[1,0,0,0]);
 const expected=[[4,0],[3.5,1.5],[4,2]];
 for(const [n,mode,w]of [[4,2,expected[0]],[2,0,expected[1]],[2,2,expected[2]],[2,1,[2,0]]]){
  const a=interestRun(n,mode,10000,.2).w;for(let j=0;j<2;j++)near(a[j],w[j],.001);
 }
 near(.9**10,.3486784401,1e-12);
});
test('Ch9: actual full source budgets and independent C++ fixtures stay attached to browser data',()=>{
 assert.deepEqual(W,read('measurements_ch9_walk.json'));
 assert.equal(W.mc.episodes,100000);assert.equal(W.mc.steps,8302094);
 assert.equal(W.nstep.runs,100);assert.equal(W.nstep.episodes,10);assert.equal(W.nstep.results.mean.length,210);
 assert.equal(W.bases.runs,30);assert.equal(W.bases.episodes,5000);assert.equal(W.bases.distinct_path_steps,12465376);assert.equal(W.bases.curves.length,8);
 const refs=read('measurements_ch9_independent.json');assert.ok(refs.nstep_210_max_difference<1e-11);assert.ok(refs.bases_8_max_difference<1e-11);
});
