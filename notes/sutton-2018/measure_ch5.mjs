// Actual browser algorithms + cross-check against independent NumPy references.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {rng} from '../../site/js/labs/plot.js';
import {reference,simulate,index} from '../../site/js/labs/blackjackmc.js';
import {loopCurves,blackjackErrors,advancedErrors,binomial} from '../../site/js/labs/importancemc.js';
import {walks,harmonic} from '../../site/js/labs/soapdirichlet.js';
import {buildTrack,trainTrack,trackReference,rollout} from '../../site/js/labs/racetrackmc.js';
const here=path.dirname(fileURLToPath(import.meta.url)),P=JSON.parse(fs.readFileSync(process.argv[2]||path.join(here,'measurements_ch5_python.json'))),out={};
const r=reference(),diff=(a,b)=>Math.max(...a.flat(Infinity).map((v,i)=>Math.abs(v-b.flat(Infinity)[i])));
out.blackjack={reference:r,prediction:simulate(500000,1),control:simulate(1000000,1,true)};
out.cross_language={blackjack_prediction:diff(r.prediction.V,P.blackjack.prediction),blackjack_optimal:diff(r.optimal.V,P.blackjack.optimal),blackjack_Q:diff(r.optimal.Q,P.blackjack.optimal_Q),dealer:diff(r.dealer,P.blackjack.dealer)};
if(Math.max(...Object.values(out.cross_language))>1e-12)throw Error('Blackjack NumPy/JS mismatch');
out.blackjack.prediction_rmse=Math.sqrt(out.blackjack.prediction.V.reduce((s,v,i)=>s+(v-r.prediction.V[i])**2,0)/200);out.blackjack.control_rmse=Math.sqrt(out.blackjack.control.V.reduce((s,v,i)=>s+(v-r.optimal.V[i])**2,0)/200);out.blackjack.agreement=out.blackjack.control.policy.filter((a,i)=>a===r.optimal.policy[i]).length/200;
out.racetrack=[];
for(let k=0;k<2;k++){const m=buildTrack(k),d=trainTrack(k),ref=trackReference(k),ev=trackReference(k,.1,d.policy),p=P.racetrack[k],err=Math.max(...p.flat_state_ids.map((s,i)=>Math.abs(ref.V[s]-p.V[i])));if(err>1e-9)throw Error('Track NumPy/JS mismatch');const traces=m.starts.map((s,i)=>rollout(m,d.policy,s,1+i,0)),optimalTraces=m.starts.map((s,i)=>rollout(m,ref.policy,s,1+i,0));if([...traces,...optimalTraces].some(t=>!t.finished))throw Error('Did not finish');out.racetrack.push({track:k,episodes:50000,seed:1,epsilon:.1,noise:.1,steps:d.steps,returns:d.returns,policy:Array.from(d.policy),learned_value:ev.mean,optimal_value:ref.mean,residual:ref.residual,model_iterations:ref.iterations,numpy_js_maxdiff:err,traces,optimalTraces});}
out.blackjack_importance=blackjackErrors(10000,100,rng(1));out.infinite_variance=loopCurves(100000000,.9,.5,rng(1));out.discount=advancedErrors(10000,100,0,rng(1));out.soap={harmonic:harmonic(),walks:walks(5000,.5,0,.15,1)};
fs.writeFileSync(process.argv[3]||path.join(here,'measurements_ch5_js.json'),JSON.stringify(out,null,2)+'\n');console.log('Cross-language',out.cross_language,'tracks',out.racetrack.map(t=>[t.optimal_value,t.learned_value,t.numpy_js_maxdiff]),'all start trajectories finished');
