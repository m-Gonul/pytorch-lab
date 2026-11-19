"""Chapter11 read-only browser regression. Requires Playwright and Chromium."""
from pathlib import Path
import json,argparse
from playwright.sync_api import sync_playwright
parser=argparse.ArgumentParser();parser.add_argument('--base',default='http://localhost:8000');parser.add_argument('--out',default='/tmp/ch11-ab-shots');args=parser.parse_args()
O=Path(args.out);O.mkdir(parents=True,exist_ok=True)
CASES=[('m11-6-dqn','l12',{'mode':0,'alpha':.01},{'mode':4,'alpha':.01}),('m11-6-dqn','l12',{'mode':1},{'mode':5}),('m11-6-dqn','l12',{'mode':3,'alpha':.03},{'mode':7,'alpha':.03}),('m11-6-dqn','l12',{'mode':8,'N':10000},{'mode':4,'N':1000}),('m11-6-dqn','l13',{'goal':0},{'goal':2,'mu':1}),('m11-6-dqn','l14',{'mode':0,'algorithm':1},{'mode':1,'algorithm':2}),('m11-6-dqn','l14',{'mode':2},{'mode':3}),('m11-6-dqn','l15',{'mode':0},{'mode':1,'T':3}),('m11-6-dqn','l15',{'mode':1,'T':10},{'mode':2,'T':20})]
with sync_playwright()as p:
 b=p.chromium.launch(executable_path='/usr/bin/chromium',args=['--no-sandbox'])
 for w in[1280,390]:
  page=b.new_page(viewport=dict(width=w,height=900));page.set_default_timeout(120000);errs=[];page.on('pageerror',lambda e:errs.append(str(e)));current=None
  for i,(mod,lab,A,B)in enumerate(CASES):
   if mod!=current:page.goto(f'{args.base.rstrip(chr(47))}/{mod}.html',wait_until='networkidle');current=mod
   sec=page.locator('#lab-'+lab);sec.locator('[data-k="reset"]').click();sec.locator('[data-k="cmp"]').check()
   for side,params in [('A',A),('B',B)]:
    for key,value in params.items():
     e=sec.locator(f'[data-set="{side}"] [data-s="{key}"]')
     if e.count():e.select_option(str(value));e.dispatch_event('input')
     else:
      e=sec.locator(f'[data-set="{side}"] [data-n="{key}"]');e.fill(str(value));e.dispatch_event('input')
   page.evaluate("() => {for(const e of document.querySelectorAll('body *'))if(['fixed','sticky'].includes(getComputedStyle(e).position))e.style.visibility='hidden'}");page.wait_for_timeout(350);sec.locator('.lab-ui').screenshot(path=str(O/f'{w}-{i}.png'));ov=page.evaluate('document.documentElement.scrollWidth-innerWidth');assert ov==0,(w,i,ov)
   assert not sec.locator('.lab-issues').inner_text().strip();assert not errs,errs
   print(json.dumps(dict(width=w,case=i,mod=mod,lab=lab,overflow=ov,errors=errs)),flush=True)
  page.close()
 b.close()
