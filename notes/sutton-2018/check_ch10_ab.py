"""Chapter10 read-only browser regression. Requires Playwright and Chromium."""
from pathlib import Path
import json,argparse
from playwright.sync_api import sync_playwright
parser=argparse.ArgumentParser();parser.add_argument('--base',default='http://localhost:8000');parser.add_argument('--out',default='/tmp/ch10-ab-shots');args=parser.parse_args()
O=Path(args.out);O.mkdir(parents=True,exist_ok=True)
CASES=[('m11-6-dqn','l10',{'mode':0},{'mode':3,'frame':0}),('m11-6-dqn','l10',{'mode':2},{'mode':1}),('m11-6-dqn','l11',{'n':0},{'n':3,'alpha':.3}),('m11-13-average-reward','l1',{'kind':0},{'kind':2}),('m11-13-average-reward','l1',{'kind':0},{'kind':1}),('m11-13-average-reward','l2',{'steps':10000,'order':0},{'steps':10000,'order':1}),('m11-13-average-reward','l3',{'g':.5},{'g':.9}),('m11-13-average-reward','l4',{'n':2},{'n':1})]
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
