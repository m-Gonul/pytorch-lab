"""Chapter10 read-only browser regression. Requires Playwright and Chromium."""
from pathlib import Path
import json,argparse
from playwright.sync_api import sync_playwright
parser=argparse.ArgumentParser();parser.add_argument('--base',default='http://localhost:8000');parser.add_argument('--out',default='/tmp/ch10-svg-shots');args=parser.parse_args()
O=Path(args.out);O.mkdir(parents=True,exist_ok=True)
with sync_playwright()as p:
 b=p.chromium.launch(executable_path='/usr/bin/chromium',args=['--no-sandbox'])
 for mod,start in[('m11-6-dqn',17),('m11-13-average-reward',0)]:
  for w in[1280,390]:
   page=b.new_page(viewport=dict(width=w,height=900));page.goto(f'{args.base.rstrip(chr(47))}/{mod}.html',wait_until='networkidle');page.evaluate('document.fonts.ready');page.evaluate("() => {for(const e of document.querySelectorAll('body *'))if(['fixed','sticky'].includes(getComputedStyle(e).position))e.style.visibility='hidden'}");bad=[]
   for i in range(start,start+4):
    e=page.locator('figure.fig').nth(i);e.scroll_into_view_if_needed();page.wait_for_timeout(60)
    bad+=e.evaluate('''(f)=>[...f.querySelectorAll('svg text')].map(e=>{const r=e.getBBox(),v=e.ownerSVGElement.viewBox.baseVal;return r.x<-.5||r.y<-.5||r.x+r.width>v.width+.5||r.y+r.height>v.height+.5?{text:e.textContent,bounds:[r.x,r.y,r.width,r.height],view:[v.width,v.height]}:null}).filter(Boolean)''')
    if mod=='m11-6-dqn'and i==17:e.screenshot(path=str(O/f'{w}-corrected-valley.png'))
   assert not bad,bad
   print(json.dumps(dict(module=mod,width=w,SVG_text_bounds_errors=bad)),flush=True);page.close()
 b.close()
