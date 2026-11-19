"""Chapter10 read-only browser regression. Requires Playwright and Chromium."""
from pathlib import Path
import json,argparse
from playwright.sync_api import sync_playwright
parser=argparse.ArgumentParser();parser.add_argument('--base',default='http://localhost:8000');parser.add_argument('--out',default='/tmp/ch10-shots');args=parser.parse_args()
O=Path(args.out);O.mkdir(parents=True,exist_ok=True)
with sync_playwright() as p:
 b=p.chromium.launch(executable_path='/usr/bin/chromium',headless=True,args=['--no-sandbox'])
 for mod,labs,figs,cards in [('m11-6-dqn',[('l10',5),('l11',3)],list(range(17,21)),list(range(15,18))),('m11-13-average-reward',[('l1',4),('l2',4),('l3',3),('l4',4)],list(range(4)),list(range(1,10)))]:
  for w in [1280,390]:
   page=b.new_page(viewport=dict(width=w,height=900));page.set_default_timeout(120000);errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
   page.goto(f'{args.base.rstrip(chr(47))}/{mod}.html',wait_until='networkidle');page.wait_for_selector('#lab-l1 svg',state='attached');page.evaluate('document.fonts.ready')
   page.evaluate('''() => {for(const e of document.querySelectorAll('body *'))if(['fixed','sticky'].includes(getComputedStyle(e).position))e.style.visibility='hidden'}''')
   for i in figs:page.locator('figure.fig').nth(i).screenshot(path=str(O/f'{w}-{mod}-fig{i+1}.png'))
   for lab,n in labs:
    for i in range(n):
     page.locator(f'#lab-{lab} [data-preset="{i}"]').click();page.wait_for_timeout(400)
     page.locator(f'#lab-{lab} .lab-ui').screenshot(path=str(O/f'{w}-{mod}-{lab}-{i}.png'))
     issue=page.evaluate('document.documentElement.scrollWidth-innerWidth')
     print(json.dumps(dict(mod=mod,width=w,lab=lab,preset=i,overflow=issue)),flush=True)
     assert issue==0
   for i in cards:
    page.locator(f'#pr-p{i} .pr-sol summary').click();page.locator(f'#pr-p{i}').screenshot(path=str(O/f'{w}-{mod}-p{i}.png'))
   result=page.evaluate('''() => ({width:innerWidth,overflow:document.documentElement.scrollWidth-innerWidth,badMath:[...document.querySelectorAll('.katex')].filter(e=>!e.closest('.katex-display')).filter(e=>{const r=e.getBoundingClientRect();const t=e.closest('table'),p=t?.parentElement;if(t&&(t.scrollWidth>t.clientWidth+1||(p&&p.scrollWidth>p.clientWidth+1)))return false;return r.width&&(r.right>innerWidth+1||r.left<0)}).map(e=>e.innerText.slice(0,100)),mathErrors:[...document.querySelectorAll('.katex-error')].map(e=>e.textContent),duplicates:[...document.querySelectorAll('[id]')].map(e=>e.id).filter((id,i,a)=>a.indexOf(id)!==i)})''');result.update(pageerrors=errors,mod=mod);print(json.dumps(result,ensure_ascii=False),flush=True)
   assert result['overflow']==0 and not result['badMath'] and not result['mathErrors'] and not result['duplicates'] and not errors
   page.close()
 b.close()
