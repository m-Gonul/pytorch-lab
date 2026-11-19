from pathlib import Path
import json,argparse
from playwright.sync_api import sync_playwright
ap=argparse.ArgumentParser();ap.add_argument('--base',default='http://localhost:8000');ap.add_argument('--out',default='/tmp/ch11-shots');a=ap.parse_args();O=Path(a.out);O.mkdir(parents=True,exist_ok=True)
with sync_playwright()as p:
 b=p.chromium.launch(executable_path='/usr/bin/chromium',args=['--no-sandbox'])
 for width in [1280,390]:
  page=b.new_page(viewport=dict(width=width,height=900));page.set_default_timeout(120000);errs=[];page.on('pageerror',lambda e:errs.append(str(e)));page.goto(a.base+'/m11-6-dqn.html',wait_until='networkidle');page.evaluate('document.fonts.ready');page.evaluate("() => {for(const e of document.querySelectorAll('body *'))if(['fixed','sticky'].includes(getComputedStyle(e).position))e.style.visibility='hidden'}")
  for lab,n in [('l12',9),('l13',4),('l14',6),('l15',5)]:
   for i in range(n):
    sec=page.locator('#lab-'+lab);sec.locator(f'[data-preset="{i}"]').click();page.wait_for_timeout(250);sec.locator('.lab-ui').screenshot(path=str(O/f'{width}-{lab}-{i}.png'));issue=sec.locator('.lab-issues').inner_text().strip();assert not issue,(lab,i,issue)
    ov=page.evaluate('document.documentElement.scrollWidth-innerWidth');assert ov==0,(width,lab,i,ov);print(json.dumps(dict(width=width,lab=lab,preset=i,overflow=ov)),flush=True)
  for i in range(18,26):
   card=page.locator('#pr-p'+str(i));card.locator('.pr-sol summary').click();card.screenshot(path=str(O/f'{width}-p{i}.png'))
  bad=[]
  for i in range(21,33):
   fig=page.locator('figure.fig').nth(i);fig.scroll_into_view_if_needed();page.wait_for_timeout(60);fig.screenshot(path=str(O/f'{width}-fig{i+1}.png'))
   bad+=fig.evaluate('''f=>[...f.querySelectorAll('svg text')].map(e=>{const r=e.getBBox(),v=e.ownerSVGElement.viewBox.baseVal;return r.x<-.5||r.y<-.5||r.x+r.width>v.width+.5||r.y+r.height>v.height+.5?e.textContent:null}).filter(Boolean)''')
  result=page.evaluate('''()=>({overflow:document.documentElement.scrollWidth-innerWidth,mathErrors:[...document.querySelectorAll('.katex-error')].map(e=>e.textContent),badInline:[...document.querySelectorAll('.katex-base')].filter(e=>!e.closest('.math-display')).filter(e=>{const r=e.getBoundingClientRect();if(e.closest('table'))return false;for(let p=e.parentElement;p&&p!==document.body;p=p.parentElement)if(['auto','scroll','hidden','clip'].includes(getComputedStyle(p).overflowX))return false;return r.width&&(r.right>innerWidth+1||r.left<0)}).map(e=>e.innerText.slice(0,100)),duplicates:[...document.querySelectorAll('[id]')].map(e=>e.id).filter((id,i,a)=>a.indexOf(id)!==i)})''');result.update(width=width,SVGbad=bad,pageerrors=errs);print(json.dumps(result,ensure_ascii=False),flush=True);assert not any(result[k]for k in ['overflow','mathErrors','badInline','duplicates','SVGbad','pageerrors']),result;page.close()
 b.close()
