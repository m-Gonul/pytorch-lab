"""Actual A/B regression checks. Serve dist; pass base URL and Chromium executable.
python notes/sutton-2018/check_ch9_browser.py http://localhost:8010 /usr/bin/chromium
"""
import sys,json
from playwright.sync_api import sync_playwright
base=sys.argv[1].rstrip('/');exe=sys.argv[2]if len(sys.argv)>2 else None
with sync_playwright()as p:
 browser=p.chromium.launch(**({'executable_path':exe}if exe else{}),headless=True,args=['--no-sandbox'])
 for width in [1280,390]:
  page=browser.new_page(viewport=dict(width=width,height=900));errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  page.goto(base+'/m11-6-dqn.html',wait_until='networkidle');page.wait_for_selector('#lab-l3 svg')
  def mode(lab,side,value):page.locator(f'#lab-{lab} [data-set="{side}"] select[data-s="mode"]').select_option(str(value));page.wait_for_timeout(100)
  # Different plot counts AND units. Both directions must retain all plots.
  page.locator('#lab-l3 [data-preset="0"]').click();page.locator('#lab-l3 [data-k="cmp"]').check();mode('l3','B',1)
  for a,b in [(0,1),(1,0)]:
   mode('l3','A',a);mode('l3','B',b)
   assert page.locator('#lab-l3 .lab-out svg').count()==3
   rows=page.locator('#lab-l3 .lab-tbl tbody tr').all_text_contents();assert len(rows)==8
   cells=page.locator('#lab-l3 .lab-tbl tbody tr').evaluate_all('(rs)=>rs.map(r=>[...r.cells].map(c=>c.textContent))')
   # An unavailable metric must not inherit another metric's value by row index.
   for label,av,bv in cells:
    if label=='Seçili n':assert(av=='—')==(a==0)and(bv=='—')==(b==0)
  # Every new lab's default A/B route, including sparse image and kernel cases.
  for lab in ['l4','l5','l6','l7','l8','l9']:
   page.locator(f'#lab-{lab} [data-k="cmp"]').check();page.wait_for_timeout(100)
   assert page.locator(f'#lab-{lab} .lab-out svg').count()>=1
  images=page.locator('#lab-l6 .lab-pair image').evaluate_all('(xs)=>xs.map(x=>({w:+x.getAttribute("width"),h:+x.getAttribute("height")}))')
  assert len(images)==2 and all(abs(x['w']-x['h'])<1e-9 for x in images)
  # A single image also retains the square state-domain aspect.
  page.locator('#lab-l6 [data-k="cmp"]').uncheck();page.wait_for_timeout(100)
  one=page.locator('#lab-l6 image').evaluate('(x)=>({w:+x.getAttribute("width"),h:+x.getAttribute("height")})');assert one['w']==one['h']
  overflow=page.evaluate('document.documentElement.scrollWidth-innerWidth');assert overflow==0 and not errors
  print(json.dumps(dict(width=width,AB_cases=8,square_images=True,overflow=overflow,errors=errors)),flush=True);page.close()
 browser.close()
