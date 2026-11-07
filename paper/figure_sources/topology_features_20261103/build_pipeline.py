#!/usr/bin/env python3
"""GFP-inspired visual grammar adapted to static account-level TopoAML.

Transaction rows are aggregated to account rows. All tables and graphs are
schematic. This is neither a streaming procedure nor a temporal split diagram.
"""
from pathlib import Path
import os
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'output'
QA = ROOT / 'qa'
OUT.mkdir(exist_ok=True)
QA.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(QA / 'mplconfig'))
os.environ.setdefault('XDG_CACHE_HOME', str(QA / 'cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch, Circle, Polygon
from PIL import Image

# Portable fixed style; do not load workstation-local skill files.
matplotlib.rcParams.update({'font.family':'sans-serif',
    'font.sans-serif':['Arial','DejaVu Sans'], 'font.size':22,
    'axes.labelsize':22, 'axes.titlesize':22, 'legend.fontsize':18,
    'xtick.labelsize':14, 'ytick.labelsize':14,
    'xtick.direction':'out', 'ytick.direction':'out',
    'axes.spines.top':False, 'axes.spines.right':False,
    'axes.spines.left':False, 'axes.spines.bottom':False,
    'legend.frameon':False, 'savefig.dpi':300,
    'pdf.fonttype':42, 'ps.fonttype':42})
matplotlib.rcParams.update({'savefig.bbox':None, 'svg.fonttype':'none',
                           'svg.hashsalt':'TopoAML-GFP-grammar'})
W,H=1800,1020
BLUE='#173F73'; ORANGE='#C76418'; RED='#982F3A'
INK='#202A35'; GREY='#536171'; LINE='#CFD8E2'; PALE='#F5F8FC'
ORANGE_PALE='#FCF6EF'; NEUTRAL='#F6F7F8'
TITLE_SIZE=22; MODULE_SIZE=18; BODY_SIZE=14; PANEL_SIZE=24
fig=plt.figure(figsize=(18,10.2),dpi=100,facecolor='white')
ax=fig.add_axes([0,0,1,1]); ax.set(xlim=(0,W),ylim=(H,0)); ax.axis('off')
texts=[]; bounded_labels=[]

def text(x,y,s,size=MODULE_SIZE,color=INK,ha='left',weight='normal',bounds=None,**kw):
    t=ax.text(x,y,s,fontsize=size,color=color,ha=ha,va='center',
              fontweight=weight,linespacing=1.25,**kw); texts.append(t)
    if bounds is not None: bounded_labels.append((t,bounds))
    return t

def line(a,b,color=GREY,lw=1.5,ls='-',z=1):
    ax.plot([a[0],b[0]],[a[1],b[1]],color=color,lw=lw,ls=ls,
            solid_capstyle='round',zorder=z)

def arrow(a,b,color=BLUE,lw=2,rad=0):
    ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=17,
                 shrinkA=rad,shrinkB=rad,color=color,linewidth=lw,zorder=2))

def box(x,y,w,h,color=BLUE,fill='white',rounded=True,lw=1.8,ls='-'):
    patch=(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=10',
                         facecolor=fill,edgecolor=color,lw=lw,linestyle=ls)
           if rounded else Rectangle((x,y),w,h,facecolor=fill,edgecolor=color,lw=lw,linestyle=ls))
    ax.add_patch(patch)

def node(p,filled=False,color=BLUE,r=8):
    ax.add_patch(Circle(p,r,facecolor=color if filled else 'white',edgecolor=color,lw=1.8,zorder=4))

def graph(p,edges,centre='v',directed=True,color=BLUE):
    for a,b in edges:
        if directed: arrow(p[a],p[b],color=color,lw=1.5,rad=8)
        else: line(p[a],p[b],color=color,lw=1.5)
    for k,pt in p.items(): node(pt,k==centre,color=color)

text(28,37,'TopoAML: transaction records become explicit account representations',size=TITLE_SIZE,color=BLUE,weight='bold')
text(1772,77,'Static, transductive account classification',size=14,color=GREY,ha='right')

# A: the GFP-style grammar, with the unit changing from transactions to accounts.
text(28,105,'A',size=PANEL_SIZE,weight='bold',color=BLUE)
text(64,105,'Observed transactions',size=TITLE_SIZE,weight='bold')
tx_x,tx_y=28,146
widths=[70,70,95,90,100]; headers=['Payer','Payee','Amount','Currency','Format']
rows=[['A','B','…','USD','Wire'],['A','C','…','USD','Wire'],['D','A','…','EUR','ACH']]
xx=tx_x
for ww,hd in zip(widths,headers):
    box(xx,tx_y,ww,42,color=LINE,fill=NEUTRAL,rounded=False,lw=1)
    text(xx+ww/2,tx_y+21,hd,size=14,ha='center',weight='bold')
    xx+=ww
for rr,row in enumerate(rows):
    xx=tx_x
    for ww,val in zip(widths,row):
        box(xx,tx_y+42*(rr+1),ww,42,color=LINE,rounded=False,lw=1)
        text(xx+ww/2,tx_y+42*(rr+1)+21,val,size=14,ha='center'); xx+=ww
text(240,341,'One row per transaction',size=14,ha='center',color=GREY)

arrow((473,230),(516,230),color=GREY)
box(534,176,345,108,color=BLUE,lw=2.3)
text(706.5,209,'TopoAML',size=TITLE_SIZE,weight='bold',ha='center',color=BLUE,bounds=(534,176,345,108))
text(706.5,249,'account feature extractor',size=MODULE_SIZE,ha='center',color=BLUE,bounds=(534,176,345,108))
arrow((898,230),(955,230),color=GREY)

text(986,105,'Account feature matrix',size=TITLE_SIZE,weight='bold')
mx,my=986,146
mwidths=[102,142,142,80,80,80,142]
mh=['Account','T\n16','G\n16','M\n5','K\n3','E\n4','P\n12']
mf=[NEUTRAL,NEUTRAL,PALE,PALE,PALE,PALE,ORANGE_PALE]
mc=[GREY,GREY,BLUE,BLUE,BLUE,BLUE,ORANGE]
xx=mx
for ww,hd,fill,col in zip(mwidths,mh,mf,mc):
    box(xx,my,ww,60,color=LINE,fill=fill,rounded=False,lw=1)
    text(xx+ww/2,my+30,hd,size=14,ha='center',color=col,weight='bold'); xx+=ww
for rr,acct in enumerate(['A','B','…']):
    xx=mx
    for cc,(ww,fill,col) in enumerate(zip(mwidths,mf,mc)):
        box(xx,my+60+36*rr,ww,36,color=LINE,fill='white' if cc==0 else fill,rounded=False,lw=1)
        if cc==0: text(xx+ww/2,my+78+36*rr,acct,size=14,ha='center')
        else:
            for frac in [0.25,0.5,0.75]: line((xx+ww*frac,my+60+36*rr),(xx+ww*frac,my+96+36*rr),color=LINE,lw=0.7)
        xx+=ww
text(1370,341,'One row per account · Full = 56 columns',size=14,ha='center',color=GREY)

# Four input selections are nested column subsets, not concatenated matrices.
text(1370,399,'Compare column subsets',size=MODULE_SIZE,color=BLUE,ha='center',weight='bold')
for yy,label in [(439,'Base: T (16)'),(477,'Base+G: T + G (32)'),
                 (515,'Base+Struct: T + S (44)'),(553,'Full: T + S + P (56)')]:
    text(1124,yy,label,size=14)
arrow((1370,359),(1370,383),color=GREY)
arrow((1370,573),(1370,613),color=GREY)
box(1136,628,565,118,color=RED,lw=2)
text(1418,662,'Tabular classifier',size=TITLE_SIZE,weight='bold',color=RED,ha='center',bounds=(1136,628,565,118))
text(1418,706,'LightGBM · XGBoost · random forest',size=MODULE_SIZE,ha='center',bounds=(1136,628,565,118))
arrow((1418,761),(1418,800),color=GREY)
box(1214,815,410,112,color=RED,fill='#FDF9FA',lw=1.8)
text(1419,848,'Account risk ranking',size=TITLE_SIZE,weight='bold',color=RED,ha='center',bounds=(1214,815,410,112))
text(1419,891,'Top-budget account alerts',size=MODULE_SIZE,ha='center',bounds=(1214,815,410,112))
text(1419,968,'Shared account splits; tune each input independently',size=14,ha='center',color=GREY)

# B: internal extraction expands the processor instead of another text-only flow.
line((550,292),(400,395),color=LINE,lw=1.5,ls='--')
line((862,292),(1092,395),color=LINE,lw=1.5,ls='--')
box(28,397,1064,566,color=BLUE,lw=1.5,ls='--')
text(48,431,'B',size=PANEL_SIZE,weight='bold',color=BLUE)
text(86,431,'Inside the account feature extractor',size=TITLE_SIZE,weight='bold',color=BLUE)
text(86,469,'Fixed graph from all observed transactions; labels never enter extraction',size=14,color=GREY)

# Transaction branch is explicitly distinct from graph-derived branches.
box(52,516,264,411,color=GREY,lw=1.3)
text(184,549,'Transaction layer T',size=MODULE_SIZE,weight='bold',ha='center',color=GREY,bounds=(52,516,264,411))
text(184,582,'16 columns',size=14,ha='center',color=GREY)
for i in range(4):
    box(93,627+32*i,182,32,color=LINE,fill=NEUTRAL if i==0 else 'white',rounded=False,lw=1)
    for z in [0.33,0.66]: line((93+182*z,627+32*i),(93+182*z,659+32*i),color=LINE,lw=0.7)
text(184,794,'Counts / amounts',size=14,ha='center')
text(184,826,'Counterparty diversity',size=14,ha='center')
text(184,858,'Currencies / formats',size=14,ha='center')
text(184,900,'Repeated / self transfers kept',size=14,ha='center',color=GREY)

text(691,521,'Simple account graphs D and U',size=MODULE_SIZE,weight='bold',ha='center',color=BLUE)
gp={'v':(650,579),'a':(580,555),'b':(580,614),'c':(723,548),'d':(723,614),'e':(799,579)}
graph(gp,[('a','v'),('b','v'),('v','c'),('v','d'),('c','e'),('d','e')],centre='v')
text(470,580,'D',size=14,color=BLUE)
text(932,580,'U: ignore direction',size=14,color=BLUE,ha='center')
arrow((530,623),(530,659),color=BLUE)
arrow((912,623),(912,659),color=ORANGE)

box(344,674,370,253,color=BLUE,lw=1.3)
text(529,709,'Directed structural S · 28',size=MODULE_SIZE,weight='bold',ha='center',color=BLUE,bounds=(344,674,370,253))
smallp={'v':(391,781),'a':(453,747),'b':(453,815),'w':(514,781)}
graph(smallp,[('v','a'),('v','b'),('a','w'),('b','w')],centre='v')
text(604,766,'G 16',size=14,ha='center',color=BLUE)
text(604,801,'M 5 · K 3 · E 4',size=14,ha='center',color=BLUE)
text(529,865,'Directed reach / reconvergence / hubs',size=14,ha='center')
text(529,900,'Untruncated local graph',size=14,ha='center',color=GREY)

box(730,674,346,253,color=ORANGE,lw=1.3)
text(903,709,'Persistent homology P · 12',size=MODULE_SIZE,weight='bold',ha='center',color=ORANGE,bounds=(730,674,346,253))
sq=[(780,754),(837,754),(837,811),(780,811)]
for i in range(4): line(sq[i],sq[(i+1)%4],color=ORANGE,lw=1.8)
for pt in sq: node(pt,color=ORANGE,r=6)
arrow((861,782),(902,782),color=GREY,lw=1.3)
for yy in [756,776,796]: line((926,yy),(1010,yy),color=ORANGE,lw=2)
text(903,865,'Radii 2 / 3 · sampled H0 / H1',size=14,ha='center')
text(903,900,'Shared BFS prefix · cap 40',size=14,ha='center',color=GREY)

text(28,997,'Feature blocks are schematic; S + P = 40 topology-aware descriptors. E is a deterministic transform of G.',size=14,color=GREY)

fig.canvas.draw(); renderer=fig.canvas.get_renderer()
boxes=[(t,t.get_window_extent(renderer)) for t in texts]
overlaps=[]
for i,(ta,ba) in enumerate(boxes):
    for tb,bb in boxes[i+1:]:
        if ba.overlaps(bb): overlaps.append([ta.get_text(),tb.get_text()])
outside=[t.get_text() for t,b in boxes if b.x0<0 or b.y0<0 or b.x1>W or b.y1>H]
label_overflow=[]
for t,(x,y,w,h) in bounded_labels:
    b=t.get_window_extent(renderer)
    corners=ax.transData.transform([(x,y+h),(x+w,y)])
    x0,y0=corners[0]; x1,y1=corners[1]
    if b.x0<x0+5 or b.x1>x1-5 or b.y0<y0+5 or b.y1>y1-5:
        label_overflow.append({'text':t.get_text(),'text_width_px':round(b.width,1),
                               'box_width_px':w,'padding_px':5})
stem='TopoAML_GFP_style_framework'
fig.savefig(OUT/f'{stem}.svg',metadata={'Title':'TopoAML static account-level feature enrichment',
             'Description':'Original artwork using feature-table enrichment and extractor-detail visual grammar inspired by Blanusa et al., GFP (2024). All tables and graphs are schematic.'})
fig.savefig(OUT/f'{stem}.pdf')
fig.savefig(OUT/f'{stem}_preview.png',dpi=100)
fig.savefig(OUT/f'{stem}.png',dpi=300)
preview=Image.open(OUT/f'{stem}_preview.png')
for name,crop in {'pipeline_panel_a':(16,80,1775,355),
                  'pipeline_panel_b':(16,388,1102,970),
                  'pipeline_classifier_output':(1112,619,1725,982)}.items():
    preview.crop(crop).save(QA/f'{name}.png')
svg=ET.parse(OUT/f'{stem}.svg').getroot()
report={'passed':not overlaps and not outside and not label_overflow,
        'text_overlaps':overlaps,'outside_canvas':outside,'label_box_overflow':label_overflow,
        'svg_text_elements':sum(e.tag.endswith('text') for e in svg.iter()),
        'svg_contains_raster_images':any(e.tag.endswith('image') for e in svg.iter()),
        'font':'Arial','typography_pt':{'titles':TITLE_SIZE,'module_titles':MODULE_SIZE,
                                      'body':BODY_SIZE,'panel_letters':PANEL_SIZE},
        'background_fills':{'structural':PALE,'PH':ORANGE_PALE,'neutral':NEUTRAL,
                            'risk_output':'#FDF9FA'},
        'foreground_alpha':1,'account_matrix_rows':True,'static_transductive':True,
        'feature_counts':{'T':16,'G':16,'M':5,'K':3,'E':4,'P':12},
        'topology_descriptors':40,'full_columns':56,'source':'../FlowTopo_manuscript.tex:225-307',
        'reference':'https://arxiv.org/abs/2402.08593',
        'review_crops':['pipeline_panel_a.png','pipeline_panel_b.png',
                        'pipeline_classifier_output.png'],
        'visual_review':'Render-only checks above; dated visual inspection is recorded separately in qa/.'}
(QA/'pipeline_validation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
