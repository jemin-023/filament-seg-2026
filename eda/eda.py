import json, sys, os, collections, statistics as st
root=sys.argv[1]
d=json.load(open(f"{root}/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
print("keys",list(d)); print("info",d.get("info")); print("cats",d["categories"])
ims=d["images"]; anns=d["annotations"]
print("n images entries",len(ims),"n anns",len(anns))
print("image example",ims[0]); a=anns[0]; print("ann keys",list(a), {k:(v if k not in('segmentation','spine') else str(v)[:120]) for k,v in a.items()})
fn=collections.Counter(i["file_name"] for i in ims)
print("unique files",len(fn),"annotations-per-file dist",collections.Counter(fn.values()))
prefix=collections.Counter(i["id"].split("-")[0] for i in ims); print("n annotator-batch prefixes",len(prefix), prefix.most_common(8))
inst=collections.Counter(f[14:16] for f in fn); print("train instruments",inst)
yrs=collections.Counter(f[:4] for f in fn); print("train years",sorted(yrs.items()))
tf=os.listdir(f"{root}/test/test_images"); print("n test",len(tf))
print("test instruments",collections.Counter(f[14:16] for f in tf)); print("test years",sorted(collections.Counter(f[:4] for f in tf).items()))
print("test file overlap with train",len(set(tf)&set(fn)))
tr_dates=set(f[:8] for f in fn); te_dates=set(f[:8] for f in tf); print("test dates also in train",len(te_dates&tr_dates),"/",len(te_dates))
per=collections.Counter(a["image_id"] for a in anns)
ids=[i["id"] for i in ims]; cnt=[per.get(i,0) for i in ids]
print("filaments per annotation-entry: mean %.1f median %s max %s zero %d"%(st.mean(cnt),st.median(cnt),max(cnt),sum(c==0 for c in cnt)))
ar=sorted(a["area"] for a in anns); n=len(ar); print("area pct",[ar[int(n*q)] for q in (0,.05,.25,.5,.75,.95,.99)], "max",ar[-1])
print("cat counts",collections.Counter(a["category_id"] for a in anns))
print("multi-poly segs",sum(len(a["segmentation"])>1 for a in anns))
import math
el=[]
for a in anns:
    w,h=a["bbox"][2],a["bbox"][3]; el.append(a["area"]/max(1,w*h))
el.sort(); print("area/bbox fill pct",[round(el[int(n*q)],2) for q in (.05,.25,.5,.75,.95)])
