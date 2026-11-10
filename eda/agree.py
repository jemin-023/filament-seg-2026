import json, sys, collections, itertools, numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as ndi
root=sys.argv[1]
d=json.load(open(f"{root}/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
by_img=collections.defaultdict(list)
for a in d["annotations"]: by_img[a["image_id"]].append(a)
files=collections.defaultdict(list)
for i in d["images"]: files[i["file_name"]].append(i["id"])
S=2048
def labmap(anns):
    im=Image.new("I",(S,S),0); dr=ImageDraw.Draw(im)
    for k,a in enumerate(anns,1):
        p=a["segmentation"][0]; dr.polygon(list(zip(p[0::2],p[1::2])),fill=k)
    return np.array(im,dtype=np.int32)
def pq(gt,pr):
    ng,np_=gt.max(),pr.max()
    if ng==0 and np_==0: return 1,0,0,0,1
    j=gt.astype(np.int64)*(np_+1)+pr; c=np.bincount(j.ravel(),minlength=(ng+1)*(np_+1)).reshape(ng+1,np_+1)
    ag=c.sum(1); ap=c.sum(0); inter=c[1:,1:]; union=ag[1:,None]+ap[None,1:]-inter; iou=inter/np.maximum(union,1)
    m=iou>0.5; tp=m.sum(); s=iou[m].sum(); fp=np_-tp; fn=ng-tp
    g=gt>0; p=pr>0; dice=2*(g&p).sum()/max(1,g.sum()+p.sum())
    # one-to-many: gt overlapping >1 pred (any overlap >10% of pred)
    return s/max(1e-9,tp+.5*fp+.5*fn), tp, fp, fn, dice
res=[];touch=[];
for f,ids in files.items():
    if len(ids)<2: continue
    L=[labmap(by_img[i]) for i in ids]
    for a,b in itertools.combinations(range(len(ids)),2): res.append(pq(L[a],L[b]))
    for l in L:
        n=l.max(); cc=ndi.label(l>0)[1]; touch.append((n,cc))
r=np.array(res,dtype=float)
print("pairs",len(r),"inter-annotator PQ mean %.3f median %.3f p25 %.3f p75 %.3f"%(r[:,0].mean(),np.median(r[:,0]),*np.percentile(r[:,0],[25,75])))
print("SQ (mean IoU of matches) %.3f  RQ %.3f"%(r[:,0].sum()/ (r[:,0]/ (r[:,0]+1e-9)).sum() if False else (r[:,0]/( r[:,1]/(r[:,1]+.5*r[:,2]+.5*r[:,3]+1e-9)+1e-9)).mean(), (r[:,1]/(r[:,1]+.5*r[:,2]+.5*r[:,3])).mean()))
print("semantic Dice between annotators mean %.3f"%r[:,4].mean())
print("matched frac of instances: tp/(tp+fn) %.3f"%(r[:,1].sum()/(r[:,1]+r[:,3]).sum()))
t=np.array(touch); print("instances vs connected comps (same annotator): inst",t[:,0].sum(),"cc",t[:,1].sum(),"entries with cc<inst",(t[:,1]<t[:,0]).mean())
