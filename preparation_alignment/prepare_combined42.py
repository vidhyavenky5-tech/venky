"""Prepare a separate, source-balanced development dataset. No GPU training."""
import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm
from leaf_preproc import preprocess

HERE = Path(__file__).resolve().parent
BASE = HERE / 'preprocessed/combined42'
PD = HERE / 'preprocessed/external_candidates/PlantDoc'
MAPPING = {
'Apple Scab Leaf':'Apple___Apple_scab', 'Apple leaf':'Apple___healthy',
'Apple rust leaf':'Apple___Cedar_apple_rust', 'Bell_pepper leaf':'Pepper,_bell___healthy',
'Bell_pepper leaf spot':'Pepper,_bell___Bacterial_spot', 'Blueberry leaf':'Blueberry___healthy',
'Cherry leaf':'Cherry___healthy', 'Corn Gray leaf spot':'Corn___Cercospora_leaf_spot Gray_leaf_spot',
'Corn leaf blight':'Corn___Northern_Leaf_Blight', 'Corn rust leaf':'Corn___Common_rust',
'Peach leaf':'Peach___healthy', 'Potato leaf early blight':'Potato___Early_blight',
'Potato leaf late blight':'Potato___Late_blight', 'Raspberry leaf':'Raspberry___healthy',
'Soyabean leaf':'Soybean___healthy', 'Squash Powdery mildew leaf':'Squash___Powdery_mildew',
'Strawberry leaf':'Strawberry___healthy', 'Tomato Early blight leaf':'Tomato___Early_blight',
'Tomato Septoria leaf spot':'Tomato___Septoria_leaf_spot', 'Tomato leaf':'Tomato___healthy',
'Tomato leaf bacterial spot':'Tomato___Bacterial_spot', 'Tomato leaf late blight':'Tomato___Late_blight',
'Tomato leaf mosaic virus':'Tomato___Tomato_mosaic_virus', 'Tomato leaf yellow virus':'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
'Tomato mold leaf':'Tomato___Leaf_Mold', 'Tomato two spotted spider mites leaf':'Tomato___Spider_mites Two-spotted_spider_mite',
'grape leaf':'Grape___healthy', 'grape leaf black rot':'Grape___Black_rot'}

def sha(x):
    return hashlib.sha256(x.tobytes()).hexdigest()

def process(path):
    with Image.open(path) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert('RGB'))
    small = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY),(9,8))
    dh = int.from_bytes(np.packbits(small[:,1:] > small[:,:-1]).tobytes(),'big')
    pixels, info = preprocess(rgb,size=64)
    return pixels, dict(path=str(path),label=path.parent.name,original_split=path.parent.parent.name,
        pixel_hash=sha(pixels),decoded_hash=sha(rgb),dhash=dh,mask_ok=info['mask_ok'],reason=info['reason'])

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workers',type=int,default=4)
    a=ap.parse_args()
    if (BASE/'dataset.json').exists():
        raise SystemExit('Prepared dataset already exists. Reuse it; do not change splits during an experiment.')
    BASE.mkdir(parents=True,exist_ok=True)
    oldbase=HERE/'preprocessed/reliable_42class'
    cm=json.loads((oldbase/'class_map.json').read_text())
    classes=cm['classes']; ids=cm['class_to_id']
    assert set(MAPPING.values()) <= set(classes)
    (BASE/'plantdoc_label_mapping.json').write_text(json.dumps(MAPPING,indent=2))
    paths=[p for sp in ('train','test') for p in sorted((PD/sp).rglob('*')) if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}]
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        processed=list(tqdm(pool.map(process,paths),total=len(paths),desc='PlantDoc preprocessing'))
    X=np.stack([x for x,r in processed]); rows=[r for x,r in processed]
    y=np.array([ids[MAPPING[r['label']]] for r in rows],dtype=np.int64)
    # Conservative near-duplicate grouping, including across class labels.
    parent=list(range(len(rows)))
    def root(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]; i=parent[i]
        return i
    for i in tqdm(range(len(rows)),desc='PlantDoc duplicate grouping'):
        for j in range(i):
            if rows[i]['decoded_hash']==rows[j]['decoded_hash'] or rows[i]['pixel_hash']==rows[j]['pixel_hash'] or (rows[i]['dhash'] ^ rows[j]['dhash']).bit_count() <= 4:
                parent[root(i)]=root(j)
    groups={}
    for i in range(len(rows)): groups.setdefault(root(i),[]).append(i)
    candidates=[]; test=[]
    for group,members in groups.items():
        if len(set(y[members]))>1:
            for i in members: rows[i]['status']='quarantined_conflicting_labels'; rows[i]['group']=group
            continue
        tests=[i for i in members if rows[i]['original_split']=='test']
        chosen=(tests or members)[0]
        (test if tests else candidates).append(chosen)
        for i in members:
            rows[i]['status']='test' if i==chosen and tests else 'development' if i==chosen else 'duplicate_excluded'
            rows[i]['group']=group
    # At least one development training example per represented class; tiny classes get no validation.
    rng=np.random.default_rng(42); train=[]; val=[]
    for c in sorted(set(y[candidates])):
        arr=np.array([i for i in candidates if y[i]==c]); rng.shuffle(arr)
        nval=max(1,round(.15*len(arr))) if len(arr)>=5 else 0
        val.extend(arr[:nval]); train.extend(arr[nval:])
    pdparts={'train':np.array(train,int),'val':np.array(val,int),'test':np.array(test,int)}
    source={}
    with np.load(oldbase/'leaf_cache_64px_masked_v2.npz') as z:
        for sp in ('train','val','test'):
            source[sp]=(z['X_'+sp],z['y_'+sp].astype(np.int64),z['ok_'+sp].astype(bool))
    # Keep all held-out images out of both sources' training/validation by model-input hash.
    hashes={sp:[sha(x) for x in source[sp][0]] for sp in source}
    pd_test_hashes={rows[i]['pixel_hash'] for i in pdparts['test']}
    test_hashes=set(hashes['test']) | pd_test_hashes
    existing_val_hashes=set(hashes['val'])
    for sp in ('train','val'):
        kept=[]
        for i in pdparts[sp]:
            if rows[i]['pixel_hash'] in test_hashes or rows[i]['pixel_hash'] in existing_val_hashes:
                rows[i]['status']='excluded_existing_holdout_overlap'
            else: kept.append(i)
        pdparts[sp]=np.array(kept,int)
    val_hashes=existing_val_hashes | {rows[i]['pixel_hash'] for i in pdparts['val']}
    summary={}
    for sp in ('train','val','test'):
        xx,yy,ok=source[sp]
        keep=np.array([h not in test_hashes and (sp!='train' or h not in val_hashes) for h in hashes[sp]]) if sp!='test' else np.ones(len(yy),bool)
        xx,yy,ok=xx[keep],yy[keep],ok[keep]
        np.savez_compressed(BASE/f'existing_{sp}.npz',X=xx,y=yy,ok=ok)
        indices=pdparts[sp]
        np.savez_compressed(BASE/f'plantdoc_{sp}.npz',X=X[indices],y=y[indices],ok=np.array([rows[i]['mask_ok'] for i in indices]),group=np.array([rows[i]['group'] for i in indices]))
        for i in indices: rows[i]['status']=sp
        summary[sp]={'existing':len(yy),'plantdoc':len(indices),'existing_removed_overlap':int((~keep).sum())}
    pd.DataFrame(rows).to_csv(BASE/'plantdoc_manifest.csv',index=False)
    (BASE/'class_map.json').write_text(json.dumps(cm,indent=2))
    counts=[]
    for domain in ('existing','plantdoc'):
        for sp in ('train','val','test'):
            with np.load(BASE/f'{domain}_{sp}.npz') as z:
                for i,c in enumerate(classes): counts.append(dict(source=domain,split=sp,label=c,n=int((z['y']==i).sum())))
    pd.DataFrame(counts).to_csv(BASE/'class_counts.csv',index=False)
    metadata=dict(counts=summary,preprocessing='leaf_preproc.preprocess size=64; existing cached preprocessing preserved',sources=['existing42 (PlantVillage plus prior external additions)','PlantDoc classification repository'],mapping_note='Conventional disease aliases; broader PlantDoc names such as corn leaf blight and apple rust require label review before scientific claims.',duplicates='PlantDoc decoded/pixel hash and dHash distance <=4; conflicting groups quarantined, one representative per group. Cross-source exact preprocessed-image overlap removed. Cross-source near duplicates and same-plant groups not fully audited.',test_note='Earlier test-informed class selection and possible prior checkpoint exposure limit independent evaluation claims. Source test sets are never used for checkpoint selection.')
    (BASE/'dataset.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2))

if __name__=='__main__': main()
