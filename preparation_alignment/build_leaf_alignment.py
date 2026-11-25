"""Build annotation-assisted leaf views and quantify development-domain differences.
This is a crop-classification study, not an automatic leaf-detection benchmark.
"""
import argparse,json,hashlib,shutil
from pathlib import Path
import cv2,numpy as np,pandas as pd
from PIL import Image,ImageDraw
from tqdm import tqdm
from leaf_canonical import canonicalize
from prepare_combined42 import MAPPING
HERE=Path(__file__).resolve().parent
BASE=HERE/'preprocessed/combined42'
PD=HERE/'preprocessed/external_candidates/PlantDoc'
OUT=HERE/'preprocessed/leaf_aligned42'


def features(x):
    hsv=cv2.cvtColor(x,cv2.COLOR_RGB2HSV)
    fg=np.max(np.abs(x.astype(float)-128),axis=2)>12
    border=np.ones(x.shape[:2],bool);border[4:-4,4:-4]=False
    gray=cv2.cvtColor(x,cv2.COLOR_RGB2GRAY)
    return [float(fg.mean()),float(x[border].std()),float(hsv[:,:,1].mean()),float(hsv[:,:,2].mean()),float(cv2.Laplacian(gray,cv2.CV_64F).var())]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--include-test',action='store_true',help='Apply frozen pipeline to test; no scoring or tuning');a=ap.parse_args()
    OUT.mkdir(exist_ok=True)
    classes=json.loads((BASE/'class_map.json').read_text())['classes'];ids={c:i for i,c in enumerate(classes)}
    rev={str(Path(r['local'])):Path(r['original']).name for r in json.loads((PD/'filename_mapping.json').read_text())}
    ann=pd.concat([pd.read_csv(HERE/'preprocessed/external_candidates/PlantDoc_annotations'/f'{s}_labels.csv') for s in ('train','test')]).drop_duplicates()
    grouped={name:f for name,f in ann.groupby('filename')}
    manifest=pd.read_csv(BASE/'plantdoc_manifest.csv')
    reuse=a.include_test and (OUT/'crop_manifest.csv').exists()
    records=pd.read_csv(OUT/'crop_manifest.csv').to_dict('records') if reuse else []
    feature_rows=pd.read_csv(OUT/'development_features.csv').to_dict('records') if reuse else []
    previews={};mapping=dict(MAPPING,**{'Potato leaf':'Potato___healthy'})
    splits=('train','val','test') if a.include_test else ('train','val')
    detector=OUT/'detector'
    for sp in splits:
        if reuse and (OUT/f'plantdoc_{sp}.npz').exists(): continue
        selected=manifest[manifest.status==sp];images=[];labels=[];parents=[];plain=[]
        for r in tqdm(selected.itertuples(),total=len(selected),desc='Annotated leaves '+sp):
            path=Path(r.path);name=rev[str(path.relative_to(PD))]
            with Image.open(path) as im:
                rgb=np.asarray(im.convert('RGB'));orientation=im.getexif().get(274,1)
            h,w=rgb.shape[:2];boxes=grouped.get(name,pd.DataFrame())
            if orientation!=1 or len(boxes)==0 or not ((boxes.width==w)&(boxes.height==h)).all():
                records.append(dict(source=str(path),split=sp,status='review_dimensions_or_orientation',n_boxes=0));continue
            boxes=boxes.sort_values(['xmin','ymin','xmax','ymax']).drop_duplicates(['xmin','ymin','xmax','ymax','class'])
            valid=[]
            for b in boxes.to_dict('records'):
                target=mapping.get(b['class'])
                if target not in ids:continue
                x0,y0,x1,y1=[int(b[k]) for k in ('xmin','ymin','xmax','ymax')]
                if not (0<=x0<x1<=w and 0<=y0<y1<=h) or min(x1-x0,y1-y0)<8:continue
                valid.append((b,target,(x0,y0,x1,y1)))
            if not valid:
                records.append(dict(source=str(path),split=sp,status='review_no_valid_leaf_box',n_boxes=0));continue
            # Detector labels collapse ALL annotated leaf conditions to a single leaf class.
            dst=detector/'images'/sp/(path.stem+path.suffix);dst.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,dst)
            txt=detector/'labels'/sp/(path.stem+'.txt');txt.parent.mkdir(parents=True,exist_ok=True)
            txt.write_text('\n'.join(f'0 {(box[0]+box[2])/(2*w):.8f} {(box[1]+box[3])/(2*h):.8f} {(box[2]-box[0])/w:.8f} {(box[3]-box[1])/h:.8f}' for _,_,box in valid))
            for j,(b,target,box) in enumerate(valid):
                crop,info=canonicalize(rgb,box=box);uncut,_=canonicalize(rgb,box=box,mask=False)
                dest=OUT/'crops'/sp/target/f'{path.stem}_{j}.png';dest.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(crop).save(dest)
                images.append(crop);plain.append(uncut);labels.append(ids[target]);parents.append(r.group)
                records.append(dict(source=str(path),prepared=str(dest),split=sp,label=target,status='crop',n_boxes=len(valid),box=json.dumps(box),mask_ok=info['mask_ok'],reason=info['reason'],parent_group=r.group))
                if sp!='test':
                    for method,x in [('annotated_crop',uncut),('canonical',crop)]:feature_rows.append(dict(source='plantdoc',split=sp,label=target,method=method,**dict(zip(['foreground_fraction','border_std','saturation','brightness','sharpness'],features(x)))))
                if sp=='val' and target not in previews: previews[target]=(rgb,uncut,crop)
        if images:
            for name,X in [('plantdoc',images),('plantdoc_box',plain)]:np.savez_compressed(OUT/f'{name}_{sp}.npz',X=np.stack(X),y=np.array(labels),group=np.array(parents))
    # Existing training caches are already 64px; canonical framing cannot recover lost resolution.
    for sp in splits:
        if reuse and (OUT/f'existing_{sp}.npz').exists(): continue
        with np.load(BASE/f'existing_{sp}.npz') as z:X=z['X'];y=z['y']
        transformed=[]
        for i,x in enumerate(tqdm(X,desc='Existing canonical '+sp)):
            target=classes[y[i]]
            image=x if target=='Background_without_leaves' else canonicalize(x,cached=True)[0]
            transformed.append(image)
            if sp!='test' and i%20==0:
                for method,pixels in [('original',x),('canonical',image)]:feature_rows.append(dict(source='existing',split=sp,label=target,method=method,**dict(zip(['foreground_fraction','border_std','saturation','brightness','sharpness'],features(pixels)))))
        np.savez_compressed(OUT/f'existing_{sp}.npz',X=np.stack(transformed),y=y)
    pd.DataFrame(records).to_csv(OUT/'crop_manifest.csv',index=False)
    f=pd.DataFrame(feature_rows);f.to_csv(OUT/'development_features.csv',index=False)
    f.groupby(['source','method'])[['foreground_fraction','border_std','saturation','brightness','sharpness']].median().to_csv(OUT/'domain_summary.csv')
    (OUT/'class_map.json').write_text(json.dumps({'classes':classes,'class_to_id':ids},indent=2))
    (detector/'data.yaml').write_text('path: '+json.dumps(str(detector.resolve()).replace('\\','/'))+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: leaf\n')
    # Visual comparisons use validation only and identical categories across datasets.
    with np.load(BASE/'existing_val.npz') as z:oldx=z['X'];oldy=z['y']
    chosen=[c for c in previews if c in {'Apple___healthy','Apple___Apple_scab','Corn___Northern_Leaf_Blight','Tomato___Late_blight','Grape___Black_rot','Potato___Early_blight','Pepper,_bell___Bacterial_spot','Strawberry___healthy'}]
    canvas=Image.new('RGB',(1050,50+len(chosen)*180),'white');draw=ImageDraw.Draw(canvas)
    for i,title in enumerate(['PlantDoc source','Annotated leaf','Shared leaf view','Existing input','Existing shared view']):draw.text((i*210+5,10),title,fill='black')
    for j,c in enumerate(chosen):
        rgb,plain,crop=previews[c];original=oldx[np.flatnonzero(oldy==ids[c])[0]];new=canonicalize(original,cached=True)[0]
        for i,img in enumerate([rgb,plain,crop,original,new]):
            im=Image.fromarray(img);im.thumbnail((200,150));canvas.paste(im,(i*210,50+j*180))
        draw.text((5,50+j*180+152),c,fill='black')
    if chosen: canvas.save(OUT/'comparison.png')
    report=dict(crops=dict(pd.DataFrame(records).query("status=='crop'").groupby('split').size()),rejected_source_records=int(sum(r['status']!='crop' for r in records)),test_processed=a.include_test,warning='Annotation-assisted crop dataset. Not automatic field inference. Gray masks are foreground proposals, not verified semantic masks. Existing input resolution limited to cached 64px. No colour matching or symptom recolouring.',annotation_source='https://github.com/pratikkayal/PlantDoc-Object-Detection-Dataset',license='CC BY 4.0; retained in PlantDoc_annotations/LICENSE.txt')
    report['crops']={k:int(v) for k,v in report['crops'].items()}
    (OUT/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
