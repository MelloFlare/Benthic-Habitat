import os, sys, glob, random, gc, zipfile, warnings
warnings.filterwarnings('ignore')
import numpy as np
import matplotlib
matplotlib.use('Agg')  # no GUI — saves memory
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import rasterio
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, Subset

DATA_ROOT = 'Dataset\sets\'
PATCH_SIZE = 64
PATCH_STRIDE = 48
BATCH_SIZE = 2
MAX_TRAIN = 300
MAX_VAL = 60
NUM_EPOCHS = 20
LR = 1e-4
SEED = 42

torch.set_num_threads(2)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device('cpu')

NUM_CLASSES = 7
CLASS_NAMES = ['Background','Seagrass','Coral','Macroalgae','Sand','Land','Ocean']
CLASS_COLORS = ['#1a1a1a','#2d8a3e','#ff6b35','#1a6b2d','#f5deb3','#8b4513','#87ceeb']
CLASS_FREQ = np.array([0.0350,0.0306,0.0765,0.1191,0.0860,0.3789,0.2741])
cmap = ListedColormap(CLASS_COLORS)

TRAIN_IMG_DIR = os.path.join(DATA_ROOT, 'raw_labeled_data', 'images')
TRAIN_ANN_DIR = os.path.join(DATA_ROOT, 'raw_labeled_data', 'annotations')
TEST1_IMG_DIR = os.path.join(DATA_ROOT, 'test_data', '1', 'images')
TEST3_IMG_DIR = os.path.join(DATA_ROOT, 'test_data', '3', 'images')
TEST1_ANN_DIR = os.path.join(DATA_ROOT, 'test_data', '1', 'annotations')
TEST3_ANN_DIR = os.path.join(DATA_ROOT, 'test_data', '3', 'annotations')

print('=== Benthic Habitat Segmentation ===')
for p in [TRAIN_IMG_DIR, TRAIN_ANN_DIR]:
    print(f'  {"OK" if os.path.isdir(p) else "MISSING"}: {p}')

print('\nComputing band statistics...')
def compute_band_stats(img_dir):
    tifs = sorted(glob.glob(os.path.join(img_dir, '*.tif')))
    assert len(tifs) > 0, f'No tifs in {img_dir}'
    accum = [[] for _ in range(8)]
    for p in tifs:
        with rasterio.open(p) as src:
            for b in range(8):
                data = src.read(b+1).astype(np.float32).ravel()
                idx = np.random.choice(len(data), size=min(10000,len(data)), replace=False)
                accum[b].append(data[idx])
    lo = np.array([np.percentile(np.concatenate(accum[b]), 2)  for b in range(8)], dtype=np.float32)
    hi = np.array([np.percentile(np.concatenate(accum[b]), 98) for b in range(8)], dtype=np.float32)
    del accum; gc.collect()
    return lo, hi

BAND_LO, BAND_HI = compute_band_stats(TRAIN_IMG_DIR)
print('Band stats done.')

class BenthicDataset(Dataset):
    def __init__(self, img_dir, ann_dir, band_lo, band_hi,
                 patch_size=64, stride=48, augment=True):
        self.img_dir=img_dir; self.ann_dir=ann_dir
        self.band_lo=band_lo; self.band_hi=band_hi
        self.patch_size=patch_size; self.augment=augment
        self.index=[]
        img_paths = sorted(glob.glob(os.path.join(img_dir,'*.tif')))
        assert len(img_paths)>0
        for ip in img_paths:
            with rasterio.open(ip) as src:
                H,W=src.height,src.width
            added=False
            for y in range(0,max(H-patch_size+1,1),stride):
                for x in range(0,max(W-patch_size+1,1),stride):
                    self.index.append((ip,y,x)); added=True
            if not added: self.index.append((ip,0,0))
        print(f'  {len(img_paths)} images -> {len(self.index)} patches')

    def __len__(self): return len(self.index)

    def __getitem__(self, idx):
        ip,y,x = self.index[idx]
        ap = os.path.join(self.ann_dir, os.path.basename(ip))
        ps = self.patch_size
        with rasterio.open(ip) as src:
            H,W=src.height,src.width
            y2,x2=min(y+ps,H),min(x+ps,W)
            win=rasterio.windows.Window(x,y,x2-x,y2-y)
            patch=src.read(window=win).astype(np.float32)
        with rasterio.open(ap) as src:
            mask=src.read(1,window=rasterio.windows.Window(x,y,x2-x,y2-y)).astype(np.int64)
        img_p=np.zeros((8,ps,ps),dtype=np.float32)
        msk_p=np.zeros((ps,ps),dtype=np.int64)
        img_p[:,:y2-y,:x2-x]=patch
        msk_p[:y2-y,:x2-x]=np.clip(mask,0,NUM_CLASSES-1)
        for b in range(8):
            rng=float(self.band_hi[b]-self.band_lo[b])
            img_p[b]=np.clip(img_p[b],self.band_lo[b],self.band_hi[b])
            if rng>0: img_p[b]=(img_p[b]-self.band_lo[b])/rng
        img_t=torch.from_numpy(img_p)
        msk_t=torch.from_numpy(msk_p)
        if self.augment:
            if random.random()>0.5: img_t=torch.flip(img_t,dims=[2]); msk_t=torch.flip(msk_t,dims=[1])
            if random.random()>0.5: img_t=torch.flip(img_t,dims=[1]); msk_t=torch.flip(msk_t,dims=[0])
        return img_t, msk_t

print('\nBuilding dataset...')
full_ds = BenthicDataset(TRAIN_IMG_DIR,TRAIN_ANN_DIR,BAND_LO,BAND_HI,
                          patch_size=PATCH_SIZE,stride=PATCH_STRIDE,augment=True)
n_val=max(1,int(0.20*len(full_ds))); n_train=len(full_ds)-n_val
train_ds,val_ds=random_split(full_ds,[n_train,n_val],
                              generator=torch.Generator().manual_seed(SEED))
t_idx=torch.randperm(len(train_ds))[:MAX_TRAIN].tolist()
v_idx=torch.randperm(len(val_ds))[:MAX_VAL].tolist()
train_loader=DataLoader(Subset(train_ds,t_idx),batch_size=BATCH_SIZE,shuffle=True, num_workers=0,pin_memory=False)
val_loader  =DataLoader(Subset(val_ds,  v_idx),batch_size=BATCH_SIZE,shuffle=False,num_workers=0,pin_memory=False)
print(f'Train: {MAX_TRAIN}  Val: {MAX_VAL}')

class DoubleConv(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1,bias=False),
            nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,padding=1,bias=False),
            nn.BatchNorm2d(out_ch),nn.ReLU(inplace=True))
    def forward(self,x): return self.net(x)

class Down(nn.Module):
    def __init__(self,in_ch,out_ch):
        super().__init__()
        self.net=nn.Sequential(nn.MaxPool2d(2),DoubleConv(in_ch,out_ch))
    def forward(self,x): return self.net(x)

class Up(nn.Module):
    def __init__(self,in_ch,skip_ch,out_ch):
        super().__init__()
        self.up=nn.Upsample(scale_factor=2,mode='bilinear',align_corners=True)
        self.conv=DoubleConv(in_ch+skip_ch,out_ch)
    def forward(self,x,skip):
        x=self.up(x)
        if x.shape!=skip.shape:
            x=F.interpolate(x,size=skip.shape[2:],mode='bilinear',align_corners=True)
        return self.conv(torch.cat([skip,x],dim=1))

class UNet8(nn.Module):
    def __init__(self,in_channels=8,num_classes=7):
        super().__init__()
        self.enc1=DoubleConv(in_channels,16)
        self.enc2=Down(16,32)
        self.bottleneck=Down(32,64)
        self.dec2=Up(64,32,32)
        self.dec1=Up(32,16,16)
        self.head=nn.Conv2d(16,num_classes,1)
    def forward(self,x):
        s1=self.enc1(x); s2=self.enc2(s1)
        bn=self.bottleneck(s2)
        x=self.dec2(bn,s2); x=self.dec1(x,s1)
        return self.head(x)

gc.collect()
model=UNet8(in_channels=8,num_classes=NUM_CLASSES).to(device)
print(f'\nModel ready. Params: {sum(p.numel() for p in model.parameters()):,}')

class_weights=torch.tensor(
    (1.0/(CLASS_FREQ+1e-6))/(1.0/(CLASS_FREQ+1e-6)).sum(),dtype=torch.float32)
loss_fn =nn.CrossEntropyLoss(weight=class_weights)
optimizer=optim.AdamW(model.parameters(),lr=LR,weight_decay=1e-4)
scheduler=optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=NUM_EPOCHS,eta_min=1e-6)

def pixel_accuracy(logits,targets):
    return (logits.argmax(dim=1)==targets).float().mean().item()

def mean_iou(logits,targets):
    preds=logits.argmax(dim=1).numpy().ravel(); tgts=targets.numpy().ravel()
    ious=[]
    for c in range(NUM_CLASSES):
        inter=int(np.sum((preds==c)&(tgts==c))); union=int(np.sum((preds==c)|(tgts==c)))
        if union>0: ious.append(inter/union)
    return float(np.mean(ious)) if ious else 0.0

history={'train_loss':[],'val_loss':[],'train_acc':[],'val_acc':[],'val_miou':[]}
best_val_loss=float('inf'); best_model_path='best_unet8.pth'

print(f'\nTraining {NUM_EPOCHS} epochs...')
print(f'{"Ep":>4}  {"TLoss":>7}  {"VLoss":>7}  {"TAcc":>7}  {"VAcc":>7}  {"mIoU":>7}')
print('-'*50)

for epoch in range(1,NUM_EPOCHS+1):
    model.train()
    tl,ta,nb=0.0,0.0,0
    for imgs,masks in train_loader:
        optimizer.zero_grad()
        out=model(imgs); loss=loss_fn(out,masks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
        optimizer.step()
        tl+=loss.item(); ta+=pixel_accuracy(out,masks); nb+=1

    model.eval()
    vl,va,vm,nv=0.0,0.0,0.0,0
    with torch.no_grad():
        for imgs,masks in val_loader:
            out=model(imgs)
            vl+=loss_fn(out,masks).item()
            va+=pixel_accuracy(out,masks)
            vm+=mean_iou(out,masks); nv+=1

    tl/=nb; ta=ta/nb*100; vl/=nv; va=va/nv*100; vm=vm/nv*100
    history['train_loss'].append(tl); history['val_loss'].append(vl)
    history['train_acc'].append(ta);  history['val_acc'].append(va)
    history['val_miou'].append(vm)
    flag=' *' if vl<best_val_loss else ''
    if vl<best_val_loss:
        best_val_loss=vl; torch.save(model.state_dict(),best_model_path)
    scheduler.step()
    print(f'{epoch:>4}  {tl:>7.4f}  {vl:>7.4f}  {ta:>6.1f}%  {va:>6.1f}%  {vm:>6.1f}%{flag}')
    sys.stdout.flush(); gc.collect()

print(f'\nDone. Best val loss: {best_val_loss:.4f}')

epochs=range(1,NUM_EPOCHS+1)
fig,axes=plt.subplots(1,3,figsize=(15,4))
axes[0].plot(epochs,history['train_loss'],label='Train',linewidth=2)
axes[0].plot(epochs,history['val_loss'],  label='Val',  linewidth=2,linestyle='--')
axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(epochs,history['train_acc'],label='Train',linewidth=2)
axes[1].plot(epochs,history['val_acc'],  label='Val',  linewidth=2,linestyle='--')
axes[1].set_title('Pixel Accuracy (%)'); axes[1].legend(); axes[1].grid(alpha=0.3)
axes[2].plot(epochs,history['val_miou'],color='green',linewidth=2)
axes[2].set_title('Val mIoU (%)'); axes[2].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('training_curves.png',dpi=120,bbox_inches='tight')
print('Saved -> training_curves.png')

model.load_state_dict(torch.load(best_model_path,map_location='cpu'))
model.eval()

def read_mask(path):
    with rasterio.open(path) as src: return src.read(1).astype(np.int64)

def predict_full_image(model,img_path):
    ps=PATCH_SIZE
    with rasterio.open(img_path) as src:
        H,W=src.height,src.width; img=src.read().astype(np.float32)
    for b in range(8):
        rng=float(BAND_HI[b]-BAND_LO[b])
        img[b]=np.clip(img[b],BAND_LO[b],BAND_HI[b])
        if rng>0: img[b]=(img[b]-BAND_LO[b])/rng
    acc=np.zeros((NUM_CLASSES,H,W),dtype=np.float32)
    cnt=np.zeros((H,W),dtype=np.float32)
    stride=ps//2
    for y in range(0,max(H-ps+1,1),stride):
        for x in range(0,max(W-ps+1,1),stride):
            y2,x2=min(y+ps,H),min(x+ps,W); ph,pw=y2-y,x2-x
            patch=np.zeros((8,ps,ps),dtype=np.float32)
            patch[:,:ph,:pw]=img[:,y:y2,x:x2]
            t=torch.from_numpy(patch).unsqueeze(0)
            with torch.no_grad(): logit=model(t).squeeze(0).numpy()
            acc[:,y:y2,x:x2]+=logit[:,:ph,:pw]; cnt[y:y2,x:x2]+=1
    return (acc/np.maximum(cnt,1)[np.newaxis]).argmax(axis=0)

def per_class_iou(pred,gt):
    ious={}
    for c in range(NUM_CLASSES):
        inter=np.sum((pred==c)&(gt==c)); union=np.sum((pred==c)|(gt==c))
        ious[CLASS_NAMES[c]]=inter/union if union>0 else float('nan')
    return ious

print('\nEvaluating on labeled images...')
img_paths=sorted(glob.glob(os.path.join(TRAIN_IMG_DIR,'*.tif')))
all_acc,all_miou=[],[]
for ip in img_paths:
    fname=os.path.basename(ip); ap=os.path.join(TRAIN_ANN_DIR,fname)
    pred=predict_full_image(model,ip)
    gt=np.clip(read_mask(ap),0,NUM_CLASSES-1)
    acc=np.mean(pred==gt)*100
    ious=per_class_iou(pred,gt); miou=np.nanmean(list(ious.values()))*100
    all_acc.append(acc); all_miou.append(miou)
    print(f'{fname}  Acc={acc:.1f}%  mIoU={miou:.1f}%')
    for cls,iou in ious.items():
        print(f'  {cls:12s}: {iou*100:.1f}%' if not np.isnan(iou) else f'  {cls:12s}: n/a')
    gc.collect()
print(f'\nMEAN  Acc={np.mean(all_acc):.1f}%  mIoU={np.mean(all_miou):.1f}%')

print('\nGenerating comparison plots...')
for ip in img_paths[:2]:
    fname=os.path.basename(ip); ap=os.path.join(TRAIN_ANN_DIR,fname)
    with rasterio.open(ip) as src: img=src.read().astype(np.float32)
    for b in range(8):
        rng=float(BAND_HI[b]-BAND_LO[b]); img[b]=np.clip(img[b],BAND_LO[b],BAND_HI[b])
        if rng>0: img[b]=(img[b]-BAND_LO[b])/rng
    rgb=np.clip(img[[4,2,1]].transpose(1,2,0),0,1)
    gt=np.clip(read_mask(ap),0,NUM_CLASSES-1)
    pred=predict_full_image(model,ip)
    fig,axes=plt.subplots(1,3,figsize=(15,5))
    fig.suptitle(fname)
    axes[0].imshow(rgb);  axes[0].set_title('RGB'); axes[0].axis('off')
    axes[1].imshow(gt,  cmap=cmap,vmin=0,vmax=6); axes[1].set_title('Ground Truth'); axes[1].axis('off')
    axes[2].imshow(pred,cmap=cmap,vmin=0,vmax=6); axes[2].set_title('Prediction');   axes[2].axis('off')
    lp=[mpatches.Patch(facecolor=CLASS_COLORS[c],label=CLASS_NAMES[c]) for c in range(NUM_CLASSES)]
    fig.legend(handles=lp,loc='lower center',ncol=7,fontsize=8)
    plt.tight_layout(rect=[0,0.08,1,1])
    out=f'comparison_{fname.replace(".tif",".png")}'
    fig.savefig(out,dpi=100,bbox_inches='tight'); plt.close()
    print(f'Saved -> {out}'); gc.collect()

print('\nGenerating confusion matrix...')
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
all_p,all_g=[],[]
for ip in img_paths:
    ap=os.path.join(TRAIN_ANN_DIR,os.path.basename(ip))
    pred=predict_full_image(model,ip); gt=np.clip(read_mask(ap),0,NUM_CLASSES-1)
    flat_p=pred.ravel(); flat_g=gt.ravel()
    idx=np.random.choice(len(flat_p),size=min(50000,len(flat_p)),replace=False)
    all_p.append(flat_p[idx]); all_g.append(flat_g[idx]); gc.collect()
cm=confusion_matrix(np.concatenate(all_g),np.concatenate(all_p),
                    labels=list(range(NUM_CLASSES)),normalize='true')
fig,ax=plt.subplots(figsize=(9,7))
ConfusionMatrixDisplay(cm,display_labels=CLASS_NAMES).plot(ax=ax,cmap='Blues',values_format='.2f')
ax.set_title('Normalised Confusion Matrix')
plt.xticks(rotation=30,ha='right'); plt.tight_layout()
plt.savefig('confusion_matrix.png',dpi=120,bbox_inches='tight'); plt.close()
print('Saved -> confusion_matrix.png')

print('\nGenerating error map...')
ip=img_paths[0]; ap=os.path.join(TRAIN_ANN_DIR,os.path.basename(ip))
with rasterio.open(ip) as src: img=src.read().astype(np.float32)
for b in range(8):
    rng=float(BAND_HI[b]-BAND_LO[b]); img[b]=np.clip(img[b],BAND_LO[b],BAND_HI[b])
    if rng>0: img[b]=(img[b]-BAND_LO[b])/rng
rgb=np.clip(img[[4,2,1]].transpose(1,2,0),0,1)
gt=np.clip(read_mask(ap),0,NUM_CLASSES-1); pred=predict_full_image(model,ip)
err=(pred!=gt).astype(np.uint8)
fig,axes=plt.subplots(1,4,figsize=(20,5))
axes[0].imshow(rgb);  axes[0].set_title('RGB');          axes[0].axis('off')
axes[1].imshow(gt,  cmap=cmap,vmin=0,vmax=6); axes[1].set_title('Ground Truth'); axes[1].axis('off')
axes[2].imshow(pred,cmap=cmap,vmin=0,vmax=6); axes[2].set_title('Prediction');   axes[2].axis('off')
axes[3].imshow(rgb); axes[3].imshow(err,cmap='Reds',alpha=0.6,vmin=0,vmax=1)
axes[3].set_title(f'Errors: {err.mean()*100:.1f}% wrong'); axes[3].axis('off')
lp=[mpatches.Patch(facecolor=CLASS_COLORS[c],label=CLASS_NAMES[c]) for c in range(NUM_CLASSES)]
fig.legend(handles=lp,loc='lower center',ncol=7,fontsize=8)
plt.tight_layout(rect=[0,0.07,1,1])
plt.savefig('error_map_sample.png',dpi=110,bbox_inches='tight'); plt.close()
print('Saved -> error_map_sample.png')

print('\nRunning test inference...')
def run_test_set(img_dir,ann_dir,region,out_dir):
    os.makedirs(out_dir,exist_ok=True)
    paths=sorted(glob.glob(os.path.join(img_dir,'*.tif')))
    if not paths: print(f'  No tifs in {img_dir}'); return
    accs,mious=[],[]
    for ip in paths:
        fname=os.path.basename(ip); stem=fname.replace('.tif','')
        pred=predict_full_image(model,ip)
        plt.imsave(os.path.join(out_dir,f'{stem}_pred.png'),pred,cmap=cmap,vmin=0,vmax=6)
        ap_tif=os.path.join(ann_dir,fname); ap_png=os.path.join(ann_dir,f'{stem}.png')
        gt=None
        if os.path.exists(ap_tif): gt=np.clip(read_mask(ap_tif),0,NUM_CLASSES-1)
        elif os.path.exists(ap_png):
            from PIL import Image
            raw=np.array(Image.open(ap_png))
            gt=np.clip(raw if raw.ndim==2 else raw[:,:,0],0,NUM_CLASSES-1).astype(np.int64)
        if gt is not None and gt.shape==pred.shape:
            acc=np.mean(pred==gt)*100
            miou=np.nanmean(list(per_class_iou(pred,gt).values()))*100
            accs.append(acc); mious.append(miou)
            print(f'  [{region}] {fname}  Acc={acc:.1f}%  mIoU={miou:.1f}%')
        else:
            print(f'  [{region}] {fname}  saved')
        gc.collect()
    if accs: print(f'  [{region}] MEAN  Acc={np.mean(accs):.1f}%  mIoU={np.mean(mious):.1f}%')

run_test_set(TEST1_IMG_DIR,TEST1_ANN_DIR,'Agana', 'output_masks/agana')
run_test_set(TEST3_IMG_DIR,TEST3_ANN_DIR,'Manell','output_masks/manell')

with zipfile.ZipFile('predicted_masks.zip','w') as zf:
    for folder in ['output_masks/agana','output_masks/manell']:
        for f in glob.glob(os.path.join(folder,'*.png')):
            zf.write(f,arcname=os.path.relpath(f,'output_masks'))
print('Saved -> predicted_masks.zip')

print('\n'+'='*50)
print('FINAL SUMMARY')
print('='*50)
print(f'  Model     : Lightweight U-Net (8-channel)')
print(f'  Params    : {sum(p.numel() for p in model.parameters()):,}')
print(f'  Patch     : {PATCH_SIZE}x{PATCH_SIZE}')
print(f'  Loss      : Weighted CrossEntropy')
print(f'  Best val  : {best_val_loss:.4f}')
print(f'  Best mIoU : {max(history["val_miou"]):.2f}%')
print(f'  Best acc  : {max(history["val_acc"]):.2f}%')
print('='*50)
print('\nAll done! Output files:')
print('  best_unet8.pth, training_curves.png,')
print('  confusion_matrix.png, error_map_sample.png,')
print('  predicted_masks.zip')