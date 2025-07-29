# train.py

import os
import glob
import random
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from dataset import GoProDataset
from model   import MultiStageDeblurNet
from losses  import PerceptualLoss, EdgeLoss, combined_loss
from pytorch_msssim import ms_ssim
import torchvision.transforms.functional as TF
from PIL import Image

def train(
    root_dir: str,
    epochs: int = 50,
    batch_size: int = 4,
    lr_max: float = 1e-3,
    weight_decay: float = 1e-4,
    ema_decay: float = 0.999,
    save_path: str = 'deblur_model_ema.pth'
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds_train = GoProDataset(root_dir, 'train', augment=True)
    ds_val   = GoProDataset(root_dir, 'test',  augment=False)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,  num_workers=2, pin_memory=True)
    dl_val   = DataLoader(ds_val,   batch_size=1,         shuffle=False, num_workers=2, pin_memory=True)

    model     = MultiStageDeblurNet().to(device)
    perc_loss = PerceptualLoss().to(device)
    edge_loss = EdgeLoss().to(device)
    opt   = AdamW(model.parameters(), lr=lr_max/25, weight_decay=weight_decay)
    sched = OneCycleLR(opt,
                       max_lr=lr_max,
                       total_steps=epochs * len(dl_train),
                       pct_start=0.1, div_factor=25, final_div_factor=1e4)

    # EMA buffer
    ema_state = {k: v.clone().detach() for k,v in model.state_dict().items()}

    for ep in range(1, epochs+1):
        ds_train.gb_prob = 0.1 + 0.4*(ep-1)/(epochs-1)

        model.train()
        t = tqdm(dl_train, desc=f"Epoch {ep}/{epochs} [Train]")
        for blur, sharp in t:
            blur, sharp = blur.to(device), sharp.to(device)
            coarse = torch.sigmoid(model.stage1(blur))
            final  = model(blur)

            loss = combined_loss(coarse, final, sharp, perc_loss, edge_loss)
            opt.zero_grad(); loss.backward()
            clip_grad_norm_(model.parameters(), 0.5)
            opt.step(); sched.step()

            # update EMA
            for k, v in model.state_dict().items():
                ema_state[k] = ema_decay * ema_state[k] + (1-ema_decay)*v.detach()

            s = ms_ssim(final, sharp, data_range=1.0).item()
            t.set_postfix(loss=f"{loss.item():.4f}", ssim=f"{s:.4f}")

        # validate with EMA weights
        backup = model.state_dict()
        model.load_state_dict(ema_state)
        model.eval()
        tot = 0.0
        v = tqdm(dl_val, desc=f"Epoch {ep}/{epochs} [Val]")
        with torch.no_grad():
            for blur, sharp in v:
                blur, sharp = blur.to(device), sharp.to(device)
                out = model(blur)
                s = ms_ssim(out, sharp, data_range=1.0).item()
                tot += s
                v.set_postfix(ssim=f"{s:.4f}")
        mean_ssim = tot / len(dl_val)
        print(f"Epoch {ep:02d}: Mean Val SSIM = {mean_ssim:.4f}")
        model.load_state_dict(backup)

    # final save
    model.load_state_dict(ema_state)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Saved EMA model to {save_path}")


def inference(
    root_dir: str,
    weight_path: str,
    out_path: str = 'deblur_demo.png',
    image: str = None  # NEW: path to a single blur image
):
    # Gather blur paths
    if image:
        blur_paths = [image]
    else:
        blur_dir = os.path.join(root_dir, 'test', 'blur')
        if not os.path.isdir(blur_dir):
            raise FileNotFoundError(f"No such directory: {blur_dir}")
        blur_paths = glob.glob(os.path.join(blur_dir, '*.png'))

    if not blur_paths:
        raise RuntimeError(
            f"No blur images found in {blur_paths!r}. "
            "Either pass --image /path/to/your.png or check --root_dir."
        )

    # Pick one
    blur_path = random.choice(blur_paths)
    sharp_path = blur_path.replace(os.sep + 'blur' + os.sep,
                                   os.sep + 'sharp' + os.sep)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = MultiStageDeblurNet().to(device)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()

    # Load & preprocess
    img = Image.open(blur_path).convert('RGB')
    tensor = TF.to_tensor(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        out = model(tensor).squeeze().cpu().clamp(0,1)

    # Save result
    TF.to_pil_image(out).save(out_path)
    print("Blur:      ", blur_path)
    if os.path.exists(sharp_path):
        print("Sharp GT:  ", sharp_path)
    else:
        print("Sharp GT:   (not found)")
    print("Deblurred: ", out_path)

    # If GT exists, compute SSIM
    if os.path.exists(sharp_path):
        gt = TF.to_tensor(Image.open(sharp_path).convert('RGB')).unsqueeze(0).to(device)
        ssim_gt = ms_ssim(out.unsqueeze(0), gt, data_range=1.0).item()
        print(f"SSIM vs GT: {ssim_gt:.4f}")
    return blur_path, sharp_path, out_path



if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode',       choices=['train','infer'], default='train')
    parser.add_argument('--root_dir',   type=str, required=False, default=".",
                        help="(train mode) path to GoPro root; in infer mode you can also use --image alone")
    parser.add_argument('--weights',    type=str, default='deblur_model_ema.pth')
    parser.add_argument('--epochs',     type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--out_demo',   type=str, default='deblur_demo.png')

    # ← ADD THIS LINE ↓
    parser.add_argument('--image',      type=str, default=None,
                        help="(infer mode) path to a single blur image")

    args = parser.parse_args()

    if args.mode=='train':
        train(
            root_dir   = args.root_dir,
            epochs     = args.epochs,
            batch_size = args.batch_size,
            save_path  = args.weights
        )
    else:
        inference(
            root_dir    = args.root_dir,
            weight_path = args.weights,
            out_path    = args.out_demo,
            image       = args.image    # pass it here
        )

