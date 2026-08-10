"""
Smoke test: build a tiny fake dataset (2 classes × 2 train samples × 5 views
+ 2 test samples × 5 views), use a random ViT-like backbone to verify the
data flow, memory bank, inference, mask writing, csv writing all work.
"""
import os
import sys
import shutil
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dataset import CSIGImageDataset, CSIGSampleDataset, build_train_transform
from src.patchcore import MultiClassPatchCore, bilinear_upsample, gaussian_smooth2d
from src.multiview import aggregate_image_scores, multiview_mask_vote
from src.utils import PerClassPercentileCalibrator, save_mask_png, save_submission_csv

ROOT = Path("/tmp/csig_smoke")
if ROOT.exists():
    shutil.rmtree(ROOT)
IN_SIZE = 112  # small so patch=14 → 8×8 grid
PATCH = 14
CLASSES = ["battery", "blade_switch"]


def make_fake_images():
    rng = np.random.default_rng(0)
    for split, n_s in [("Train", 2), ("Test_A", 2)]:
        for cls in CLASSES:
            for s in range(1, n_s + 1):
                d = ROOT / split / cls / f"S{s:04d}"
                d.mkdir(parents=True, exist_ok=True)
                for v in range(5):
                    arr = rng.integers(0, 255, size=(64+v*8, 64+v*8, 3), dtype=np.uint8)
                    Image.fromarray(arr).save(d / f"{v}.png")


class TinyFakeBackbone(nn.Module):
    def __init__(self, embed_dim=64, patch_size=14, n_layers=12):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = n_layers
        self.conv = nn.Conv2d(3, embed_dim, patch_size, patch_size)
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(n_layers)])
        self.norm = nn.Identity()
        # positional embedding-like buffer to make features consistent
        self.pos = nn.Parameter(torch.randn(1, 64, embed_dim) * 0.02)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
        # CLS token
        self.cls = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

    @torch.no_grad()
    def forward(self, x):
        B, _, H, W = x.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size
        tok = self.conv(x).flatten(2).transpose(1, 2)  # (B, N, D)
        tok = tok + self.pos[:, :tok.shape[1], :]
        tok = torch.cat([self.cls.expand(B, -1, -1), tok], dim=1)

        # Harvest multiple "layers" by applying small random transforms
        # (simulates different transformer blocks)
        patch_layers = []
        layers_to_get = (8, 9, 10, 11)
        out = tok
        for i, blk in enumerate(self.blocks):
            # fake "layer output" with different noise scale per layer
            out = out + torch.randn_like(out) * (0.02 * (i + 1) / len(self.blocks))
            if i in layers_to_get:
                p = out[:, 1:].reshape(B, Hp, Wp, -1).permute(0, 3, 1, 2)
                patch_layers.append(p)
        cls_tok = F.normalize(out[:, 0], dim=-1)
        fused = F.normalize(torch.cat(patch_layers, dim=1), dim=1)
        return {"cls": cls_tok, "patch": fused, "patch_layers": patch_layers,
                "Hp": Hp, "Wp": Wp}


def run():
    make_fake_images()
    tfm = build_train_transform(IN_SIZE)
    train_ds = CSIGImageDataset(ROOT/"Train", transform=tfm, classes=CLASSES)
    test_ds = CSIGSampleDataset(ROOT/"Test_A", transform=tfm, classes=CLASSES)
    print(f"train images: {len(train_ds)}, test samples: {len(test_ds)}")

    bb = TinyFakeBackbone(embed_dim=64, patch_size=PATCH).eval()
    bank = MultiClassPatchCore(coreset_ratio=0.5, coreset_max=128,
                               neighbourhood_size=3, n_neighbours=3,
                               high_res=IN_SIZE, smooth_kernel=5,
                               smooth_sigma=2.0, cls_bank_weight=0.25,
                               random_proj_dim=32)

    # Fit
    from torch.utils.data import DataLoader
    ld = DataLoader(train_ds, batch_size=2, shuffle=False, num_workers=0)
    for b in ld:
        out = bb(b["image"])
        for i in range(b["image"].shape[0]):
            bank.add(b["cls_name"][i], out["patch"][i:i+1], out["cls"][i:i+1])
    bank.build()
    print(f"built banks for: {list(bank.banks.keys())}, "
          f"sizes: {[(c, b.features.shape[0]) for c,b in bank.banks.items()]}")

    # Calibrator
    cal = PerClassPercentileCalibrator()
    for b in DataLoader(CSIGSampleDataset(ROOT/"Train", transform=tfm, classes=CLASSES),
                        batch_size=1, num_workers=0):
        views = b["views"].squeeze(0)
        out = bb(views)
        cls = b["cls_name"][0]
        res = bank.predict(cls, out["patch"], out["cls"], return_map=False)
        cal.update(cls, res["image_score"].mean().unsqueeze(0))

    # Predict and write outputs
    out_dir = ROOT/"submission"
    mask_dir = out_dir/"predicted_masks"
    rows = []
    for b in DataLoader(test_ds, batch_size=1, num_workers=0):
        views = b["views"].squeeze(0)
        cls = b["cls_name"][0]
        out = bb(views)
        res = bank.predict(cls, out["patch"], out["cls"])
        pv_scores = res["image_score"]
        amap = res["anomaly_map"]  # (V,H,W)
        amap = multiview_mask_vote(amap.unsqueeze(0), beta=1.5).squeeze(0)
        img_score = aggregate_image_scores(pv_scores.unsqueeze(0), "robust_mean")
        img_score = cal.apply(cls, img_score).item()
        gf = b["group_folder"][0]
        rows.append((gf, float(img_score)))
        sd = mask_dir / cls / b["sample_id"][0]
        sd.mkdir(parents=True, exist_ok=True)
        for v in range(amap.shape[0]):
            save_mask_png(amap[v], sd/f"{v}_mask.png", target_size=448)  # test resize

    save_submission_csv(rows, out_dir/"submission.csv")
    print("CSV:")
    print((out_dir/"submission.csv").read_text())
    saved = list(mask_dir.rglob("*_mask.png"))
    print(f"saved {len(saved)} mask PNGs, e.g. {saved[0]} size: "
          f"{Image.open(saved[0]).size}")
    print("SMOKE TEST PASSED ✅")
    shutil.rmtree(ROOT)


if __name__ == "__main__":
    run()
