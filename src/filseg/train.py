"""Train one fold. Validation = full-image inference + v0 post-processing + official-equivalent PQ.

Usage: python -m filseg.train --name r34_full_f0 --arch r34 --mode full --size 1024 --epochs 40 --bs 4
"""
import argparse
import copy
import json
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .data import CACHE, ROOT, TRAIN_IMG, load_index
from .dataset import FilamentDataset
from .infer import predict_probs
from .losses import FilamentLoss
from .metrics import Score, gt_insts, label_to_insts
from .model import build_model
from .postprocess import simple

VAL_THRS = (0.3, 0.4, 0.5, 0.6)
AMP = torch.float16 if os.environ.get("FILSEG_AMP") == "fp16" else torch.bfloat16   # T4/P100 lack bf16


class EMA:
    def __init__(self, model, decay=0.999):
        self.m = copy.deepcopy(model).eval()
        self.decay, self.n = decay, 0
        for p in self.m.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        for e, p in zip(self.m.state_dict().values(), model.state_dict().values()):
            if e.dtype.is_floating_point:
                e.lerp_(p, 1 - d)
            else:
                e.copy_(p)


def load_val_gt(stems):
    idx = load_index()
    return {s: [(e.entry_id, gt_insts(e.anns)) for e in idx[s]] for s in stems}


@torch.no_grad()
def validate(model, stems, gt, cfg):
    model.eval()
    model.cfg = cfg
    scores = {t: Score() for t in VAL_THRS}
    for s in stems:
        img = cv2.imread(str(TRAIN_IMG / f"{s}.jpeg"), cv2.IMREAD_GRAYSCALE)
        prob = predict_probs([model], img).astype(np.float32)
        for t in VAL_THRS:
            pred = label_to_insts(simple(prob, thr=t, min_area=100))
            for eid, g in gt[s]:
                scores[t].add(eid, g, pred)
    return {t: sc.summary() for t, sc in scores.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arch", default="effv2s")
    ap.add_argument("--mode", default="crop", choices=["crop", "full"])
    ap.add_argument("--crop", type=int, default=768)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--spi", type=int, default=4, help="samples per training image per epoch")
    ap.add_argument("--val-every", type=int, default=3)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--all-data", action="store_true", help="train on all folds (no holdout)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--throttle", type=float, default=0.0, help="sleep this fraction of each step time (quiet mode)")
    ap.add_argument("--max-minutes", type=float, default=1e9)
    cfg = vars(ap.parse_args())

    out = ROOT / "runs" / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "cfg.json").write_text(json.dumps(cfg, indent=1))
    folds = pd.read_csv(CACHE / "folds.csv")
    tr = folds.stem.tolist() if cfg["all_data"] else folds[folds.fold != cfg["fold"]].stem.tolist()
    va = folds[folds.fold == cfg["fold"]].stem.tolist()
    gt = load_val_gt(va)

    torch.manual_seed(cfg["seed"])
    if cfg["seed"]:
        import random
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    ds = FilamentDataset(tr, cfg["mode"], cfg["crop"], cfg["size"], cfg["spi"])
    dl = DataLoader(ds, batch_size=cfg["bs"], shuffle=True, num_workers=cfg["workers"], pin_memory=True,
                    drop_last=True, persistent_workers=True)
    model = build_model(cfg["arch"]).cuda().to(memory_format=torch.channels_last)
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    dec = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    opt = torch.optim.AdamW([{"params": enc, "lr": cfg["lr"] / 3}, {"params": dec, "lr": cfg["lr"]}],
                            weight_decay=1e-2)
    total = cfg["epochs"] * len(dl)
    warm = len(dl)
    base = [g["lr"] for g in opt.param_groups]
    loss_fn = FilamentLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=AMP == torch.float16)
    ema = EMA(model)
    best, step, t0, start = -1.0, 0, time.time(), 0
    rs = out / "resume.pt"
    if rs.exists():
        st = torch.load(rs, map_location="cpu", weights_only=False)
        model.load_state_dict(st["model"])
        ema.m.load_state_dict(st["ema"])
        ema.n = st["ema_n"]
        opt.load_state_dict(st["opt"])
        best, step, start = st["best"], st["step"], st["epoch"]
        print(f"resumed from epoch {start} (best {best:.4f})", flush=True)
    log = open(out / "log.jsonl", "a")
    for ep in range(start, cfg["epochs"]):
        model.train()
        te, agg = time.time(), []
        for x, y in dl:
            ts = time.time()
            x = x.cuda(non_blocking=True).to(memory_format=torch.channels_last)
            y = y.cuda(non_blocking=True)
            f = step / warm if step < warm else 0.5 * (1 + math.cos(math.pi * (step - warm) / (total - warm)))
            for g, b in zip(opt.param_groups, base):
                g["lr"] = max(b * f, 1e-6)
            with torch.autocast("cuda", dtype=AMP):
                logits = model(x)
            loss, parts = loss_fn(logits, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            agg.append([loss.item(), parts["sem"], parts["spine"], parts["bnd"]])
            step += 1
            if cfg["throttle"] > 0:
                torch.cuda.synchronize()
                time.sleep((time.time() - ts) * cfg["throttle"])
        rec = {"epoch": ep + 1, "loss": np.mean(agg, 0).round(4).tolist(), "ep_min": round((time.time() - te) / 60, 2)}
        last = ep + 1 == cfg["epochs"] or (time.time() - t0) / 60 > cfg["max_minutes"]
        if (ep + 1) % cfg["val_every"] == 0 or last:
            tv = time.time()
            res = validate(ema.m, va, gt, cfg)
            thr, r = max(res.items(), key=lambda kv: kv[1]["pq"])
            rec.update(val=res, best_thr=thr, pq=r["pq"], val_min=round((time.time() - tv) / 60, 2))
            sd = {k: v.half() if v.dtype.is_floating_point else v for k, v in ema.m.state_dict().items()}
            ck = {"cfg": cfg, "state_dict": sd, "epoch": ep + 1, "pq": r["pq"], "thr": thr}
            torch.save(ck, out / "last.pt")
            if r["pq"] > best:
                best = r["pq"]
                torch.save(ck, out / "best.pt")
        torch.save({"model": model.state_dict(), "ema": ema.m.state_dict(), "ema_n": ema.n, "opt": opt.state_dict(),
                    "best": best, "step": step, "epoch": ep + 1}, out / "resume.tmp")
        (out / "resume.tmp").replace(rs)
        print(json.dumps({k: v for k, v in rec.items() if k != "val"}), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if last:
            break
    rs.unlink(missing_ok=True)
    print(f"done. best PQ {best:.4f}  total {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
