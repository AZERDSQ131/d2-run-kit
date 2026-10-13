#!/usr/bin/env python3
# D2 — étape 3 : entraînement sur le blend (tête NON LIÉE + sélection OOD).
# Delta vs d0_07 : lm_head fraîche non liée, métriques par primitif (qtype),
# éval OOD (typed-decisions train) à chaque palier, best.pt sur OOD, --device.
# Usage : python3 d2_03_train.py --smoke
#   plein : python3 d2_03_train.py --epochs 3 --batch-size 8 --eval-every 500
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

TRAIN = os.environ.get("D0_TRAIN", "/mnt/nas/nano-gpt/common/training")
if TRAIN not in sys.path:
    sys.path.insert(0, TRAIN)
from config import ModelConfig          # noqa: E402
from model import NanoGPT, load_state_dict_compat   # noqa: E402

ROOT = Path(os.environ.get("D0_ROOT", "/mnt/nas/nano-gpt/D"))
TOK = ROOT / "data" / "tokenized_d2"
INIT = os.environ.get(
    "D0_INIT", "/mnt/nas/nano-gpt/G/G2-nano/pretrain_3b/checkpoints/final.pt")
BUCKETS = [128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
QTNAME = ["choice", "score", "noul"]


def load_split(split):
    d = np.load(TOK / (split + ".npz"))
    return {k: d[k] for k in d.files}


def bucket_of(n):
    for i, b in enumerate(BUCKETS):
        if n <= b:
            return i
    return None


def build_batch(d, chunk, L):
    off, ids = d["offsets"], d["ids"]
    pos, poff = d["pos"], d["pos_off"]
    let, tgt = d["letters"], d["target"]
    qty = d["qtype"]
    B = len(chunk)
    K = max(int(poff[i + 1] - poff[i]) for i in chunk)
    x = np.zeros((B, L), dtype=np.int64)
    P = np.zeros((B, K), dtype=np.int64)
    ok = np.zeros((B, K), dtype=np.bool_)
    QTY = np.zeros((B, K), dtype=np.int8)
    LET = np.zeros((B, K, 26), dtype=np.int64)
    TGT = np.zeros((B, K, 26), dtype=np.float32)
    for r, i in enumerate(chunk):
        seq = ids[off[i]:off[i + 1]]
        x[r, :len(seq)] = seq
        p = pos[poff[i]:poff[i + 1]]
        q0 = int(poff[i])
        for k in range(len(p)):
            P[r, k] = p[k]
            ok[r, k] = True
            QTY[r, k] = qty[q0 + k]
            LET[r, k] = let[q0 + k]
            TGT[r, k] = tgt[q0 + k]
    return x, P, ok, LET, TGT, QTY


def batches(d, batch_size, rng):
    lens = np.diff(d["offsets"])
    by = {}
    for i, n in enumerate(lens):
        b = bucket_of(int(n))
        if b is not None:
            by.setdefault(b, []).append(i)
    order = list(by.keys())
    rng.shuffle(order)
    for b in order:
        idxs = by[b]
        rng.shuffle(idxs)
        for s in range(0, len(idxs), batch_size):
            yield build_batch(d, idxs[s:s + batch_size], BUCKETS[b])


def count_batches(d, batch_size):
    return sum(1 for _ in batches(d, batch_size, np.random.default_rng(0)))


def letter_logp(logits, LET):
    """log-softmax restreint aux lettres d'options de chaque question."""
    idx = torch.clamp(LET, min=0)
    sel = logits.gather(2, idx)
    m = (LET >= 0).to(sel.dtype)
    sel = sel.masked_fill(m == 0, -1e9)   # -inf * 0 ferait NaN dans la loss
    return F.log_softmax(sel.float(), dim=-1), m


def step_loss(logits, LET, TGT, ok):
    logp, m = letter_logp(logits, LET)
    loss = -(logp * TGT * m).sum(-1)
    good = ok & (m.sum(-1) > 0)
    return loss[good].sum() / good.sum().clamp(min=1), logp, m, good


@torch.no_grad()
def evaluate(model, d, batch_size, device, max_batches=40):
    model.eval()
    rng = np.random.default_rng(0)
    tot = correct = nb = 0
    brier = 0.0
    bins = np.zeros(10)
    accs = np.zeros(10)
    confs = np.zeros(10)
    per = {t: [0, 0] for t in QTNAME}
    use_amp = device.startswith("cuda")
    for x, P, ok, LET, TGT, QTY in batches(d, batch_size, rng):
        if nb >= max_batches:
            break
        nb += 1
        xb = torch.from_numpy(x).to(device)
        pb = torch.from_numpy(P).to(device)
        lb = torch.from_numpy(LET).to(device)
        tb = torch.from_numpy(TGT).to(device)
        ob = torch.from_numpy(ok).to(device)
        qb = QTY
        ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
               if use_amp else torch.cpu.amp.autocast(enabled=False))
        with ctx:
            logits, _ = model(xb, keep_positions=pb)
        _, logp, m, good = step_loss(logits, lb, tb, ob)
        probs = logp.exp()
        conf, pred = probs.max(-1)
        hit = pred == tb.argmax(-1)
        tot += int(good.sum().item())
        correct += int((hit & good).sum().item())
        brier += float(((((probs - tb) ** 2) * m).sum(-1))[good].sum().item())
        g = good.cpu().numpy()
        h = (hit & good).cpu().numpy()
        for ti, t in enumerate(QTNAME):
            s = g & (qb == ti)
            per[t][0] += int(s.sum())
            per[t][1] += int(h[s].sum())
        c = conf[good].cpu().numpy()
        hh = hit[good].cpu().numpy().astype(float)
        for i in range(10):
            s = (c >= i / 10.0) & (c < (i + 1) / 10.0)
            if i == 9:
                s = s | (c >= 1.0)
            bins[i] += s.sum()
            accs[i] += hh[s].sum()
            confs[i] += c[s].sum()
    ece = 0.0
    for i in range(10):
        if bins[i] > 0:
            ece += (bins[i] / max(tot, 1)) * abs(accs[i] / bins[i]
                                                 - confs[i] / bins[i])
    model.train()
    return {"questions": tot, "acc": round(correct / max(tot, 1), 4),
            "brier": round(brier / max(tot, 1), 4), "ece": round(ece, 4),
            "by_type": {t: round(c / max(n, 1), 4) for t, (n, c) in per.items()},
            "batches": nb}


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / max(1, a.warmup)
    p = (step - a.warmup) / max(1, a.steps - a.warmup)
    return a.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, p))))


def save_ckpt(model, mcfg, step, out, name):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "model_config": vars(mcfg),
                "step": step, "head": "untied"}, out / name)
    print("[ckpt] " + str(out / name), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=INIT)
    ap.add_argument("--out", default=str(ROOT / "d2-nano"))
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--steps", type=int, default=0,
                    help="0 = calculé depuis --epochs")
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--optim", default="adamw", choices=["8bit", "adamw"])
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--log-every", type=int, default=20)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=20261013)
    args = ap.parse_args()

    device = (args.device if args.device != "auto"
              else ("cuda" if torch.cuda.is_available() else "cpu"))
    use_amp = device.startswith("cuda")
    if args.optim == "8bit":
        from optim8bit import AdamW8bit
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    ck = torch.load(args.init, map_location="cpu", mmap=True, weights_only=False)
    mcfg = ModelConfig(**ck["model_config"])
    mcfg.max_seq_len = 2048
    mcfg.grad_checkpoint = (device != "cpu")  # ckpt+CPU = init device interdite
    model = NanoGPT(mcfg)
    load_state_dict_compat(model, ck["model"])
    if hasattr(model, "lm_proj"):
        raise SystemExit("D2 pilote = tête pleine (output_rank 0 requis)")
    model.lm_head = nn.Linear(mcfg.d_model, mcfg.vocab_size, bias=False)
    nn.init.normal_(model.lm_head.weight, mean=0.0, std=0.02)
    model.to(device)
    model.train()
    n = sum(p.numel() for p in model.parameters())
    print("init=%s step=%s params=%.2fM L=%d d=%d head=untied device=%s"
          % (args.init, ck.get("step"), n / 1e6, mcfg.n_layers, mcfg.d_model,
             device), flush=True)

    if args.optim == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    else:
        opt = AdamW8bit(model.parameters(), lr=args.lr)
        try:
            opt.init_states()
        except Exception as e:
            print("init_states skip: " + type(e).__name__, flush=True)
    print("optim=" + args.optim, flush=True)

    tr = load_split("train")
    if args.steps > 0:
        max_steps = 25 if args.smoke else args.steps
    else:
        per_epoch = count_batches(tr, args.batch_size)
        max_steps = 25 if args.smoke else max(1, int(per_epoch * args.epochs))
    args.steps = max_steps  # lr_at() lit a.steps : ne jamais le laisser à 0
    print(f"steps={max_steps} (epochs={args.epochs})", flush=True)
    ood = None if args.smoke else load_split("ood")
    val = None if args.smoke else load_split("val")
    done = toks = 0
    best = -1.0
    t0 = time.time()
    mem = (lambda: torch.cuda.max_memory_allocated() / 1e9) if use_amp else (lambda: 0.0)
    if use_amp:
        torch.cuda.reset_peak_memory_stats()
    stop = False
    while not stop:
        for x, P, ok, LET, TGT, QTY in batches(tr, args.batch_size, rng):
            xb = torch.from_numpy(x).to(device)
            pb = torch.from_numpy(P).to(device)
            lb = torch.from_numpy(LET).to(device)
            tb = torch.from_numpy(TGT).to(device)
            ob = torch.from_numpy(ok).to(device)
            lr = lr_at(done, args)
            for g in opt.param_groups:
                g["lr"] = lr
            opt.zero_grad(set_to_none=True)
            ctx = (torch.autocast(device_type="cuda",
                                                dtype=torch.bfloat16)
                   if use_amp else torch.cpu.amp.autocast(enabled=False))
            with ctx:
                logits, _ = model(xb, keep_positions=pb)
            loss, logp, m, good = step_loss(logits, lb, tb, ob)
            loss.backward()
            if args.grad_clip and args.grad_clip > 0:
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                    args.grad_clip)
            else:
                gn = torch.zeros(1)
            opt.step()
            done += 1
            toks += int(xb.numel())
            if done % args.log_every == 0 or done == 1:
                el = time.time() - t0
                print("step %d loss %.4f grad %.2f lr %.2e tok/s %.0f peak_GB %.2f"
                      % (done, loss.item(), float(gn), lr, toks / max(el, 1e-9),
                         mem()), flush=True)
            if not args.smoke and args.eval_every and done % args.eval_every == 0:
                ev = evaluate(model, val, args.batch_size, device,
                              max_batches=args.eval_batches)
                eo = evaluate(model, ood, args.batch_size, device,
                              max_batches=args.eval_batches)
                print("EVAL val=" + json.dumps(ev), flush=True)
                print("EVAL ood=" + json.dumps(eo), flush=True)
                if eo["acc"] > best:
                    best = eo["acc"]
                    save_ckpt(model, mcfg, done, args.out, "best.pt")
            if not args.smoke and args.save_every and done % args.save_every == 0:
                save_ckpt(model, mcfg, done, args.out, "latest.pt")
            if done >= max_steps:
                stop = True
                break
    el = time.time() - t0
    print("done steps=%d tokens=%d %.1f min tok/s=%.0f peak_GB=%.2f best_ood=%.4f"
          % (done, toks, el / 60, toks / max(el, 1e-9), mem(), best), flush=True)
    if args.smoke:
        print("smoke: pas de checkpoint")
        return
    save_ckpt(model, mcfg, done, args.out, "d2-nano-step%d.pt" % done)


if __name__ == "__main__":
    main()
