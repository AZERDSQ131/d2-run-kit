#!/usr/bin/env python3
# D2 — étape 4 : température de calibration par primitif (T fittée, argmax inchangé).
# Usage : python3 d2_04_calibrate.py --ckpt <best.pt> [--split val] [--out temps.json]
#   test math sans GPU : python3 d2_04_calibrate.py --selftest
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

TRAIN = os.environ.get("D0_TRAIN", "/mnt/nas/nano-gpt/common/training")
if TRAIN not in sys.path:
    sys.path.insert(0, TRAIN)

ROOT = Path(os.environ.get("D0_ROOT", "/mnt/nas/nano-gpt/D"))
TOK = ROOT / "data" / "tokenized_d2"
QTNAME = ["choice", "score", "noul"]


def fit_T(logits_list, gold_list):
    """Grille 0.25..4 + raffinement x2. Retourne (T, nll_avant, nll_apres)."""
    import torch
    import torch.nn.functional as F
    L = max(len(a) for a in logits_list)
    pad = lambda a, v: np.pad(np.asarray(a, dtype=np.float64), (0, L - len(a)),
                              constant_values=v)
    z = torch.tensor(np.stack([pad(a, -1e9) for a in logits_list]),
                     dtype=torch.float64)
    g = torch.tensor(np.stack([pad(a, 0.0) for a in gold_list]),
                     dtype=torch.float64)
    g = g / g.sum(-1, keepdim=True).clamp(min=1e-12)

    def nll(T):
        lp = F.log_softmax(z / T, dim=-1)
        return float((-(lp * g)).sum(-1).mean())

    best_T, best_n = 1.0, nll(1.0)
    grid = [round(x * 0.25, 2) for x in range(1, 17)]
    for T in grid:
        v = nll(T)
        if v < best_n:
            best_n, best_T = v, T
    for _ in range(2):
        for T in (best_T * 0.9, best_T * 1.1):
            v = nll(T)
            if v < best_n:
                best_n, best_T = v, T
    return round(float(best_T), 3), round(nll(1.0), 4), round(best_n, 4)


def selftest():
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2.0, size=(2000, 4))
    g = np.exp(z / 2.0)
    g = g / g.sum(-1, keepdims=True)   # vraie loi = softmax(z/2) -> T*=2
    T, avant, apres = fit_T([z[i] for i in range(2000)],
                            [g[i] for i in range(2000)])
    print(json.dumps({"T": T, "nll_avant": avant, "nll_apres": apres}))
    assert apres < avant and 1.5 < T < 2.5, "fit T suspect"
    print("[ok] calibration math validée")


def collect(ckpt, split, max_batches, batch_size, device):
    import torch
    import torch.nn as nn
    sys.path.insert(0, str(Path(__file__).parent))
    from d2_03_train import (load_split, batches, letter_logp, ModelConfig,
                             NanoGPT, load_state_dict_compat)
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    mcfg = ModelConfig(**ck["model_config"])
    model = NanoGPT(mcfg)
    load_state_dict_compat(model, ck["model"])
    if "head" not in ck or ck["head"] != "untied":
        model.lm_head = nn.Linear(mcfg.d_model, mcfg.vocab_size, bias=False)
    else:
        # le state contient déjà la tête non liée : rien à remplacer
        pass
    # recharge après éventuel remplacement (poids tête dans le ckpt si untied)
    try:
        load_state_dict_compat(model, ck["model"])
    except Exception as e:
        print("reload note: " + type(e).__name__, flush=True)
    model.to(device).eval()
    d = load_split(split)
    rng = np.random.default_rng(1)
    per = {t: ([], []) for t in QTNAME}
    nb = 0
    with torch.no_grad():
        for x, P, ok, LET, TGT, QTY in batches(d, batch_size, rng):
            if nb >= max_batches:
                break
            nb += 1
            xb = torch.from_numpy(x).to(device)
            pb = torch.from_numpy(P).to(device)
            logits, _ = model(xb, keep_positions=pb)
            logp, m = letter_logp(logits, torch.from_numpy(LET).to(device))
            z = (logp.exp().log()).float().cpu().numpy()
            t = TGT
            for r in range(x.shape[0]):
                for k in range(P.shape[1]):
                    if not ok[r, k]:
                        continue
                    n = int((LET[r, k] >= 0).sum())
                    per[QTNAME[int(QTY[r, k])]][0].append(z[r, k, :n])
                    per[QTNAME[int(QTY[r, k])]][1].append(t[r, k, :n])
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="")
    ap.add_argument("--split", default="val")
    ap.add_argument("--max-batches", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--out", default=str(ROOT / "d2-nano" / "temps.json"))
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    device = (args.device if args.device != "auto"
              else ("cuda" if __import__("torch").cuda.is_available() else "cpu"))
    per = collect(args.ckpt, args.split, args.max_batches, args.batch_size,
                  device)
    out = {}
    for t in QTNAME:
        zl, gl = per[t]
        if len(zl) < 20:
            out[t] = {"T": 1.0, "n": len(zl), "note": "trop peu, T=1"}
            continue
        T, avant, apres = fit_T(zl, gl)
        out[t] = {"T": T, "nll_avant": avant, "nll_apres": apres,
                  "n": len(zl)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
