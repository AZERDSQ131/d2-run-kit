# d2-run-kit — kit pilote D2 (G-serie)

Tout pour lancer le pilote beat-Jev sur une 4090, tirage datacenter.

## Récupération (1-2 min sur l'instance)

```bash
git clone https://github.com/AZERDSQ131/d2-run-kit.git d2 && cd d2
cat base-weights.pt.* > base-weights.pt && rm base-weights.pt.a?
mkdir -p data/tokenized_d2 && mv data/*.npz data/tokenized_d2/
python3 -c "import torch; c=torch.load('base-weights.pt',map_location='cpu',weights_only=False); print(round(sum(p.numel() for p in c['model'].values())/1e6,1),'M')"
```

## Lancement (~15 min, ~0,15 $)

```bash
D0_TRAIN=$PWD D0_ROOT=$PWD PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32 \
nohup python3 d2_03_train.py --init $PWD/base-weights.pt --out $PWD/d2-nano \
  --epochs 3 --batch-size 8 --eval-every 500 --log-every 20 > train.log 2>&1 &
```

Puis calibration + destroy (voir G-serie `D/D2-RUN.md`).

Contenu : `d2_03_train.py` (tête non liée, OOD, best.pt), `d2_04_calibrate.py`
(T par primitif), `model.py` + `config.py` (NanoGPT), `data/` (blend v2
tokenizé : 18k questions + teacher Bosun), `base-weights.pt` (G2-nano, 60M,
découpé pour GitHub).
