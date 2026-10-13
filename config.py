from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 16384
    d_model: int = 640
    n_layers: int = 12
    n_heads: int = 10
    n_kv_heads: int = 2
    head_dim: int = 64
    mlp_hidden: int = 1728
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    dropout: float = 0.0
    qk_norm: bool = False  # RMSNorm sur Q/K avant RoPE -- ablation V2, defaut False pour compatibilite V1
    output_rank: int = 0  # 0 = tied full-rank head; >0 = low-rank output head
    # Pattern type Kimi (PROGRESS 2026-07-27) : 3 couches locales / 1 globale.
    # local_window=0 : comportement historique (attention complete partout).
    local_window: int = 0
    global_every: int = 4
    rope_trained_len: int = 0  # 0 = pas de NTK ; sinon theta NTK si seq > cette longueur
    grad_checkpoint: bool = False  # recompute activations au backward (train.py --grad-checkpoint)
    grad_checkpoint_every: int = 1  # 1 = chaque bloc ; 2 = 1 bloc/2 (meme memoire, -14% compute)
    # MoE style DeepSeek (G3-nano V3) : couches MoE, N experts, top-k, 1 partagé.
    # moe_layers=() : comportement historique (dense partout).
    moe_layers: tuple = ()
    moe_num_experts: int = 4
    moe_top_k: int = 2
    moe_shared: bool = True
    moe_aux_coef: float = 0.01
    # Engram (G3-nano V4) : mémoire externe de lookup n-gram, appliquée
    # une fois après l'embedding. engram=False : comportement historique.
    engram: bool = False
    engram_orders: tuple = (2, 3, 4)
    engram_buckets: int = 2048
    # mHC (G3-nano V5) : 2 flux résiduels parallèles, mélanges contraints
    # à la variété (paires de norme unitaire). mhc=False : historique.
    mhc: bool = False


def model_config_from_ckpt(raw) -> "ModelConfig":
    if not isinstance(raw, dict):
        raw = vars(raw)
    allowed = {f.name for f in ModelConfig.__dataclass_fields__.values()}
    return ModelConfig(**{k: v for k, v in raw.items() if k in allowed})


V2_CANDIDATES = {
    # Variantes issues de V2_PLAN.md, section "Architecture V2 recommandee" --
    # a comparer en ablation courte (meme volume de tokens) avant tout run long.
    # Toutes conservent vocab_size=16384 et max_seq_len=1024 (contexte natif
    # etendu = etape ulterieure du plan, apres avoir fixe profondeur/largeur).
    "v2-14L": ModelConfig(
        n_layers=14, d_model=576, n_heads=9, n_kv_heads=3, head_dim=64, mlp_hidden=1664,
    ),
    "v2-16L": ModelConfig(
        n_layers=16, d_model=512, n_heads=8, n_kv_heads=4, head_dim=64, mlp_hidden=1664,
    ),
    "v2-18L": ModelConfig(
        n_layers=18, d_model=480, n_heads=8, n_kv_heads=4, head_dim=60, mlp_hidden=1600,
    ),
}


# G2 REDEFINI 2026-09-14 : palier Jetson ~500M, scale homothetique de
# G2-nano (v2-14L + kv=1). 28L, d=1280, 20/1 tetes, SwiGLU 3584 (x2,80),
# ctx 2048, embeddings lies -> 502,73M verifies via model.py.
# L'ancien _G2_ARCH 377M (32L, d=1024, pilote DGX clos) est ecrase ;
# les .pt historiques portent leur config embarquee (model_config_from_ckpt).
_G2_ARCH = ModelConfig(
    vocab_size=16384, n_layers=28, d_model=1280,
    n_heads=20, n_kv_heads=1, head_dim=64, mlp_hidden=3584,
    max_seq_len=2048,
)

G2_CANDIDATES = {
    # G2 : 500M Jetson from scratch — voir G/G2/PLAN.md.
    "g2": _G2_ARCH,
}

# G3-test 500M — test de distillation live on-policy (student genere,
# teacher Qwen3-4B corrige, preference-train en boucle). Meme architecture
# que G2 (alias du meme _G2_ARCH, ~502,7M) sous un nom distinct pour ne
# pas polluer la campagne G2 from scratch. Jamais un nom de release.
# Voir G/G3-test/PLAN.md.
G3_TEST_500_CANDIDATES = {
    "g3-test-500": _G2_ARCH,
}

# G3-test 100M — deuxieme eleve de la boucle live black-box (teacher Ollama
# qwen3:4b, meme protocole que g3-test-500). Etudiant compact repris du
# design Qwen3 a l'echelle 100M : vocab 151936 (tokenizer Qwen3 repris tel
# quel — 151643 entrees utilisees + ids reserves jamais emis, cf.
# prepare_qwen_data.py ; embeddings lies obligatoires : 151936 x 512 = 77,8M), GQA 8/2,
# SwiGLU x2,0, RMSNorm pre-norm + qk-norm, RoPE theta=1M (copie teacher),
# sans biais, attention dense, ctx natif 2048 -> ~100,08M verifies via
# model.py (dont ~22,3M hors embeddings). KV-cache 2k bf16 ≈ 10,5 Mo.
# Jamais un nom de release. Voir G/G3-test/PLAN.md.
G3_TEST_100M_CANDIDATES = {
    "g3-test-100m": ModelConfig(
        vocab_size=151936, n_layers=10, d_model=512,
        n_heads=8, n_kv_heads=2, head_dim=64, mlp_hidden=1024,
        max_seq_len=2048, rope_theta=1000000.0, rms_norm_eps=1e-6,
        qk_norm=True,
        grad_checkpoint=True,
    ),
}

# G3-test live 0.6B — eleve MIROIR du teacher Qwen3-0.6B (decision 29/09 :
# meme geometrie que le prof au lieu du pari MLP-large initial ; la KL ne voit
# que les logits, et une geometrie identique rend la distribution du prof
# realisable par l'eleve. Squelette : vocab 151936 embeddings lies, 28L,
# d=1024, GQA 16/8, head_dim=128, SwiGLU 3072, RMSNorm + qk-norm,
# RoPE theta=1M, sans biais, ctx 2048 (entrainement seq 256 en logits-live).
# Init aleatoire (pas les poids Qwen) : le but est de mesurer la transmission.
# Voir G/G3-test/LOGITS-LIVE-06B.md. Jamais un nom de release.
G3_LIVE_06B_CANDIDATES = {
    "g3-live-06b": ModelConfig(
        vocab_size=151936, n_layers=28, d_model=1024,
        n_heads=16, n_kv_heads=8, head_dim=128, mlp_hidden=3072,
        max_seq_len=2048, rope_theta=1000000.0, rms_norm_eps=1e-6,
        qk_norm=True,
        grad_checkpoint=True,
    ),
}

# Palier Jetson ~60M, sortie groupée avec G2 1B — G/G2-nano/PLAN.md.
# Même forme que G1-nano (v2-14L + n_kv_heads=1 + ctx 2048).
# Ne pas confondre avec ARCHIVED_CANDIDATES["g2-nano"] (2 couches, tête factorisée).
G2_NANO_60_CANDIDATES = {
    "g2-nano-60": ModelConfig(
        n_layers=14, d_model=576, n_heads=9, n_kv_heads=1, head_dim=64,
        mlp_hidden=1664, max_seq_len=2048,
    ),
}

# G3-nano 100M — palier Jetson, decoder-only, DeepSeek-inspired (V1 dense).
# 12L, d=768, 12/4 têtes GQA, SwiGLU 3072, ctx 2048, embeddings liés -> ~116M.
# Voir G/G3-nano/PLAN.md pour le plan itératif complet (V1→V5).
G3_NANO_100_CANDIDATES = {
    "g3-nano-100": ModelConfig(
        vocab_size=16384, n_layers=12, d_model=768,
        n_heads=12, n_kv_heads=4, head_dim=64, mlp_hidden=3072,
        max_seq_len=2048,
        grad_checkpoint=True,
    ),
}

# G3-nano V3 — V1 + MoE couches 4 et 8 : 4 experts, top-2, 1 partagé.
# ~173M params totaux (~145M actifs/token). Voir G/G3-nano/PLAN.md.
G3_NANO_100_MOE_CANDIDATES = {
    "g3-nano-100-moe": ModelConfig(
        vocab_size=16384, n_layers=12, d_model=768,
        n_heads=12, n_kv_heads=4, head_dim=64, mlp_hidden=3072,
        max_seq_len=2048,
        grad_checkpoint=True,
        moe_layers=(4, 8), moe_num_experts=4, moe_top_k=2,
        moe_shared=True, moe_aux_coef=0.01,
    ),
}

# G3-nano V4 — V1 + Engram : lookup 2/3/4-grams, 2048 buckets × 768.
# ~4,72M params additionnels. Voir G/G3-nano/PLAN.md.
G3_NANO_100_ENGRAM_CANDIDATES = {
    "g3-nano-100-engram": ModelConfig(
        vocab_size=16384, n_layers=12, d_model=768,
        n_heads=12, n_kv_heads=4, head_dim=64, mlp_hidden=3072,
        max_seq_len=2048,
        grad_checkpoint=True,
        engram=True, engram_orders=(2, 3, 4), engram_buckets=2048,
    ),
}

# G3-nano V5 — V1 + mHC : 2 flux résiduels, mélanges de norme unitaire
# (init [1, 0] exacte = V1 à l'init). +74 scalaires, ~116,26M au total.
# Voir G/G3-nano/PLAN.md.
G3_NANO_100_MHC_CANDIDATES = {
    "g3-nano-100-mhc": ModelConfig(
        vocab_size=16384, n_layers=12, d_model=768,
        n_heads=12, n_kv_heads=4, head_dim=64, mlp_hidden=3072,
        max_seq_len=2048,
        grad_checkpoint=True,
        mhc=True,
    ),
}

# G3-distill — eleves ~100M distilles de Qwen3-4B (tokenizer Qwen3, vocab
# 151936). Contrainte structurante : embeddings lies obligatoires, d<=512
# (151936 x 512 = 77,8M). Design copie du teacher (GQA, SwiGLU, RMSNorm
# pre-norm + qk-norm, RoPE theta=1M, sans biais, attention dense plein 2k).
# Voir G/G3/PLAN.md. Boucle d'entrainement : distill.py (pas train.py :
# bins uint32 + perte KL chunkee), mais les variantes restent utilisables
# dans train.py pour une baseline CE pure.
G3_DISTILL_CANDIDATES = {
    # S1 « equilibree » (recommandee) : 10L, d=512, GQA 8/2, SwiGLU 1024
    # -> ~100,08M (dont ~22,3M hors embeddings). KV-cache 2k ≈ 10,5 Mo.
    "g3-distill-s1": ModelConfig(
        vocab_size=151936, n_layers=10, d_model=512,
        n_heads=8, n_kv_heads=2, head_dim=64, mlp_hidden=1024,
        max_seq_len=2048, rope_theta=1000000.0, rms_norm_eps=1e-6,
        qk_norm=True,
        grad_checkpoint=True,
    ),
    # S0 « profonde-fine » : 12L, d=448, GQA 8/2, SwiGLU 1280 -> ~95,6M.
    # A/B contre S1 si le budget le permet.
    "g3-distill-s0": ModelConfig(
        vocab_size=151936, n_layers=12, d_model=448,
        n_heads=8, n_kv_heads=2, head_dim=64, mlp_hidden=1280,
        max_seq_len=2048, rope_theta=1000000.0, rms_norm_eps=1e-6,
        qk_norm=True,
        grad_checkpoint=True,
    ),
    # S2 « FFN large » (ablation) : 8L, d=512, GQA 8/4, SwiGLU 1536 (x3,0,
    # proche du ratio teacher x3,8) -> ~103M. Hypothese « MLP > profondeur ».
    "g3-distill-s2": ModelConfig(
        vocab_size=151936, n_layers=8, d_model=512,
        n_heads=8, n_kv_heads=4, head_dim=64, mlp_hidden=1536,
        max_seq_len=2048, rope_theta=1000000.0, rms_norm_eps=1e-6,
        qk_norm=True,
        grad_checkpoint=True,
    ),
}

ARCHIVED_CANDIDATES = {
    # Pistes archivees sous G/archive/ -- reprise de checkpoints historiques.
    "g1-a": ModelConfig(
        n_layers=14, d_model=2560, n_heads=40, n_kv_heads=1, head_dim=64, mlp_hidden=7398,
    ),
    "g1-b": ModelConfig(
        n_layers=24, d_model=1792, n_heads=28, n_kv_heads=7, head_dim=64, mlp_hidden=6224,
        max_seq_len=2048,
    ),
    "g1": ModelConfig(
        n_layers=26, d_model=1280, n_heads=20, n_kv_heads=5, head_dim=64, mlp_hidden=3456,
        max_seq_len=2048,
    ),
    "g1-pro": ModelConfig(
        n_layers=34, d_model=2304, n_heads=36, n_kv_heads=9, head_dim=64, mlp_hidden=6144,
    ),
    "g2-nano": ModelConfig(
        n_layers=2, d_model=1536, n_heads=24, n_kv_heads=1, head_dim=64,
        mlp_hidden=2200, max_seq_len=2048, output_rank=64,
    ),
    "g2-nano-8l": ModelConfig(
        n_layers=8, d_model=832, n_heads=13, n_kv_heads=1, head_dim=64,
        mlp_hidden=1200, max_seq_len=2048, output_rank=64,
    ),
    "g2-draft": ModelConfig(
        n_layers=2, d_model=64, n_heads=1, n_kv_heads=1, head_dim=64,
        mlp_hidden=192, max_seq_len=2048, output_rank=8,
    ),
    "g3-nano": ModelConfig(
        n_layers=2, d_model=64, n_heads=1, n_kv_heads=1, head_dim=64,
        mlp_hidden=192, max_seq_len=2048, output_rank=8,
    ),
    # Alias deprecie : run DGX lance sous le nom G3 avant renommage 2026-08-30.
    "g3": _G2_ARCH,
}

# Compatibilite imports historiques
G1_CANDIDATES = {k: ARCHIVED_CANDIDATES[k] for k in ("g1-a", "g1-b", "g1", "g1-pro")}
G2_NANO_CANDIDATES = {k: ARCHIVED_CANDIDATES[k] for k in ("g2-nano", "g2-nano-8l")}
G2_DRAFT_CANDIDATES = {"g2-draft": ARCHIVED_CANDIDATES["g2-draft"]}
G3_NANO_CANDIDATES = {"g3-nano": ARCHIVED_CANDIDATES["g3-nano"]}
G3_NANO_100 = G3_NANO_100_CANDIDATES  # alias
G3_CANDIDATES = {"g3": ARCHIVED_CANDIDATES["g3"]}


@dataclass
class TrainConfig:
    data_dir: str = "data/tokenized"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    batch_size: int = 2          # micro-batch (per step) -- reduit vs le max calibre (4) pour garder de la marge
                                  # face aux fluctuations memoire des autres services du Jetson sur une run longue
    grad_accum_steps: int = 64   # effective batch = batch_size * grad_accum_steps = 128
    seq_len: int = 1024

    max_steps: int = 100_000
    warmup_steps: int = 1_000
    lr: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95

    eval_interval: int = 500
    eval_iters: int = 50
    log_interval: int = 20
    checkpoint_interval: int = 500

    dtype: str = "bfloat16"
    compile: bool = True
    seed: int = 1337


@dataclass
class SFTConfig:
    data_dir: str = "data/sft_tokenized"
    checkpoint_dir: str = "checkpoints/sft"
    log_dir: str = "logs/sft"

    batch_size: int = 2          # a recalibrer sur Jetson (seq_len plus court que le pretraining)
    grad_accum_steps: int = 32   # effective batch = batch_size * grad_accum_steps = 64
    seq_len: int = 512

    epochs: int = 3
    max_steps: int = 0           # calcule a partir de epochs et de la taille du dataset (cf. sft_train.py)
    warmup_steps: int = 50
    lr: float = 3e-5             # nettement plus faible que le pretraining (3e-4), fine-tuning depuis un modele deja entraine
    min_lr: float = 3e-6
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95

    eval_interval: int = 100
    eval_iters: int = 20
    log_interval: int = 10
    checkpoint_interval: int = 200

    dtype: str = "bfloat16"
    compile: bool = True
    seed: int = 1337
