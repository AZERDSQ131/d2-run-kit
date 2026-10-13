import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from config import ModelConfig

import functools


@functools.lru_cache(maxsize=1)
def _is_tegra_sm87() -> bool:
    """True sur Jetson Orin (sm_87, uniquement Tegra).

    Sur sm_87 les kernels SDPA fusionnes refusent Q != KV avec
    ``enable_gqa=True`` (repli silencieux sur le backend MATH ~2x plus
    lent) et le backward flash sous inductor exige des K/V contigus
    (``repeat_interleave``, pas ``expand``). Partout ailleurs (Ampere+,
    MPS, CPU) le GQA natif est disponible : pas de replication 20x du
    KV (critique a ``n_kv_heads=1`` : 20x de trafic memoire en moins).
    sm_87 n'existe que sur Tegra Orin (pas de GPU desktop en 8.7),
    donc ``cap == (8, 7)`` discrimine sans acces driver specifique.
    """
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.get_device_capability() == (8, 7)
    except Exception:
        return False


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype,
                     trained_len: int = 0):
    if trained_len and seq_len > trained_len:
        # NTK-aware (alternative documentee si KDA trop lourd, docs/PROGRESS.md).
        scale = seq_len / float(trained_len)
        theta = theta * (scale ** (head_dim / (head_dim - 2)))
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)
    cos = freqs.cos().to(dtype)
    sin = freqs.sin().to(dtype)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_heads, T, head_dim)
    x1, x2 = x[..., ::2], x[..., 1::2]
    cos = cos[None, None, :x.size(2), :]
    sin = sin[None, None, :x.size(2), :]
    rotated1 = x1 * cos - x2 * sin
    rotated2 = x1 * sin + x2 * cos
    out = torch.stack([rotated1, rotated2], dim=-1).flatten(-2)
    return out.to(x.dtype)


def _sdpa_maybe_gqa(q, k, v, *, attn_mask=None, is_causal=False, dropout_p=0.0):
    kwargs = {"dropout_p": dropout_p}
    if attn_mask is None:
        kwargs["is_causal"] = is_causal
    else:
        kwargs["attn_mask"] = attn_mask
    try:
        return F.scaled_dot_product_attention(q, k, v, enable_gqa=True, **kwargs)
    except TypeError:
        n_rep = q.size(1) // k.size(1)
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        return F.scaled_dot_product_attention(q, k, v, **kwargs)


def sliding_window_sdpa(q, k, v, window: int, query_start: int, dropout_p: float):
    """Attention causale a fenetre, sans masque (T,T) plein (OOM Jetson a 4k)."""
    Tq = q.size(2)
    chunk = min(max(int(window), 1), 256)
    out = q.new_empty(q.shape)
    device = q.device
    for t0 in range(0, Tq, chunk):
        t1 = min(Tq, t0 + chunk)
        abs_q0 = query_start + t0
        abs_q1 = query_start + t1
        k0 = max(0, abs_q0 - window + 1)
        k1 = abs_q1
        qpos = torch.arange(abs_q0, abs_q1, device=device)
        kpos = torch.arange(k0, k1, device=device)
        mask = (kpos.unsqueeze(0) <= qpos.unsqueeze(1)) & (
            (qpos.unsqueeze(1) - kpos.unsqueeze(0)) < window
        )
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out[:, :, t0:t1] = _sdpa_maybe_gqa(
                q[:, :, t0:t1], k[:, :, k0:k1], v[:, :, k0:k1],
                attn_mask=mask, dropout_p=dropout_p,
            )
    return out


def chunked_causal_sdpa(q, k, v, query_start: int, dropout_p: float, chunk: int = 128):
    """Causal complet sans Flash a grande T (8k global OOM sur Orin)."""
    Tq = q.size(2)
    out = q.new_empty(q.shape)
    device = q.device
    for t0 in range(0, Tq, chunk):
        t1 = min(Tq, t0 + chunk)
        abs_q0 = query_start + t0
        abs_q1 = query_start + t1
        k1 = abs_q1
        qpos = torch.arange(abs_q0, abs_q1, device=device)
        kpos = torch.arange(0, k1, device=device)
        mask = kpos.unsqueeze(0) <= qpos.unsqueeze(1)
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
            out[:, :, t0:t1] = _sdpa_maybe_gqa(
                q[:, :, t0:t1], k[:, :, :k1], v[:, :, :k1],
                attn_mask=mask, dropout_p=dropout_p,
            )
    return out


def _recompute_block(block: "Block", x: torch.Tensor,
                    cos: torch.Tensor, sin: torch.Tensor,
                    amp_dtype=None) -> torch.Tensor:
    """Gradient checkpointing : ne garde que l'entree de couche, recalcule
    le reste au backward. amp_dtype = dtype autocast capture au forward
    (le backward n'est pas sous autocast) ; None = pas d'autocast."""
    def fn(x_, cos_, sin_):
        if amp_dtype is None:
            return block(x_, cos_, sin_)
        with torch.autocast(device_type="cuda" if x_.is_cuda else "cpu",
                            dtype=amp_dtype):
            return block(x_, cos_, sin_)
    return torch.utils.checkpoint.checkpoint(fn, x, cos, sin, use_reentrant=False)


class GQAAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.n_rep = cfg.n_heads // cfg.n_kv_heads
        self.local_window = 0

        self.wq = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.wo = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)
        self.dropout = cfg.dropout

        self.qk_norm = cfg.qk_norm
        if self.qk_norm:
            self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)
            self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                past_key_value=None, use_cache: bool = False, attn_mask=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        past_len = 0 if past_key_value is None else past_key_value[0].size(2)
        q = apply_rope(q, cos[past_len:past_len + T], sin[past_len:past_len + T])
        k = apply_rope(k, cos[past_len:past_len + T], sin[past_len:past_len + T])

        if past_key_value is not None:
            k = torch.cat((past_key_value[0], k), dim=2)
            v = torch.cat((past_key_value[1], v), dim=2)
        present = (k, v) if use_cache else None

        local_w = int(getattr(self, "local_window", 0) or 0)
        drop = self.dropout if self.training else 0.0
        if local_w > 0 and attn_mask is None:
            out = sliding_window_sdpa(q, k, v, local_w, past_len, drop)
        elif past_key_value is None and attn_mask is None:
            # Chemin pretraining pur (pas de masque, pas de cache).
            # Tegra/sm87 : enable_gqa=True est REFUSE par les kernels
            # fusionnes (Q != KV -> repli MATH lent + scores T*T).
            # repeat_interleave CONTIGU (pas expand : le backward flash
            # sous inductor exige des strides normaux) pour debloquer
            # EFFICIENT_ATTENTION en eager et flash sous compile.
            # Cout : ~5 Mo/couche, negligeable.
            # Hors Tegra (Ampere+/datacenter) : GQA natif via
            # _sdpa_maybe_gqa, sans replication (20x de trafic KV en moins
            # a n_kv_heads=1) et backend flash/cudnn auto. Voir
            # _is_tegra_sm87().
            if k.size(2) <= 4096:
                if self.n_rep > 1 and _is_tegra_sm87():
                    k = k.repeat_interleave(self.n_rep, dim=1)
                    v = v.repeat_interleave(self.n_rep, dim=1)
                    out = F.scaled_dot_product_attention(
                        q, k, v, is_causal=True, dropout_p=drop)
                else:
                    out = _sdpa_maybe_gqa(
                        q, k, v, is_causal=True, dropout_p=drop)
            else:
                out = chunked_causal_sdpa(q, k, v, past_len, drop)
        elif attn_mask is not None and past_key_value is None:
            # SFT packe ou fenetre locale (pattern Kimi-lite 3:1).
            # Tegra : repeat contigu + backend force (pas de flash GQA).
            # Hors Tegra : GQA natif, backend auto (flash/cudnn inclus).
            if _is_tegra_sm87():
                with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
                    if self.n_rep > 1:
                        k = k.repeat_interleave(self.n_rep, dim=1)
                        v = v.repeat_interleave(self.n_rep, dim=1)
                    out = F.scaled_dot_product_attention(
                        q, k, v, attn_mask=attn_mask, dropout_p=drop)
            else:
                out = _sdpa_maybe_gqa(
                    q, k, v, attn_mask=attn_mask, dropout_p=drop)
        else:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
            if attn_mask is None:
                key_len = k.size(2)
                query_positions = torch.arange(T, device=x.device) + past_len
                key_positions = torch.arange(key_len, device=x.device)
                attn_mask = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=drop,
            )
        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        result = self.wo(out)
        return (result, present) if use_cache else result


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.w_up = nn.Linear(cfg.d_model, cfg.mlp_hidden, bias=False)
        self.w_down = nn.Linear(cfg.mlp_hidden, cfg.d_model, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class MoELayer(nn.Module):
    """MoE style DeepSeek : N experts SwiGLU + 1 expert partagé, routing top-k.

    Retourne un tenseur (pas de tuple) pour rester compatible avec le
    gradient checkpointing de _recompute_block ; l'aux-loss Switch est
    stockée dans self._aux_loss et additionnée dans NanoGPT.forward.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.num_experts = int(getattr(cfg, "moe_num_experts", 4) or 4)
        self.top_k = int(getattr(cfg, "moe_top_k", 2) or 2)
        self.has_shared = bool(getattr(cfg, "moe_shared", True))
        self.router = nn.Linear(cfg.d_model, self.num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(cfg) for _ in range(self.num_experts)])
        self.shared = SwiGLU(cfg) if self.has_shared else None
        self._aux_loss = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gates = torch.softmax(self.router(x), dim=-1)
        top_gates, top_idx = torch.topk(gates, self.top_k, dim=-1)
        top_gates = top_gates / top_gates.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        stacked = torch.stack([e(x) for e in self.experts], dim=0)  # (N,B,T,D)
        w = torch.zeros_like(gates).scatter_(-1, top_idx, top_gates)  # (B,T,N)
        out = torch.einsum("btn,nbtd->btd", w, stacked)
        if self.shared is not None:
            out = out + self.shared(x)
        # aux-loss Switch : N * sum(fraction_routée * prob_moyenne)
        with torch.no_grad():
            flat_idx = top_idx.reshape(-1)
            counts = torch.zeros(
                self.num_experts, device=x.device, dtype=torch.float32
            ).scatter_add_(
                0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32)
            )
            frac = counts / max(flat_idx.numel(), 1)
        mean_prob = gates.float().mean(dim=(0, 1))
        self._aux_loss = (frac.to(mean_prob.dtype) * mean_prob).sum() * self.num_experts
        return out


class EngramMemory(nn.Module):
    """G3-nano V4 : mémoire externe de lookup n-gram (ordres 2/3/4).

    Pour chaque position t et chaque ordre n, le n-gramme causal
    (tokens t-n+1..t) est hashé vers un bucket dont l'embedding appris
    est ajouté à l'état caché via une porte initialisée à zéro : le
    modèle démarre donc exactement comme sa baseline dense (A/B propre).

    Causalité : seuls les tokens <= t contribuent (contexte gauche
    manquant = zéro). En décodage incrémental (past + 1 token) le
    contexte gauche absent est approximé à zéro — exact en prefill,
    en train et en éval lm-eval (forward complet). Entraînable par
    gradient via les lookups d'embeddings.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.orders = tuple(int(n) for n in (getattr(cfg, "engram_orders", (2, 3, 4)) or (2, 3, 4)))
        self.buckets = int(getattr(cfg, "engram_buckets", 2048) or 2048)
        mult = 9113823
        self.pow_mult = [1]
        for _ in range(max(self.orders)):
            self.pow_mult.append((self.pow_mult[-1] * mult) % self.buckets)
        self.tables = nn.ModuleList(
            [nn.Embedding(self.buckets, cfg.d_model) for _ in self.orders]
        )
        self.gate = nn.Parameter(torch.zeros(len(self.orders), cfg.d_model))

    def _hash_ngram(self, idx: torch.Tensor, n: int) -> torch.Tensor:
        h = idx % self.buckets
        for j in range(1, n):
            shifted = torch.zeros_like(idx)
            if idx.size(1) > j:
                shifted[:, j:] = idx[:, :-j]
            h = (h + self.pow_mult[j] * shifted) % self.buckets
        return h

    def forward(self, idx: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        mem = torch.zeros_like(x)
        for k, (n, table) in enumerate(zip(self.orders, self.tables)):
            mem = mem + self.gate[k] * table(self._hash_ngram(idx, n))
        return x + mem


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int = 0):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = GQAAttention(cfg)
        window = int(getattr(cfg, "local_window", 0) or 0)
        every = int(getattr(cfg, "global_every", 4) or 4)
        is_global = window <= 0 or (layer_idx % every == every - 1) or (
            layer_idx == cfg.n_layers - 1
        )
        self.attn.local_window = 0 if is_global else window
        self.mlp_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        moe_layers = set(getattr(cfg, "moe_layers", ()) or ())
        self.is_moe = layer_idx in moe_layers
        self.mlp = MoELayer(cfg) if self.is_moe else SwiGLU(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                past_key_value=None, use_cache: bool = False, attn_mask=None):
        attn_result = self.attn(self.attn_norm(x), cos, sin, past_key_value, use_cache, attn_mask)
        if use_cache:
            attn_result, present = attn_result
        x = x + attn_result
        x = x + self.mlp(self.mlp_norm(x))
        return (x, present) if use_cache else x


class NanoGPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        # Engram créé APRÈS l'init des poids partagés : à seed égal, tous
        # les poids communs avec la baseline dense sont bit-identiques, et
        # la porte à zéro donne une identité fonctionnelle stricte à l'init.
        self.engram = None
        self.blocks = nn.ModuleList(
            [Block(cfg, layer_idx=i) for i in range(cfg.n_layers)]
        )
        # mHC (G3-nano V5) : mélanges 2-flux contraints à la variété
        # (paires de norme unitaire). Constantes [1, 0] : aucun tirage RNG,
        # poids partagés bit-identiques à V1 à seed égal, et identité
        # fonctionnelle stricte à l'init — mais gradient non nul sur le
        # 2e flux dès le pas 0 (pas d'endormissement comme au softmax).
        self.mhc_read = None
        self.mhc_write = None
        self.mhc_out = None
        if bool(getattr(cfg, "mhc", False)):
            self.mhc_read = nn.ParameterList(
                [nn.Parameter(torch.tensor([1.0, 0.0]))
                 for _ in range(cfg.n_layers)]
            )
            self.mhc_write = nn.ParameterList(
                [nn.Parameter(torch.tensor([1.0, 0.0]))
                 for _ in range(cfg.n_layers)]
            )
            self.mhc_out = nn.Parameter(torch.tensor([1.0, 0.0]))
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        if cfg.output_rank:
            # A low-rank output head reduces the per-token vocab projection on
            # very small edge models.  It is opt-in so G0/G1 checkpoints keep
            # their exact tied-embedding layout.
            self.lm_proj = nn.Linear(cfg.d_model, cfg.output_rank, bias=False)
            self.lm_head = nn.Linear(cfg.output_rank, cfg.vocab_size, bias=False)
        else:
            self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
            self.lm_head.weight = self.tok_emb.weight  # embeddings partagés

        self.apply(self._init_weights)
        # scaled init for residual projections, cf. GPT-2 / Llama
        for name, p in self.named_parameters():
            if name.endswith("w_down.weight") or name.endswith("wo.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))
        if bool(getattr(cfg, "engram", False)):
            self.engram = EngramMemory(cfg)
            with torch.no_grad():
                gen = torch.Generator().manual_seed(0xE64A4)
                for table in self.engram.tables:
                    table.weight.copy_(torch.randn(table.weight.shape, generator=gen) * 0.02)

        self._rope_cache = {}

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _get_rope(self, seq_len: int, device, dtype):
        key = (seq_len, device, dtype)
        if key not in self._rope_cache:
            self._rope_cache[key] = build_rope_cache(
                seq_len, self.cfg.head_dim, self.cfg.rope_theta, device, dtype,
                trained_len=int(getattr(self.cfg, "rope_trained_len", 0) or 0),
            )
        return self._rope_cache[key]

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None,
                past_key_values=None, use_cache: bool = False, attn_mask=None,
                keep_positions: torch.Tensor = None):
        B, T = idx.shape
        past_len = 0 if past_key_values is None else past_key_values[0][0].size(2)
        assert past_len + T <= self.cfg.max_seq_len, \
            f"sequence length {past_len + T} > max_seq_len {self.cfg.max_seq_len}"

        x = self.dropout(self.tok_emb(idx))
        if self.engram is not None:
            x = self.engram(idx, x)
        cos, sin = self._get_rope(past_len + T, idx.device, x.dtype)

        presents = []
        use_ckpt = bool(getattr(self.cfg, "grad_checkpoint", False)) \
            and self.training \
            and (targets is not None or keep_positions is not None) \
            and past_key_values is None and not use_cache
        ckpt_every = max(int(getattr(self.cfg, "grad_checkpoint_every", 1) or 1), 1)
        amp_dtype = None
        if use_ckpt and torch.is_autocast_enabled():
            try:
                amp_dtype = torch.get_autocast_dtype(x.device.type)
            except Exception:
                amp_dtype = torch.bfloat16
        n_blocks = len(self.blocks)
        use_mhc = (
            self.mhc_read is not None
            and not use_cache
            and past_key_values is None
        )
        # Garde anti-NaN (mémoire CUDA fraîche = zéros : norme nulle
        # possible si les mélanges n'ont jamais été initialisés).
        def _unit2(v):
            return v / v.norm(p=2).clamp_min(1e-6)
        if use_mhc:
            # V5 mHC : 2 flux résiduels //, mélange lecture (simplexe
            # sphérique) → bloc V1 → redistribution. Mixage position par
            # position : causalité préservée. Mode cache (génération) =
            # repli V1 ci-dessous ; train + éval lm-eval passent ici.
            s0, s1 = x, x.clone()
            for i, block in enumerate(self.blocks):
                wr = _unit2(self.mhc_read[i])
                h = wr[0] * s0 + wr[1] * s1
                if use_ckpt and i % ckpt_every == 0:
                    y = _recompute_block(block, h, cos, sin, amp_dtype)
                else:
                    y = block(h, cos, sin, None, False, attn_mask)
                d = y - h
                ww = _unit2(self.mhc_write[i])
                s0 = s0 + ww[0] * d
                s1 = s1 + ww[1] * d
            wo = _unit2(self.mhc_out)
            x = wo[0] * s0 + wo[1] * s1
        else:
            for i, block in enumerate(self.blocks):
                past = None if past_key_values is None else past_key_values[i]
                if use_ckpt and i % ckpt_every == 0:
                    x = _recompute_block(block, x, cos, sin, amp_dtype)
                    continue
                block_result = block(x, cos, sin, past, use_cache, attn_mask)
                if use_cache:
                    x, present = block_result
                    presents.append(present)
                else:
                    x = block_result
        x = self.final_norm(x)

        if targets is not None:
            logits = self._logits(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
            aux = None
            for b in self.blocks:
                a = getattr(getattr(b, "mlp", None), "_aux_loss", None)
                if a is not None:
                    aux = a if aux is None else aux + a
            if aux is not None:
                loss = loss + float(getattr(self.cfg, "moe_aux_coef", 0.01)) * aux
            return (logits, loss, presents) if use_cache else (logits, loss)
        else:
            if keep_positions is not None:
                gi = keep_positions.unsqueeze(-1).expand(-1, -1, x.size(-1))
                logits = self._logits(x.gather(1, gi))
            else:
                logits = self._logits(x[:, [-1], :])
            return (logits, None, presents) if use_cache else (logits, None)

    def _logits(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.output_rank:
            x = self.lm_proj(x)
        return self.lm_head(x)

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                 top_k: int = None, repetition_penalty: float = 1.0, stop_token_id: int = None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.cfg.max_seq_len else idx[:, -self.cfg.max_seq_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    seen = idx[b].unique()
                    scores = logits[b, seen]
                    logits[b, seen] = torch.where(scores > 0, scores / repetition_penalty, scores * repetition_penalty)

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_id], dim=1)
            if stop_token_id is not None and (next_id == stop_token_id).all():
                break
        return idx

    @torch.no_grad()
    def generate_stream(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
                        top_k: int = None, repetition_penalty: float = 1.0, stop_token_id: int = None):
        """Yield token ids incrementally while reusing a per-layer KV cache."""
        if idx.size(1) == 0:
            raise ValueError("generate_stream requires at least one prompt token")
        temperature = max(float(temperature), 1e-5)
        _, _, past_key_values = self(idx, use_cache=True)
        generated = idx

        for _ in range(max_new_tokens):
            logits, _, past_key_values = self(
                generated[:, -1:], past_key_values=past_key_values, use_cache=True
            )
            logits = logits[:, -1, :] / temperature

            if repetition_penalty != 1.0:
                for b in range(generated.size(0)):
                    seen = generated[b].unique()
                    scores = logits[b, seen]
                    logits[b, seen] = torch.where(
                        scores > 0, scores / repetition_penalty, scores * repetition_penalty
                    )
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            next_id = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
            generated = torch.cat([generated, next_id], dim=1)
            yield next_id[:, 0], past_key_values
            if stop_token_id is not None and (next_id == stop_token_id).all():
                break


def enable_kimi_lite(
    model: "NanoGPT",
    *,
    local_window: int = 1024,
    global_every: int = 4,
    max_seq_len: Optional[int] = None,
    rope_trained_len: Optional[int] = None,
) -> None:
    """Pattern documenté Kimi / G1-nano PLAN : 3 couches fenêtre + 1 globale.

    Pas de KDA / FlashKDA (Tegra). NTK si seq > rope_trained_len.
    local_window=0 (défaut ModelConfig) : no-op historique.
    """
    cfg = model.cfg
    native = int(cfg.max_seq_len)
    if rope_trained_len is None:
        rope_trained_len = native if native > 0 else 2048
    if max_seq_len is not None:
        cfg.max_seq_len = int(max_seq_len)
    cfg.local_window = int(local_window)
    cfg.global_every = int(global_every)
    cfg.rope_trained_len = int(rope_trained_len)
    n = cfg.n_layers
    every = max(int(global_every), 1)
    for i, block in enumerate(model.blocks):
        is_global = (i % every == every - 1) or (i == n - 1)
        block.attn.local_window = 0 if is_global else int(local_window)
    model._rope_cache.clear()


def build_cuda(cfg: "ModelConfig", dtype=torch.bfloat16) -> "NanoGPT":
    """Init directement sur GPU, tenseur par tenseur.

    Sur Tegra unifie, toute allocation CUDA unique ~1 Go echoue
    (NvMap error 12) — y compris le `.to(device)` d'un 500M fp32.
    Ici : squelette meta -> `to_empty` (allocs unitaires, la plus grosse
    ~84 Mo) -> init manuelle identique a `NanoGPT.__init__`.
    Poids finaux en `dtype` (bf16 : ~1 Go a 500M). Seed CUDA a fixer
    par l'appelant (`torch.cuda.manual_seed_all`) pour reproduire.
    """
    with torch.device("meta"):
        model = NanoGPT(cfg)
    model = model.to_empty(device="cuda")
    model = model.to(dtype=dtype)
    if not cfg.output_rank:
        # to_empty ne preserve pas le partage : re-lier (21M a 500M).
        model.lm_head.weight = model.tok_emb.weight
    with torch.no_grad():
        model.apply(model._init_weights)
        for name, p in model.named_parameters():
            if name.endswith("w_down.weight") or name.endswith("wo.weight"):
                nn.init.normal_(
                    p, mean=0.0,
                    std=0.02 / math.sqrt(2 * cfg.n_layers))
        if bool(getattr(cfg, "mhc", False)):
            # Les mélanges mHC ne sont ni Linear ni Embedding : apply ne les
            # touche pas — sans ceci ils restent à l'état vide (zéros →
            # norme nulle → NaN). Reset explicite à l'identité [1, 0].
            for v in (*model.mhc_read, *model.mhc_write, model.mhc_out):
                v.copy_(torch.tensor([1.0, 0.0], device=v.device,
                                      dtype=v.dtype))
    return model


def load_state_dict_compat(model, state_dict):
    """Charge un state_dict, y compris ceux produits par torch.compile (_orig_mod.*)."""
    if any(k.startswith("_orig_mod.") for k in state_dict):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)


if __name__ == "__main__":
    cfg = ModelConfig()
    model = NanoGPT(cfg)
    print(f"params total: {model.num_params() / 1e6:.2f}M")
    print(f"params non-embedding: {model.num_params(non_embedding=True) / 1e6:.2f}M")
