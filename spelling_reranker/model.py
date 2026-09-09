"""~28M byte-level bidirectional Transformer spelling reranker."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from spelling_reranker.byte_encoding import MASK_ID, N_CANDIDATE_SLOTS, PAD_ID, VOCAB_SIZE


@dataclass
class ModelConfig:
    vocab_size: int = VOCAB_SIZE
    n_layers: int = 8
    d_model: int = 512
    n_heads: int = 8
    head_dim: int = 64
    ffn_hidden: int = 1536
    dropout: float = 0.10
    attn_dropout: float = 0.00
    max_seq_len: int = 448
    n_candidates: int = N_CANDIDATE_SLOTS
    score_hidden: int = 320
    score_dropout: float = 0.10
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    pad_id: int = PAD_ID
    #: Auxiliary masked-byte head. Training-only; adds ~0.2M parameters and is
    #: never used at inference.
    mlm_head: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ModelConfig":
        known = {k: data[k] for k in cls.__dataclass_fields__ if k in data}
        return cls(**known)


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x_f = x.float()
        rms = x_f.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return (x_f * rms * self.weight.float()).to(orig_dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: [B, H, T, D]; cos/sin: [T, D]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin
    return q, k


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float = 10000.0) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("RoPE head_dim must be even")
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, max_seq_len: int) -> None:
        t = torch.arange(max_seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        if seq_len > self.cos_cached.shape[0]:
            self._build_cache(seq_len)
        cos = self.cos_cached[:seq_len].to(device=device, dtype=dtype)
        sin = self.sin_cached[:seq_len].to(device=device, dtype=dtype)
        return cos, sin


class BidirectionalAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        if cfg.d_model != cfg.n_heads * cfg.head_dim:
            raise ValueError("d_model must equal n_heads * head_dim")
        self.n_heads = cfg.n_heads
        self.head_dim = cfg.head_dim
        self.attn_dropout = cfg.attn_dropout
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        qkv = self.qkv(x).view(batch, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, k, cos, sin)

        # attention_mask: [B, T] with 1 = keep. Build boolean pad mask for SDPA.
        pad = attention_mask == 0
        # SDPA bool mask True means "keep" when using is_causal=False with attn_mask
        # Use float mask: [B, 1, 1, T] additive -inf on pads (key positions).
        additive = torch.zeros(
            batch, 1, 1, seq_len, device=x.device, dtype=q.dtype
        )
        additive = additive.masked_fill(pad.unsqueeze(1).unsqueeze(1), torch.finfo(q.dtype).min)

        dropout_p = self.attn_dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=additive, dropout_p=dropout_p, is_causal=False
        )
        out = out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.proj(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int) -> None:
        super().__init__()
        self.w_gate = nn.Linear(d_model, hidden, bias=False)
        self.w_up = nn.Linear(d_model, hidden, bias=False)
        self.w_down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.attn = BidirectionalAttention(cfg)
        self.norm2 = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.ffn = SwiGLU(cfg.d_model, cfg.ffn_hidden)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x), attention_mask, cos, sin))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


#: Features fed to the scoring head, per candidate:
#: [candidate, typo, context(CLS), candidate*typo, |candidate-typo|].
#: The last two are the standard sentence-pair matching interaction terms; they
#: let the head compare a candidate against the typo directly instead of having
#: to rediscover the comparison inside a single linear layer.
SCORE_FEATURES = 5


class ScoringHead(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model * SCORE_FEATURES, cfg.score_hidden),
            nn.SiLU(),
            nn.Dropout(cfg.score_dropout),
            nn.Linear(cfg.score_hidden, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


class ByteSpellingReranker(nn.Module):
    def __init__(self, cfg: ModelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or ModelConfig()
        cfg = self.cfg
        # Fail here, on CPU, with a readable message. A config whose vocab is
        # too small does not fail at construction: it fails as an asynchronous
        # device-side gather assert on the first batch that happens to contain
        # the offending id, which is both unreadable and arbitrarily delayed.
        if cfg.vocab_size < VOCAB_SIZE:
            raise ValueError(
                f"vocab_size={cfg.vocab_size} is smaller than the byte vocabulary "
                f"({VOCAB_SIZE}); ids up to {VOCAB_SIZE - 1} (MASK={MASK_ID}) would "
                "index out of bounds"
            )
        if cfg.n_candidates > N_CANDIDATE_SLOTS:
            raise ValueError(
                f"n_candidates={cfg.n_candidates} exceeds the {N_CANDIDATE_SLOTS} "
                "CAND tokens defined in the vocabulary"
            )
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.embed_dropout = nn.Dropout(cfg.dropout)
        self.rope = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_norm_eps)
        self.score_head = ScoringHead(cfg)
        # Auxiliary masked-byte objective. The reranking signal alone teaches
        # the encoder very little English: it only ever says which of a handful
        # of candidates fits. Predicting masked context bytes in the same
        # forward pass gives the encoder an actual language-modelling gradient,
        # which is what the "pick the candidate that fits the sentence" half of
        # the task depends on. Weight is decayed to zero during training so the
        # final phase optimises reranking alone.
        self.mlm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False) if cfg.mlm_head else None

    def encode(self, token_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        hidden = self.embed_dropout(self.embed(token_ids))
        cos, sin = self.rope(token_ids.size(1), hidden.device, hidden.dtype)
        for block in self.blocks:
            hidden = block(hidden, attention_mask, cos, sin)
        return self.final_norm(hidden)

    def _masked_mean(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weights = mask.to(hidden.dtype).unsqueeze(-1)
        denom = weights.sum(dim=-2).clamp(min=1e-6)
        return (hidden * weights).sum(dim=-2) / denom

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        typo_mask: torch.Tensor,
        candidate_masks: torch.Tensor,
        candidate_valid: torch.Tensor,
        return_hidden: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Return logits of shape [batch, n_candidates]. Invalid slots are -inf."""
        hidden = self.encode(token_ids, attention_mask)
        context_repr = hidden[:, 0, :]
        typo_repr = self._masked_mean(hidden, typo_mask)

        # candidate_masks: [B, C, T]. Pooling as a batched matmul keeps the
        # intermediate at [B, C, D]; the elementwise form would materialise
        # [B, C, T, D] (~1 GB for B=128, C=16, T=448, D=512) and was what drove
        # the 22 GiB peak in the first experiment.
        cand_weights = candidate_masks.to(hidden.dtype)
        cand_denom = cand_weights.sum(dim=2).clamp(min=1e-6).unsqueeze(-1)
        cand_repr = torch.bmm(cand_weights, hidden) / cand_denom

        typo_exp = typo_repr.unsqueeze(1).expand_as(cand_repr)
        ctx_exp = context_repr.unsqueeze(1).expand_as(cand_repr)
        features = torch.cat(
            [
                cand_repr,
                typo_exp,
                ctx_exp,
                cand_repr * typo_exp,
                (cand_repr - typo_exp).abs(),
            ],
            dim=-1,
        )
        logits = self.score_head(features)
        invalid = candidate_valid == 0
        logits = logits.masked_fill(invalid, torch.finfo(logits.dtype).min)
        if return_hidden:
            return logits, hidden
        return logits


def apply_byte_masking(
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    typo_mask: torch.Tensor,
    candidate_masks: torch.Tensor,
    *,
    probability: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask a fraction of *context* bytes for the auxiliary objective.

    Only raw byte positions in the context are eligible: the typo, the
    candidate strings and every structural token are left intact, so the
    reranking task itself is never made unanswerable.

    Returns (masked token_ids, labels) where labels is -100 off the masked
    positions.
    """
    eligible = (
        (attention_mask != 0)
        & (typo_mask == 0)
        & (candidate_masks.sum(dim=1) == 0)
        & (token_ids < 256)
    )
    draw = torch.rand(token_ids.shape, device=token_ids.device, generator=generator)
    selected = eligible & (draw < probability)
    labels = torch.where(selected, token_ids, torch.full_like(token_ids, -100))
    masked = torch.where(selected, torch.full_like(token_ids, MASK_ID), token_ids)
    return masked, labels


def masked_byte_loss(hidden: torch.Tensor, head: nn.Linear, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over the masked byte positions only."""
    selected = labels != -100
    if not bool(selected.any()):
        return hidden.sum() * 0.0
    logits = head(hidden[selected])
    return F.cross_entropy(logits, labels[selected])


def masked_cross_entropy(logits: torch.Tensor, gold_index: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, gold_index)


def topk_accuracy(
    logits: torch.Tensor,
    gold_index: torch.Tensor,
    k: int = 1,
    candidate_valid: torch.Tensor | None = None,
) -> torch.Tensor:
    if candidate_valid is not None:
        logits = logits.masked_fill(candidate_valid == 0, torch.finfo(logits.dtype).min)
    pred = logits.topk(k, dim=-1).indices
    return (pred == gold_index.unsqueeze(-1)).any(dim=-1).float().mean()
