"""Módulo de Cross-Attention Causal em Sub-bandas (Dual-Conditioning Frente 2).

Projetado para condicionamento biométrico fino na região de voz humana (300 Hz - 4 kHz,
correspondente às sub-bandas ERB 3..19 do DeepFilterNet3), mantendo estrita causalidade
(zero lookahead) e latência de inferência < 0.2 ms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

# Frequências e sub-bandas ERB do DeepFilterNet3 (sr=48000, fft_size=960, 32 bandas ERB):
# Banda  3: 300 Hz - 400 Hz (bin  6 a  8)
# ...
# Banda 19: 3400 Hz - 4000 Hz (bin 68 a 80)
# Banda 20: 4000 Hz - 4650 Hz (bin 80 a 93)
# Portanto, a fatia 3:20 cobre exatamente a faixa de voz humana 300 Hz a 4000 Hz (17 sub-bandas).
DEFAULT_VOICE_ERB_START: int = 3
DEFAULT_VOICE_ERB_END: int = 20
DEFAULT_TOTAL_ERB_BANDS: int = 32


@dataclass(frozen=True)
class SubbandCrossAttentionConfig:
    """Configuração do bloco de Cross-Attention Causal em Sub-bandas."""

    query_dim: int = 512  # Dimensão dos embeddings temporais do encoder (emb)
    embedding_dim: int = 192  # Dimensão do embedding biométrico do locutor (ECAPA-TDNN)
    proj_dim: int = 64  # Dimensão de projeção de baixa ordem para tempo real
    num_heads: int = 4  # Número de cabeças de atenção (head_dim = proj_dim // num_heads = 16)
    num_subbands: int = DEFAULT_TOTAL_ERB_BANDS  # Total de bandas ERB (32)
    voice_band_start: int = DEFAULT_VOICE_ERB_START  # Início da região vocal (300 Hz)
    voice_band_end: int = DEFAULT_VOICE_ERB_END  # Fim da região vocal (4000 Hz)
    gain_bound: float = 0.5  # Limite máximo de modulação fina de ganho em sub-bandas
    dropout: float = 0.0  # Dropout (0.0 para inferência determinística)
    zero_init: bool = True  # Inicialização com zero para identidade exata no início

    @property
    def num_voice_bands(self) -> int:
        return self.voice_band_end - self.voice_band_start

    @property
    def head_dim(self) -> int:
        return self.proj_dim // self.num_heads

    def validate(self) -> SubbandCrossAttentionConfig:
        if self.proj_dim % self.num_heads != 0:
            raise ValueError(
                f"proj_dim ({self.proj_dim}) deve ser divisível por num_heads ({self.num_heads})"
            )
        if not (0 <= self.voice_band_start < self.voice_band_end <= self.num_subbands):
            raise ValueError(
                f"Fatia de bandas vocais inválida: [{self.voice_band_start}:{self.voice_band_end}] "
                f"com total={self.num_subbands}"
            )
        if self.gain_bound <= 0.0:
            raise ValueError(f"gain_bound deve ser positivo, obtido {self.gain_bound}")
        return self


@dataclass(frozen=True)
class SubbandAttentionOutput:
    """Saída estruturada do módulo de Cross-Attention Causal em Sub-bandas."""

    output: Tensor  # Embeddings temporais condicionados [B, T, query_dim]
    attn_weights: Tensor  # Pesos de atenção sobre as sub-bandas [B, num_heads, T, num_voice_bands]
    subband_gains: Tensor  # Ganhos finos para as 32 bandas ERB [B, 1, T, num_subbands]


class CausalSubbandCrossAttention(nn.Module):
    """Bloco de Cross-Attention Causal ultraleve focado nas sub-bandas de voz humana.

    Princípios arquiteturais:
    1. Causalidade estrita: A saída no instante temporal t depende exclusivamente de
       informações no tempo <= t e do embedding biométrico estático, sem qualquer
       lookahead no eixo do tempo.
    2. Baixa dimensão (dim 64, 4 cabeças): Mantém custo computacional mínimo (~70 kFLOPs/frame)
       e latência de inferência < 0.2 ms para execução em tempo real a 48 kHz.
    3. Foco nas sub-bandas vocais (300 Hz - 4 kHz): A atenção mapeia o vetor biométrico
       para representações nas sub-bandas ERB 3..19, modulando finamente o ganho espectral
       dessas bandas sem interferir nas baixas ou ultra-altas frequências fora da voz.
    4. Identidade no marco zero (zero-init): Projeções de saída inicializadas em zero garantem
       que na ausência de treino ou com máscara zero, a transformação seja a identidade exata.
    """

    def __init__(self, cfg: SubbandCrossAttentionConfig | None = None):
        super().__init__()
        self.cfg = (cfg or SubbandCrossAttentionConfig()).validate()

        num_voice = self.cfg.num_voice_bands
        proj_dim = self.cfg.proj_dim
        head_dim = self.cfg.head_dim

        # Projeção de Query (a partir dos embeddings temporais do encoder)
        self.q_proj = nn.Linear(self.cfg.query_dim, proj_dim, bias=True)

        # Projeção do vetor biométrico da voz para chaves e valores nas sub-bandas
        # Cada sub-banda vocal recebe seu próprio token biométrico projetado
        self.bio_to_kv = nn.Linear(self.cfg.embedding_dim, num_voice * proj_dim * 2, bias=True)

        # Projeção de saída temporal (residual para os embeddings do encoder)
        self.out_proj = nn.Linear(proj_dim, self.cfg.query_dim, bias=True)

        # Cabeça de ganho fino para as sub-bandas de voz
        self.subband_gain_head = nn.Linear(proj_dim, num_voice, bias=True)

        self.scale = 1.0 / math.sqrt(float(head_dim))
        self.dropout = nn.Dropout(self.cfg.dropout) if self.cfg.dropout > 0.0 else nn.Identity()

        if self.cfg.zero_init:
            nn.init.zeros_(self.out_proj.weight)
            nn.init.zeros_(self.out_proj.bias)
            nn.init.zeros_(self.subband_gain_head.weight)
            nn.init.zeros_(self.subband_gain_head.bias)

    def forward(
        self,
        query: Tensor,
        embedding: Tensor,
        mask: Tensor | None = None,
    ) -> SubbandAttentionOutput:
        """Executa a Cross-Attention causal em sub-bandas.

        Args:
            query: Embeddings temporais do encoder [B, T, query_dim]
            embedding: Vetor biométrico do locutor [B, embedding_dim] ou [B, 1, embedding_dim]
            mask: Máscara de condicionamento / perfil [B, 1] ou [B, 1, 1] (0.0=bypass, 1.0=ativo)

        Returns:
            SubbandAttentionOutput com:
              - output: [B, T, query_dim]
              - attn_weights: [B, num_heads, T, num_voice_bands]
              - subband_gains: [B, 1, T, num_subbands]
        """
        b, t, _ = query.shape
        num_voice = self.cfg.num_voice_bands
        num_heads = self.cfg.num_heads
        head_dim = self.cfg.head_dim
        proj_dim = self.cfg.proj_dim

        # Adequação de formato do embedding biométrico
        if embedding.ndim == 3:
            embedding = embedding.squeeze(1)

        # Máscara padrão (1.0 se não fornecida)
        if mask is None:
            mask_t = torch.ones(b, 1, 1, device=query.device, dtype=query.dtype)
        else:
            mask_t = mask.view(b, 1, 1)

        # 1. Projeção de Query temporal: causalidade estrita porque a projeção é pontual por frame
        q = self.q_proj(query)  # [B, T, proj_dim]
        q = q.view(b, t, num_heads, head_dim).transpose(1, 2)  # [B, H, T, head_dim]

        # 2. Projeção do vetor biométrico da voz em tokens de sub-banda Key e Value
        bio_kv = self.bio_to_kv(embedding)  # [B, num_voice * proj_dim * 2]
        bio_k, bio_v = bio_kv.chunk(2, dim=-1)
        k = bio_k.view(b, num_voice, num_heads, head_dim).permute(0, 2, 1, 3)  # [B, H, num_voice, head_dim]
        v = bio_v.view(b, num_voice, num_heads, head_dim).permute(0, 2, 1, 3)  # [B, H, num_voice, head_dim]

        # 3. Scaled Dot-Product Cross-Attention: cada frame de áudio atende às sub-bandas vocais
        # scores: [B, H, T, num_voice]
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        attn_dropped = self.dropout(attn)

        # 4. Contexto atendido
        ctx = torch.matmul(attn_dropped, v)  # [B, H, T, head_dim]
        ctx = ctx.transpose(1, 2).contiguous().view(b, t, proj_dim)  # [B, T, proj_dim]

        # 5. Condicionamento temporal fino (residual nos embeddings do encoder)
        delta_emb = self.out_proj(ctx) * mask_t  # [B, T, query_dim]
        out_emb = query + delta_emb

        # 6. Modulação fina nas sub-bandas ERB de voz (300 Hz - 4000 Hz)
        # Ganhos vocais delta limitados por tanh e gain_bound
        delta_voice = torch.tanh(self.subband_gain_head(ctx)) * (self.cfg.gain_bound * mask_t)  # [B, T, num_voice]
        voice_gains = (1.0 + delta_voice).unsqueeze(1)  # [B, 1, T, num_voice]

        # Montagem funcional do tensor completo de ganhos ERB (32 bandas) sem mutações in-place
        # Bandas 0..voice_band_start (infra-voz): ganho neutro 1.0
        # Bandas voice_band_start..voice_band_end (voz): ganho biométrico voice_gains
        # Bandas voice_band_end..num_subbands (ultra-voz): ganho neutro 1.0
        n_low = self.cfg.voice_band_start
        n_high = self.cfg.num_subbands - self.cfg.voice_band_end

        parts: list[Tensor] = []
        if n_low > 0:
            parts.append(torch.ones(b, 1, t, n_low, device=query.device, dtype=query.dtype))
        parts.append(voice_gains)
        if n_high > 0:
            parts.append(torch.ones(b, 1, t, n_high, device=query.device, dtype=query.dtype))

        subband_gains = torch.cat(parts, dim=-1)  # [B, 1, T, num_subbands]

        return SubbandAttentionOutput(
            output=out_emb,
            attn_weights=attn,
            subband_gains=subband_gains,
        )
