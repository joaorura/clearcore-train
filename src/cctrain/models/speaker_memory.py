import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LearnedSpeakerAttentionPooler(nn.Module):
    """Módulo de pooling atencional com consultas aprendidas para extrair K tokens de locutor."""

    def __init__(self, k: int = 16, embed_dim: int = 128, num_heads: int = 4):
        super().__init__()
        self.k = k
        self.embed_dim = embed_dim
        self.queries = nn.Parameter(torch.randn(1, k, embed_dim) * (1.0 / math.sqrt(embed_dim)))
        self.mha = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, features: Tensor) -> Tensor:
        """Extrai K tokens através de Cross-Attention entre consultas e features.

        Args:
            features: [B, T, D] ou [B, D, T]

        Returns:
            tokens: [B, K, embed_dim]
        """
        if features.ndim == 2:
            features = features.unsqueeze(0)
        elif features.ndim == 3 and features.shape[1] < features.shape[2]:
            features = features.transpose(1, 2)

        B, T, D_in = features.shape
        if D_in != self.embed_dim:
            if D_in > self.embed_dim:
                features = features[..., : self.embed_dim]
            else:
                features = F.pad(features, (0, self.embed_dim - D_in))

        q = self.queries.expand(B, -1, -1)
        tokens, _ = self.mha(q, features, features)
        return self.norm(tokens)


def extract_speaker_tokens(
    features: Tensor,
    k: int = 16,
    embed_dim: int = 128,
    method: str = "temporal_pool",
    pooler: LearnedSpeakerAttentionPooler | None = None,
) -> Tensor:
    """Extrai K tokens de memória de locutor a partir de características acústicas (STFT/fbank)
    de uma gravação de enrollment.

    Args:
        features: [B, D, T] ou [B, T, D] tensor de features.
        k: Número de tokens a extrair (default 16 para Fase 2 SOTA).
        embed_dim: Dimensão de cada token (default 128 para Fase 2 SOTA).
        method: 'temporal_pool' (divide temporalmente em K janelas e aplica média),
                'kmeans' (k-means iterativo), ou
                'attention_pool' (pooling atencional por consultas harmônicas/aprendidas).
        pooler: Opcional. Instância de LearnedSpeakerAttentionPooler.

    Returns:
        tokens: [B, K, embed_dim]
    """
    if pooler is not None:
        return pooler(features)

    # Padroniza shape para [B, T, D]
    if features.ndim == 3:
        if features.shape[1] < features.shape[2]:  # [B, D, T] onde D < T
            features = features.transpose(1, 2)
    elif features.ndim == 2:  # [T, D] ou [D, T]
        features = features.unsqueeze(0)
        if features.shape[1] < features.shape[2]:
            features = features.transpose(1, 2)

    B, T, D_in = features.shape

    # Ajusta dimensões para embed_dim
    if D_in != embed_dim:
        if D_in > embed_dim:
            features = features[..., :embed_dim]
        else:
            features = F.pad(features, (0, embed_dim - D_in))

    D = embed_dim

    if method == "temporal_pool":
        chunk_size = max(1, T // k)
        tokens = []
        for i in range(k):
            start = i * chunk_size
            end = start + chunk_size if i < k - 1 else T
            chunk = features[:, start:end, :]
            if chunk.shape[1] == 0:
                token = torch.zeros(B, D, device=features.device, dtype=features.dtype)
            else:
                token = chunk.mean(dim=1)
            tokens.append(token)
        return torch.stack(tokens, dim=1)  # [B, K, D]

    elif method == "kmeans":
        indices = torch.randint(0, max(1, T), (B, k), device=features.device)
        indices = indices.clamp(0, T - 1)
        centroids = torch.gather(features, 1, indices.unsqueeze(-1).expand(-1, -1, D))

        for _ in range(10):
            dists = torch.cdist(features, centroids)  # [B, T, K]
            assignments = dists.argmin(dim=-1)  # [B, T]

            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(B, k, 1, device=features.device, dtype=features.dtype)

            assign_exp = assignments.unsqueeze(-1).expand(-1, -1, D)
            new_centroids.scatter_add_(1, assign_exp, features)
            counts.scatter_add_(1, assignments.unsqueeze(-1), torch.ones(B, T, 1, device=features.device, dtype=features.dtype))

            counts = counts.clamp_min(1.0)
            centroids = new_centroids / counts

        return centroids

    elif method == "attention_pool":
        t_steps = torch.arange(T, device=features.device, dtype=features.dtype).unsqueeze(0)  # [1, T]
        k_freqs = torch.arange(1, k + 1, device=features.device, dtype=features.dtype).unsqueeze(1)  # [k, 1]
        phase = (k_freqs * t_steps * math.pi) / max(1.0, float(T))  # [k, T]
        weights = torch.softmax(torch.cos(phase), dim=-1).unsqueeze(0).expand(B, -1, -1)  # [B, k, T]
        tokens = torch.matmul(weights, features)  # [B, k, D]
        return tokens

    else:
        raise ValueError(f"Método desconhecido: {method}")
