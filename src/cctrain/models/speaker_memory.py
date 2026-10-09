import torch
import torch.nn.functional as F
from torch import Tensor

def extract_speaker_tokens(
    features: Tensor, 
    k: int = 8, 
    embed_dim: int = 64, 
    method: str = 'temporal_pool'
) -> Tensor:
    """
    Extrai K tokens de memória de locutor a partir de características acústicas (STFT/fbank)
    de uma gravação de enrollment.
    
    Args:
        features: [B, D, T] ou [B, T, D] tensor de features.
        k: Número de tokens a extrair.
        embed_dim: Dimensão desejada para cada token. Se a dimensão for diferente, 
                   será ajustada via truncamento ou padding.
        method: 'temporal_pool' (divide a sequência temporal em K sub-janelas e aplica mean-pool) 
                ou 'kmeans' (k-means simples iterativo).
                
    Returns:
        tokens: [B, K, embed_dim]
    """
    # Padroniza shape para [B, T, D]
    # Se dimensão -1 for T e dimensão 1 for D (ou vice-versa), vamos tentar inferir.
    # Geralmente STFT/fbank tem T longo, D curto ou fixo.
    if features.ndim == 3:
        if features.shape[1] > features.shape[2]: # assumindo que T > D na maioria dos casos
            features = features.transpose(1, 2)
    elif features.ndim == 2: # [T, D] ou [D, T]
        features = features.unsqueeze(0)
        if features.shape[1] > features.shape[2]:
            features = features.transpose(1, 2)

    B, T, D_in = features.shape

    # Ajusta dimensões para embed_dim
    if D_in != embed_dim:
        if D_in > embed_dim:
            features = features[..., :embed_dim]
        else:
            features = F.pad(features, (0, embed_dim - D_in))

    D = embed_dim

    if method == 'temporal_pool':
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
        return torch.stack(tokens, dim=1) # [B, K, D]
    
    elif method == 'kmeans':
        # features: [B, T, D]
        indices = torch.randint(0, T, (B, k), device=features.device)
        centroids = torch.gather(features, 1, indices.unsqueeze(-1).expand(-1, -1, D))
        
        for _ in range(10):
            dists = torch.cdist(features, centroids) # [B, T, K]
            assignments = dists.argmin(dim=-1) # [B, T]
            
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(B, k, 1, device=features.device, dtype=features.dtype)
            
            assign_exp = assignments.unsqueeze(-1).expand(-1, -1, D)
            new_centroids.scatter_add_(1, assign_exp, features)
            counts.scatter_add_(1, assignments.unsqueeze(-1), torch.ones(B, T, 1, device=features.device, dtype=features.dtype))
            
            counts = counts.clamp_min(1.0)
            centroids = new_centroids / counts
            
        return centroids
    
    else:
        raise ValueError(f"Método desconhecido: {method}")
