"""ECAPA-TDNN (embedding 192-d), com attentive statistics pooling de contexto global (spec §3.1).

Os padrões (C=512, padding de zeros) são o protótipo do M0. O encoder do M1 é o SpeechBrain
`spkrec-ecapa-voxceleb` portado por `cctrain.models.sb_port.port_ecapa`: C=1024, mfa 3072, `pad_mode="reflect"`
(o `Conv1d(padding="same")` do SpeechBrain reflete as bordas).

context_mode="expand" usa Expand sobre o eixo de tempo (como o SpeechBrain); "broadcast" usa x*0 + stat,
sem operadores de forma. O M0 prototipa os dois no tract (Task 16).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class TDNNBlock(nn.Module):
    def __init__(self, cin: int, cout: int, kernel: int, dilation: int = 1, pad_mode: str = "zeros"):
        super().__init__()
        if pad_mode not in ("zeros", "reflect"):
            raise ValueError(f"pad_mode inválido: {pad_mode}")
        self.pad = dilation * (kernel - 1) // 2
        self.pad_mode = pad_mode
        # zeros: padding embutido no Conv (grafo do protótipo do M0); reflect: Pad explícito, como o SpeechBrain
        self.conv = nn.Conv1d(cin, cout, kernel, dilation=dilation, padding=self.pad if pad_mode == "zeros" else 0)
        self.bn = nn.BatchNorm1d(cout)
        # Reflexão por Gather com índices constantes: o tract 0.19 ignora o `mode` do Pad e preencheria com zeros.
        # Esquerda x[p], ..., x[1]; direita x[T-2], ..., x[T-1-p] (índices negativos do Gather).
        self.register_buffer("_left", torch.arange(self.pad, 0, -1), persistent=False)
        self.register_buffer("_right", -torch.arange(2, self.pad + 2), persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        if self.pad_mode == "reflect" and self.pad:
            x = torch.cat([x[..., self._left], x, x[..., self._right]], dim=-1)
        return self.bn(torch.relu(self.conv(x)))


class Res2Block(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, scale: int = 8, pad_mode: str = "zeros"):
        super().__init__()
        if channels % scale:
            raise ValueError("channels deve ser múltiplo de scale")
        self.scale = scale
        width = channels // scale
        self.blocks = nn.ModuleList([TDNNBlock(width, width, kernel, dilation, pad_mode) for _ in range(scale - 1)])

    def forward(self, x: Tensor) -> Tensor:
        chunks = torch.chunk(x, self.scale, dim=1)
        ys = [chunks[0]]
        y = chunks[0]
        for i, block in enumerate(self.blocks, start=1):
            y = block(chunks[i] if i == 1 else chunks[i] + y)
            ys.append(y)
        return torch.cat(ys, dim=1)


class SEBlock(nn.Module):
    def __init__(self, channels: int, bottleneck: int = 128):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, bottleneck, 1)
        self.conv2 = nn.Conv1d(bottleneck, channels, 1)

    def forward(self, x: Tensor) -> Tensor:
        s = x.mean(dim=2, keepdim=True)
        return x * torch.sigmoid(self.conv2(torch.relu(self.conv1(s))))


class SERes2Block(nn.Module):
    def __init__(self, channels: int, kernel: int, dilation: int, scale: int = 8, pad_mode: str = "zeros"):
        super().__init__()
        self.tdnn1 = TDNNBlock(channels, channels, 1)
        self.res2 = Res2Block(channels, kernel, dilation, scale, pad_mode)
        self.tdnn2 = TDNNBlock(channels, channels, 1)
        self.se = SEBlock(channels)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.se(self.tdnn2(self.res2(self.tdnn1(x))))


class AttentiveStatsPool(nn.Module):
    def __init__(self, channels: int, attention: int = 128, context_mode: str = "expand", eps: float = 1e-12):
        super().__init__()
        if context_mode not in ("expand", "broadcast"):
            raise ValueError(f"context_mode inválido: {context_mode}")
        self.context_mode = context_mode
        self.eps = eps
        self.tdnn = TDNNBlock(3 * channels, attention, 1)
        self.conv = nn.Conv1d(attention, channels, 1)

    def _broadcast(self, stat: Tensor, x: Tensor) -> Tensor:
        if self.context_mode == "expand":
            return stat.expand(-1, -1, x.shape[-1])
        return x * 0.0 + stat

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=2, keepdim=True)
        std = torch.sqrt(torch.clamp(((x - mean) ** 2).mean(dim=2, keepdim=True), min=self.eps))
        ctx = torch.cat([x, self._broadcast(mean, x), self._broadcast(std, x)], dim=1)
        w = torch.softmax(self.conv(torch.tanh(self.tdnn(ctx))), dim=2)
        mu = (w * x).sum(dim=2)
        sg = torch.sqrt(torch.clamp((w * (x - mu.unsqueeze(2)) ** 2).sum(dim=2), min=self.eps))  # fórmula do SpeechBrain
        return torch.cat([mu, sg], dim=1)


class ECAPATDNN(nn.Module):
    def __init__(self, n_mels: int = 80, channels: int = 512, mfa_channels: int = 1536, embedding_dim: int = 192,
                 attention: int = 128, context_mode: str = "expand", pad_mode: str = "zeros"):
        super().__init__()
        self.layer1 = TDNNBlock(n_mels, channels, 5, 1, pad_mode)
        self.layer2 = SERes2Block(channels, 3, 2, pad_mode=pad_mode)
        self.layer3 = SERes2Block(channels, 3, 3, pad_mode=pad_mode)
        self.layer4 = SERes2Block(channels, 3, 4, pad_mode=pad_mode)
        self.mfa = TDNNBlock(3 * channels, mfa_channels, 1)
        self.pool = AttentiveStatsPool(mfa_channels, attention, context_mode)
        self.pool_bn = nn.BatchNorm1d(2 * mfa_channels)
        self.fc = nn.Linear(2 * mfa_channels, embedding_dim)

    def forward(self, feats: Tensor) -> Tensor:
        x1 = self.layer1(feats)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x = self.mfa(torch.cat([x2, x3, x4], dim=1))
        return self.fc(self.pool_bn(self.pool(x)))
