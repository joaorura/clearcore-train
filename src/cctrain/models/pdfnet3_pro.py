"""pDFNet3Pro = pDFNet3 com Dual-Conditioning (FiLM Macro + Causal Cross-Attention Fina em Sub-bandas).

Combina:
1. Condicionamento Macro (FiLM): escalares gamma/beta injetados nos GRUs de encoder e df_decoder.
2. Condicionamento Fino (Cross-Attention Causal em Sub-bandas): atenção pontual nos embeddings temporais
   do encoder direcionada às sub-bandas de voz humana (300 Hz - 4 kHz, ERB 3..19), com modulação fina
   de ganho no ERB decoder.
3. Fallback Gracioso: se cross_attention=False (ou se desabilitado), comporta-se de forma estritamente
   idêntica ao pDFNet3 aprovado.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from cctrain.models.cross_attention import (
    CausalSubbandCrossAttention,
    SubbandAttentionOutput,
    SubbandCrossAttentionConfig,
)
from cctrain.models.film import FilmGenerator
from cctrain.models.pdfnet3 import PDFNet3



class FrequencyDWConvMixer(nn.Module):
    """Dual-Path Frequency Mixer (Depthwise na frequência + Pointwise)."""
    def __init__(self, in_channels: int = 1, kernel_size: int = 9, zero_init: bool = True):
        super().__init__()
        pad_f = kernel_size // 2
        
        # Convolução Depthwise ao longo do eixo de frequência (T, F)
        self.dw_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, kernel_size),
            padding=(0, pad_f),
            groups=in_channels,
            bias=False
        )
        
        self.norm = nn.BatchNorm2d(in_channels)
        self.act = nn.GELU()
        
        # Convolução Pointwise 1x1 combinando canais
        self.pw_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            bias=False
        )
        
        if zero_init:
            nn.init.zeros_(self.pw_conv.weight)

    def forward(self, x: Tensor) -> Tensor:
        # x: [B, C, T, F]
        res = x
        x = self.dw_conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = self.pw_conv(x)
        return x + res

class PDFNet3Pro(PDFNet3):

    """Modelo pDFNet3 Pro com Dual-Conditioning: FiLM Macro + Cross-Attention Fina em Sub-bandas."""

    def __init__(
        self,
        base: nn.Module,
        generator: FilmGenerator,
        cross_attention: CausalSubbandCrossAttention | None = None,
        cross_attention_config: SubbandCrossAttentionConfig | None = None,
        enable_cross_attention: bool = True,
    ):
        super().__init__(base, generator)
        if cross_attention is not None:
            self.cross_attn = cross_attention
        else:
            cfg = cross_attention_config or SubbandCrossAttentionConfig()
            self.cross_attn = CausalSubbandCrossAttention(cfg)
        self.enable_cross_attention = enable_cross_attention
        zero_init = self.cross_attn.cfg.zero_init
        self.freq_mixer = FrequencyDWConvMixer(in_channels=1, kernel_size=9, zero_init=zero_init)

    def forward(
        self,
        spec: Tensor,
        feat_erb: Tensor,
        feat_spec: Tensor,
        embedding: Tensor,
        mask: Tensor,
        cross_attention: bool | None = None,
        return_attention_details: bool = False,
        speaker_tokens: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor] | tuple[tuple[Tensor, Tensor, Tensor, Tensor], SubbandAttentionOutput]:
        """Forward do pDFNet3Pro com suporte a Dual-Conditioning e fallback gracioso.

        Args:
            spec: Espectro complexo [B, 1, T, F, 2]
            feat_erb: Features de magnitude ERB [B, 1, T, Fe]
            feat_spec: Features de espectrograma complexo [B, 1, T, Fc, 2]
            embedding: Vetor biométrico de voz [B, embedding_dim]
            mask: Máscara de perfil / gating [B, 1]
            cross_attention: Override explícito para ligar/desligar cross-attention (None usa flag padrão)
            return_attention_details: Se True, retorna também o objeto SubbandAttentionOutput
            speaker_tokens: Opcional. Dicionário de memória do falante multi-token [B, K, token_dim].

        Returns:
            (spec_e, m, lsnr, df_coefs) ou ((spec_e, m, lsnr, df_coefs), attn_details)
        """
        use_ca = self.enable_cross_attention if cross_attention is None else cross_attention

        # 1. Macro Conditioning: Gerador FiLM (gamma / beta para encoder e df_decoder)
        v = self.film(embedding, mask)

        # Fallback gracioso: Comportamento idêntico ao pDFNet3 base
        if not use_ca:
            out = self.base(
                spec,
                feat_erb,
                feat_spec,
                film_enc=(v.gamma_enc, v.beta_enc),
                film_df=(v.gamma_df, v.beta_df),
            )
            if return_attention_details:
                # Retorna detalhes nulos / neutros
                b, t = spec.shape[0], spec.shape[2]
                dummy_attn = SubbandAttentionOutput(
                    output=torch.zeros((b, t, self.cross_attn.cfg.query_dim), device=spec.device),
                    attn_weights=torch.zeros((b, self.cross_attn.cfg.num_heads, t, self.cross_attn.cfg.num_voice_bands), device=spec.device),
                    subband_gains=torch.ones((b, 1, t, self.cross_attn.cfg.num_subbands), device=spec.device),
                )
                return out, dummy_attn
            return out

        # 2. Dual-Conditioning ativo:
        # Prepara features do espectro complexo
        feat_spec_perm = feat_spec.squeeze(1).permute(0, 3, 1, 2)
        pad_erb = self.base.pad_feat(feat_erb)
        pad_spec = self.base.pad_feat(feat_spec_perm)

        # Aplica o FrequencyDWConvMixer no fluxo antes da GRU temporal stateful
        delta_erb = self.freq_mixer(pad_erb) - pad_erb
        mask_t = mask.view(-1, 1, 1, 1)
        pad_erb = pad_erb + delta_erb * mask_t

        # Executa Encoder com FiLM macro
        e0, e1, e2, e3, emb, c0, lsnr = self.base.enc(
            pad_erb,
            pad_spec,
            film=(v.gamma_enc, v.beta_enc),
        )

        # 3. Fine Conditioning: Cross-Attention Causal em Sub-bandas
        # Condiciona os embeddings temporais e calcula os ganhos finos nas bandas de voz
        ca_input = speaker_tokens if speaker_tokens is not None else embedding
        attn_out = self.cross_attn(emb, ca_input, mask=mask)
        emb_conditioned = attn_out.output

        # LSNR Dropout (se habilitado no modelo base)
        if getattr(self.base, "lsnr_droput", False):
            idcs = lsnr.squeeze() > -10.0
            b, t = spec.shape[0], spec.shape[2]
            m = torch.zeros((b, 1, t, self.base.erb_bins), device=spec.device)
            df_coefs = torch.zeros((b, t, self.base.nb_df, self.base.df_order * 2), device=spec.device)
            emb_dec = emb_conditioned[:, idcs]
            e0_dec = e0[:, :, idcs]
            e1_dec = e1[:, :, idcs]
            e2_dec = e2[:, :, idcs]
            e3_dec = e3[:, :, idcs]
            c0_dec = c0[:, :, idcs]
        else:
            idcs = None
            emb_dec = emb_conditioned
            e0_dec, e1_dec, e2_dec, e3_dec, c0_dec = e0, e1, e2, e3, c0

        # 4. ERB Decoder + modulação fina de sub-bandas
        if getattr(self.base, "run_erb", True):
            if idcs is not None:
                m_sub = self.base.erb_dec(emb_dec, e3_dec, e2_dec, e1_dec, e0_dec)
                m[:, :, idcs] = m_sub * attn_out.subband_gains[:, :, idcs]
            else:
                m_sub = self.base.erb_dec(emb_dec, e3_dec, e2_dec, e1_dec, e0_dec)
                m = m_sub * attn_out.subband_gains
            spec_m = self.base.mask(spec, m)
        else:
            m = torch.zeros((), device=spec.device)
            spec_m = torch.zeros_like(spec)

        # 5. DF Decoder com FiLM macro e embeddings condicionados
        if getattr(self.base, "run_df", True):
            if idcs is not None:
                df_coefs[:, idcs] = self.base.df_dec(
                    emb_dec, c0_dec, film=(v.gamma_df, v.beta_df)
                )
            else:
                df_coefs = self.base.df_dec(
                    emb_dec, c0_dec, film=(v.gamma_df, v.beta_df)
                )
            df_coefs = self.base.df_out_transform(df_coefs)
            spec_e = self.base.df_op(spec.clone(), df_coefs)
            spec_e[..., self.base.nb_df :, :] = spec_m[..., self.base.nb_df :, :]
        else:
            df_coefs = torch.zeros((), device=spec.device)
            spec_e = spec_m

        # 6. Post-Filter (se ativo no modelo base)
        if getattr(self.base, "post_filter", False):
            beta = self.base.post_filter_beta
            eps = 1e-12
            spec_e_c = torch.view_as_complex(spec_e)
            spec_c = torch.view_as_complex(spec)
            mask_pf = (spec_e_c.abs() / spec_c.abs().add(eps)).clamp(eps, 1)
            mask_sin = mask_pf * torch.sin(math.pi * mask_pf / 2).clamp_min(eps)
            pf = (1 + beta) / (1 + beta * mask_pf.div(mask_sin).pow(2))
            spec_e = spec_e * pf.unsqueeze(-1)

        result = (spec_e, m, lsnr, df_coefs)
        if return_attention_details:
            return result, attn_out
        return result

    def conditioning_sites(self) -> dict[str, nn.Module]:
        """Retorna todos os sítios de injeção de condicionamento."""
        return {
            "film_enc": self.base.enc.emb_gru.linear_in,
            "film_df": self.base.df_dec.df_gru.linear_in,
            "cross_attn": self.cross_attn,
        }
