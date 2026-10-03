"""pDFNet3 = DFNet3 upstream com FiLM nos dois sites + gerador FiLM (spec §3.2-§3.4).

O gerador não entra nos três grafos exportados; ele vai no enrollment.onnx.
"""

from __future__ import annotations

from torch import Tensor, nn

from cctrain.models.film import FilmGenerator


class PDFNet3(nn.Module):
    def __init__(self, base: nn.Module, generator: FilmGenerator):
        super().__init__()
        self.base = base
        self.film = generator

    def forward(self, spec: Tensor, feat_erb: Tensor, feat_spec: Tensor, embedding: Tensor, mask: Tensor):
        v = self.film(embedding, mask)
        return self.base(spec, feat_erb, feat_spec, film_enc=(v.gamma_enc, v.beta_enc),
                         film_df=(v.gamma_df, v.beta_df))

    def film_sites(self) -> dict[str, nn.Module]:
        return {"enc": self.base.enc.emb_gru.linear_in, "df_dec": self.base.df_dec.df_gru.linear_in}
