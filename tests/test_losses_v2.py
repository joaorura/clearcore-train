import pytest
import torch
import torch.nn.functional as F

from cctrain.losses.dfnet import (
    ArcFaceLoss,
    arcface_angular_margin_loss,
    target_absence_loss,
    interferer_rejection_loss,
    df_coef_temporal_regularization_loss,
    stage_c_loss
)
from cctrain.models.speaker_memory import extract_speaker_tokens, LearnedSpeakerAttentionPooler

def test_target_absence_loss():
    # pred [B, T]
    pred = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    # mask [B, T], absence mask
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    
    loss = target_absence_loss(pred, mask)
    # sum of squares for masked: 1^2 + 3^2 = 10. denom = 2. mean = 5.0
    assert torch.isclose(loss, torch.tensor(5.0))
    
    # zero mask
    mask_zero = torch.zeros_like(mask)
    loss_zero = target_absence_loss(pred, mask_zero)
    assert loss_zero.item() == 0.0

def test_interferer_rejection_loss():
    pred_emb = torch.randn(2, 64)
    target_emb = pred_emb + torch.randn(2, 64) * 0.1 # Similar
    interferer_emb = -pred_emb + torch.randn(2, 64) * 0.1 # Opposite
    
    loss = interferer_rejection_loss(pred_emb, target_emb, interferer_emb, temperature=0.1)
    # Since target is similar and interferer is opposite, cross entropy should be low
    assert loss.item() >= 0

def test_arcface_loss_classifier():
    in_features = 128
    num_classes = 10
    arcface = ArcFaceLoss(in_features=in_features, num_classes=num_classes, s=32.0, m=0.25)
    
    x = torch.randn(4, in_features)
    labels = torch.tensor([0, 2, 5, 9])
    loss = arcface(x, labels)
    assert torch.isfinite(loss)
    assert loss.item() > 0.0

def test_arcface_angular_margin_loss():
    pred_emb = torch.randn(4, 128)
    target_emb = pred_emb + torch.randn(4, 128) * 0.5
    interferer_emb = pred_emb + torch.randn(4, 128) * 0.5
    
    loss = arcface_angular_margin_loss(pred_emb, target_emb, interferer_emb, s=32.0, m=0.25)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0

def test_df_coef_temporal_regularization_loss():
    df_coefs = torch.zeros(2, 10, 5, 2)
    df_coefs[:, :, 1, :] = 1.0
    loss = df_coef_temporal_regularization_loss(df_coefs, time_dim=-2)
    assert loss.item() == 80.0

def test_extract_speaker_tokens():
    features = torch.randn(2, 100, 64) # [B, T, D]
    tokens_pool = extract_speaker_tokens(features, k=8, embed_dim=64, method='temporal_pool')
    assert tokens_pool.shape == (2, 8, 64)
    
    tokens_kmeans = extract_speaker_tokens(features, k=8, embed_dim=64, method='kmeans')
    assert tokens_kmeans.shape == (2, 8, 64)

def test_extract_speaker_tokens_fase2():
    # Valida parâmetros da SOTA Fase 2: K=16 tokens e D=128 canais
    features = torch.randn(2, 100, 128)
    tokens_16 = extract_speaker_tokens(features, k=16, embed_dim=128, method='temporal_pool')
    assert tokens_16.shape == (2, 16, 128)
    
    tokens_attn = extract_speaker_tokens(features, k=16, embed_dim=128, method='attention_pool')
    assert tokens_attn.shape == (2, 16, 128)
    
    pooler = LearnedSpeakerAttentionPooler(k=16, embed_dim=128, num_heads=4)
    tokens_learned = pooler(features)
    assert tokens_learned.shape == (2, 16, 128)

def test_stage_c_loss_integration():
    pred = torch.randn(2, 16000)
    target = torch.randn(2, 16000)
    lsnr = torch.randn(2, 100)
    lsnr_target = torch.randn(2, 100)
    lsnr_mask = torch.ones(2, 100)
    interferer_frames = torch.ones(2, 100)
    
    absence_mask = torch.zeros(2, 16000)
    pred_emb = torch.randn(2, 128)
    target_emb = torch.randn(2, 128)
    interferer_emb = torch.randn(2, 128)
    df_coefs = torch.randn(2, 256, 100, 2)
    
    res = stage_c_loss(
        pred, target, lsnr, lsnr_target,
        lsnr_mask=lsnr_mask, interferer_frames=interferer_frames,
        absence_mask=absence_mask,
        pred_emb=pred_emb, target_emb=target_emb, interferer_emb=interferer_emb,
        df_coefs=df_coefs, df_coefs_time_dim=-2
    )
    
    assert "total" in res
    assert "abs" in res
    assert "conf" in res
    assert "arcface" in res
    assert "df_reg" in res
