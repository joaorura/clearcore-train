# Patches sobre o DeepFilterNet v0.5.6

Cada trecho alterado é marcado no código com `CCTRAIN-PATCH(<id>)`. Licença do upstream mantida (MIT OR Apache-2.0).

| id | arquivo | motivo |
|---|---|---|
| io-audiometadata | `DeepFilterNet/df/io.py` | `torchaudio.backend.common.AudioMetaData` não existe no torchaudio >= 2.1 (R02 §1.6) |

## Arquivos removidos

| arquivo | motivo |
|---|---|
| `DeepFilterNet/README.md` | symlink quebrado para `../README.md`, que não foi vendorizado; removido |
