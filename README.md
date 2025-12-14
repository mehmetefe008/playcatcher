# playcatcher

Bu repo, videolarda “start” anını bir şablon görsel (template) ile **multi-scale template matching** yaparak tespit edip,
tespit edilen anların **birkaç saniye sonrasından** ffmpeg ile **tek kare PNG** çıkaran bir Python scripti içerir.

## Gereksinimler

- Python **3.10+** (scriptte `ROI | None` gibi type syntax var)
- `ffmpeg` (sistemde kurulu ve komut satırından çalışıyor olmalı)
- Python paketleri: `numpy`, `tqdm`, `opencv-python`

## Kurulum

### 1) Repo’yu indir
```bash
git clone https://github.com/mehmetefe008/playcatcher.git
cd playcatcher
