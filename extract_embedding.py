import os, sys, torch, torchaudio, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fairseq-local"))
np.float = float

from types import SimpleNamespace
from omegaconf import OmegaConf
from fairseq.data.dictionary import Dictionary
from fairseq.models.wav2vec import Wav2Vec2Model
from fairseq.checkpoint_utils import load_checkpoint_to_cpu

# 白名单自定义类
torch.serialization._package_registry.append(
    ("fairseq.data.dictionary", "Dictionary", Dictionary)
)

def load_audio(p, sr=16000):
    wav, s = torchaudio.load(p)
    if s != sr:
        wav = torchaudio.functional.resample(wav, s, sr)
    return wav

def extract_embedding(wav_path, ckpt_path, out_path):
    from types import SimpleNamespace

    state = torch.load(ckpt_path, map_location="cpu")      # <‑ 读取 .pt

    cfg_model = state["cfg"]["model"]                      # 已是 dict
    args = SimpleNamespace(**cfg_model)                    # dict → Namespace

    model = Wav2Vec2Model.build_model(args, task=None)
    # ---------- 过滤并加载权重 ---------- #
    model_sd   = model.state_dict()
    ckpt_sd    = state["model"]

    loaded, skipped = 0, 0
    new_sd = {}

    for k, v in ckpt_sd.items():
        if k in model_sd and v.shape == model_sd[k].shape:
            new_sd[k] = v
            loaded += 1
        else:
            skipped += 1          # 形状不符或模型里没有 → 丢弃

    print(f"→ 兼容加载: {loaded} tensors   |   跳过: {skipped}")

    model_sd.update(new_sd)       # 用能对得上的权重替换
    model.load_state_dict(model_sd, strict=False)
    # ----------------------------------- #

    model.eval()


    # ── inference ────────────────────────────────────
    with torch.no_grad():
        wav = load_audio(wav_path)
        feat = model.feature_extractor(wav)               # (1, C, T')
        embedding = feat.mean(dim=2).squeeze(0).numpy()   # 直接对 C×T' 做均值
        # ⤴ 不再用 model.post_extract_proj


    np.save(out_path, embedding)
    print(f"✅ Saved {out_path}  | shape={embedding.shape}")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_path",  required=True)
    ap.add_argument("--ckpt_path",   required=True)
    ap.add_argument("--output_path", required=True)
    args = ap.parse_args()
    extract_embedding(args.audio_path, args.ckpt_path, args.output_path)
