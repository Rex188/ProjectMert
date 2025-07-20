"""
batch_extract_mert.py
────────────────────────────────────────────────────────
• 读取 fma_small_labels.csv（含 track_id 与 genre_top）
• 使用 wav16k/<track_id>.wav 提取 MERT embedding
• 每首歌保存 emb/<track_id>_emb.npy
• initializer 方案：3 个子进程并行，每进程只加载 1 份模型
"""

import os, csv, pathlib, multiprocessing as mp
from types import SimpleNamespace

import numpy as np
import torch, torchaudio
from tqdm import tqdm

# ---------- 可按需调整 ----------
CSV_PATH    = "fma_small_labels.csv"    # 标签文件
WAV_DIR     = pathlib.Path("wav16k")    # WAV 目录
EMB_DIR     = pathlib.Path("emb")       # 输出目录
CKPT_PATH   = "checkpoints/checkpoint_best.pt"
NUM_WORKERS = 3                         # 并行进程数，建议 2‑4
TARGET_SR   = 16000                     # 采样率
# ----------------------------------

# —— 让子进程仅初始化一次模型 ——
def init_worker():
    global MODEL
    import sys, os, torch
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "fairseq-local"))

    from fairseq.data.dictionary import Dictionary
    from fairseq.models.wav2vec import Wav2Vec2Model
    torch.serialization._package_registry.append(
        ("fairseq.data.dictionary", "Dictionary", Dictionary)
    )

    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    cfg  = SimpleNamespace(**ckpt["cfg"]["model"])
    MODEL = Wav2Vec2Model.build_model(cfg, task=None)

    # ---- 仅加载形状匹配的参数 ----
    model_sd = MODEL.state_dict()
    ok_sd = {k: v for k, v in ckpt["model"].items()
             if k in model_sd and v.shape == model_sd[k].shape}
    MODEL.load_state_dict(ok_sd, strict=False)
    # --------------------------------
    MODEL.eval()
    print(f"[PID {os.getpid()}] MERT 初始化完毕，加载 {len(ok_sd)} 张量")

def load_audio(path: pathlib.Path):
    wav, sr = torchaudio.load(str(path))
    if sr != TARGET_SR:
        wav = torchaudio.functional.resample(wav, sr, TARGET_SR)
    return wav

def extract_one(row):
    """row = (track_id, wav_path)"""
    track_id, wav_path = row
    wav_path  = pathlib.Path(wav_path)
    emb_path  = EMB_DIR / f"{track_id}_emb.npy"

    if emb_path.exists():
        return "skip"
    if not wav_path.exists():
        return "missing"

    try:
        wav = load_audio(wav_path)
        with torch.no_grad():
            feat = MODEL.feature_extractor(wav)
            emb  = feat.mean(dim=2).squeeze(0).numpy()     # 512‑D
        np.save(emb_path, emb)
        return "ok"
    except Exception as e:
        return f"err:{e}"

def main():
    EMB_DIR.mkdir(exist_ok=True)

    # -------- 构造任务列表 --------
    tasks = []
    with open(CSV_PATH, newline='', encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = row["track_id"]
            wav = WAV_DIR / f"{tid}.wav"
            tasks.append((tid, wav))

    print(f"共 {len(tasks)} 首，开始提取 embedding …")

    # -------- 多进程池并行 --------
    with mp.Pool(processes=NUM_WORKERS, initializer=init_worker) as pool:
        results = list(tqdm(pool.imap_unordered(extract_one, tasks),
                            total=len(tasks)))

    print("✅ 完成！",
          results.count("ok"),       "成功，",
          results.count("skip"),     "已存在，",
          results.count("missing"),  "缺失文件，",
          len([r for r in results if str(r).startswith('err')]), "错误")

if __name__ == "__main__":
    mp.freeze_support()   # Windows 兼容
    main()
