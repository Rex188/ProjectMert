import subprocess, pathlib as p, multiprocessing as mp, tqdm, os

SRC_ROOT = p.Path("fma_small")
DST_ROOT = p.Path("wav16k")
DST_ROOT.mkdir(exist_ok=True)

MP3_LIST = list(SRC_ROOT.rglob("*.mp3"))

def convert(mp3_path):
    mp3_path = p.Path(mp3_path)
    tid = mp3_path.stem
    wav_path = DST_ROOT / f"{tid}.wav"
    if wav_path.exists():
        return
    cmd = [
        "ffmpeg", "-loglevel", "quiet",
        "-i", str(mp3_path),
        "-ac", "1", "-ar", "16000", "-sample_fmt", "s16",
        str(wav_path)
    ]
    subprocess.run(cmd, check=False)

def main():
    with mp.Pool(8) as pool:
        list(tqdm.tqdm(pool.imap_unordered(convert, MP3_LIST),
                       total=len(MP3_LIST)))
    print("✅ MP3 已全部转 WAV")

if __name__ == "__main__":
    mp.freeze_support()          # 兼容 Windows exe 打包，可保留
    main()
