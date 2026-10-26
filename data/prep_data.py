"""Download FineWeb-Edu sample/350BT and tokenize each parquet file as soon as it lands. Resumable.

Layout: $DATA_ROOT/raw/<name>.parquet -> $DATA_ROOT/tok/<name>.{bin,idx}   (DATA_ROOT default /mnt/nvme/data)
The last file (sorted) is the held-out validation shard. Downloads go in file order so the first
shards (needed by the isoFLOP sweep) are ready first.
"""
import json
import os
import subprocess
import time
import urllib.request
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

REPO = "HuggingFaceFW/fineweb-edu"
SUB = "sample/350BT"
DATA_ROOT = os.environ.get("DATA_ROOT", "/mnt/nvme/data")
RAW, TOK = f"{DATA_ROOT}/raw", f"{DATA_ROOT}/tok"
DL_WORKERS = int(os.environ.get("DL_WORKERS", 12))
TOK_WORKERS = int(os.environ.get("TOK_WORKERS", 100))


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def list_files():
    with urllib.request.urlopen(f"https://huggingface.co/api/datasets/{REPO}/tree/main/{SUB}") as r:
        items = json.load(r)
    return sorted((x["path"], x["size"]) for x in items if x["path"].endswith(".parquet"))


def download(path, size):
    name = os.path.basename(path)[: -len(".parquet")]
    dst = f"{RAW}/{name}.parquet"
    if os.path.exists(dst) and os.path.getsize(dst) == size:
        return name
    url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{path}"
    for attempt in range(8):
        rc = subprocess.call(["curl", "-sSfL", "--retry", "5", "-o", dst + ".part", url])
        if rc == 0 and os.path.getsize(dst + ".part") == size:
            os.replace(dst + ".part", dst)
            return name
        log(f"download retry {attempt} {name} rc={rc}")
        time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"download failed: {name}")


def tokenize(name):
    if os.path.exists(f"{TOK}/{name}.idx"):
        return name, "cached"
    out = subprocess.run(
        ["python", os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokenize_parquet.py"), f"{RAW}/{name}.parquet", f"{TOK}/{name}"],
        capture_output=True, text=True,
    )
    if out.returncode != 0:
        return name, "FAILED " + out.stderr[-500:]
    return name, out.stdout.strip().splitlines()[-1]


def main():
    os.makedirs(RAW, exist_ok=True)
    os.makedirs(TOK, exist_ok=True)
    files = list_files()
    log(f"{len(files)} files, {sum(s for _, s in files) / 1e9:.1f} GB; val shard = {files[-1][0]}")
    # Validation shard first, then training shards in order.
    order = [files[-1]] + files[:-1]
    done = failed = 0
    with ThreadPoolExecutor(DL_WORKERS) as dl, ProcessPoolExecutor(TOK_WORKERS) as tp:
        tok_futs = []
        for fut in [dl.submit(download, p, s) for p, s in order]:
            try:
                tok_futs.append(tp.submit(tokenize, fut.result()))
            except Exception as e:  # noqa: BLE001
                failed += 1
                log(f"ERROR {e}")
        for f in tok_futs:
            name, msg = f.result()
            done += 1
            if msg.startswith("FAILED"):
                failed += 1
            log(f"tok {done}/{len(order)} {name}: {msg}")
    log(f"ALL DONE failed={failed}")


if __name__ == "__main__":
    main()
