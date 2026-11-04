"""Segmented, resumable download (N parallel HTTP range connections per file) for the EgoExOR legacy HF files.

HF gives only ~6 MB/s per connection, so splitting a file into ranges speeds it up roughly N-fold.
Usage: python scripts/download_segmented.py miss_1.h5 miss_3.h5 [--segments 8]
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import egoexor_io as io  # noqa: E402

URL = "https://huggingface.co/datasets/ardamamur/EgoExOR/resolve/main/{}"


def fetch_segment(url, dst, start, end, state):
    """Download bytes [start, end] into dst at offset start; resumable via a per-segment progress file."""
    prog = Path(str(dst) + f".seg{start}")
    done = int(prog.read_text()) if prog.exists() else 0
    pos = start + done
    while pos <= end:
        try:
            with requests.get(url, headers={"Range": f"bytes={pos}-{end}"}, stream=True, timeout=30) as r:
                r.raise_for_status()
                with open(dst, "r+b") as f:
                    f.seek(pos)
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                        pos += len(chunk)
                        state["bytes"] += len(chunk)
                        prog.write_text(str(pos - start))
        except (requests.RequestException, OSError) as e:
            print(f"retry segment {start}: {e}", flush=True)
            time.sleep(3)
    prog.write_text(str(end - start + 1))


def download(name, segments):
    size = io.EXPECTED_BYTES[name]
    dst = io.RAW_DIR / name
    if dst.exists() and dst.stat().st_size == size:
        print("already complete:", name)
        return
    url = URL.format(name)
    if not dst.exists() or dst.stat().st_size != size:
        # (re)allocate full size; resume info lives in the .seg files. A stale shorter file from curl is discarded.
        for p in io.RAW_DIR.glob(name + ".seg*"):
            p.unlink()
        with open(dst, "wb") as f:
            f.truncate(size)
    seg = -(-size // segments)
    state = {"bytes": 0}
    threads = []
    for i in range(segments):
        s, e = i * seg, min((i + 1) * seg, size) - 1
        t = threading.Thread(target=fetch_segment, args=(url, dst, s, e, state), daemon=True)
        t.start()
        threads.append(t)
    t0 = time.time()
    while any(t.is_alive() for t in threads):
        time.sleep(10)
        print(
            f"{name}: {state['bytes'] / 1e9:.2f}/{size / 1e9:.2f} GB  {state['bytes'] / (time.time() - t0) / 1e6:.1f} MB/s",
            flush=True,
        )
    for p in io.RAW_DIR.glob(name + ".seg*"):
        p.unlink()
    print("DONE", name, flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--segments", type=int, default=8)
    a = ap.parse_args()
    ths = [threading.Thread(target=download, args=(n, a.segments)) for n in a.files]
    [t.start() for t in ths]
    [t.join() for t in ths]
