"""Tokenize the downloaded FineWeb-Edu sample/350BT parquet shards with GPT-2
BPE into per-file uint16 .bin shards (nanoGPT layout).

One worker process per parquet file (resumable: a finished shard writes a
<name>.bin.done marker and is skipped on re-run). Documents are joined with the
GPT-2 end-of-text token (50256).

Usage:
  python tokenize_fineweb.py --workers 96
"""

import argparse
import glob
import os
import time
from multiprocessing import Pool

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import tiktoken

# Cap pyarrow's internal thread pool: with many worker processes its default
# (= all cores) would oversubscribe the CPU massively.
pa.set_cpu_count(1)

BASE = os.path.dirname(__file__)
RAW_DIR = os.path.join(BASE, "data_fineweb", "raw")
OUT_DIR = os.path.join(BASE, "data_fineweb", "bin")

_enc = tiktoken.get_encoding("gpt2")
EOT = _enc.eot_token  # 50256


def tokenize_file(pf):
    out = os.path.join(OUT_DIR, os.path.basename(pf).replace(".parquet", ".bin"))
    done = out + ".done"
    if os.path.exists(done):
        return (os.path.basename(pf), 0, "skip")
    # Stream the parquet in small row-batches and write tokens incrementally,
    # so peak RAM is bounded by one batch (~a few MB) regardless of file size.
    tmp = out + ".tmp"
    ntok = 0
    pqf = pq.ParquetFile(pf)
    with open(tmp, "wb") as fh:
        for batch in pqf.iter_batches(batch_size=2048, columns=["text"], use_threads=False):
            texts = batch.column("text").to_pylist()
            parts = []
            for ids in _enc.encode_ordinary_batch(texts, num_threads=1):
                ids.append(EOT)
                parts.append(np.asarray(ids, dtype=np.uint16))
            if parts:
                arr = np.concatenate(parts)
                fh.write(arr.tobytes())
                ntok += int(arr.size)
    os.replace(tmp, out)
    open(done, "w").close()
    return (os.path.basename(pf), ntok, "ok")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=96)
    args = p.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    files = sorted(glob.glob(os.path.join(RAW_DIR, "**", "*.parquet"), recursive=True))
    print(f"tokenizing {len(files)} parquet files with {args.workers} workers")

    total_tokens = 0
    done_files = 0
    t0 = time.time()
    with Pool(args.workers) as pool:
        for name, ntok, status in pool.imap_unordered(tokenize_file, files):
            done_files += 1
            total_tokens += ntok
            if done_files % 10 == 0 or status == "skip":
                dt = time.time() - t0
                rate = total_tokens / max(dt, 1e-9)
                print(f"[{done_files}/{len(files)}] {name} {status} "
                      f"| {total_tokens/1e9:.2f}B tok | {rate/1e6:.1f} M tok/s | {dt:.0f}s")
    print(f"DONE: {total_tokens/1e9:.2f}B tokens across {len(files)} shards "
          f"in {(time.time()-t0)/60:.1f} min -> {OUT_DIR}")


if __name__ == "__main__":
    main()
