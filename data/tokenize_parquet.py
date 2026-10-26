"""Tokenize one FineWeb-Edu parquet file into a Megatron indexed dataset (<prefix>.bin/.idx).

Matches tools/preprocess_data.py with HuggingFaceTokenizer + --append-eod: no BOS/special tokens,
EOS appended to every document, int32 storage (vocab 131072 > uint16).
"""
import os
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
import numpy as np
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder

MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16"
REVISION = "bf77c3174f68ad409e1c2aa60daeb46e32d1c606"  # the snapshot the study used


def main(src: str, prefix: str) -> None:
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    eod = tok.convert_tokens_to_ids(tok.eos_token)
    tmp = prefix + ".partial"
    builder = IndexedDatasetBuilder(tmp + ".bin", dtype=np.int32)
    ntok = ndoc = 0
    for batch in pq.ParquetFile(src).iter_batches(batch_size=1000, columns=["text"]):
        for ids in tok(batch.column(0).to_pylist(), add_special_tokens=False)["input_ids"]:
            ids.append(eod)
            builder.add_document(np.asarray(ids, dtype=np.int32), [len(ids)])
            ntok += len(ids)
            ndoc += 1
    builder.finalize(tmp + ".idx")
    # Publish atomically so a half-written file is never picked up by training.
    os.replace(tmp + ".bin", prefix + ".bin")
    os.replace(tmp + ".idx", prefix + ".idx")
    print(f"{os.path.basename(src)} docs={ndoc} tokens={ntok}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
