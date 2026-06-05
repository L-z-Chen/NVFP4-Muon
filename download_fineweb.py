"""Download all parquet shards of HuggingFaceFW/fineweb-edu sample/350BT
(~1.0 TB, 472 files) to data_fineweb/raw. Resumable: re-running skips files
already present."""

import os

from huggingface_hub import snapshot_download

os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

path = snapshot_download(
    "HuggingFaceFW/fineweb-edu",
    repo_type="dataset",
    allow_patterns="sample/350BT/*.parquet",
    local_dir=os.path.join(os.path.dirname(__file__), "data_fineweb", "raw"),
    max_workers=32,
)
print("downloaded to", path)
