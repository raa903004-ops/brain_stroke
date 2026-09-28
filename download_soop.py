"""Download the annotated TRACE, ADC, and acute masks from SOOP."""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import time

os.environ.setdefault("HF_HOME", r"D:\hf-cache")
from huggingface_hub import hf_hub_download


ROOT = Path(r"D:\SOOP")
MANIFEST = ROOT / "subjects.jsonl"
REPO = "MedOtter/SOOP"


def fetch(path):
    target = ROOT / path
    if target.is_file() and target.stat().st_size:
        return path
    for attempt in range(5):
        try:
            hf_hub_download(REPO, path, repo_type="dataset", local_dir=ROOT)
            return path
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2 ** attempt)


def main():
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding="utf-8").splitlines()]
    usable = [row for row in rows if row.get("evaluable_lesionAcute") and row.get("adc_usable")]
    paths = sorted({row[key] for row in usable for key in
                    ("image_trace", "image_adc", "mask_lesionAcute")})
    print(f"SOOP usable subjects: {len(usable)}; files: {len(paths)}", flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(fetch, path) for path in paths]
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            if index % 50 == 0 or index == len(paths):
                print(f"Downloaded {index}/{len(paths)} ({100*index/len(paths):.0f}%)", flush=True)
    print("SOOP download complete", flush=True)


if __name__ == "__main__":
    main()
