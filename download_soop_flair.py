"""Download SOOP FLAIR volumes from the public OpenNeuro S3 release.

Only subjects already selected for the DWI/ADC acute-lesion cohort are fetched.
Downloads are resumable, size-checked, and atomically installed.
"""

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import threading
import time
import xml.etree.ElementTree as ET

import requests
import truststore

truststore.inject_into_ssl()

ROOT = Path(r"D:\SOOP")
S3 = "https://s3.amazonaws.com/openneuro.org/"
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
MANIFEST = ROOT / "flair_manifest.json"
thread_local = threading.local()


def session():
    if not hasattr(thread_local, "session"):
        thread_local.session = requests.Session()
    return thread_local.session


def inventory():
    wanted = {
        row["subject_id"]
        for line in (ROOT / "subjects.jsonl").read_text(encoding="utf-8").splitlines()
        if (row := json.loads(line)).get("evaluable_lesionAcute") and row.get("adc_usable")
    }
    items = {}
    token = None
    page = 0
    while True:
        params = {"list-type": "2", "prefix": "ds004889/", "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        response = session().get(S3, params=params, timeout=60)
        response.raise_for_status()
        doc = ET.fromstring(response.content)
        for entry in doc.findall("s3:Contents", NS):
            key = entry.findtext("s3:Key", namespaces=NS)
            parts = key.split("/")
            if (len(parts) == 4 and parts[1] in wanted and parts[2] == "anat"
                    and parts[3].endswith("_FLAIR.nii.gz")):
                items[parts[1]] = {
                    "key": key,
                    "size": int(entry.findtext("s3:Size", namespaces=NS)),
                    "etag": entry.findtext("s3:ETag", namespaces=NS).strip('"'),
                }
        page += 1
        if page % 10 == 0:
            print(f"Listed {page} S3 pages; matching FLAIR files: {len(items)}", flush=True)
        if doc.findtext("s3:IsTruncated", namespaces=NS) != "true":
            break
        token = doc.findtext("s3:NextContinuationToken", namespaces=NS)
    MANIFEST.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Inventory: {len(items)}/{len(wanted)} subjects with FLAIR; "
          f"{sum(item['size'] for item in items.values()) / 2**30:.2f} GiB", flush=True)
    return items


def fetch(subject, item):
    target = ROOT / item["key"].removeprefix("ds004889/")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == item["size"]:
        return subject, 0
    partial = target.with_name(target.name + ".part")
    for attempt in range(5):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > item["size"]:
                partial.unlink()
                offset = 0
            headers = {"Range": f"bytes={offset}-"} if offset else {}
            with session().get(S3 + item["key"], headers=headers, stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                if offset and response.status_code != 206:
                    raise RuntimeError(f"Resume ignored for {subject}")
                with partial.open("ab" if offset else "wb") as output:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            output.write(chunk)
            if partial.stat().st_size != item["size"]:
                raise RuntimeError(f"Size mismatch for {subject}")
            if len(item["etag"]) == 32 and "-" not in item["etag"]:
                digest = hashlib.md5()
                with partial.open("rb") as source:
                    for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
                        digest.update(chunk)
                if digest.hexdigest() != item["etag"]:
                    partial.unlink()
                    raise RuntimeError(f"Checksum mismatch for {subject}")
            os.replace(partial, target)
            return subject, item["size"]
        except Exception as exc:
            if attempt == 4:
                raise RuntimeError(f"{subject}: {exc}") from exc
            time.sleep(2 ** attempt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    items = inventory() if not MANIFEST.exists() else json.loads(MANIFEST.read_text(encoding="utf-8"))
    if args.inventory_only:
        print(f"Manifest: {len(items)} files, {sum(x['size'] for x in items.values()) / 2**30:.2f} GiB")
        return
    missing = [(subject, item) for subject, item in items.items()
               if not (ROOT / item["key"].removeprefix("ds004889/")).is_file()
               or (ROOT / item["key"].removeprefix("ds004889/")).stat().st_size != item["size"]]
    print(f"Downloading {len(missing)} FLAIR volumes with {args.workers} workers", flush=True)
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, subject, item) for subject, item in missing]
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 25 == 0 or done == len(missing):
                print(f"FLAIR {done}/{len(missing)} ({100 * done / max(len(missing), 1):.1f}%)", flush=True)
    print("SOOP FLAIR download complete", flush=True)


if __name__ == "__main__":
    main()
