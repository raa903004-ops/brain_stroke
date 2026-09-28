"""Download the public ISLES 2022 archive with resumable range requests."""

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import shutil
import time

import requests


URL = "https://zenodo.org/api/records/7153326/files/ISLES-2022.zip/content"
SIZE = 1_692_717_470
MD5 = "302ee280373cdd5c190ab763d72a7a50"
TARGET = Path(r"D:\ISLES-2022.zip")
PARTS = Path(r"D:\ISLES-2022.parts")
COUNT = 170


def download_part(index):
    start = SIZE * index // COUNT
    end = SIZE * (index + 1) // COUNT - 1
    path = PARTS / f"{index:02d}.part"
    length = end - start + 1
    for attempt in range(10):
        have = path.stat().st_size if path.exists() else 0
        if have == length:
            return index
        if have > length:
            path.unlink()
            have = 0
        try:
            with requests.get(URL, headers={"Range": f"bytes={start + have}-{end}"},
                              stream=True, timeout=(30, 120)) as response:
                response.raise_for_status()
                expected = f"bytes {start + have}-{end}/{SIZE}"
                if response.status_code != 206 or response.headers.get("Content-Range") != expected:
                    raise RuntimeError(f"Unexpected range response: {response.status_code} {response.headers.get('Content-Range')}")
                with path.open("ab") as output:
                    for block in response.iter_content(1024 * 1024):
                        if block:
                            output.write(block)
        except (requests.RequestException, RuntimeError) as exc:
            print(f"part {index}: retry {attempt + 1}: {exc}", flush=True)
            time.sleep(min(2 ** attempt, 30))
            continue
    if path.stat().st_size != length:
        raise RuntimeError(f"Part {index} incomplete")
    return index


def main():
    PARTS.mkdir(exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(download_part, i) for i in range(COUNT)]
        for future in as_completed(futures):
            print(f"complete part {future.result() + 1}/{COUNT}", flush=True)
    digest = hashlib.md5()
    with TARGET.open("wb") as output:
        for index in range(COUNT):
            with (PARTS / f"{index:02d}.part").open("rb") as source:
                while block := source.read(4 * 1024 * 1024):
                    output.write(block)
                    digest.update(block)
    if TARGET.stat().st_size != SIZE or digest.hexdigest() != MD5:
        raise RuntimeError(f"Checksum mismatch: {digest.hexdigest()}")
    print(f"verified {TARGET} ({SIZE} bytes, md5 {MD5})", flush=True)
    shutil.rmtree(PARTS)


if __name__ == "__main__":
    main()
