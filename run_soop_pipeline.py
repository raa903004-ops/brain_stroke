"""Wait for SOOP download, then resume GPU training until 150 epochs finish."""

import os
from pathlib import Path
import subprocess
import sys
import time

from datasets.soop_dataset import build_manifest


ROOT = Path(__file__).resolve().parent
LAST = ROOT / "weights" / "soop_dwi_adc_last.pth"
DOWNLOAD_LOG = ROOT / "soop_download.log"
DOWNLOAD_ERROR = ROOT / "soop_download_error.log"


def finished():
    if not LAST.is_file():
        return False
    import torch
    checkpoint = torch.load(LAST, map_location="cpu", weights_only=True)
    return int(checkpoint["epoch"]) >= 150


def main():
    while True:
        count = len(build_manifest())
        print(f"SOOP ready cases: {count}/1449", flush=True)
        if count == 1449:
            break
        failed = DOWNLOAD_ERROR.is_file() and "Traceback (most recent call last)" in DOWNLOAD_ERROR.read_text(errors="replace")
        stalled = DOWNLOAD_LOG.is_file() and time.time() - DOWNLOAD_LOG.stat().st_mtime > 20 * 60
        if failed or stalled:
            print("SOOP download stopped; resuming missing files", flush=True)
            DOWNLOAD_ERROR.write_text("")
            subprocess.run([sys.executable, "-u", "download_soop.py"], cwd=ROOT,
                           env=os.environ.copy())
        time.sleep(60)
    failures = 0
    while not finished():
        result = subprocess.run([sys.executable, "-u", "train_soop.py", "--epochs", "150"],
                                cwd=ROOT, env=os.environ.copy())
        if result.returncode == 0:
            break
        failures += 1
        print(f"Training exited {result.returncode}; restart {failures}/5 from last checkpoint", flush=True)
        if failures >= 5:
            raise RuntimeError("SOOP training repeatedly failed; inspect log")
        time.sleep(15)
    if not finished():
        raise RuntimeError("Training stopped before epoch 150")
    print("SOOP 150 epochs complete", flush=True)


if __name__ == "__main__":
    main()
