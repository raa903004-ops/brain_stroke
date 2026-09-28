"""Resumable FLAIR download, registration, and separate three-channel training."""

import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PYTHON = r"D:\brain-stroke-gpu-venv\Scripts\python.exe"


def stage(name, args, attempts=1):
    for attempt in range(1, attempts + 1):
        print(f"\n=== {name} (attempt {attempt}/{attempts}) ===", flush=True)
        result = subprocess.run([PYTHON, "-u", *args], cwd=ROOT, check=False)
        if result.returncode == 0:
            print(f"{name} complete", flush=True)
            return
        print(f"{name} exited {result.returncode}", flush=True)
    raise RuntimeError(f"{name} failed after {attempts} attempts")


def main():
    stage("FLAIR download", ["download_soop_flair.py", "--workers", "32"], attempts=5)
    stage("FLAIR registration", ["prepare_soop_flair.py"])
    stage("DWI+ADC+FLAIR training", ["train_soop_flair.py", "--epochs", "150", "--patience", "12"], attempts=5)
    stage("External evaluation", ["evaluate_soop_flair.py"])
    print("SOOP FLAIR pipeline complete; new model remains separate pending external validation", flush=True)


if __name__ == "__main__":
    main()
