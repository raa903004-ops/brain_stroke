"""Download, verify, and extract the ISLES 2022 dataset to drive D."""

from pathlib import Path
import zipfile

from download_isles import TARGET, main as download


DESTINATION = Path(r"D:\ISLES-2022")


def main():
    download()
    with zipfile.ZipFile(TARGET) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"Corrupt archive member: {bad}")
        names = archive.namelist()
        nested = names and all(name.startswith("ISLES-2022/") for name in names)
        base = DESTINATION.parent if nested else DESTINATION
        base = base.resolve()
        for name in names:
            target = (base / name).resolve()
            if not target.is_relative_to(base):
                raise RuntimeError(f"Unsafe archive member: {name}")
        archive.extractall(base)
    cases = list(DESTINATION.glob("sub-strokecase*"))
    print(f"extracted {len(cases)} cases to {DESTINATION}", flush=True)
    if len(cases) < 200:
        raise RuntimeError("Expected at least 200 cases after extraction")


if __name__ == "__main__":
    main()
