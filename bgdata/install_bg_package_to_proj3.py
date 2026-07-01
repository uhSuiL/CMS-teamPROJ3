"""Install a bg-only package into a PROJ3 CellMap data directory.

Run this on the target machine after uploading/extracting the bg-only package.

Steps:
1. Validate that every package crop*/bg directory has all declared scale arrays.
2. Delete all existing target crop*/bg directories.
3. Copy the package crop*/bg directories into the matching target crop folders.

Only bg label directories are deleted/copied. Other labels and raw data are not
touched.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path


# Edit these two paths after uploading to the supercomputer if needed.
BG_ONLY_PACKAGE_DIR = Path("/root/autodl-tmp/bgloader/data_bg3_bg_only")
TARGET_PROJ3_DATA_DIR = Path("/root/autodl-tmp/CMS-teamPROJ3/data")
BACKGROUND_LABEL = "bg"


def print_progress(label: str, current: int, total: int, start_time: float) -> None:
    if total <= 0:
        return
    ratio = current / total
    filled = int(30 * ratio)
    bar = "#" * filled + "-" * (30 - filled)
    elapsed = time.monotonic() - start_time
    eta = (elapsed / current * (total - current)) if current else 0
    print(
        f"\r{label} [{bar}] {current}/{total} "
        f"({ratio:6.2%}) elapsed {elapsed:6.1f}s ETA {eta:6.1f}s",
        end="",
        flush=True,
    )
    if current == total:
        print()


def get_multiscale_paths(label_dir: Path) -> list[str]:
    zattrs = label_dir / ".zattrs"
    if not zattrs.exists():
        return ["s0"]

    attrs = json.loads(zattrs.read_text(encoding="utf-8"))
    multiscales = attrs.get("multiscales", [])
    if not multiscales:
        return ["s0"]

    datasets = multiscales[0].get("datasets", [])
    paths = [dataset["path"] for dataset in datasets if "path" in dataset]
    return paths or ["s0"]


def get_missing_scale_paths(label_dir: Path) -> list[str]:
    return [
        scale_path
        for scale_path in get_multiscale_paths(label_dir)
        if not (label_dir / scale_path).exists()
    ]


def iter_package_bg_dirs(package_dir: Path):
    for groundtruth_dir in package_dir.glob("*/**/recon-1/labels/groundtruth"):
        for bg_dir in sorted(groundtruth_dir.glob(f"crop*/{BACKGROUND_LABEL}")):
            if bg_dir.is_dir():
                yield bg_dir


def iter_target_bg_dirs(target_data_dir: Path):
    for groundtruth_dir in target_data_dir.glob("*/**/recon-1/labels/groundtruth"):
        for bg_dir in sorted(groundtruth_dir.glob(f"crop*/{BACKGROUND_LABEL}")):
            if bg_dir.is_dir():
                yield bg_dir


def get_target_bg_dir(package_bg_dir: Path) -> Path:
    relative_path = package_bg_dir.relative_to(BG_ONLY_PACKAGE_DIR)
    return TARGET_PROJ3_DATA_DIR / relative_path


def validate_package(package_bg_dirs: list[Path]) -> None:
    incomplete = [
        (bg_dir, get_missing_scale_paths(bg_dir))
        for bg_dir in package_bg_dirs
        if get_missing_scale_paths(bg_dir)
    ]
    if incomplete:
        print("Incomplete package bg directories:")
        for path, missing in incomplete[:20]:
            print(f"  {path}: missing {missing}")
        if len(incomplete) > 20:
            print(f"  ... and {len(incomplete) - 20} more")
        raise RuntimeError("Refusing to install incomplete bg package.")

    missing_targets = [
        get_target_bg_dir(bg_dir).parent
        for bg_dir in package_bg_dirs
        if not get_target_bg_dir(bg_dir).parent.exists()
    ]
    if missing_targets:
        print("Missing target crop directories:")
        for path in missing_targets[:20]:
            print(f"  {path}")
        if len(missing_targets) > 20:
            print(f"  ... and {len(missing_targets) - 20} more")
        raise RuntimeError("Refusing to install until target crop directories exist.")


def clear_target_bg_dirs() -> int:
    target_bg_dirs = list(iter_target_bg_dirs(TARGET_PROJ3_DATA_DIR))
    if not target_bg_dirs:
        print("No existing target bg directories to clear.")
        return 0

    print(f"Clearing {len(target_bg_dirs)} existing target bg directories.")
    start_time = time.monotonic()
    for index, target_bg_dir in enumerate(target_bg_dirs, start=1):
        if target_bg_dir.name != BACKGROUND_LABEL:
            raise ValueError(f"Refusing to remove non-bg directory: {target_bg_dir}")
        shutil.rmtree(target_bg_dir)
        print_progress("Clearing target bg", index, len(target_bg_dirs), start_time)
    return len(target_bg_dirs)


def install_package() -> None:
    if not BG_ONLY_PACKAGE_DIR.exists():
        raise FileNotFoundError(f"Missing bg-only package: {BG_ONLY_PACKAGE_DIR}")
    if not TARGET_PROJ3_DATA_DIR.exists():
        raise FileNotFoundError(f"Missing target data dir: {TARGET_PROJ3_DATA_DIR}")

    package_bg_dirs = list(iter_package_bg_dirs(BG_ONLY_PACKAGE_DIR))
    if not package_bg_dirs:
        raise RuntimeError(f"No bg directories found in {BG_ONLY_PACKAGE_DIR}")

    validate_package(package_bg_dirs)

    print(f"Package: {BG_ONLY_PACKAGE_DIR}")
    print(f"Target: {TARGET_PROJ3_DATA_DIR}")
    print(f"Installing {len(package_bg_dirs)} bg directories.")

    cleared = clear_target_bg_dirs()
    start_time = time.monotonic()
    copied = 0
    for index, package_bg_dir in enumerate(package_bg_dirs, start=1):
        target_bg_dir = get_target_bg_dir(package_bg_dir)
        target_bg_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(package_bg_dir, target_bg_dir)
        copied += 1
        print_progress("Copying package bg", index, len(package_bg_dirs), start_time)

    print(f"Done. Cleared {cleared} old bg dirs and installed {copied} new bg dirs.")


if __name__ == "__main__":
    install_package()
