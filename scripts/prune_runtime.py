"""Remove development-only files from an installed scientific runtime.

The script deliberately operates only inside selected third-party package
directories. It never removes modules, shared libraries, package metadata, or
data outside those allow-listed roots.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import sysconfig
from pathlib import Path


PACKAGE_NAMES = ("mcubes", "numpy", "plotly", "pyscf", "scipy")
DEVELOPMENT_DIRECTORY_NAMES = frozenset({"benchmarks", "test", "tests"})
DEVELOPMENT_FILE_SUFFIXES = (".pyi",)
DEVELOPMENT_FILENAMES = frozenset({"py.typed"})


def allocated_bytes(path: Path) -> int:
    """Return ordinary file bytes beneath a path before it is removed."""
    if path.is_file():
        return path.stat().st_size
    return sum(
        item.stat().st_size
        for item in path.rglob("*")
        if item.is_file() and not item.is_symlink()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-system-prefix",
        action="store_true",
        help="permit pruning an isolated container's system Python",
    )
    arguments = parser.parse_args()

    environment_prefix = Path(sys.prefix).resolve()
    base_prefix = Path(sys.base_prefix).resolve()
    if environment_prefix == base_prefix and not arguments.allow_system_prefix:
        raise SystemExit(
            "Refusing to modify the system Python. Run this script with the "
            "application virtual environment."
        )

    library_roots = {
        Path(path).resolve()
        for key, path in sysconfig.get_paths().items()
        if key in {"purelib", "platlib"} and path
    }
    reclaimed = 0
    removed = 0
    for library_root in library_roots:
        for package_name in PACKAGE_NAMES:
            package_root = library_root / package_name
            if not package_root.is_dir():
                continue

            removable_directories = sorted(
                (
                    child
                    for child in package_root.rglob("*")
                    if child.is_dir()
                    and child.name.lower() in DEVELOPMENT_DIRECTORY_NAMES
                ),
                key=lambda path: len(path.parts),
                reverse=True,
            )
            for directory in removable_directories:
                if not directory.exists():
                    continue
                reclaimed += allocated_bytes(directory)
                shutil.rmtree(directory)
                removed += 1

            for file_path in package_root.rglob("*"):
                if not file_path.is_file() or file_path.is_symlink():
                    continue
                if (
                    file_path.name in DEVELOPMENT_FILENAMES
                    or file_path.suffix in DEVELOPMENT_FILE_SUFFIXES
                ):
                    reclaimed += file_path.stat().st_size
                    file_path.unlink()
                    removed += 1

    print(
        f"Removed {removed} development-only paths; "
        f"reclaimed {reclaimed / 1024**2:.1f} MiB."
    )


if __name__ == "__main__":
    main()
