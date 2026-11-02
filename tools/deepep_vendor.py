#!/usr/bin/env python3
"""Reuse installed DeepEP vendor bytes when rebuilding only its host bridge."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def manifest(root):
    root = Path(root).resolve(strict=True)
    files = {}

    def visit(directory, ancestors):
        canonical = directory.resolve(strict=True)
        if not canonical.is_relative_to(root) or canonical in ancestors:
            raise ValueError(f"External or cyclic vendor directory: {directory}")
        for path in sorted(directory.iterdir()):
            if not path.resolve(strict=True).is_relative_to(root):
                raise ValueError(f"Vendor path escapes source: {path}")
            if path.is_dir():
                visit(path, ancestors | {canonical})
            elif path.is_file():
                with path.open("rb") as stream:
                    files[path.relative_to(root).as_posix()] = hashlib.file_digest(
                        stream, "sha256"
                    ).hexdigest()

    visit(root, set())
    if "op_api/lib/libcust_opapi.so" not in files:
        raise ValueError("DeepEP vendor lacks op_api/lib/libcust_opapi.so")
    return files


def stage(source, package_root):
    source = Path(source).resolve(strict=True)
    package = Path(package_root).resolve()
    vendors = package / "vendors"
    target = vendors / "hwcomputing"
    if vendors.is_symlink() or target.is_symlink():
        raise ValueError("Vendor destination must not be a symlink")
    if source.is_relative_to(target.resolve()) or target.resolve().is_relative_to(
        source
    ):
        raise ValueError("Vendor source and destination must not overlap")
    before = manifest(source)
    if target.exists():
        shutil.rmtree(target)
    vendors.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, symlinks=False)
    if manifest(target) != before:
        raise ValueError("Copied DeepEP vendor differs from installed bytes")
    return dict(source=str(source), destination=str(target), files=before)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--package-root", type=Path)
    mode.add_argument("--compare", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()
    if args.compare:
        before = manifest(args.source)
        if manifest(args.compare) != before:
            raise ValueError("DeepEP vendor bytes changed")
        print(json.dumps(dict(status="identical", files=len(before))))
    else:
        result = stage(args.source, args.package_root)
        if args.receipt:
            args.receipt.write_text(json.dumps(result, indent=2) + "\n")
        print(
            json.dumps(dict(status="copied_and_verified", files=len(result["files"])))
        )


if __name__ == "__main__":
    main()
