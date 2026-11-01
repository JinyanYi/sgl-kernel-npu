"""Build an isolated Torch wrapper for an installed donor A5 ACLNN kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(root: Path, *args: str) -> str | None:
    result = subprocess.run(
        ["git", "-C", str(root), *args], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        help="use an explicit donor source tree instead of bundled sources",
    )
    parser.add_argument(
        "--source-manifest", type=Path, help="manifest from prepare_original.py"
    )
    parser.add_argument("--build-dir", required=True, type=Path)
    args = parser.parse_args()
    build = args.build_dir.resolve()
    if build.exists():
        parser.error("build directory exists; preserve it and choose a new directory")
    source_manifest = args.source_manifest
    if args.source_root:
        source = args.source_root.resolve(strict=True)
    else:
        prepared = build.parent / (build.name + "-source")
        prepare_script = Path(__file__).resolve().with_name("prepare_original.py")
        subprocess.run(
            [sys.executable, str(prepare_script), "--output-root", str(prepared)],
            check=True,
        )
        source = prepared / "source"
        source_manifest = prepared / "SOURCE_MANIFEST.json"
    if source_manifest and not source_manifest.is_file():
        parser.error("source manifest does not exist")
    csrc = source / "csrc"
    adapter = csrc / "attention/fused_quant_lightning_indexer_manage"
    bridge_sources = [
        csrc / "aclnn_torch_adapter/NPUBridge.cpp",
        csrc / "aclnn_torch_adapter/NPUStorageImpl.cpp",
    ]
    required = [
        adapter / "fused_quant_lightning_indexer_manage_torch_adpt.h",
        adapter / "fused_quant_lightning_indexer_manage_ordered.h",
        csrc / "aclnn_torch_adapter/op_api_common.h",
    ] + bridge_sources
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        parser.error("donor A5 source files missing: " + ", ".join(missing))
    ascend_home = os.environ.get("ASCEND_HOME_PATH")
    if not ascend_home:
        parser.error("source the Ascend environment before building the wrapper")

    import torch
    import torch_npu
    from torch.utils.cpp_extension import load

    build.mkdir(parents=True)
    here = Path(__file__).resolve().parent
    npu_root = Path(torch_npu.__file__).resolve().parent
    result = load(
        name="a5_original_lim_binding",
        sources=[str(here / "original_binding.cpp")]
        + [str(path) for path in bridge_sources],
        extra_include_paths=[
            str(csrc),
            str(npu_root.parent),
            str(npu_root / "include"),
            str(Path(ascend_home) / "include"),
        ],
        extra_cflags=["-O3", "-std=c++17", "-fvisibility=hidden"],
        extra_ldflags=[
            "-L" + str(npu_root / "lib"),
            "-ltorch_npu",
            "-L" + str(Path(ascend_home) / "lib64"),
            "-lascendcl",
            "-ldl",
        ],
        build_directory=str(build),
        is_python_module=False,
        verbose=True,
    )
    library = Path(result) if isinstance(result, str) else None
    if library is None or not library.is_file():
        matches = list(build.glob("a5_original_lim_binding*.so"))
        if len(matches) != 1:
            raise RuntimeError("cannot identify the built original A5 wrapper")
        library = matches[0]
    donor_files = sorted(path for path in adapter.rglob("*") if path.is_file())
    git_top = git_value(source, "rev-parse", "--show-toplevel")
    source_is_git_root = bool(git_top and Path(git_top).resolve() == source)
    manifest = {
        "kind": "donor_adapter_wrapper_not_a_rebuilt_device_kernel",
        "library": str(library.resolve()),
        "library_sha256": checksum(library),
        "torch_op": "a5_original_lim.run",
        "source_root": str(source),
        "source_revision": (
            git_value(source, "rev-parse", "HEAD") if source_is_git_root else None
        ),
        "source_status": (
            git_value(source, "status", "--short") if source_is_git_root else None
        ),
        "source_manifest": str(source_manifest.resolve()) if source_manifest else None,
        "source_manifest_sha256": (
            checksum(source_manifest) if source_manifest else None
        ),
        "source_provenance": (
            json.loads(source_manifest.read_text(encoding="utf-8"))
            if source_manifest
            else None
        ),
        "donor_files": {
            str(path.relative_to(source)): checksum(path) for path in donor_files
        },
        "wrapper_sha256": checksum(here / "original_binding.cpp"),
        "torch": torch.__version__,
        "torch_npu": torch_npu.__version__,
        "ascend_home": ascend_home,
        "ascend_custom_opp_path": os.environ.get("ASCEND_CUSTOM_OPP_PATH"),
        "note": (
            "The donor ACLNN/OPP device kernel must already be installed. "
            "This wrapper preserves the donor adapter, including its ordered "
            "submission, and does not substitute the SGLang direct launcher."
        ),
    }
    (build / "binding_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
