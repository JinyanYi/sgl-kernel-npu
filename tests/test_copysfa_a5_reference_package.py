"""The offline original reference must preserve pinned upstream source bytes."""

import hashlib
import json
import tarfile
from pathlib import Path


def test_original_reference_archive_and_dependencies_match_manifest():
    root = Path(__file__).parents[1] / "benchmark/copysfa_a5_ab/reference"
    provenance = json.loads((root / "PROVENANCE.json").read_text(encoding="utf-8"))
    assert provenance["revision"] == "1b180deb840484c2fe33cbdd7abb68a615085172"
    for name, expected in provenance["dependencies"].items():
        assert (
            hashlib.sha256((root / "vendor" / name).read_bytes()).hexdigest()
            == expected
        ), name
    archive = root / "vendor/original-copysfa-pr18490-source.tar.xz"
    expected = {entry["path"]: entry["sha256"] for entry in provenance["files"]}
    with tarfile.open(archive) as source:
        files = {
            member.name: member for member in source.getmembers() if member.isfile()
        }
        assert files.keys() == expected.keys()
        assert not any("fused_quant_lightning_indexer_manage" in name for name in files)
        for name, member in files.items():
            assert (
                hashlib.sha256(source.extractfile(member).read()).hexdigest()
                == expected[name]
            ), name
    binding = (root / "original_binding.cpp").read_text(encoding="utf-8")
    assert "EXEC_OWNED_KERNEL_CMD" not in binding
    assert (
        "a5_original_copysfa_vllm::npu_fused_quant_scatter_copy_sparse_flash_attention"
        in "".join(binding.split())
    )
    assert "at::empty({0}" in binding and "NPUStorageImpl" in binding
