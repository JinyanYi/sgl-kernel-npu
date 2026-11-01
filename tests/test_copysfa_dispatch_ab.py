"""CPU checks for the diagnostic input contract and unchanged accuracy gates."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location(
    "copysfa_dispatch_ab", ROOT / "benchmark/copysfa_a5_ab/dispatch_ab.py"
)
dispatch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dispatch)

from fused_quant_scatter_copy_sparse_flash_attention_golden import make_test_data


def test_dispatch_cases_cover_both_dimensions_and_execution_modes():
    assert len(set(dispatch.cases())) == 12
    assert set(dispatch.cases()) == {
        f"{dtype}_h{heads}_{mode}"
        for dtype in ("float16", "bfloat16")
        for heads in (8, 16)
        for mode in ("eager", "graph")
    } | {
        f"float16_h{heads}_{mode}_typedkv"
        for heads in (8, 16)
        for mode in ("eager", "graph")
    }


def test_small_absolute_error_still_fails_original_relative_l2_gate():
    golden = torch.full((1, 8, 512), 0.001)
    actual = golden * 1.48
    result = dispatch.compare(
        actual,
        golden,
        atol=dispatch.CPU_ATOL,
        rtol=dispatch.CPU_RTOL,
        max_relative_l2=dispatch.CPU_REL_L2,
    )
    assert result["allclose"]
    assert not result["passed"]
    assert result["max_relative_l2"] == pytest.approx(0.48)
    assert (dispatch.CPU_ATOL, dispatch.CPU_RTOL, dispatch.CPU_REL_L2) == (
        0.008,
        0.05,
        0.08,
    )
    assert (dispatch.PARITY_ATOL, dispatch.PARITY_RTOL) == (0.002, 0.005)


def test_nonfinite_outputs_fail_and_remain_json_serializable():
    result = dispatch.compare(
        torch.full((1, 8, 512), float("nan")),
        torch.ones(1, 8, 512),
        atol=0.008,
        rtol=0.05,
    )
    assert not result["passed"]
    json.dumps(result, allow_nan=False)


def test_dtype_head_cross_keeps_identical_kv_and_failing_seed():
    data = [
        make_test_data(8, query_dtype=dtype, query_heads=heads, seed=dispatch.SEED)
        for dtype in (torch.float16, torch.bfloat16)
        for heads in (8, 16)
    ]
    original_failure = make_test_data(
        8, query_dtype=torch.float16, query_heads=8, seed=20261032
    )
    torch.testing.assert_close(data[0].query, original_failure.query, rtol=0, atol=0)
    for value in data:
        for field in (
            "dram",
            "initial_hbm",
            "expected_hbm",
            "hbm_block_table",
            "dram_block_table",
            "topk_src_ids",
            "topk_dst_slots",
            "miss_src_ids",
            "miss_dst_slots",
            "miss_counts",
        ):
            assert dispatch.tensor_digest(
                getattr(value, field)
            ) == dispatch.tensor_digest(getattr(original_failure, field))
    torch.testing.assert_close(data[0].query, data[1].query[:, :8], rtol=0, atol=0)
    torch.testing.assert_close(data[2].query, data[3].query[:, :8], rtol=0, atol=0)


def test_driver_runs_all_cases_after_accuracy_failure(tmp_path, monkeypatch):
    def fake_process(command, **kwargs):
        output = Path(command[command.index("--output-root") + 1])
        case = command[command.index("--case") + 1]
        output.mkdir()
        failed = case == dispatch.cases()[0]
        dispatch.write_json(
            output / "case.json", dict(status="accuracy_failed" if failed else "pass")
        )
        return type("Result", (), dict(returncode=int(failed)))()

    monkeypatch.setattr(dispatch.subprocess, "run", fake_process)
    monkeypatch.setattr(
        sys, "argv", ["dispatch_ab.py", "--output-root", str(tmp_path / "result")]
    )
    assert dispatch.main() == 1
    report = json.loads((tmp_path / "result/summary.json").read_text())
    assert len(report["cases"]) == report["expected_cases"] == 12
    assert report["status"] == "failed"
    assert [case["exit_code"] for case in report["cases"]] == [1] + [0] * 11


def test_matching_fp16_kv_changes_rope_encoding_but_keeps_queries_and_scales():
    legacy = make_test_data(8, query_dtype=torch.float16, seed=dispatch.SEED)
    typed = make_test_data(
        8, query_dtype=torch.float16, seed=dispatch.SEED, kv_dtype=torch.float16
    )
    torch.testing.assert_close(typed.query, legacy.query, rtol=0, atol=0)
    assert torch.equal(legacy.dram[..., :512], typed.dram[..., :512])
    assert torch.equal(legacy.dram[..., 640:], typed.dram[..., 640:])
    assert not torch.equal(legacy.dram[..., 512:640], typed.dram[..., 512:640])
    assert torch.isfinite(typed.attention_golden).all()


def test_legacy_fp16_byte_interpretation_reproduces_reported_error():
    legacy = make_test_data(8, query_dtype=torch.float16, seed=dispatch.SEED)
    result = dispatch.compare(
        dispatch.kernel_layout_golden(legacy),
        legacy.attention_golden,
        atol=dispatch.CPU_ATOL,
        rtol=dispatch.CPU_RTOL,
        max_relative_l2=dispatch.CPU_REL_L2,
    )
    assert not result["passed"]
    assert result["max_relative_l2"] == pytest.approx(0.481742, abs=0.001)
    typed = make_test_data(
        8, query_dtype=torch.float16, seed=dispatch.SEED, kv_dtype=torch.float16
    )
    torch.testing.assert_close(
        dispatch.kernel_layout_golden(typed), typed.attention_golden, atol=0, rtol=0
    )
