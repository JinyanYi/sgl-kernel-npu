"""CPU-only report checks using synthetic summary fixtures, without NPU access."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1] / "benchmark/copysfa_a5_ab"
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location(
    "copysfa_result_report", ROOT / "report.py"
)
reporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reporter)


def timing_row(case="example", original=100, ours=110):
    return dict(
        case=case,
        exit_code=0,
        result=dict(
            status="pass",
            timing=dict(
                original=dict(median_us=original, p95_us=120),
                port=dict(median_us=ours, p95_us=126),
            ),
        ),
    )


def save_summary(tmp_path, group, rows, status="pass", expected=None):
    path = tmp_path / group / "summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            dict(
                status=status,
                cases=rows,
                expected_cases=len(rows) if expected is None else expected,
            )
        ),
        encoding="utf-8",
    )
    return path


def test_percent_direction_p95_and_worst_case_use_actual_timings(tmp_path):
    path = save_summary(
        tmp_path, "matrix", [timing_row("slower"), timing_row("faster", ours=90)]
    )
    report, text = reporter.build_report([path])
    assert report["cases"][0]["median_delta_pct"] == pytest.approx(10)
    assert report["cases"][1]["median_delta_pct"] == pytest.approx(-10)
    assert report["cases"][0]["p95_delta_pct"] == pytest.approx(5)
    assert report["aggregates"]["matrix"]["worst_case"] == "slower"
    assert "delta=+10.00%" in text and "delta=-10.00%" in text
    assert "positive delta means ours is slower" in text


def test_failed_case_timings_are_excluded_and_missing_cases_are_visible(tmp_path):
    failed = timing_row("bad")
    failed.update(exit_code=1)
    failed["result"]["status"] = "accuracy_failed"
    path = save_summary(tmp_path, "matrix", [failed], status="failed", expected=38)
    report, text = reporter.build_report([path])
    assert "median_ratio" not in report["cases"][0]
    assert report["aggregates"]["matrix"]["measured_cases"] == 0
    assert report["aggregates"]["matrix"]["unexpected_failures"] == 1
    assert "completed=1/38" in text


def test_legacy_failure_is_recognized_without_rewriting_original_exit_code(tmp_path):
    check = dict(
        passed=False,
        finite=True,
        payload_bitwise=True,
        max_relative_l2=0.48,
        kernel_layout_diagnostic_only=dict(passed=True),
    )
    row = dict(
        case="float16_h8_eager",
        exit_code=1,
        result=dict(
            status="accuracy_failed",
            eager=dict(
                original=check,
                port=check,
                parity=dict(passed=True),
                dram_unchanged=True,
            ),
        ),
    )
    path = save_summary(tmp_path, "dispatch", [row], status="failed")
    before = path.read_bytes()
    text = reporter.write_report([path], tmp_path)
    assert path.read_bytes() == before
    assert "verified legacy-format negative control; recorded exit=1" in text
    result = json.loads((tmp_path / "performance_report.json").read_text())
    assert result["sources"][0]["status"] == "failed"
    assert result["aggregates"]["dispatch"]["expected_format_mismatches"] == 1
    assert (tmp_path / "performance_report.csv").is_file()
    row["exit_code"] = 139
    crashed_path = save_summary(tmp_path, "dispatch", [row], status="failed")
    crashed, _ = reporter.build_report([crashed_path])
    assert crashed["aggregates"]["dispatch"]["expected_format_mismatches"] == 0
    assert crashed["aggregates"]["dispatch"]["unexpected_failures"] == 1


def test_dispatch_nested_timings_and_cli_root_detection(tmp_path, monkeypatch, capsys):
    row = timing_row("float16_h8_eager_typedkv", ours=125)
    row["result"]["performance"] = dict(
        valid=True, statistics=row["result"].pop("timing")
    )
    save_summary(tmp_path, "dispatch", [row])
    monkeypatch.setattr(sys, "argv", ["report.py", str(tmp_path)])
    assert reporter.main() == 0
    data = json.loads((tmp_path / "performance_report.json").read_text())
    assert data["aggregates"]["dispatch"]["performance_warnings"] == 1
    assert "delta=+25.00%" in capsys.readouterr().out


def test_nonfinite_timing_is_rejected(tmp_path):
    path = save_summary(tmp_path, "matrix", [timing_row(ours=float("nan"))])
    with pytest.raises(ValueError, match="Invalid timing"):
        reporter.build_report([path])
