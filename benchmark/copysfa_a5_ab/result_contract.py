"""Acceptance rules for valid inputs and the retained legacy-format diagnostic."""


def is_legacy_negative(case):
    return case in {
        f"float16_h{heads}_{mode}" for heads in (8, 16) for mode in ("eager", "graph")
    }


def format_mismatch_reproduced(case, result):
    """Require a specific finite format mismatch, not just any test failure."""
    if not is_legacy_negative(case):
        return False
    if result.get("status") not in ("accuracy_failed", "expected_format_mismatch"):
        return False
    stages = ("eager", "graph") if case.endswith("_graph") else ("eager",)
    for stage in stages:
        checks = result.get(stage, {})
        if not checks.get("dram_unchanged") or not checks.get("parity", {}).get(
            "passed"
        ):
            return False
        for name in ("original", "port"):
            check = checks.get(name, {})
            if not (
                check.get("finite")
                and check.get("payload_bitwise")
                and check.get("passed") is False
                and check.get("max_relative_l2", 0) > 0.08
                and check.get("kernel_layout_diagnostic_only", {}).get("passed")
            ):
                return False
    return True


def accepted(row):
    if row.get("exit_code") != 0:
        return False
    result = row["result"]
    if is_legacy_negative(row["case"]):
        return result.get(
            "status"
        ) == "expected_format_mismatch" and format_mismatch_reproduced(
            row["case"], result
        )
    return result.get("status") in ("pass", "performance_warning")
