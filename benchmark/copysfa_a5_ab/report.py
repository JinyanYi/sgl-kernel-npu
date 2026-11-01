#!/usr/bin/env python3
"""Print and export per-case CopySFA comparisons from existing summary.json files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from result_contract import format_mismatch_reproduced


def load_rows(paths):
    sources, rows = [], []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        group = path.parent.name
        sources.append(
            dict(
                path=str(path.resolve()),
                status=data["status"],
                completed=len(data["cases"]),
                expected=data["expected_cases"],
            )
        )
        for item in data["cases"]:
            result = item["result"]
            negative = item["exit_code"] in (0, 1) and format_mismatch_reproduced(
                item["case"], result
            )
            row = dict(
                group=group,
                case=item["case"],
                status=result["status"],
                exit_code=item["exit_code"],
                expected_format_mismatch=negative,
            )
            perf = result.get("performance", {})
            timing = perf.get("statistics", result.get("timing", {}))
            valid = (
                item["exit_code"] == 0
                and result["status"] in ("pass", "performance_warning")
                and perf.get("valid", True)
            )
            if valid and "original" in timing and "port" in timing:
                original, port = timing["original"], timing["port"]
                values = [
                    original["median_us"],
                    port["median_us"],
                    original["p95_us"],
                    port["p95_us"],
                ]
                if not all(
                    isinstance(value, (int, float))
                    and math.isfinite(value)
                    and value > 0
                    for value in values
                ):
                    raise ValueError(f"Invalid timing values for {item['case']}")
                row.update(
                    vllm_median_us=original["median_us"],
                    ours_median_us=port["median_us"],
                    median_ratio=port["median_us"] / original["median_us"],
                    median_delta_pct=(port["median_us"] / original["median_us"] - 1)
                    * 100,
                    vllm_p95_us=original["p95_us"],
                    ours_p95_us=port["p95_us"],
                    p95_delta_pct=(port["p95_us"] / original["p95_us"] - 1) * 100,
                    warning_threshold_pct=result.get(
                        "regression_threshold_pct",
                        perf.get("regression_warning_pct", 20),
                    ),
                )
                row["performance_warning"] = (
                    row["median_delta_pct"] > row["warning_threshold_pct"]
                )
            rows.append(row)
    return sources, rows


def case_line(row):
    label = f"[{row['group']}] {row['case']}"
    if "median_delta_pct" not in row:
        kind = (
            "verified legacy-format negative control"
            if row["expected_format_mismatch"]
            else row["status"]
        )
        return f"{label}\n  {kind}; recorded exit={row['exit_code']}; no valid timing"
    return (
        f"{label} [{row['status']}]\n"
        f"  median us: vLLM={row['vllm_median_us']:.3f} ours={row['ours_median_us']:.3f} "
        f"delta={row['median_delta_pct']:+.2f}% ratio={row['median_ratio']:.4f}; "
        f"p95 us: vLLM={row['vllm_p95_us']:.3f} ours={row['ours_p95_us']:.3f} "
        f"delta={row['p95_delta_pct']:+.2f}%"
    )


def build_report(paths):
    sources, rows = load_rows(paths)
    aggregates = {}
    for group in dict.fromkeys(row["group"] for row in rows):
        selected = [row for row in rows if row["group"] == group]
        timed = [row for row in selected if "median_delta_pct" in row]
        aggregate = dict(
            cases=len(selected),
            measured_cases=len(timed),
            expected_format_mismatches=sum(
                row["expected_format_mismatch"] for row in selected
            ),
            unexpected_failures=sum(
                not row["expected_format_mismatch"]
                and (
                    row["exit_code"] != 0
                    or row["status"] not in ("pass", "performance_warning")
                )
                for row in selected
            ),
        )
        if timed:
            worst = max(timed, key=lambda row: row["median_delta_pct"])
            best = min(timed, key=lambda row: row["median_delta_pct"])
            aggregate.update(
                worst_median_delta_pct=worst["median_delta_pct"],
                worst_case=worst["case"],
                best_median_delta_pct=best["median_delta_pct"],
                best_case=best["case"],
                performance_warnings=sum(row["performance_warning"] for row in timed),
            )
        aggregates[group] = aggregate
    lines = [
        "CopySFA comparison: positive delta means ours is slower; negative means faster.",
        "Delta (%) = (ours / vLLM - 1) * 100. Times are microseconds.",
        "Median and p95 describe standalone operator/replay latency, not model throughput.",
        "Original summary statuses and exit codes are preserved, including older negative-control failures.",
    ]
    for source in sources:
        lines.append(
            f"Source: {source['path']} status={source['status']} completed={source['completed']}/{source['expected']}"
        )
    lines.extend(case_line(row) for row in rows)
    lines.append("Aggregates:")
    for group, aggregate in aggregates.items():
        lines.append(
            f"  {group}: {aggregate['measured_cases']} measured, {aggregate['expected_format_mismatches']} verified negative controls, {aggregate['unexpected_failures']} unexpected failures"
        )
        if aggregate["measured_cases"]:
            lines.append(
                f"    median delta range: {aggregate['best_median_delta_pct']:+.2f}% to {aggregate['worst_median_delta_pct']:+.2f}%; warnings={aggregate['performance_warnings']}; worst={aggregate['worst_case']}"
            )
    return (
        dict(sources=sources, aggregates=aggregates, cases=rows),
        "\n".join(lines) + "\n",
    )


def write_report(paths, output_dir):
    report, rendered = build_report(paths)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "performance_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (output_dir / "performance_report.txt").write_text(rendered, encoding="utf-8")
    fields = [
        "group",
        "case",
        "status",
        "exit_code",
        "expected_format_mismatch",
        "vllm_median_us",
        "ours_median_us",
        "median_ratio",
        "median_delta_pct",
        "vllm_p95_us",
        "ours_p95_us",
        "p95_delta_pct",
        "warning_threshold_pct",
        "performance_warning",
    ]
    with (output_dir / "performance_report.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report["cases"])
    return rendered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help="diagnostics root, matrix/dispatch directory, or summary.json",
    )
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.input.is_file():
        paths = [args.input]
        output = args.input.parent
    else:
        output = args.input
        paths = [
            path
            for path in (
                output / "dispatch/summary.json",
                output / "matrix/summary.json",
            )
            if path.is_file()
        ]
        if not paths and (output / "summary.json").is_file():
            paths = [output / "summary.json"]
    if not paths:
        parser.error("No summary.json found; supply a completed results path")
    rendered = write_report(paths, args.output_dir or output)
    print(rendered, end="")
    print(f"Reports: {args.output_dir or output}/performance_report.{{txt,csv,json}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
