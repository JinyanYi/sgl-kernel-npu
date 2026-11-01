#!/usr/bin/env python3
"""Run isolated A5 CopySFA cases, retaining every exit code and raw sample."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

from fixture import Config, matrix
from report import case_line, load_rows, write_report


def cases(batch=16, heads=8):
    configs = matrix(batch=batch, heads=heads)
    # Additional functional coverage is independent of the hit-rate matrix.
    for graph in (False, True):
        for hit in (95, 91):
            configs.append(
                Config(
                    capacity=16384,
                    query_width=4,
                    hit_percent=hit,
                    batch=batch,
                    heads=heads,
                    graph=graph,
                    query_overlap_percent=90,
                )
            )
        for misses in (0, 512, 513, 8192):
            configs.append(
                Config(
                    capacity=8192,
                    query_width=4,
                    batch=3,
                    heads=8,
                    prefix=65536,
                    graph=graph,
                    shared_dram=True,
                    padding=True,
                    forced_misses=misses,
                )
            )
        configs.append(
            Config(
                capacity=8192,
                query_width=4,
                batch=3,
                heads=8,
                prefix=9088,
                graph=graph,
                shared_dram=True,
                padding=True,
            )
        )
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--original-library", type=Path)
    parser.add_argument("--original-opp-root", type=Path)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    selected = cases(args.batch, args.heads)
    if args.smoke:
        selected = [
            Config(
                capacity=8192,
                query_width=4,
                batch=3,
                heads=8,
                prefix=9088,
                graph=True,
                shared_dram=True,
                padding=True,
                forced_misses=513,
            )
        ]
    if args.dry_run:
        print(
            json.dumps(
                [dataclasses.asdict(c) | dict(case=c.name) for c in selected], indent=2
            )
        )
        return 0
    if not args.output_root:
        parser.error("--output-root is required")
    args.output_root.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    report = dict(
        status="running",
        mode="smoke" if args.smoke else "formal",
        expected_cases=len(selected),
        cases=[],
    )
    for config in selected:
        output = args.output_root / config.name
        command = [sys.executable, str(root / "bench.py"), "--output", str(output)]
        for key, value in dataclasses.asdict(config).items():
            flag = "--" + key.replace("_", "-")
            if isinstance(value, bool):
                if value:
                    command.append(flag)
            elif value is not None:
                command.extend((flag, str(value)))
        command.extend(
            (
                "--warmup",
                str(2 if args.smoke else args.warmup),
                "--rounds",
                str(1 if args.smoke else args.rounds),
                "--iterations",
                str(2 if args.smoke else args.iterations),
            )
        )
        if args.original_library:
            command.extend(("--original-library", str(args.original_library)))
        if args.original_opp_root:
            command.extend(("--original-opp-root", str(args.original_opp_root)))
        (args.output_root / (config.name + ".command.json")).write_text(
            json.dumps(command, indent=2) + "\n"
        )
        with (args.output_root / (config.name + ".log")).open("w") as log:
            try:
                process = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=3600,
                    env=dict(os.environ, PYTHONFAULTHANDLER="1", PYTHONUNBUFFERED="1"),
                )
                exit_code = process.returncode
            except subprocess.TimeoutExpired:
                exit_code = 124
                log.write("\nCase exceeded the 3600-second timeout.\n")
        (args.output_root / (config.name + ".exitcode")).write_text(
            str(exit_code) + "\n"
        )
        path = output / "case.json"
        result = (
            json.loads(path.read_text())
            if path.exists()
            else dict(status="missing_result")
        )
        report["cases"].append(
            dict(case=config.name, exit_code=exit_code, result=result)
        )
        (args.output_root / "summary.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        _, rows = load_rows([args.output_root / "summary.json"])
        print(case_line(rows[-1]), flush=True)
        if exit_code or result["status"] not in (
            "pass",
            "performance_warning",
        ):
            report["status"] = "failed"
            break
    else:
        report["status"] = (
            "pass_with_performance_warnings"
            if any(
                c["result"]["status"] == "performance_warning" for c in report["cases"]
            )
            else "pass"
        )
    (args.output_root / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    write_report([args.output_root / "summary.json"], args.output_root)
    print(
        json.dumps(
            dict(
                status=report["status"],
                completed=len(report["cases"]),
                expected=len(selected),
            ),
            indent=2,
        )
    )
    return 1 if report["status"] == "failed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
