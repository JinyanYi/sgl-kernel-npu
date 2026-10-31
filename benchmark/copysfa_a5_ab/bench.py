#!/usr/bin/env python3
"""Compare the unchanged PR18490 ACLNN operator with the SGLang A5 port."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import statistics
import sys
import traceback
from pathlib import Path

from fixture import Config, fingerprint, make_fixture, matrix, remap_physical_pages


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def stats(samples):
    ordered = sorted(samples)
    return dict(
        median_us=statistics.median(samples),
        mean_us=statistics.fmean(samples),
        min_us=min(samples),
        max_us=max(samples),
        p95_us=ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--case")
    parser.add_argument("--capacity", type=int, default=16384)
    parser.add_argument("--query-width", type=int, choices=(1, 4), default=4)
    parser.add_argument("--hit-percent", type=int, choices=(91, 95), default=95)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--prefix", type=int, default=65536)
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--shared-dram", action="store_true")
    parser.add_argument("--padding", action="store_true")
    parser.add_argument("--forced-misses", type=int)
    parser.add_argument("--query-overlap-percent", type=int, default=0)
    parser.add_argument(
        "--original-library",
        type=Path,
        default=os.environ.get("ORIGINAL_COPYSFA_BINDING"),
    )
    parser.add_argument(
        "--original-opp-root", type=Path, default=os.environ.get("ORIGINAL_OPP_ROOT")
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--regression-threshold-pct", type=float, default=20)
    parser.add_argument("--strict-performance", action="store_true")
    parser.add_argument("--parity-atol", type=float, default=0.002)
    parser.add_argument("--parity-rtol", type=float, default=0.005)
    args = parser.parse_args()
    if args.list:
        print(
            json.dumps(
                [
                    dataclasses.asdict(c) | dict(case=c.name)
                    for c in matrix(batch=args.batch, heads=args.heads)
                ],
                indent=2,
            )
        )
        return 0
    if args.case:
        choices = {c.name: c for c in matrix(batch=args.batch, heads=args.heads)}
        if args.case not in choices:
            parser.error("Unknown case; use --list")
        config = choices[args.case]
    else:
        config = Config(
            args.capacity,
            args.query_width,
            args.hit_percent,
            args.batch,
            args.heads,
            args.prefix,
            args.dtype,
            args.graph,
            args.shared_dram,
            args.padding,
            args.forced_misses,
            args.query_overlap_percent,
        )
    if not args.output or not args.original_library or not args.original_opp_root:
        parser.error(
            "--output, --original-library and --original-opp-root are required"
        )
    if args.rounds < 1 or args.iterations < 1 or args.warmup < 1:
        parser.error("Warmup, rounds and iterations must be positive")
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(
        case=config.name,
        config=dataclasses.asdict(config),
        status="running",
        reference_revision="1b180deb840484c2fe33cbdd7abb68a615085172",
        regression_threshold_pct=args.regression_threshold_pct,
        parity_atol=args.parity_atol,
        parity_rtol=args.parity_rtol,
        command=sys.argv,
        strict_performance=args.strict_performance,
        warmup=args.warmup,
        rounds=args.rounds,
        iterations=args.iterations,
        timing_boundary="NPU event around one operator/replay; HBM restore, output clear, validation and fixture construction excluded",
    )
    fixture = None
    (args.output / "case.json").write_text(json.dumps(report, indent=2) + "\n")
    try:
        import acl
        import sgl_kernel_npu
        import torch
        import torch_npu
        from sgl_kernel_npu.sparsity_driven_kv_offload import (
            create_shm_tensor,
            free_shm,
        )

        torch.set_num_threads(8)
        torch.npu.set_device(0)
        if acl.rt.get_device_info(0, 601) != (3510, 0):
            raise RuntimeError(
                "Actual original/port C8 accuracy and timing require Ascend950"
            )
        library = args.original_library.resolve(strict=True)
        opp_root = args.original_opp_root.resolve(strict=True)
        torch.ops.load_library(str(library))
        original = torch.ops.a5_original_copysfa.run.default
        port = torch.ops.npu.fused_quant_scatter_copy_sparse_flash_attention.default
        resolved = dict(torch.ops.a5_original_copysfa.resolve_libraries())
        for symbol, path in resolved.items():
            if not Path(path).resolve().is_relative_to(opp_root):
                raise RuntimeError(
                    f"{symbol} resolved outside selected private OPP: {path}"
                )
        port_so = Path(sgl_kernel_npu.__file__).parent / "lib/libsgl_kernel_npu.so"
        report["provenance"] = dict(
            python=sys.version,
            torch=torch.__version__,
            torch_npu=torch_npu.__version__,
            kernel_module=sgl_kernel_npu.__file__,
            kernel_distribution=importlib.metadata.version("sgl-kernel-npu"),
            port_so=str(port_so),
            port_so_sha256=digest(port_so),
            original_library=str(library),
            original_library_sha256=digest(library),
            original_opp_root=str(opp_root),
            aclnn_libraries={
                name: dict(path=path, sha256=digest(path))
                for name, path in resolved.items()
            },
            original_schema=str(original._schema),
            port_schema=str(port._schema),
        )
        fixture, metrics = make_fixture(config)
        report["hit_rates"] = metrics
        report["fixture_sha256"] = fingerprint(fixture)
        report["fixture_recipe_sha256"] = digest(Path(__file__).with_name("fixture.py"))
        (args.output / "case.json").write_text(json.dumps(report, indent=2) + "\n")
        owner, _, pointer = create_shm_tensor(
            tuple(fixture["dram"].shape), torch.int8, device_id=0
        )
        try:
            owner.copy_(fixture["dram"])
            common = {
                name: t.to("npu")
                for name, t in fixture.items()
                if name not in ("dram", "expected", "golden")
            }
            dram_view = torch.ops.a5_original_copysfa.registered_dram_view(
                owner, common["query"], pointer
            )
            assert dram_view.device.type == "npu" and dram_view.data_ptr() == pointer
            assert (
                torch.ops.a5_original_copysfa.validate_registered_dram_view(dram_view)
                == pointer
            )
            buffers = {
                name: (
                    common["initial"].clone(),
                    torch.zeros_like(common["query"][..., :512]),
                )
                for name in ("original", "port")
            }

            def call(name):
                hbm, out = buffers[name]
                tail = [
                    common[k]
                    for k in (
                        "hbm_table",
                        "dram_table",
                        "dst",
                        "src",
                        "route_misses",
                        "query_ends",
                        "lengths",
                        "cache_lengths",
                        "miss_src",
                        "miss_dst",
                        "misses",
                    )
                ]
                if name == "original":
                    original(
                        common["query"], hbm, dram_view, *tail, 1 / (576**0.5), out
                    )
                else:
                    port(
                        common["query"], hbm, owner, pointer, *tail, 1 / (576**0.5), out
                    )

            def reset(name):
                buffers[name][0].copy_(common["initial"])
                buffers[name][1].zero_()

            def validate(name):
                torch.npu.synchronize()
                hbm, out = buffers[name]
                torch.testing.assert_close(
                    hbm.cpu(), fixture["expected"], atol=0, rtol=0
                )
                value = out.cpu().float()
                assert bool(torch.isfinite(value).all())
                torch.testing.assert_close(
                    value, fixture["golden"], atol=0.005, rtol=0.02
                )
                delta = value - fixture["golden"]
                return dict(
                    payload_bitwise=True,
                    golden_max_abs=float(delta.abs().max()),
                    golden_rmse=float(delta.square().mean().sqrt()),
                )

            report["correctness"] = {}
            for name in buffers:
                reset(name)
                call(name)
                report["correctness"][name] = validate(name)
            torch.testing.assert_close(
                buffers["port"][1].cpu(),
                buffers["original"][1].cpu(),
                atol=args.parity_atol,
                rtol=args.parity_rtol,
            )
            delta = (
                buffers["port"][1].cpu().float() - buffers["original"][1].cpu().float()
            )
            report["parity"] = dict(
                passed=True,
                bitwise=torch.equal(
                    buffers["port"][1].cpu(), buffers["original"][1].cpu()
                ),
                max_abs=float(delta.abs().max()),
                rmse=float(delta.square().mean().sqrt()),
            )
            graphs = {}
            for name in buffers:
                for _ in range(args.warmup):
                    reset(name)
                    call(name)
                torch.npu.synchronize()
                if config.graph:
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        call(name)
                    graphs[name] = graph
                    reset(name)
                    graph.replay()
                    report["correctness"][name]["graph_replay"] = validate(name)
            torch.testing.assert_close(
                buffers["port"][1].cpu(),
                buffers["original"][1].cpu(),
                atol=args.parity_atol,
                rtol=args.parity_rtol,
            )
            if config.padding:
                original_fixture = fixture
                changed, _ = make_fixture(dataclasses.replace(config, padding=False))
                fixture = remap_physical_pages(changed)
                owner.copy_(fixture["dram"])
                for key, tensor in common.items():
                    tensor.copy_(fixture[key])
                mutation = dict(
                    fixture_sha256=fingerprint(fixture),
                    padding_activated=True,
                    physical_pages_remapped=True,
                )
                for name in buffers:
                    reset(name)
                    graphs[name].replay() if config.graph else call(name)
                    mutation[name] = validate(name)
                torch.testing.assert_close(
                    buffers["port"][1].cpu(),
                    buffers["original"][1].cpu(),
                    atol=args.parity_atol,
                    rtol=args.parity_rtol,
                )
                report["input_mutation"] = mutation
                fixture = original_fixture
                owner.copy_(fixture["dram"])
                for key, tensor in common.items():
                    tensor.copy_(fixture[key])
            report["status"] = "accuracy_passed_timing_pending"
            (args.output / "case.json").write_text(json.dumps(report, indent=2) + "\n")
            samples = {name: [] for name in buffers}
            round_stats = []
            for round_index in range(args.rounds):
                current = {name: [] for name in buffers}
                for iteration in range(args.iterations):
                    order = (
                        ("original", "port")
                        if (round_index + iteration) % 2 == 0
                        else ("port", "original")
                    )
                    for name in order:
                        reset(name)
                        start = torch.npu.Event(enable_timing=True)
                        end = torch.npu.Event(enable_timing=True)
                        start.record()
                        graphs[name].replay() if config.graph else call(name)
                        end.record()
                        end.synchronize()
                        current[name].append(float(start.elapsed_time(end)) * 1000)
                for name in buffers:
                    report["correctness"][name][f"round_{round_index}"] = validate(name)
                    samples[name].extend(current[name])
                torch.testing.assert_close(
                    buffers["port"][1].cpu(),
                    buffers["original"][1].cpu(),
                    atol=args.parity_atol,
                    rtol=args.parity_rtol,
                )
                round_stats.append(
                    {name: stats(values) for name, values in current.items()}
                )
            report["raw_samples_us"] = samples
            report["round_statistics"] = round_stats
            report["timing"] = {name: stats(values) for name, values in samples.items()}
            ratio = (
                report["timing"]["port"]["median_us"]
                / report["timing"]["original"]["median_us"]
            )
            report["port_over_original"] = ratio
            report["regression_pct"] = (ratio - 1) * 100
            report["performance_gate_passed"] = (
                ratio - 1
            ) * 100 <= args.regression_threshold_pct
            report["status"] = (
                "pass" if report["performance_gate_passed"] else "performance_warning"
            )
        finally:
            torch.npu.synchronize()
            free_shm()
    except Exception:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        if fixture is not None:
            import torch

            torch.save(fixture, args.output / "failure_fixture.pt")
    (args.output / "case.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ("raw_samples_us", "provenance")
            },
            indent=2,
        ),
        flush=True,
    )
    if report["status"] == "performance_warning":
        return 2 if args.strict_performance else 0
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
