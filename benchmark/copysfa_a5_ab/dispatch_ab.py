#!/usr/bin/env python3
"""Diagnose FP16/BF16 dispatch against original PR18490 on identical inputs."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

from result_contract import accepted, format_mismatch_reproduced, is_legacy_negative

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))

REVISION = "1b180deb840484c2fe33cbdd7abb68a615085172"
SEED = 20261032
# Keep the failing pytest's CPU gates unchanged, including relative L2.
CPU_ATOL, CPU_RTOL, CPU_REL_L2 = 0.008, 0.05, 0.08
PARITY_ATOL, PARITY_RTOL = 0.002, 0.005


def cases():
    return [
        f"{dtype}_h{heads}_{mode}"
        for mode in ("eager", "graph")
        for dtype in ("float16", "bfloat16")
        for heads in (8, 16)
    ] + [
        f"float16_h{heads}_{mode}_typedkv"
        for mode in ("eager", "graph")
        for heads in (8, 16)
    ]


def write_json(path, value):
    path.write_text(
        json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def tensor_digest(tensor):
    import torch

    value = tensor.detach().cpu().contiguous()
    digestor = hashlib.sha256(f"{value.dtype}:{tuple(value.shape)}:".encode())
    digestor.update(value.view(torch.uint8).numpy().tobytes())
    return digestor.hexdigest()


def compare(actual, golden, *, atol, rtol, max_relative_l2=None):
    """Return every accuracy metric without hiding a failed backend's output."""
    import torch

    actual, golden = actual.detach().cpu().float(), golden.detach().cpu().float()
    if actual.shape != golden.shape:
        return dict(
            passed=False,
            reason="shape",
            actual=list(actual.shape),
            golden=list(golden.shape),
        )
    finite = bool(torch.isfinite(actual).all() & torch.isfinite(golden).all())
    if not finite:
        return dict(passed=False, reason="non_finite", finite=False)
    delta = actual - golden
    relative_l2 = delta.flatten(1).norm(dim=1) / golden.flatten(1).norm(
        dim=1
    ).clamp_min(1e-7)
    allclose = torch.allclose(actual, golden, atol=atol, rtol=rtol)
    relative = float(relative_l2.max())
    return dict(
        passed=allclose and (max_relative_l2 is None or relative <= max_relative_l2),
        finite=True,
        allclose=allclose,
        max_abs=float(delta.abs().max()),
        rmse=float(delta.square().mean().sqrt()),
        max_relative_l2=relative,
        bitwise=torch.equal(actual, golden),
        atol=atol,
        rtol=rtol,
        relative_l2_limit=max_relative_l2,
    )


def kernel_layout_golden(data):
    """Diagnostic only: decode the existing packed bytes as the kernel does."""
    import torch
    from fused_quant_scatter_copy_sparse_flash_attention_golden import (
        physical_rows,
        unpack_rows,
    )

    sources = data.topk_src_ids[0, 0]
    rows = data.dram.view(-1, 656)[physical_rows(data.dram_block_table, sources)]
    key, value = unpack_rows(rows, data.query.dtype)
    return (
        torch.softmax(data.query[0].float() @ key.T * data.scale_value, dim=-1) @ value
    ).unsqueeze(0)


def worker(args):
    output = args.output_root
    output.mkdir(parents=True, exist_ok=False)
    report = dict(
        case=args.case,
        status="running",
        command=sys.argv,
        reference_revision=REVISION,
        test_kind=(
            "legacy_format_negative"
            if is_legacy_negative(args.case)
            else "numerical_acceptance"
        ),
    )
    write_json(output / "case.json", report)
    try:
        import acl
        import sgl_kernel_npu
        import torch
        import torch_npu
        from bench import stats
        from fused_quant_scatter_copy_sparse_flash_attention_golden import (
            make_test_data,
        )
        from sgl_kernel_npu.sparsity_driven_kv_offload import (
            create_shm_tensor,
            free_shm,
        )

        torch.set_num_threads(8)
        torch.npu.set_device(0)
        if acl.rt.get_device_info(0, 601) != (3510, 0):
            raise RuntimeError("Original and port C8 attention require Ascend950 (A5)")
        library = Path(os.environ["ORIGINAL_COPYSFA_BINDING"]).resolve(strict=True)
        opp_root = Path(os.environ["ORIGINAL_OPP_ROOT"]).resolve(strict=True)
        manifest_path = Path(os.environ["ORIGINAL_BINDING_MANIFEST"]).resolve(
            strict=True
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        provenance = manifest.get("source_provenance") or {}
        if provenance.get("upstream", {}).get("revision") != REVISION:
            raise RuntimeError("Reference binding is not the pinned PR18490 build")
        if manifest["library_sha256"] != digest(library):
            raise RuntimeError("Reference binding hash differs from its build manifest")
        torch.ops.load_library(str(library))
        original = torch.ops.a5_original_copysfa.run.default
        port = torch.ops.npu.fused_quant_scatter_copy_sparse_flash_attention.default
        resolved = dict(torch.ops.a5_original_copysfa.resolve_libraries())
        for path in resolved.values():
            if not Path(path).resolve().is_relative_to(opp_root):
                raise RuntimeError(
                    f"Reference symbol resolved outside private OPP: {path}"
                )
        port_so = Path(sgl_kernel_npu.__file__).parent / "lib/libsgl_kernel_npu.so"
        dtype, head_text, mode, *layout = args.case.split("_")
        kv_dtype = getattr(torch, dtype) if layout else torch.bfloat16
        data = make_test_data(
            8,
            query_dtype=getattr(torch, dtype),
            query_heads=int(head_text[1:]),
            seed=SEED,
            kv_dtype=kv_dtype,
        )
        fixture = dataclasses.asdict(data)
        interpreted_golden = kernel_layout_golden(data)
        fixture["kernel_layout_golden_diagnostic_only"] = interpreted_golden
        torch.save(fixture, output / "fixture.pt")
        report.update(
            fixture_sha256={
                key: tensor_digest(value)
                for key, value in fixture.items()
                if isinstance(value, torch.Tensor)
            },
            fixture_file_sha256=digest(output / "fixture.pt"),
            fixture_recipe_sha256=digest(
                ROOT / "tests/fused_quant_scatter_copy_sparse_flash_attention_golden.py"
            ),
            config=dict(
                dtype=dtype,
                heads=data.query.shape[1],
                mode=mode,
                batch=1,
                queries=1,
                misses=8,
                seed=SEED,
                kv_dtype=str(kv_dtype),
                layout="query_dtype_kv" if layout else "legacy_bf16_kv",
            ),
            provenance=dict(
                python=sys.version,
                torch=torch.__version__,
                torch_npu=torch_npu.__version__,
                kernel_module=sgl_kernel_npu.__file__,
                port_so=str(port_so),
                port_so_sha256=digest(port_so),
                reference_binding=str(library),
                reference_binding_sha256=digest(library),
                binding_manifest=str(manifest_path),
                binding_manifest_sha256=digest(manifest_path),
                original_opp_root=str(opp_root),
                aclnn_libraries={
                    symbol: dict(path=path, sha256=digest(path))
                    for symbol, path in resolved.items()
                },
                port_schema=str(port._schema),
                original_schema=str(original._schema),
                visible_devices=os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
            ),
        )
        write_json(output / "case.json", report)
        owner, _, pointer = create_shm_tensor(data.dram.shape, torch.int8, device_id=0)
        try:
            owner.copy_(data.dram)
            query = data.query.to("npu")
            dram_view = torch.ops.a5_original_copysfa.registered_dram_view(
                owner, query, pointer
            )
            assert (
                torch.ops.a5_original_copysfa.validate_registered_dram_view(dram_view)
                == pointer
            )
            keys = (
                "hbm_block_table",
                "dram_block_table",
                "topk_dst_slots",
                "topk_src_ids",
                "topk_miss_counts",
                "actual_seq_lengths_query",
                "actual_seq_lengths_kv",
                "num_cache_tokens",
                "miss_src_ids",
                "miss_dst_slots",
                "miss_counts",
            )
            tail = [getattr(data, key).to("npu") for key in keys]
            initial = data.initial_hbm.to("npu")
            buffers = {
                name: (
                    initial.clone(),
                    torch.empty(
                        (*query.shape[:-1], 512), dtype=query.dtype, device=query.device
                    ),
                )
                for name in ("original", "port")
            }

            def reset(name):
                hbm, out = buffers[name]
                hbm.copy_(initial)
                out.fill_(float("nan"))

            def call(name):
                hbm, out = buffers[name]
                if name == "original":
                    original(query, hbm, dram_view, *tail, data.scale_value, out)
                else:
                    port(query, hbm, owner, pointer, *tail, data.scale_value, out)

            def snapshot():
                torch.npu.synchronize()
                values = {
                    name: dict(hbm=hbm.cpu(), attention=out.cpu())
                    for name, (hbm, out) in buffers.items()
                }
                checks = {}
                for name, value in values.items():
                    check = compare(
                        value["attention"],
                        data.attention_golden,
                        atol=CPU_ATOL,
                        rtol=CPU_RTOL,
                        max_relative_l2=CPU_REL_L2,
                    )
                    check["payload_bitwise"] = torch.equal(
                        value["hbm"], data.expected_hbm
                    )
                    check["passed"] &= check["payload_bitwise"]
                    check["kernel_layout_diagnostic_only"] = compare(
                        value["attention"],
                        interpreted_golden,
                        atol=CPU_ATOL,
                        rtol=CPU_RTOL,
                        max_relative_l2=CPU_REL_L2,
                    )
                    checks[name] = check
                checks["parity"] = compare(
                    values["port"]["attention"],
                    values["original"]["attention"],
                    atol=PARITY_ATOL,
                    rtol=PARITY_RTOL,
                    max_relative_l2=CPU_REL_L2,
                )
                checks["dram_unchanged"] = torch.equal(owner, data.dram)
                checks["passed"] = checks["dram_unchanged"] and all(
                    checks[name]["passed"] for name in ("original", "port", "parity")
                )
                return checks, values

            for name in buffers:
                reset(name)
                call(name)
            report["eager"], values = snapshot()
            torch.save(values, output / "outputs_eager.pt")
            graphs = {}
            if mode == "graph":
                for name in buffers:
                    for _ in range(2):
                        reset(name)
                        call(name)
                    torch.npu.synchronize()
                    reset(name)
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        call(name)
                    graphs[name] = graph
                    reset(name)
                    graph.replay()
                report["graph"], values = snapshot()
                torch.save(values, output / "outputs_graph.pt")
            report["accuracy_passed"] = report["eager"]["passed"] and (
                mode == "eager" or report["graph"]["passed"]
            )
            write_json(output / "case.json", report)
            if is_legacy_negative(args.case):
                report["status"] = "accuracy_failed"
                report["expectation_passed"] = format_mismatch_reproduced(
                    args.case, report
                )
                report["status"] = (
                    "expected_format_mismatch"
                    if report["expectation_passed"]
                    else "unexpected_diagnostic_result"
                )
                report["performance"] = dict(
                    skipped="Legacy-format negative control; excluded from performance acceptance"
                )
            elif report["accuracy_passed"]:

                def enqueue(name):
                    graphs[name].replay() if graphs else call(name)

                report["performance"] = dict(
                    valid=False,
                    warmup=args.warmup,
                    rounds=args.rounds,
                    iterations=args.iterations,
                    boundary="NPU event around one operator/replay; resets, CPU transfers, validation and metadata preparation excluded",
                    regression_warning_pct=20,
                    raw_samples_us={name: [] for name in buffers},
                    round_statistics=[],
                )
                for name in buffers:
                    for _ in range(args.warmup):
                        reset(name)
                        enqueue(name)
                torch.npu.synchronize()
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
                            start, end = torch.npu.Event(
                                enable_timing=True
                            ), torch.npu.Event(enable_timing=True)
                            start.record()
                            enqueue(name)
                            end.record()
                            end.synchronize()
                            current[name].append(float(start.elapsed_time(end)) * 1000)
                    check, values = snapshot()
                    perf = report["performance"]
                    perf["round_statistics"].append(
                        dict(
                            accuracy=check,
                            **{
                                name: stats(samples)
                                for name, samples in current.items()
                            },
                        )
                    )
                    for name in buffers:
                        perf["raw_samples_us"][name].extend(current[name])
                    write_json(output / "case.json", report)
                    if not check["passed"]:
                        report["accuracy_passed"] = False
                        torch.save(
                            values, output / f"outputs_failed_round_{round_index}.pt"
                        )
                        raise AssertionError(
                            "Accuracy failed after timing; latency is invalid"
                        )
                perf["statistics"] = {
                    name: stats(samples)
                    for name, samples in perf["raw_samples_us"].items()
                }
                perf["port_over_original"] = (
                    perf["statistics"]["port"]["median_us"]
                    / perf["statistics"]["original"]["median_us"]
                )
                perf["regression_pct"] = (perf["port_over_original"] - 1) * 100
                perf["warning"] = perf["regression_pct"] > 20
                perf["valid"] = True
                report["status"] = "performance_warning" if perf["warning"] else "pass"
            else:
                report["status"] = "accuracy_failed"
                report["performance"] = dict(
                    skipped="Accuracy gate failed; do not interpret latency"
                )
        finally:
            torch.npu.synchronize()
            free_shm()
    except Exception:
        report["status"] = "error"
        report["traceback"] = traceback.format_exc()
    write_json(output / "case.json", report)
    print(json.dumps(report, indent=2, allow_nan=False), flush=True)
    return 0 if accepted(dict(case=args.case, result=report, exit_code=0)) else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--case", choices=cases(), help=argparse.SUPPRESS)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    cases=cases(),
                    cpu_gates=dict(
                        atol=CPU_ATOL, rtol=CPU_RTOL, max_relative_l2=CPU_REL_L2
                    ),
                    parity_gates=dict(
                        atol=PARITY_ATOL, rtol=PARITY_RTOL, max_relative_l2=CPU_REL_L2
                    ),
                ),
                indent=2,
            )
        )
        return 0
    if not args.output_root or min(args.warmup, args.rounds, args.iterations) < 1:
        parser.error("A fresh --output-root and positive timing counts are required")
    if args.case:
        return worker(args)
    args.output_root.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", expected_cases=len(cases()), cases=[])
    for case in cases():
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--case",
            case,
            "--output-root",
            str(args.output_root / case),
            "--warmup",
            str(args.warmup),
            "--rounds",
            str(args.rounds),
            "--iterations",
            str(args.iterations),
        ]
        write_json(args.output_root / f"{case}.command.json", command)
        with (args.output_root / f"{case}.log").open("w", encoding="utf-8") as log:
            try:
                process = subprocess.run(
                    command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    timeout=1800,
                    env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONFAULTHANDLER="1"),
                )
                code = process.returncode
            except subprocess.TimeoutExpired:
                code = 124
        (args.output_root / f"{case}.exitcode").write_text(
            f"{code}\n", encoding="utf-8"
        )
        path = args.output_root / case / "case.json"
        result = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else dict(status="missing_result")
        )
        report["cases"].append(dict(case=case, exit_code=code, result=result))
        write_json(args.output_root / "summary.json", report)
        print(f"{case}: exit={code} status={result['status']}", flush=True)
        check = result.get("graph", result.get("eager", {}))
        if "original" in check and "port" in check:
            print(
                f"  CPU rel-L2: original={check['original'].get('max_relative_l2')}, "
                f"port={check['port'].get('max_relative_l2')}; "
                f"direct parity={check['parity']['passed']}",
                flush=True,
            )
        perf = result.get("performance", {})
        if perf.get("valid"):
            print(
                f"  median us: original={perf['statistics']['original']['median_us']:.3f}, "
                f"port={perf['statistics']['port']['median_us']:.3f}; "
                f"port/original={perf['port_over_original']:.4f}",
                flush=True,
            )
    report["status"] = (
        "pass" if all(accepted(row) for row in report["cases"]) else "failed"
    )
    report["counts"] = dict(
        numerical_acceptance_passed=sum(
            not is_legacy_negative(row["case"]) and accepted(row)
            for row in report["cases"]
        ),
        negative_controls_passed=sum(
            is_legacy_negative(row["case"]) and accepted(row) for row in report["cases"]
        ),
        unexpected_failures=sum(not accepted(row) for row in report["cases"]),
    )
    write_json(args.output_root / "summary.json", report)
    print(f"{report['status']}: {args.output_root / 'summary.json'}", flush=True)
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
