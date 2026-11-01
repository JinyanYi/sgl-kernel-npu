#!/usr/bin/env python3
"""A5 FP8 MTP LIM operator comparison. One isolated process per case."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

from backends import (
    call_li,
    call_lim,
    file_sha,
    load_backends,
    loaded_libraries,
    opp_libraries,
)
from fixture import (
    build_residency,
    make_matrix,
    target_union_misses,
    validate_causal_boundary,
)

B, K, C, PREFIX, ACTUAL, SOURCE = 16, 2048, 12288, 65536, 65664, 66944
INVALID = -(1 << 31)
utc_timezone = timezone.utc  # noqa: UP017 - preserve Python 3.10 support


def source_provenance(repo):
    """Record this source tree without attributing a tarball to an enclosing repo."""

    def git(*argv):
        try:
            result = subprocess.run(
                ["git", "-C", str(repo), *argv], capture_output=True, check=False
            )
        except OSError:
            return None
        return result.stdout if result.returncode == 0 else None

    top = git("rev-parse", "--show-toplevel")
    matches = bool(top and Path(top.decode().strip()).resolve() == repo.resolve())
    revision = git("rev-parse", "HEAD") if matches else None
    status = git("status", "--porcelain") if matches else None
    diff = git("diff", "--binary", "HEAD") if matches else None
    return {
        "source_commit": revision.decode().strip() if revision else None,
        "source_dirty_diff_sha256": (
            hashlib.sha256(diff).hexdigest() if diff is not None else None
        ),
        "source_status_porcelain": (
            status.decode(errors="replace") if status is not None else None
        ),
        "source_dirty_diff_scope": (
            "git diff --binary HEAD; untracked files are listed in status but not in diff hash"
            if matches
            else "no matching Git root; script and library hashes are recorded separately"
        ),
    }


def case_id(case):
    return f"q{case['q']}_{case['mode']}_{case['policy']}_hit{case['hit_percent']}"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case-id", choices=[case_id(c) for c in make_matrix()])
    p.add_argument("--output", type=Path)
    p.add_argument("--device", default="npu:0")
    p.add_argument("--kernel-python-root")
    p.add_argument("--original-library")
    p.add_argument("--original-module")
    p.add_argument("--original-op", default="a5_original_lim.run")
    p.add_argument("--original-source-revision", default="unknown")
    p.add_argument(
        "--original-opp-root",
        type=Path,
        help="donor OPP installation to inventory; does not itself select kernel",
    )
    p.add_argument("--inputs-file", type=Path)
    p.add_argument("--save-inputs", type=Path)
    p.add_argument(
        "--original-variant",
        default="unknown",
        help="original compile-time SIMT configuration; preserve unknown if unverified",
    )
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--seed", type=int, default=20261029)
    p.add_argument("--source-capacity", type=int, default=SOURCE)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--smoke",
        action="store_true",
        help="diagnostic only: 1 warmup, 1 round, 1 sample",
    )
    return p


def tensor_hash(torch, tensor):
    return hashlib.sha256(
        tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()
    ).hexdigest()


def extract_li(torch, result, routes):
    if isinstance(result, torch.Tensor):
        candidates = [result]
    else:
        candidates = list(result)
    found = [
        x
        for x in candidates
        if isinstance(x, torch.Tensor)
        and x.dtype == torch.int32
        and x.numel() == routes * K
    ]
    if len(found) != 1:
        raise RuntimeError("official LI must return exactly one int32 TopK tensor")
    return found[0].reshape(B, routes // B, K)


INPUT_NAMES = ("query", "key", "weights", "query_scale", "key_scale")


def input_data(torch, torch_npu, args, q):
    device = torch.device(args.device)
    blocks = args.source_capacity // 128
    expected = {
        "query": ((B * q, 32, 128), torch.float8_e4m3fn),
        "query_scale": ((B * q, 32), torch.float32),
        "key": ((B * blocks, 128, 1, 128), torch.float8_e4m3fn),
        "key_scale": ((B * blocks, 128, 1), torch.float32),
        "weights": ((B * q, 32), torch.bfloat16),
    }
    if args.inputs_file:
        payload = torch.load(args.inputs_file, map_location="cpu", weights_only=True)
        metadata = payload["metadata"]
        expected_metadata = {
            "schema_version": 1,
            "q": q,
            "source_capacity": args.source_capacity,
            "batch": B,
        }
        if any(
            metadata.get(key) != value for key, value in expected_metadata.items()
        ) or not isinstance(metadata.get("seed"), int):
            raise ValueError(
                f"inputs-file metadata does not match this case: {metadata}"
            )
        data = payload["tensors"]
        if set(data) != set(INPUT_NAMES):
            raise ValueError(
                "inputs-file does not contain exactly the required tensors"
            )
        for name, (shape, dtype) in expected.items():
            tensor = data[name]
            if tuple(tensor.shape) != shape or tensor.dtype != dtype:
                raise ValueError(f"inputs-file shape or dtype mismatch: {name}")
            if tensor_hash(torch, tensor) != payload["sha256"][name]:
                raise ValueError(f"inputs-file hash mismatch: {name}")
        data = {name: value.to(device) for name, value in data.items()}
    else:
        metadata = {
            "schema_version": 1,
            "q": q,
            "source_capacity": args.source_capacity,
            "batch": B,
            "seed": args.seed,
        }
        torch.manual_seed(args.seed)
        qb = torch.randn(B * q, 32, 128, dtype=torch.bfloat16).to(device)
        kb = torch.randn(B * blocks, 128, 1, 128, dtype=torch.bfloat16).to(device)
        weights = torch.randn(B * q, 32, dtype=torch.bfloat16).to(device)
        query, qs = torch_npu.npu_dynamic_quant(qb, dst_type=torch.float8_e4m3fn)
        key, ks = torch_npu.npu_dynamic_quant(kb, dst_type=torch.float8_e4m3fn)
        del qb, kb
        data = {
            "query": query.contiguous(),
            "query_scale": qs.reshape(B * q, 32).float().contiguous(),
            "key": key.contiguous(),
            "key_scale": ks.reshape(B * blocks, 128, 1).float().contiguous(),
            "weights": weights,
        }
        if args.save_inputs:
            cpu = {name: data[name].cpu() for name in INPUT_NAMES}
            payload = {
                "metadata": metadata,
                "tensors": cpu,
                "sha256": {
                    name: tensor_hash(torch, value) for name, value in cpu.items()
                },
            }
            # Exclusive creation prevents overwriting a prior or concurrently prepared fixture.
            with args.save_inputs.open("xb") as f:
                torch.save(payload, f)
            del cpu, payload
    data.update(
        table=torch.arange(B * blocks, dtype=torch.int32, device=device).reshape(
            B, blocks
        ),
        query_ends=torch.arange(q, B * q + 1, q, dtype=torch.int32, device=device),
        actual=torch.full((B,), ACTUAL, dtype=torch.int32, device=device),
        prefix=torch.full((B,), PREFIX, dtype=torch.int32, device=device),
        capacity=torch.full((B,), C, dtype=torch.int32, device=device),
        state=torch.full((B,), -1, dtype=torch.int32, device=device),
        entries=torch.arange(1, B + 1, dtype=torch.int32, device=device),
    )
    return data, metadata


def make_outputs(torch, device, q):
    return [
        torch.full(shape, -7, dtype=torch.int32, device=device)
        for shape in [
            (B * q, 1, K),
            (B * q, 1, K),
            (B * q,),
            (B, 32768),
            (B, 32768),
            (B,),
        ]
    ]


def validate_result(torch, reference, fixture, before, pool, outputs, q, label):
    src, dst, route_counts, miss_src, miss_dst, counts = [
        x.cpu().tolist() for x in outputs
    ]
    after = pool.cpu().tolist()
    if after[0] != before[0]:
        raise AssertionError(f"{label}: dummy pool row changed")
    routes = []
    for b in range(B):
        missing = set(fixture["missing"][b])
        old = before[b + 1]
        new = after[b + 1]
        n = counts[b]
        copied, slots = miss_src[b][:n], miss_dst[b][:n]
        if (
            n < 0
            or n != len(missing)
            or set(copied) != missing
            or len(set(copied)) != n
        ):
            raise AssertionError(f"{label}: request {b} copy union mismatch")
        if len(set(slots)) != n or any(x < 0 or x >= C for x in slots):
            raise AssertionError(f"{label}: request {b} invalid copy slots")
        owners = {
            slot: token for token, slot in enumerate(old[:PREFIX]) if slot != INVALID
        }
        if len(owners) != C or set(owners) != set(range(C)):
            raise AssertionError(f"{label}: invalid initial pool")
        selected = set().union(*(set(row) for row in reference[b]))
        expected = old.copy()
        for token, slot in zip(copied, slots):
            victim = owners[slot]
            if victim in selected:
                raise AssertionError(f"{label}: selected resident evicted")
            expected[victim] = INVALID
            expected[token] = slot
        if expected != new:
            raise AssertionError(
                f"{label}: request {b} post-pool differs from legal delta"
            )
        req_counts = []
        for r in range(q):
            i = b * q + r
            sources, destinations = src[i][0], dst[i][0]
            if len(set(sources)) != K or set(sources) != set(reference[b][r]):
                raise AssertionError(
                    f"{label}: request {b} route {r} differs from LI TopK set"
                )
            if any(t < 0 or t >= ACTUAL - q + r + 1 for t in sources):
                raise AssertionError(f"{label}: causal range violation at {b}/{r}")
            route_misses = sum(t in missing for t in sources)
            if route_counts[i] != route_misses:
                raise AssertionError(
                    f"{label}: request {b} route {r} miss count mismatch"
                )
            for t, slot in zip(sources, destinations):
                expected_slot = C + t - PREFIX if t >= PREFIX else new[t]
                if slot != expected_slot:
                    raise AssertionError(
                        f"{label}: wrong destination for source {t} at {b}/{r}"
                    )
            req_counts.append(route_misses)
        routes.append(req_counts)
    return {"passed": True, "request_miss_counts": counts, "route_miss_counts": routes}


def run(args, case, out):
    torch, torch_npu, li, original, sglang, versions = load_backends(args)
    q = case["q"]
    d, input_metadata = input_data(torch, torch_npu, args, q)
    device = torch.device(args.device)
    hashes = {
        name: tensor_hash(torch, d[name])
        for name in ("query", "key", "weights", "query_scale", "key_scale", "table")
    }
    ref_tensor = extract_li(torch, call_li(li, d), B * q)
    torch.npu.synchronize()
    hashes["input_sha256"] = hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode()
    ).hexdigest()
    hashes["reference"] = tensor_hash(torch, ref_tensor)
    hashes["reference_sha256"] = hashes["reference"]
    reference = ref_tensor.cpu().tolist()
    validate_causal_boundary(reference, ACTUAL)
    fixture = build_residency(
        reference,
        PREFIX,
        C,
        target_union_misses(q, case["hit_percent"]),
        case["policy"],
        args.seed,
    )
    pool_cpu = torch.full((B + 1, args.source_capacity), INVALID, dtype=torch.int32)
    for b, mapping in enumerate(fixture["maps"]):
        for token, slot in mapping.items():
            pool_cpu[b + 1, token] = slot
    hashes["pool"] = tensor_hash(torch, pool_cpu)
    hashes["residency_sha256"] = hashes["pool"]
    hashes["fixture_sha256"] = fixture["fixture_sha256"]
    seed_pool = pool_cpu.to(device)
    before = pool_cpu.tolist()
    buffers = {
        name: (seed_pool.clone(), make_outputs(torch, device, q))
        for name in ("original", "sglang")
    }
    funcs = {"li": lambda: call_li(li, d)}
    for name, op in [("original", original), ("sglang", sglang)]:
        pool, outputs = buffers[name]
        funcs[name] = lambda op=op, pool=pool, outputs=outputs: call_lim(
            op, d, pool, outputs
        )

    def restore(name):
        if name != "li":
            buffers[name][0].copy_(seed_pool)

    repo = Path(__file__).resolve().parents[2]
    source = source_provenance(repo)
    versions["original_compile_variant"] = args.original_variant
    versions["original_source_revision_hint"] = args.original_source_revision
    backend = (
        torch.ops.npu.a5_c8_backend()
        if hasattr(torch.ops.npu, "a5_c8_backend")
        else "legacy_direct"
    )
    versions["sglang_compile_variant"] = (
        "original ACLNN/OPP LIM; pinned SIMT defaults, verify source/build manifest"
        if backend == "original"
        else "direct LIM; SIMT postprocess/victim=0, verify actual built library"
    )
    versions["sglang_backend"] = backend
    versions["sglang_original_runtime_manifest"] = getattr(
        sys.modules.get("sgl_kernel_npu"), "_original_opp_manifest", None
    )
    if backend == "original":
        versions["sglang_resolved_original_libraries"] = dict(
            torch.ops.npu.resolve_a5_c8_original_libraries()
        )
    versions["variant_note"] = (
        "Both LIM entries use the pinned original implementation; inspect manifests and resolved libraries. "
        "Multiple OPP vendors may share device-op routing in one process."
        if backend == "original"
        else "Original SIMT variant may differ; no causal attribution without matching builds."
    )
    versions["environment"] = {
        k: os.environ.get(k)
        for k in (
            "ASCEND_HOME_PATH",
            "ASCEND_OPP_PATH",
            "ASCEND_CUSTOM_OPP_PATH",
            "LD_LIBRARY_PATH",
            "PYTHONPATH",
            "CANN_VERSION",
        )
    }
    result = {
        "schema_version": 1,
        "case_id": case_id(case),
        "case": case,
        "status": "running",
        "diagnostic_only": args.smoke,
        "backend_versions": versions,
        "hashes": hashes,
        "fixture": {
            "fixture_sha256": fixture["fixture_sha256"],
            "stats": fixture["stats"],
            "seed": args.seed,
            "actual_key_length": ACTUAL,
            "prefix_tokens": PREFIX,
            "source_capacity": args.source_capacity,
            "cache_capacity": C,
            "pool_rows": B + 1,
            "tail_slot_policy": "C + source - prefix; tail never resident",
        },
        "correctness": {"li": {"passed": True, "causal_boundary": True}},
        "correctness_coverage": {
            "full_outputs_and_pool": "setup, graph replay, and end of each round",
            "every_timed_lim_call": "request and route miss counts, outside events",
            "all_intermediate_outputs_fully_validated": False,
        },
        "rounds": [],
        "timing_boundary": "NPU event around operator or graph replay only; pool restore, validation and persistence excluded; LI allocates its own output",
        "input_distribution": "independent CPU BF16 normal samples; dynamic FP8 E4M3 quantization on NPU",
        "quantization": {
            "api": "torch_npu.npu_dynamic_quant",
            "dst_type": "torch.float8_e4m3fn",
            "li_query_quant_mode": 0,
            "li_key_quant_mode": 0,
            "input_seed": input_metadata["seed"],
        },
        "input_metadata": input_metadata,
        "inputs_file": str(args.inputs_file.resolve()) if args.inputs_file else None,
        "saved_inputs_file": (
            str(args.save_inputs.resolve()) if args.save_inputs else None
        ),
        "command": sys.argv,
        "script_sha256": file_sha(Path(__file__)),
        "backend_script_sha256": file_sha(Path(__file__).with_name("backends.py")),
        "fixture_script_sha256": file_sha(Path(__file__).with_name("fixture.py")),
        **source,
        "started_at_utc": datetime.now(utc_timezone).isoformat(),
        "started_at_epoch_seconds": datetime.now(utc_timezone).timestamp(),
    }

    def persist():
        temporary = out / "case.json.tmp"
        temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
        temporary.replace(out / "case.json")

    def diagnose(name):
        torch.save(
            {
                "reference": ref_tensor.cpu(),
                "initial_pool": pool_cpu,
                "pool": buffers[name][0].cpu(),
                "outputs": [x.cpu() for x in buffers[name][1]],
                "missing": fixture["missing"],
            },
            out / (name + "_diagnostic.pt"),
        )

    def validate(name, label):
        try:
            return validate_result(
                torch, reference, fixture, before, *buffers[name], q, label
            )
        except Exception:
            diagnose(name)
            raise

    def check_counts(name):
        route_counts = buffers[name][1][2].cpu().tolist()
        miss_counts = buffers[name][1][5].cpu().tolist()
        expected_requests = [len(row) for row in fixture["missing"]]
        expected_routes = [
            sum(token in missing for token in route)
            for rows, missing in zip(reference, map(set, fixture["missing"]))
            for route in rows
        ]
        if miss_counts != expected_requests or route_counts != expected_routes:
            diagnose(name)
            raise AssertionError(
                f"{name}: per-sample request or route miss counts differ"
            )

    persist()
    for name in ("original", "sglang"):
        restore(name)
        funcs[name]()
        torch.npu.synchronize()
        result["correctness"][name] = validate(name, name)
    versions["libraries"] = loaded_libraries()
    if args.original_opp_root:
        versions["original_opp_library_candidates"] = opp_libraries(
            args.original_opp_root
        )
        if not versions["original_opp_library_candidates"]:
            raise RuntimeError("original OPP root contains no shared libraries")
    else:
        versions["original_opp_library_candidates"] = None
    versions["custom_opp_library_candidates"] = {
        root: opp_libraries(root)
        for root in os.environ.get("ASCEND_CUSTOM_OPP_PATH", "").split(os.pathsep)
        if root and Path(root).is_dir()
    }
    versions["opp_candidate_note"] = (
        "Installed candidates and loaded host libraries are evidence only; device binary dispatch requires separate verification."
    )
    persist()
    # The default-stream setup and oracle work is complete before this dedicated stream starts.
    torch.npu.synchronize()
    stream = torch.npu.Stream()
    graphs = {}
    graph_outputs = {}
    if case["mode"] == "graph":
        for name in ("li", "original", "sglang"):
            with torch.npu.stream(stream):
                restore(name)
                funcs[name]()
                torch.npu.synchronize()
                restore(name)
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, stream=stream):
                    graph_outputs[name] = funcs[name]()
                graphs[name] = graph
            torch.npu.synchronize()
            with torch.npu.stream(stream):
                restore(name)
                graph.replay()
            torch.npu.synchronize()
            if name == "li":
                replay_ref = (
                    extract_li(torch, graph_outputs[name], B * q).cpu().tolist()
                )
                if any(
                    set(replay_ref[b][r]) != set(reference[b][r])
                    for b in range(B)
                    for r in range(q)
                ):
                    raise AssertionError("LI graph replay differs from eager LI")
            else:
                result["correctness"][name]["graph_replay"] = validate(
                    name, name + " graph"
                )
            persist()
    names = ["li", "original", "sglang"]

    def invoke(name):
        if case["mode"] == "graph":
            graphs[name].replay()
        else:
            return funcs[name]()

    with torch.npu.stream(stream):
        for i in range(args.warmup):
            for name in names[i % 3 :] + names[: i % 3]:
                restore(name)
                invoke(name)
        torch.npu.synchronize()
        for r in range(args.rounds):
            samples = {name: [] for name in names}
            round_data = {
                "round": r + 1,
                "samples_us": samples,
                "backend_order": [],
                "correctness": {},
            }
            result["rounds"].append(round_data)
            persist()
            for i in range(args.iterations):
                order = names[(r + i) % 3 :] + names[: (r + i) % 3]
                round_data["backend_order"].append(order)
                for name in order:
                    restore(name)
                    start, end = (
                        torch.npu.Event(enable_timing=True),
                        torch.npu.Event(enable_timing=True),
                    )
                    start.record(stream)
                    retained = invoke(name)
                    end.record(stream)
                    end.synchronize()
                    del retained  # Keep LI output alive until its event completes.
                    samples[name].append(float(start.elapsed_time(end)) * 1000)
                    persist()
                    if name != "li":
                        check_counts(name)
            # No restore: validate the state left by the last invocation of each backend.
            for name in ("original", "sglang"):
                round_data["correctness"][name] = validate(
                    name, f"{name} round {r + 1}"
                )
                persist()
    versions["libraries"] = loaded_libraries()
    result["status"] = "ok"
    result["finished_at_utc"] = datetime.now(utc_timezone).isoformat()
    result["finished_at_epoch_seconds"] = datetime.now(utc_timezone).timestamp()
    persist()
    return result


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    cases = make_matrix()
    if args.dry_run:
        print(
            json.dumps(
                [
                    {
                        "case_id": case_id(c),
                        **c,
                        "batch": B,
                        "topk": K,
                        "cache_capacity": C,
                        "prefix_tokens": PREFIX,
                        "actual_length": ACTUAL,
                        "source_capacity": args.source_capacity,
                        "union_misses_per_request": target_union_misses(
                            c["q"], c["hit_percent"]
                        ),
                    }
                    for c in cases
                ],
                indent=2,
            )
        )
        return 0
    if not args.case_id or not args.output or not args.kernel_python_root:
        p.error(
            "--case-id, --output and --kernel-python-root are required except for --dry-run"
        )
    if args.source_capacity < ACTUAL or args.source_capacity % 128:
        p.error("--source-capacity must be a multiple of 128 and >= 65664")
    if min(args.warmup, args.rounds, args.iterations) < 1:
        p.error("warmup, rounds and iterations must be positive")
    if args.inputs_file and args.save_inputs:
        p.error("--inputs-file and --save-inputs are mutually exclusive")
    if args.inputs_file and not args.inputs_file.is_file():
        p.error("--inputs-file must exist")
    if args.save_inputs and args.save_inputs.exists():
        p.error("--save-inputs must not exist")
    if args.smoke:
        args.warmup = args.rounds = args.iterations = 1
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema_version": 1,
        "case_id": args.case_id,
        "command": sys.argv,
        "status": "running",
    }
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    try:
        result = run(
            args, next(c for c in cases if case_id(c) == args.case_id), args.output
        )
        manifest["status"] = result["status"]
    except (
        Exception
    ):  # noqa: BLE001 - preserve diagnostic evidence for any backend failure
        detail = traceback.format_exc()
        manifest["status"] = "failed"
        (args.output / "error.json").write_text(
            json.dumps({"status": "failed", "traceback": detail}, indent=2),
            encoding="utf-8",
        )
        case_path = args.output / "case.json"
        failure = (
            json.loads(case_path.read_text(encoding="utf-8"))
            if case_path.exists()
            else {
                "schema_version": 1,
                "case_id": args.case_id,
                "case": next(c for c in cases if case_id(c) == args.case_id),
                "rounds": [],
            }
        )
        failure["status"] = "failed"
        failure["error_file"] = "error.json"
        (args.output / "case.json").write_text(
            json.dumps(failure, indent=2), encoding="utf-8"
        )
        print(detail, file=sys.stderr)
    (args.output / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return 0 if manifest["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
