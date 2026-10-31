"""Frozen Q1/Q4 combined-KV CopySFA microbenchmark on one assigned NPU."""

import ctypes

# Disable process THP before Torch creates or first-touches host allocations.
ctypes.CDLL(None).prctl(41, 1, 0, 0, 0)
# isort: split
import argparse
import json
import math
import os
import statistics
from pathlib import Path

import sgl_kernel_npu  # noqa: F401 -- register native operators
import torch
import torch_npu  # noqa: F401 -- register the NPU backend
from sgl_kernel_npu.sparsity_driven_kv_offload import create_shm_tensor, free_shm


def measure(call, reset, samples, warmup):
    for _ in range(warmup):
        reset()
        call()
    torch.npu.synchronize()
    pairs = []
    for _ in range(samples):
        reset()
        left = torch.npu.Event(enable_timing=True)
        right = torch.npu.Event(enable_timing=True)
        left.record()
        call()
        right.record()
        pairs.append((left, right))
    torch.npu.synchronize()
    values = [a.elapsed_time(b) * 1000 for a, b in pairs]
    return {
        "samples": samples,
        "mean_us": statistics.mean(values),
        "median_us": statistics.median(values),
        "min_us": min(values),
        "max_us": max(values),
        "values_us": values,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixture-dir", type=Path)
    ap.add_argument("--vllm-runtime", type=Path)
    ap.add_argument(
        "--address-pattern", choices=["contiguous", "random"], default="contiguous"
    )
    ap.add_argument("--candidate-so", type=Path)
    ap.add_argument("--refresh-residents", type=int, default=0)
    ap.add_argument("--backend", choices=["sglang", "vllm"], default="sglang")
    ap.add_argument(
        "--contract", choices=["full-visible", "prefix-tail"], default="full-visible"
    )
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--logical-heads", type=int)
    ap.add_argument("--context", type=int, default=65536)
    ap.add_argument("--capacity", type=int, nargs="+", default=[16384])
    ap.add_argument("--queries", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--misses", type=int, nargs="+", default=[0, 300, 500, 1500])
    ap.add_argument("--samples", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument(
        "--sharing", choices=["shared", "mixed", "disjoint"], default="mixed"
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.backend == "vllm":
        assert (
            args.contract == "prefix-tail"
        ), "native vLLM has a prefix-plus-causal-tail contract"
    torch.npu.set_device(0)
    dev = torch.device("npu:0")
    torch.set_num_threads(8)
    torch.manual_seed(42)
    B, H, S, K = args.batch, args.heads, args.context, 2048
    host, _, ptr = create_shm_tensor(
        (1, S, 1, 576), torch.bfloat16, device_id=0, name="copy_sfa_mtp_frozen_host"
    )
    logical_heads = args.logical_heads or H
    assert logical_heads <= H
    fixture_dir = args.fixture_dir or Path(__file__).parent
    fixture_dir.mkdir(parents=True, exist_ok=True)
    fixture = fixture_dir / f"frozen_base_B{B}_H{logical_heads}_S{S}.pt"
    if fixture.exists():
        frozen = torch.load(fixture, map_location="cpu", weights_only=True)
        host.copy_(frozen["host"])
        cpu_query, cpu_rope = frozen["query"], frozen["rope"]
    else:
        host.copy_(torch.randn(host.shape, dtype=torch.bfloat16))
        cpu_query = torch.randn(B * 4, logical_heads, 512, dtype=torch.bfloat16)
        cpu_rope = torch.randn(B * 4, logical_heads, 64, dtype=torch.bfloat16)
        torch.save(
            {"host": host.clone(), "query": cpu_query, "rope": cpu_rope}, fixture
        )
    if logical_heads < H:
        cpu_query = torch.nn.functional.pad(cpu_query, (0, 0, 0, H - logical_heads))
        cpu_rope = torch.nn.functional.pad(cpu_rope, (0, 0, 0, H - logical_heads))

    def describe_host_mapping(tensor):
        for line in Path("/proc/self/maps").read_text().splitlines():
            area = line.split()[0].split("-")
            if int(area[0], 16) <= tensor.data_ptr() < int(area[1], 16):
                return [
                    line
                    for line in Path("/proc/self/numa_maps").read_text().splitlines()
                    if line.startswith(area[0] + " ")
                ]
        return []

    placement = {
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "combined_host": describe_host_mapping(host),
    }
    (args.out / "host_placement.json").write_text(json.dumps(placement, indent=2))
    rows = []
    keepalive = []
    if args.backend == "vllm":
        assert H >= 8
        runtime = args.vllm_runtime or Path(__file__).parent / "vllm_runtime"
        torch.ops.load_library(str(runtime / "libvllm_ascend_kernels.so"))
        torch.ops.load_library(
            str(runtime / "vllm_ascend_C.cpython-312-aarch64-linux-gnu.so")
        )
        operator = (
            torch.ops._C_ascend.npu_fused_scatter_copy_sparse_flash_attention.default
        )
        host_ckv, _, ckv_ptr = create_shm_tensor(
            (S // 128, 128, 512), torch.bfloat16, device_id=0, name="vllm_frozen_ckv"
        )
        host_rope, _, rope_ptr = create_shm_tensor(
            (S // 128, 128, 64), torch.bfloat16, device_id=0, name="vllm_frozen_kpe"
        )
        host_ckv.copy_(host[0, :, 0, :512].reshape(S // 128, 128, 512))
        host_rope.copy_(host[0, :, 0, 512:].reshape(S // 128, 128, 64))

        def device_alias(tensor, pointer):
            if tensor.data_ptr() == pointer:
                return tensor
            buffer = (
                ctypes.c_uint8 * (tensor.numel() * tensor.element_size())
            ).from_address(pointer)
            alias = torch.frombuffer(buffer, dtype=tensor.dtype).reshape(tensor.shape)
            keepalive.extend([buffer, alias])
            assert alias.data_ptr() == pointer
            return alias

        placement.update(
            vllm_ckv=describe_host_mapping(host_ckv),
            vllm_rope=describe_host_mapping(host_rope),
        )
        (args.out / "host_placement.json").write_text(json.dumps(placement, indent=2))
        host_ckv_device = device_alias(host_ckv, ckv_ptr)
        host_rope_device = device_alias(host_rope, rope_ptr)
        print(
            "VLLM_HOST_POINTER_CONTRACT",
            host_ckv.data_ptr() == ckv_ptr,
            host_rope.data_ptr() == rope_ptr,
            flush=True,
        )
    else:
        operator = torch.ops.npu.fused_scatter_copy_sparse_flash_attention.default
        if args.candidate_so:
            torch.ops.load_library(str(args.candidate_so))
            operator = torch.ops.copysfa_candidate.run.default

    def npu(x):
        return x.to(dev)

    try:
        for C in args.capacity:
            stride = C + (128 if args.contract == "prefix-tail" else 0)
            initial_cpu = torch.zeros(B, stride, 576, dtype=torch.bfloat16)
            initial_cpu[:, :C] = host[0, :C, 0]
            if args.contract == "prefix-tail":
                initial_cpu[:, C : C + 4] = host[0, S - 4 : S, 0]
            initial = npu(initial_cpu).view(B * (stride // 128), 128, 1, 576)
            hbm = initial.clone()
            if args.backend == "vllm":
                initial_ckv = initial[..., :512].contiguous()
                initial_rope = initial[..., 512:].contiguous()
                hbm_ckv, hbm_rope = initial_ckv.clone(), initial_rope.clone()

            def reset():
                if args.backend == "vllm":
                    hbm_ckv.copy_(initial_ckv)
                    hbm_rope.copy_(initial_rope)
                else:
                    hbm.copy_(initial)

            def current_cache():
                return (
                    torch.cat((hbm_ckv, hbm_rope), dim=-1)
                    if args.backend == "vllm"
                    else hbm
                )

            hbm_table = npu(
                torch.arange(B * (stride // 128), dtype=torch.int32).view(B, -1)
            )
            dram_table = npu(torch.arange(S // 128, dtype=torch.int32).repeat(B, 1))
            cache_tokens = npu(torch.full((B,), C, dtype=torch.int32))
            for Q in args.queries:
                query = npu(
                    cpu_query.view(B, 4, H, 512)[:, :Q]
                    .reshape(B * Q, H, 512)
                    .contiguous()
                )
                rope = npu(
                    cpu_rope.view(B, 4, H, 64)[:, :Q].reshape(B * Q, H, 64).contiguous()
                )
                q_ends = npu(torch.arange(1, B + 1, dtype=torch.int32) * Q)
                # The paired native comparison uses K selected prefix rows
                # plus 1..Q causal tail tokens in both implementations.
                kv_length = C + Q if args.contract == "prefix-tail" else C + Q - 1
                kv_lens = npu(torch.full((B,), kv_length, dtype=torch.int32))
                output = torch.empty_like(query)
                for M in args.misses:
                    assert 0 <= M <= K and C + M <= S
                    src = torch.empty((B * Q, 1, K), dtype=torch.int32)
                    dst = torch.empty_like(src)
                    route_counts = torch.empty(B * Q, dtype=torch.int32)
                    request_src = torch.full((B, 32768), -1, dtype=torch.int32)
                    request_dst = torch.full_like(request_src, -1)
                    request_src[:, :M] = torch.arange(C, C + M, dtype=torch.int32)
                    request_dst[:, :M] = torch.arange(M, dtype=torch.int32)
                    for b in range(B):
                        for q in range(Q):
                            if args.sharing in ("mixed", "disjoint") and Q > 1:
                                shared = M // 4 if args.sharing == "mixed" else 0
                                mids = torch.cat(
                                    (
                                        torch.arange(shared),
                                        (
                                            torch.arange(shared + q, M, Q)
                                            if shared + q < M
                                            else torch.empty(0, dtype=torch.int64)
                                        ),
                                    )
                                ).int()
                            else:
                                mids = torch.arange(M, dtype=torch.int32)
                            missing = mids.numel()
                            common = (K - missing) * 3 // 4
                            if K == missing:
                                hits = torch.empty(0, dtype=torch.int32)
                            else:
                                hits = torch.cat(
                                    (
                                        torch.arange(M, M + common),
                                        M
                                        + common
                                        + (
                                            torch.arange(K - missing - common)
                                            + q * (K - common)
                                        )
                                        % (C - M - common),
                                    )
                                ).int()
                            row = b * Q + q
                            src[row, 0] = torch.cat((C + mids, hits))
                            dst[row, 0] = torch.cat((mids, hits))
                            route_counts[row] = missing
                    if args.address_pattern == "random":
                        for bi in range(B):
                            gen = torch.Generator().manual_seed(
                                20261031 + C * 100 + M * 10 + bi
                            )
                            random_ids = (
                                torch.randperm(S - C - 16, generator=gen)[:M].int() + C
                            )
                            request_src[bi, :M] = random_ids
                            for qi in range(Q):
                                rowi = bi * Q + qi
                                mi = int(route_counts[rowi])
                                src[rowi, 0, :mi] = random_ids[dst[rowi, 0, :mi].long()]
                    R = args.refresh_residents
                    assert 0 <= R <= Q
                    assert R == 0 or args.contract == "full-visible"
                    if R:
                        # LIM appends resident rewrites after the genuine miss union.
                        # Resident entries remain in the hit section of each TopK row.
                        request_src[:, M : M + R] = torch.arange(
                            S - R, S, dtype=torch.int32
                        )
                        request_dst[:, M : M + R] = torch.arange(
                            C - R, C, dtype=torch.int32
                        )
                        src[:, 0, -R:] = torch.arange(S - R, S, dtype=torch.int32)
                        dst[:, 0, -R:] = torch.arange(C - R, C, dtype=torch.int32)
                    source, slots, misses = npu(src), npu(dst), npu(route_counts)
                    miss_sources, miss_slots = npu(request_src), npu(request_dst)
                    miss_counts = npu(torch.full((B,), M + R, dtype=torch.int32))

                    def call():
                        if args.backend == "vllm":
                            operator(
                                rope,
                                query,
                                q_ends,
                                kv_lens,
                                cache_tokens,
                                slots,
                                source,
                                misses,
                                miss_sources,
                                miss_slots,
                                miss_counts,
                                hbm_table,
                                dram_table,
                                hbm_rope,
                                hbm_ckv,
                                host_rope_device,
                                host_ckv_device,
                                1 / math.sqrt(256),
                                output,
                            )
                        else:
                            operator(
                                query,
                                rope,
                                hbm,
                                q_ends,
                                kv_lens,
                                cache_tokens,
                                source,
                                slots,
                                misses,
                                miss_sources,
                                miss_slots,
                                miss_counts,
                                hbm_table,
                                dram_table,
                                ptr,
                                output,
                                1 / math.sqrt(256),
                                (args.contract == "full-visible"),
                            )

                    reset()
                    call()
                    torch.npu.synchronize()
                    expected_cache = initial_cpu[0].clone()
                    expected_cache[:M] = host[0, request_src[0, :M].long(), 0]
                    if R:
                        expected_cache[C - R : C] = host[0, S - R : S, 0]
                    max_abs = 0.0
                    for q in range(Q):
                        indices = dst[q, 0].long()
                        if args.contract == "prefix-tail":
                            indices = torch.cat((indices, torch.arange(C, C + q + 1)))
                        selected = expected_cache[indices].float()
                        score = (
                            cpu_query.view(B, 4, H, 512)[0, q].float()
                            @ selected[:, :512].T
                            + cpu_rope.view(B, 4, H, 64)[0, q].float()
                            @ selected[:, 512:].T
                        ) / math.sqrt(256)
                        expected = torch.softmax(score, dim=-1) @ selected[:, :512]
                        error = float((output[q].cpu().float() - expected).abs().max())
                        max_abs = max(max_abs, error)
                    actual_cache = current_cache().view(B, stride, 576)[:, :M].cpu()
                    expected_miss = torch.stack(
                        [host[0, request_src[bi, :M].long(), 0] for bi in range(B)]
                    )
                    torch.testing.assert_close(
                        actual_cache, expected_miss, rtol=0, atol=0
                    )
                    if R:
                        actual_refresh = (
                            current_cache().view(B, stride, 576)[:, C - R : C].cpu()
                        )
                        torch.testing.assert_close(
                            actual_refresh,
                            host[0, S - R : S, 0].unsqueeze(0).expand(B, -1, -1),
                            rtol=0,
                            atol=0,
                        )
                    assert max_abs < 0.08, max_abs
                    row = {
                        "backend": args.backend,
                        "B": B,
                        "Q": Q,
                        "T": B * Q,
                        "heads": H,
                        "logical_heads": logical_heads,
                        "context": S,
                        "capacity": C,
                        "request_union_misses": M,
                        "contract": args.contract,
                        "cache_stride_tokens": stride,
                        "per_route_misses": route_counts[:Q].tolist(),
                        "resident_refreshes": R,
                        "request_copy_count": M + R,
                        "experimental_candidate": bool(args.candidate_so),
                        "miss_sharing": args.sharing,
                        "address_pattern": args.address_pattern,
                        "host_layout": "shared_native_pages",
                        "kv_dtype": "bfloat16",
                        "topk": K,
                        "correctness_max_abs": max_abs,
                        "cache_payload_exact": True,
                    }
                    row["eager"] = measure(call, reset, args.samples, args.warmup)
                    reset()
                    torch.npu.synchronize()
                    graph = torch.npu.NPUGraph()
                    with torch.npu.graph(graph):
                        call()
                    torch.npu.synchronize()
                    row["graph"] = measure(
                        graph.replay, reset, args.samples, args.warmup
                    )
                    graph_cache = current_cache().view(B, stride, 576).cpu()
                    torch.testing.assert_close(
                        graph_cache[:, :M], expected_miss, rtol=0, atol=0
                    )
                    if R:
                        torch.testing.assert_close(
                            graph_cache[:, C - R : C],
                            host[0, S - R : S, 0].unsqueeze(0).expand(B, -1, -1),
                            rtol=0,
                            atol=0,
                        )
                    row["graph_cache_payload_exact"] = True
                    torch.save(
                        output[:, :logical_heads].cpu(),
                        args.out / f"output_C{C}_Q{Q}_M{M}_H{H}.pt",
                    )
                    row["attention_kv_length_metadata"] = kv_length
                    rows.append(row)
                    (args.out / "results.json").write_text(
                        json.dumps({"cases": rows}, indent=2) + "\n"
                    )
                    print(
                        json.dumps(
                            {
                                k: v
                                for k, v in row.items()
                                if k not in ["eager", "graph"]
                            }
                        )
                        + " eager_us="
                        + str(row["eager"]["mean_us"])
                        + " graph_us="
                        + str(row["graph"]["mean_us"]),
                        flush=True,
                    )
                    del graph
        print("ALL_CASES_PASSED", len(rows), flush=True)
    finally:
        torch.npu.synchronize()
        free_shm(0)


if __name__ == "__main__":
    main()
