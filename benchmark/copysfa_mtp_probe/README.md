# A3 MTP3 CopySFA: isolated source-aware refresh probe

This directory is an experimental one-die benchmark and a separately loaded
operator. It does not change the installed SGLang operator or enable a new
framework path. The paired prefix/host-KV correctness fixes remain on
`feat/mtp3-prefix-kv-fused-metadata-20261031` in both personal repositories.

## Confirmed dispatch difference

The baseline combined-cache entry point treats `full_visible && miss_count > 0`
as a batch-wide first-fill. It copies the complete request-level union,
executes a global barrier, then uses `SOURCE_AWARE=false` attention. The native
vLLM reference selects its first-fill path when the cache itself needs filling;
ordinary steady misses use its source-aware gather.

A source-aware path cannot simply replace the full-visible fallback: Q4 LIM
appends up to four rewritten resident KV entries to the request-level copy
list. They remain TopK hits, so per-query miss prefixes do not refresh them.

The experimental entry point copies only the final `min(query_count, copy_count)`
request entries before the barrier, then runs source-aware attention. This
covers every appended resident rewrite and may redundantly copy a few ordinary
misses. A batch containing a cold fill retains the complete-copy fallback.

**Required contract:** rewritten resident entries must form a suffix with at
most one entry per query. `AppendRewrittenResidents` in the paired native LIM
provides that contract, and the framework metadata adapter preserves ordering.
This is not a drop-in guarantee for arbitrary manually reordered copy plans.
The probe uses a separate device symbol and `copysfa_candidate` Torch namespace.

## Measured results on 2026-10-31

One physical A3 die; CANN 9.1; Torch 2.10; BF16 KV; B16; Q4 (MTP3); C16384;
TopK2048; 65536-row shared host cache. The exact same Q, RoPE, host mapping,
HBM mapping and miss metadata are reused by the two SGLang variants in one
process. CPU affinity is 0-7. Resetting HBM happens before the start event.
Each value below averages three alternating rounds of 150 Graph replays.
Random source IDs are sampled independently per request; route miss sets are
disjoint, with exactly 100 or 175 misses per query.

| Request miss union | Baseline full-visible | Refresh/source-aware probe | Reduction |
|---|---:|---:|---:|
| 400 | 453.031 us | 348.661 us | 23.038% |
| 700 | 539.836 us | 364.001 us | 32.572% |

The contiguous-source same-address control measured 458.392/341.832 us for
union400 and 523.419/369.438 us for union700 (baseline/probe respectively).
Every alternating output is compared across all B16 requests and four logical
heads. Request-copy and rewritten-resident payloads are checked bit-for-bit.

The independently timed native comparison uses an identical **prefix plus
causal tail** contract in both frameworks. vLLM requires at least eight heads,
so the same four logical query heads are padded with four zero heads. Its
split contiguous CKV/KPE layout is populated from the same frozen tensors.

| Request miss union | Native vLLM Graph150 | SGLang Graph150 |
|---|---:|---:|
| 400 | 331.168 us | 358.599 us |
| 700 | 362.559 us | 388.758 us |

The vLLM binary comes from the locally available
`quay.io/ascend/vllm-ascend:v0.30.0rc2-a3` image. It is loaded in the same CANN9.1
benchmark runtime. It is **not** the report's nightly r44 image. Full-visible
and prefix-plus-tail rows must not be presented as an exact semantic pair.

The operator test has no serving prefix-hit metric. Shared host pages model a
shared prefix; request HBM pools remain independent. Cache misses, resident
refreshes, and serving prefix hits are separate quantities. Whole-model
CopySFA675.749us cannot be uniquely attributed without its actual per-route
miss counts, overlap and source-address distribution. This probe has not been
validated in a full TP16 model run or a model accuracy suite.

## Reproduce

Use one allocated A3 die and the already built paired kernel package. Source
its Ascend environment first. Preserve the original PYTHONPATH when adding the
package. Do not install over a colleague's wheel for this test.

```bash
export KERNEL_REPO=/path/to/sgl-kernel-npu
export PROBE="$KERNEL_REPO/benchmark/copysfa_mtp_probe"
export TASK=/path/to/results/copysfa-mtp
export BUILD_DIR=/tmp/copysfa_probe_fresh_build
export ASCEND_RT_VISIBLE_DEVICES=0
mkdir -p "$TASK/inputs"
cmake -S "$PROBE" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD_DIR" -j8
```

Use a new build directory. Some CANN incremental preprocess stages cannot
re-link the already transformed object and report `unknown file type`.
No system-wide THP setting is changed; the benchmark disables THP only in its
own process before Torch allocations.

```bash
taskset -c 0-7 python3 "$PROBE/bench_interleaved.py"   \
  --fixture-dir "$TASK/inputs" \
  --out "$TASK/paired-random" \
  --candidate-so "$BUILD_DIR/libcopy_refresh_candidate.so" \
  --heads 4 \
  --queries 4 \
  --capacity 16384 \
  --misses 400 700 \
  --sharing disjoint \
  --address-pattern random \
  --refresh-residents 4 \
  --samples 150
```

`bench_interleaved.py` warms each DSO before capturing its graph; their host
tiling caches are independent. Its `interleaved` JSON array contains the
three baseline/candidate timing pairs. The compatibility `eager`/`graph`
fields contain an extra baseline measurement, not the paired candidate time.

For the native vLLM comparison, first extract its two libraries and custom OPP
from a matching existing image into VLLM_RUNTIME (no NPU is needed to extract):
`libvllm_ascend_kernels.so`, `vllm_ascend_C.cpython-312-aarch64-linux-gnu.so`, and
`_cann_ops_custom/vendors/custom_transformer`. Set its `LD_LIBRARY_PATH` and
`ASCEND_CUSTOM_OPP_PATH` before importing Torch. The binary must match the
Python/Torch ABI and support the target device.

```bash
export VLLM_RUNTIME=/path/to/extracted/vllm_runtime
export LD_LIBRARY_PATH="$VLLM_RUNTIME:$VLLM_RUNTIME/_cann_ops_custom/vendors/custom_transformer/op_api/lib:$LD_LIBRARY_PATH"
export ASCEND_CUSTOM_OPP_PATH="$VLLM_RUNTIME/_cann_ops_custom/vendors/custom_transformer"
taskset -c 0-7 python3 "$PROBE/bench_mtp_copy_sfa.py"   \
  --backend vllm \
  --contract prefix-tail \
  --vllm-runtime "$VLLM_RUNTIME" \
  --fixture-dir "$TASK/inputs" \
  --out "$TASK/vllm-random" \
  --heads 8 \
  --logical-heads 4 \
  --queries 4 \
  --capacity 16384 \
  --misses 400 700 \
  --sharing disjoint \
  --address-pattern random \
  --samples 150
```

Repeat that command with `--backend sglang` and a different output directory,
without `--candidate-so`. Keep `--contract prefix-tail` for both. The vLLM
adapter uses non-owning CPU tensor metadata to expose the registered device
address. Never dereference that alias on the CPU; population and reference
calculations use the original host tensor instead.

## Fixture and timing boundaries

- The persisted host/Q/RoPE fixture is reused by both implementations. Seed42
  constructs the base BF16 tensors. Per-request random miss IDs use a separate
  deterministic generator derived from capacity, union size and request ID.
- Cache HBM has independent C-token regions per request. Miss destinations are
  slots0..M-1. Hit destinations are outside that range. Resident rewrites use
  the last R slots and are appended to the request copy list.
- `disjoint` partitions the request miss union over four queries. `mixed`
  shares one quarter of the union, then partitions the rest; `shared` places
  every missing token in every query. Equal union size therefore does not imply
  equal per-query miss work. Both contiguous and random source patterns exist.
- Full-visible computes exactly the selected2048 entries per query. Prefix-tail
  computes those entries plus the corresponding1..Q causal tail entries.
- Eager timing is secondary. Graph replay is primary; HBM restoration is
  outside the measured events. Payload and attention checks precede timing;
  graph replay payload checks and saved outputs follow it.
- A FP32 attention reference checks every query in the first request. Saved
  outputs permit all-request backend comparison. The interleaved test checks
  all-request outputs and exact copied payloads after every timed round.

The CopySFA fixture bypasses LI/LIM planning; it has no LIM state=-1/-2/-3.
Cumulative query ends are Q,2Q,...,B*Q. KV-length metadata is C+Q-1
for full-visible and C+Q for prefix-tail. The full-visible flag explicitly
selects exactly2048 entries and suppresses the causal tail.
