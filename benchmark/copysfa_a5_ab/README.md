# A5 CopySFA accuracy and performance comparison

The acceptance target is numerical agreement with the original vLLM Ascend
CopySFA, followed by comparable latency. This benchmark does not invoke LIM or
a model. It passes the same frozen plans, queries and registered host KV bytes
to both backends. The reference is PR18490 at
`1b180deb840484c2fe33cbdd7abb68a615085172`; its original Torch adapter, OPP host
and device sources are retained unchanged in the reference source archive.
The reference is built under an isolated vendor and Torch namespace, without
installing vLLM or changing global OPP, DeepEP or the selected kernel wheel.

## Prepare on A5

Use a private environment with the paired Ascend950 `sgl_kernel_npu` wheel.
Source CANN and ATB before appending any Python paths. Build this reference
using that same Python/CANN environment; do not reuse a wrapper built against
another Torch ABI. All commands run manually on A5:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
export PYTHON_BIN=/path/to/task-venv/bin/python
export PATH="$(dirname "$PYTHON_BIN"):$PATH"

cd /path/to/sgl-kernel-npu
bash build.sh -a kernels Ascend950
# Install the produced wheel into the selected private interpreter with --no-deps.

bash benchmark/copysfa_a5_ab/reference/build_original.sh \
  /work/copysfa-original-001 --jobs 8
source /work/copysfa-original-001/paths.env
export ASCEND_RT_VISIBLE_DEVICES=0  # choose an actually free physical device first

python -m pytest -sv tests/test_copysfa_reference_transport.py
python -m pytest -sv \
  tests/test_fused_quant_scatter_copy_sparse_flash_attention.py \
  tests/test_fused_quant_scatter_copy_sparse_flash_attention_a5.py

python benchmark/copysfa_a5_ab/run_matrix.py --dry-run
python benchmark/copysfa_a5_ab/run_matrix.py --smoke \
  --output-root /work/copysfa-smoke-001
python benchmark/copysfa_a5_ab/run_matrix.py \
  --output-root /work/copysfa-formal-001
```

Output directories must be new. Preserve a failed directory and use a fresh
one when retrying. The smoke case is an odd-prefix-page Q4 graph with padding,
shared physical DRAM pages and 513 misses. Its two timing samples are only a
smoke check, not performance evidence.

The original adapter requires an NPU Tensor header for DRAM. A small wrapper
creates that header over the same registered host address used by the port;
it does not allocate an HBM replica. The owning CPU Tensor remains alive.
The header has a real Torch-NPU storage descriptor. It is an ACLNN operand,
not ordinary allocator-owned HBM: do not validate it with generic `.cpu()` or
`copy_`. The transport test reads it with the native byte-copy kernel instead.
The comparison verifies both resolved ACLNN symbols come from the selected
private reference OPP and records DSO hashes, schemas and source revision.

## Cases and hit-rate definition

The main 24 configurations are C=8192/16384/32384, Q1/Q4, eager/graph and
95/91 target hit rates. Defaults are B16, H8, BF16 query/output, TopK2048,
prefix65536, page128 and complete 656-byte C8 rows. `--heads 32` selects H32
for another TP geometry. Query segments represent MTP0 and MTP3 respectively.
The Q4 tail spans a page boundary and is already resident in caller-owned HBM.

The hit label means:

```text
selected-prefix C-pool hit rate =
  1 - unique selected historical misses / unique selected historical sources
```

It is not cache occupancy divided by C. Tail tokens are always HBM hits and
are excluded from that denominator. Each case reports the exact integer
counts, actual unique-prefix hit rate, individual query prefix hit rates,
all-query access hit rate and the older `1 - request_copy_count/(Q*2048)` label.
Integer rounding means 95/91 are target labels rather than exact fractions.
Cold fills separately report C copied rows, which can exceed the selected
TopK union; their actual hit rate is zero regardless of the case's label.

The main matrix deliberately has disjoint historical selections across Q4
routes. This makes the 95 and 91 configurations exercise both sides of the
original 512-miss gate. Four additional C16384 Q4 configurations have 90%
query overlap, exercising repeated miss ownership and writeback suppression.
Ten functional cases add 0/512/513/C misses, odd prefix pages, interior
padding and shared/random physical DRAM page tables, in eager and graph.
The functional cases also activate a previously padded request and permute
physical HBM/DRAM pages while retaining captured Tensor addresses. Both
backends must consume the updated metadata and preserve the logical result.
These functional cases also run both backends on two concurrent streams with
independent HBM/output buffers, including separately captured graphs. This
checks whole-core batch scheduling around their cross-core synchronization.
The full default run therefore contains 38 cases.

These are controlled operator inputs, not a model's measured hit distribution
or acceptance rate. They should not be compared directly with LIM benchmarks
that use a different denominator or different query overlap.

## Gates and timing

Before any timing, both operators must preserve expected HBM payloads bitwise,
produce finite outputs matching the packed CPU golden, and agree directly
with each other. The same checks repeat after graph replay and every timing
round. Any accuracy failure stops the matrix and saves the CPU fixture.
Padding outputs are initialized to zero by the caller, matching the operator's
empty-row contract.

Default direct parity tolerances are `atol=0.002, rtol=0.005`; the CPU golden
check uses `atol=0.005, rtol=0.02`. These are numerical checks, not an exact
generated-token gate. Maximum absolute error, RMSE and bitwise output equality
are recorded. Changing tolerances requires an explicit command and is logged.

Formal timing uses 20 warmups and 3 rounds of 50 samples per backend. Backend
order alternates. HBM restoration, output clearing and validation stay outside
the NPU event interval. Each original and port graph is captured independently.
All raw samples, round medians, median ratio and p95 are retained; samples are
not trimmed. Framework tail staging and metadata preparation are not included
in this standalone operator time.

Accuracy is the current hard gate. Latency regression is reported; above 20%
is a warning by default, not a failure. For a stricter later acceptance run,
invoke `bench.py --strict-performance --regression-threshold-pct ...` on a
selected case. No timing threshold establishes statistical equivalence.

## Evidence boundaries

A3 can execute the common metadata, registered-memory byte transport and A3
native CopySFA regressions. It can also cross-build Ascend950 sources with a
capable CANN toolchain. The original PR18490 C8 operator and this port's C8
attention require A5. A3 results do not establish A5 numerical agreement,
graph safety, model accuracy or performance. The prepared matrix is not an
A5 result until its actual `case.json`, exit codes and raw samples exist.

The offline reference dependencies and source hashes are in
[reference/PROVENANCE.json](reference/PROVENANCE.json). The private builder was
adapted from McZyWu's LIM comparison tooling; no LIM implementation is bundled
or used by this benchmark.
