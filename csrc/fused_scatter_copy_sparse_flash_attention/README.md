# Copy-SFA for SGLang NPU sparse KV offload

This folder contains the fused cache-copy and sparse-attention path used by SGLang. The native operator accepts combined-cache TND queries with one or more query rows per request. SGLang framework dispatch currently uses only Q1 decode.

## SGLang integration contract

The registered operator is `torch.ops.npu.fused_scatter_copy_sparse_flash_attention`. The SGLang manager plans request-cache metadata before the fused launch. The fused operator gathers missing KV rows from the manager's registered, device-accessible host mapping into HBM, then computes sparse flash attention. HBM cache rows are `[CKV(512) | KPE(64)]` with a 576-element token stride; the caller supplies a preallocated output tensor and the cache is updated in place.

Native operator contract:

- `query`, `query_rope`, and `attention_out` use TND `[T, H, 512]`, `[T, H, 64]`, and `[T, H, 512]`; `actual_seq_lengths_query` is the cumulative query-row prefix `[B]`. Each request has 1 to 16 query rows, and the final prefix equals `T`.
- CKV 512, KPE 64, TopK 2048, page size 128, and one KV head. `num_cache_tokens` carries the per-request cache length `C`; HBM and DRAM page tables are `[B, P]` and `[B, D]`, with positive page counts and valid physical addresses.
- FP16 or BF16 floating tensors; contiguous int32 metadata. Query, output, HBM cache, page tables, and metadata share one NPU. DRAM rows must be in the registered device-accessible host mapping used by the offload manager; arbitrary CPU allocations are unsupported.
- `combined_hbm_cache` is contiguous `[N, 128, 1, 576]`, where `N` contains enough pages for all request slots. TopK source/destination arrays are `[T, 1, 2048]`, with one miss count per query row. Request miss source/destination arrays are `[B, M]`, with one miss count per request; `M` may be larger than 2048.
- `attention_out` is caller-allocated and matches `query` shape and dtype. Active source IDs, slots, and block-table entries must be valid.

An active request may use `C == 0` for dense attention over HBM positions `[0, visible_kv)`; this path reads no TopK or miss entries. A positive sparse cache length must be block-aligned and at least 2048. In that mode, attention reads 2048 selected TopK slots and may also read the optional tail `[C, visible_kv)`. In a multi-query request, earlier query rows hide future query rows from the tail. TopK metadata must already contain valid selections for each query when `C > 0`. For existing Q1 SGLang calls, `actual_seq_lengths_kv == C` leaves the tail empty and retains full selected-TopK behavior. Request-local lengths, active IDs, and per-request query bounds are caller-owned device metadata; the wrapper does not copy their values to the CPU during dispatch.

Native multi-query attention follows the inherited sparse-prefix plus causal-tail protocol: `C` must not exceed the earliest query's visible KV length. Setting `actual_seq_lengths_kv == C` for a multi-query request violates that condition for earlier query rows. A future SGLang MTP path that selects TopK over each query's entire visible context without an appended tail needs an explicit length/selection protocol change; enabling the native multi-query shape alone does not implement it.

The SGLang path is enabled with both `SGLANG_NPU_ENABLE_SPARSE_KV_OFFLOAD=1` and `SGLANG_NPU_ENABLE_COPY_SFA=1`. Framework dispatch currently requires non-speculative Q1 decode, matching MLA/cache dimensions, and live sequence lengths of at least 2048 tokens (graph capture uses the framework's capture gate). Other calls use the existing attention path. The native multi-query contract does not imply framework MTP dispatch support.

For a graph batch containing padded rows, `actual_seq_lengths_query` remains the dense cumulative prefix `[1, 2, ..., B]` so each row keeps its output position. A padded row has zero KV/cache-token lengths, sentinel planner IDs/slots, and zero miss counts. The kernel writes zero output for that row and leaves its cache state untouched.

## Cache planner behavior

The planner mutates the source-to-slot and slot-to-source maps in place. A sort-join matches requested sources against current slot owners, emits unmatched requested sources as a miss prefix, and emits retained hits as a suffix. Unmatched old slots are paired with misses in deterministic sort order; the slot policy does not promise that invalid/free slots are preferred over all evictable slots. The manager owns map reset/reuse semantics and preallocates stable metadata buffers for graph replay.

## Future LIM handoff

The native interface can consume LIM's per-query TopK source IDs, destination slots, and miss counts together with its per-request union miss plan. This handoff must use one cache-map owner; running the current planner again after LIM would produce a second, potentially different mapping. Framework LIM dispatch is not implemented by this change.

With LIM's sparse-prefix policy, distinguish original context length `L`, indexer prefix length `P`, and resident capacity `C`. Copy-SFA needs the original host tail `[P, L)` in HBM slots `[C, C + L - P)`, with `num_cache_tokens = C` and request-local `actual_seq_lengths_kv = C + L - P`. HBM backing storage, row strides, and page tables must include this extra tail region while cache management remains restricted to `[0, C)`. Q1 needs at most one extra 128-token page for a page-aligned prefix; multi-query prefix selection and tail reservation must account for the earliest query and page crossings. These allocations and metadata addresses must be stable before graph capture.

IndexShare may share logical source IDs between layers. Reusing destination slots or miss plans also requires a maintained invariant that the layers have the same cache layout and request lifecycle; otherwise each layer needs its own mapping. KV payloads remain layer-specific.

## Invocation shape

The schema is defined in `csrc/pytorch_extensions.cpp`:

```python
torch.ops.npu.fused_scatter_copy_sparse_flash_attention(
    query, query_rope, combined_hbm_cache,
    actual_seq_lengths_query, actual_seq_lengths_kv, num_cache_tokens,
    topk_src_ids, topk_dst_slots, topk_miss_counts,
    miss_src_ids, miss_dst_slots, miss_counts,
    hbm_block_table, dram_block_table, dram_kv_dev_ptr,
    attention_out, scale_value,
)
```

`dram_kv_dev_ptr` is the registered device address for the manager's combined mapped host rows. The wrapper derives the KPE address from the CKV base and element width; do not pass an arbitrary CPU pointer.

## Upstream provenance

The inherited MLA implementation originated from `vllm-project/vllm-ascend` commit `85910cd528e2e260848e58e6337adf35cfd5d982`, under the CANN Open Software License Agreement 2.0. `UPSTREAM_PROVENANCE.json` records every original file hash and whether that path is retained in this source tree. The pristine 19-file snapshot remains in the task artifact archive with SHA-256 `ebd771541c27c4f9a5962ee4826446a378e045fba0f164b994d185ab49dcbe4b`; it is not bundled in this source folder. The retained upstream implementation files are `first_fill_scatter_copy_kernel.h` and the four MLA common/kernel/service headers. The upstream Torch adapter and unused OPP/ACLNN host/kernel sources are preserved only in the task artifact archive.

## Validation boundary

Native combined-layout Q1 integration tests are in `tests/test_copy_sfa_combined.py`; dynamic TND operator coverage is in `tests/test_copy_sfa_dynamic_tnd.py`; planner coverage is in `tests/test_sparsity_driven_kv_cache_planner.py`. The dynamic TND test matrix is adapted from vllm-ascend PR #16640 head `3326f350c0de30a5170b86ecfce851322e3216e6`, preserving its head counts, query counts, cache budgets, golden attention formula, tail visibility, payload checks, error threshold, and graph/eager equality. It additionally covers mixed first-fill and steady misses shared across MTP query rows, tail page crossings, and poisoned future tail rows to check causal visibility. Run these tests against a built extension on an available A3 NPU; no runtime pass is claimed here. CPU planner-reference tests do not validate the AscendC kernel. This interface makes no claim of split-source support, arbitrary cache dimensions, or performance improvements beyond separately reported measurements.
