import importlib.util
from pathlib import Path

import pytest
import torch

path = (
    Path(__file__).resolve().parents[1]
    / "python/sgl_kernel_npu/sgl_kernel_npu/sparsity_driven_kv_offload/full_visible_planner.py"
)
spec = importlib.util.spec_from_file_location("full_visible_planner", path)
planner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(planner)


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("device", ["cpu", "npu"])
def test_fully_visible_plan_tracks_one_union_across_query_routes(width, device):
    if device == "npu":
        pytest.importorskip("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("No NPU")
    top_k, capacity, context = 2048, 8192, 32768
    batch = 3
    dtype = torch.int32
    request_rows = torch.tensor([1, 2, 3], dtype=dtype, device=device)
    sparse = torch.tensor([True, False, False], device=device)
    source_map = torch.full((4, context + 1), -1, dtype=dtype, device=device)
    inverse = torch.full((4, capacity + 1), -1, dtype=dtype, device=device)
    sources = torch.empty((batch * width, top_k), dtype=dtype, device=device)
    destinations = torch.empty_like(sources)
    row_misses = torch.empty(batch * width, dtype=dtype, device=device)
    miss_sources = torch.empty((batch, 32768), dtype=dtype, device=device)
    miss_slots = torch.empty_like(miss_sources)
    counts = torch.empty(batch, dtype=dtype, device=device)
    pages = {}

    for step, warm in enumerate((False, True, True, False)):
        length = capacity + 17 + step
        topk = torch.full((batch, width, top_k), -1, dtype=dtype)
        for route in range(width):
            start = ((step % 3) * 1703 + route * 1399) % (capacity - top_k)
            topk[0, route] = torch.arange(start, start + top_k, dtype=dtype)
            topk[0, route, -2:] = torch.tensor(
                [length - width + route, length]
            )  # Latest visible row and an invisible row.
            topk[1, route, :2] = torch.tensor([0, route + 1])
            topk[2, route, 0] = 0  # Graph padding must not select a real row.
        previous = set(pages.values()) if warm else set()
        planner.plan_full_visible_cache(
            topk.to(device),
            torch.tensor([length, width + 2, 0], dtype=dtype, device=device),
            request_rows,
            sparse,
            torch.tensor([warm, False, False], device=device),
            source_map,
            inverse,
            sources,
            destinations,
            row_misses,
            miss_sources,
            miss_slots,
            counts,
        )
        union = set(topk[0, :, :-1].flatten().tolist())
        count, short_count, padding_count = counts.tolist()
        misses = miss_sources[0, :count].tolist()
        slots = miss_slots[0, :count].tolist()
        assert len(set(slots)) == count and all(0 <= slot < capacity for slot in slots)
        assert len(set(misses)) == count and set(misses) == union - previous
        assert short_count == padding_count == 0
        if not warm:
            pages.clear()
        pages.update(zip(slots, misses))
        for route in range(width):
            assert sources[route, :-1].tolist() == topk[0, route, :-1].tolist()
            assert sources[route, -1].item() == -1
            assert [pages[slot] for slot in destinations[route, :-1].tolist()] == topk[
                0, route, :-1
            ].tolist()
            assert row_misses[route].item() == len(
                set(topk[0, route, :-1].tolist()) - previous
            )
            assert destinations[width + route, :3].tolist() == [0, route + 1, -1]
        assert destinations[2 * width :].eq(-1).all()
        assert inverse[0].eq(-1).all()
        assert inverse[3].eq(-1).all()


@pytest.mark.parametrize("width", [1, 4])
def test_a5_quant_li_full_visible_copysfa_matches_native_attention(width):
    """Compare a mixed short/long batch with the native packed C8 attention."""
    torch_npu = pytest.importorskip("torch_npu")
    import acl
    import sgl_kernel_npu  # noqa: F401

    if not torch.npu.is_available() or acl.rt.get_device_info(0, 601) != (3510, 0):
        pytest.skip("Ascend950 required")
    device = "npu:0"
    dtype = torch.int32
    capacity, page, top_k = 8192, 128, 2048
    lengths_cpu = [16, capacity + 7]
    src_pages, dst_pages = (max(lengths_cpu) + page - 1) // page, capacity // page
    total_tokens = 2 * src_pages * page
    rng = torch.Generator().manual_seed(7409)

    def noise(*shape):
        return torch.randn(shape, generator=rng).mul_(0.125)

    li = torch.ops.npu.npu_quant_lightning_indexer
    lengths = torch.tensor(lengths_cpu, dtype=dtype, device=device)
    ends = torch.tensor([width, width * 2], dtype=dtype, device=device)
    dram_pages = torch.arange(2 * src_pages, dtype=dtype, device=device).view(2, -1)
    hbm_pages = torch.arange(2 * dst_pages, dtype=dtype, device=device).view(2, -1)
    li_query = noise(2 * width, 32, 128).to(torch.float8_e4m3fn).to(device)
    li_key = noise(2 * src_pages, page, 1, 128).to(torch.float8_e4m3fn).to(device)
    topk = (
        li(
            query=li_query,
            key=li_key,
            weights=torch.ones((2 * width, 32), dtype=torch.bfloat16, device=device),
            query_dequant_scale=torch.ones((2 * width, 32), device=device),
            key_dequant_scale=torch.ones((2 * src_pages, page, 1), device=device),
            actual_seq_lengths_query=ends,
            actual_seq_lengths_key=lengths,
            block_table=dram_pages,
            layout_query="TND",
            layout_key="PA_BSND",
            sparse_count=top_k,
            sparse_mode=3,
            query_quant_mode=0,
            key_quant_mode=0,
        )
        .reshape(2, width, top_k)
        .int()
    )
    topk_cpu = topk.cpu()
    for batch, length in enumerate(lengths_cpu):
        for q in range(width):
            visible = length - width + q + 1
            selected = topk_cpu[batch, q]
            present = selected[selected >= 0]
            assert present.numel() == min(top_k, visible)
            assert present.min() >= 0 and present.max() < visible

    source_map = torch.full((4, src_pages * page + 1), -1, dtype=dtype, device=device)
    inverse = torch.full((4, capacity + 1), -1, dtype=dtype, device=device)
    sources = torch.empty((2 * width, top_k), dtype=dtype, device=device)
    destinations = torch.empty_like(sources)
    per_query = torch.empty(2 * width, dtype=dtype, device=device)
    miss_sources = torch.empty((2, 32768), dtype=dtype, device=device)
    miss_slots = torch.empty_like(miss_sources)
    miss_counts = torch.empty(2, dtype=dtype, device=device)
    planner.plan_full_visible_cache(
        topk,
        lengths,
        torch.tensor([1, 2], dtype=dtype, device=device),
        torch.tensor([False, True], device=device),
        torch.zeros(2, dtype=torch.bool, device=device),
        source_map,
        inverse,
        sources,
        destinations,
        per_query,
        miss_sources,
        miss_slots,
        miss_counts,
    )

    packed = (
        torch.cat(
            (
                noise(total_tokens, 512).to(torch.float8_e4m3fn).view(torch.int8),
                noise(total_tokens, 64).to(torch.bfloat16).view(torch.int8),
                torch.ones((total_tokens, 4), dtype=torch.float32).view(torch.int8),
            ),
            dim=-1,
        )
        .view(-1, page, 1, 656)
        .to(device)
    )
    hbm = torch.zeros((2 * dst_pages, page, 1, 656), dtype=torch.int8, device=device)
    # Simulate the manager's short history/current staging. CopySFA fills
    # the long request's cold union from the paged source.
    hbm[:1].copy_(packed[:1])
    query = noise(2 * width, 8, 576).to(device=device, dtype=torch.bfloat16)
    output = query.new_empty((2 * width, 8, 512))
    torch.ops.npu.fused_quant_scatter_copy_sparse_flash_attention_v2(
        query[..., :512].contiguous(),
        query[..., 512:].contiguous(),
        hbm,
        packed,
        packed.data_ptr(),
        hbm_pages,
        dram_pages,
        destinations.unsqueeze(1),
        sources.unsqueeze(1),
        per_query,
        ends,
        torch.tensor([16, capacity + width - 1], dtype=dtype, device=device),
        torch.tensor([0, capacity], dtype=dtype, device=device),
        miss_sources,
        miss_slots,
        miss_counts,
        576**-0.5,
        output,
    )
    cache = packed.view(torch.float8_e4m3fn)
    reference = torch_npu.npu_kv_quant_sparse_flash_attention(
        query=query,
        key=cache,
        value=cache,
        sparse_indices=topk.reshape(-1, 1, top_k),
        scale_value=576**-0.5,
        key_quant_mode=2,
        value_quant_mode=2,
        key_dequant_scale=None,
        value_dequant_scale=None,
        actual_seq_lengths_query=ends,
        actual_seq_lengths_kv=lengths,
        block_table=dram_pages,
        sparse_block_size=1,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=3,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=128,
        rope_head_dim=64,
    )
    if isinstance(reference, tuple):
        reference = reference[0]
    torch.testing.assert_close(
        output.cpu().float(), reference.cpu().float(), atol=0.008, rtol=0.05
    )
