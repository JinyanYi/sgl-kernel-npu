"""A3/A5 registered-DRAM adapter checks without executing A5 attention."""

import os

import pytest
import torch

pytest.importorskip("torch_npu")


@pytest.mark.parametrize("graph_replay", [False, True])
def test_original_reference_header_uses_registered_host_bytes(graph_replay):
    library = os.environ.get("ORIGINAL_COPYSFA_BINDING")
    if not library:
        pytest.skip("Build the private original reference and source its paths.env")
    from sgl_kernel_npu.sparsity_driven_kv_offload import (
        create_shm_tensor,
        free_shm,
        unidex_copy_inplace,
    )

    assert torch.npu.is_available()
    torch.npu.set_device(0)
    torch.ops.load_library(library)
    host, _, pointer = create_shm_tensor((2, 128, 1, 656), torch.int8, device_id=0)
    try:
        generator = torch.Generator().manual_seed(18490)
        expected = torch.randint(
            -128, 128, host.shape, dtype=torch.int8, generator=generator
        )
        host.copy_(expected)
        query = torch.zeros(1, 8, 576, dtype=torch.bfloat16, device="npu")
        view = torch.ops.a5_original_copysfa.registered_dram_view(host, query, pointer)
        assert view.data_ptr() == pointer
        assert (
            torch.ops.a5_original_copysfa.validate_registered_dram_view(view) == pointer
        )
        output = torch.empty(host.shape, dtype=torch.int8, device="npu")
        ids = torch.arange(256, dtype=torch.int64, device="npu")
        valid = torch.ones(256, dtype=torch.bool, device="npu")

        def copy():
            unidex_copy_inplace(view, output, ids, ids, valid, 2, 2, block_dim=8)

        copy()
        torch.npu.synchronize()
        graph = None
        if graph_replay:
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                copy()
        for step in range(16):
            host.copy_((expected.to(torch.int16) + step).to(torch.int8))
            graph.replay() if graph is not None else copy()
            torch.npu.synchronize()
            torch.testing.assert_close(output.cpu(), host, rtol=0, atol=0)
    finally:
        torch.npu.synchronize()
        free_shm()
