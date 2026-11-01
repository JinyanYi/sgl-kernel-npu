import os
import pathlib
from functools import lru_cache, wraps

# Configure OPP discovery before torch_npu or the native adapter captures it.
# isort: off
from ._original_opp import activate, verify_library

_original_opp_manifest = activate(pathlib.Path(__file__).resolve().parent)

import torch
import torch_npu

# isort: on


def _load_sgl_kernel_npu():
    npu_path = pathlib.Path(__file__).parents[0]
    so_path = os.path.join(npu_path, "lib", "libsgl_kernel_npu.so")
    torch.ops.load_library(so_path)


_load_sgl_kernel_npu()

if hasattr(torch.ops.npu, "a5_c8_backend"):
    _a5_c8_backend = torch.ops.npu.a5_c8_backend()
    if _a5_c8_backend == "original":
        if _original_opp_manifest is None:
            raise ImportError(
                "the original A5 C8 backend requires its wheel-bundled OPP"
            )
        for _symbol, _library in dict(
            torch.ops.npu.resolve_a5_c8_original_libraries()
        ).items():
            verify_library(
                pathlib.Path(__file__).resolve().parent,
                _original_opp_manifest,
                _library,
            )
