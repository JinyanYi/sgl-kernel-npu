// SPDX-License-Identifier: Apache-2.0
// Register the unchanged donor A5 adapter under an isolated Torch namespace.
#include <cstdlib>
#include <dlfcn.h>
#include <limits.h>
#include <string>

#include <torch/extension.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include "aclnn_torch_adapter/op_api_common.h"

// Keep donor inline C++ symbols separate from an installed vLLM extension.
#define vllm_ascend a5_original_vllm
#include "attention/fused_quant_lightning_indexer_manage/fused_quant_lightning_indexer_manage_torch_adpt.h"
#undef vllm_ascend

thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;

// Use precisely the donor's resolver and report the defining DSO for both
// symbols. This probe does not allocate tensors or launch an operator.
c10::Dict<std::string, std::string> resolve_libraries() {
  c10::Dict<std::string, std::string> result;
  for (const char *symbol :
       {"aclnnFusedQuantLightningIndexerManageWithStridesGetWorkspaceSize",
        "aclnnFusedQuantLightningIndexerManageWithStrides"}) {
    void *address = GetOpApiFuncAddr(symbol);
    TORCH_CHECK(address != nullptr,
                "Could not resolve donor ACLNN symbol: ", symbol);
    Dl_info info{};
    TORCH_CHECK(dladdr(address, &info) != 0 && info.dli_fname != nullptr,
                "Could not identify defining library for ", symbol);
    char resolved[PATH_MAX];
    TORCH_CHECK(realpath(info.dli_fname, resolved) != nullptr,
                "Could not resolve library path for ", symbol, ": ",
                info.dli_fname);
    result.insert(std::string(symbol), std::string(resolved));
  }
  return result;
}

TORCH_LIBRARY(a5_original_lim, ops) {
  ops.def("resolve_libraries() -> Dict(str, str)", &resolve_libraries);
  ops.def("run(Tensor index_weights, Tensor query_dequant_scale, Tensor query, "
          "Tensor index_key_dequant_scale, Tensor index_key_cache, Tensor "
          "index_block_table, "
          "Tensor actual_seq_lengths_query, Tensor actual_seq_lengths_key, "
          "Tensor offload_seq_lengths_key, Tensor num_cache_tokens, Tensor "
          "request_state, "
          "Tensor req_pool_entries, Tensor(a!) cache_slots_pool, Tensor(b!) "
          "topk_src_ids, "
          "Tensor(c!) topk_dst_slots, Tensor(d!) topk_miss_counts, Tensor(e!) "
          "miss_src_ids, "
          "Tensor(f!) miss_dst_slots, Tensor(g!) miss_counts) -> ()");
  ops.impl("run", torch::kPrivateUse1,
           &a5_original_vllm::npu_fused_quant_lightning_indexer_manage);
}
