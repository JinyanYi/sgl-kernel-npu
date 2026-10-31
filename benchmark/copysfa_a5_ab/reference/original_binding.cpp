// SPDX-License-Identifier: Apache-2.0
// Isolate the unchanged PR18490 Torch adapter and its original ACLNN/OPP.
#include <cstdlib>
#include <dlfcn.h>
#include <limits.h>
#include <string>

#include <ATen/ATen.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/NPUStorageImpl.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>

#include "aclnn_torch_adapter/op_api_common.h"

#define vllm_ascend a5_original_copysfa_vllm
#include "attention/fused_quant_scatter_copy_sparse_flash_attention/fused_quant_scatter_copy_sparse_flash_attention_torch_adpt.h"
#undef vllm_ascend

thread_local char g_hashBuf[kHashBufSize];
thread_local int g_hashOffset = 0;

c10::Dict<std::string, std::string> resolve_libraries() {
  c10::Dict<std::string, std::string> result;
  for (const char *symbol :
       {"aclnnFusedQuantScatterCopySparseFlashAttentionGetWorkspaceSize",
        "aclnnFusedQuantScatterCopySparseFlashAttention"}) {
    void *address = GetOpApiFuncAddr(symbol);
    TORCH_CHECK(address != nullptr,
                "Could not resolve donor ACLNN symbol: ", symbol);
    Dl_info info{};
    TORCH_CHECK(dladdr(address, &info) != 0 && info.dli_fname != nullptr,
                "Could not identify defining library for ", symbol);
    char resolved[PATH_MAX];
    TORCH_CHECK(realpath(info.dli_fname, resolved) != nullptr,
                "Could not resolve library path");
    result.insert(std::string(symbol), std::string(resolved));
  }
  return result;
}

at::Tensor registered_dram_view(const at::Tensor &owner,
                                const at::Tensor &query,
                                int64_t device_address) {
  TORCH_CHECK(owner.device().is_cpu() && owner.scalar_type() == at::kChar &&
                  owner.is_contiguous(),
              "DRAM owner must be contiguous CPU INT8 registered storage");
  TORCH_CHECK(query.device().is_privateuseone() && device_address > 0,
              "NPU query and registered address required");
  c10_npu::OptionalNPUGuard guard(query.device());
  // Only create a Tensor header for the already registered host address.
  // No HBM allocation/copy is allowed for this reference DRAM operand.
  // Generic from_blob creates a plain StorageImpl, which lacks the NPU
  // storage descriptor required by Torch-NPU and the original ACLNN adapter.
  auto view = at::empty({0}, query.options().dtype(at::kChar));
  auto *storage = view.storage().unsafeGetStorageImpl();
  auto *keeper = new at::Tensor(owner);
  storage->set_data_ptr(c10::DataPtr(
      reinterpret_cast<void *>(static_cast<uintptr_t>(device_address)), keeper,
      [](void *context) { delete static_cast<at::Tensor *>(context); },
      query.device()));
  storage->set_nbytes(owner.nbytes());
  storage->set_resizable(false);
  view.unsafeGetTensorImpl()->set_sizes_and_strides(owner.sizes(),
                                                    owner.strides());
  auto &desc = static_cast<torch_npu::NPUStorageImpl *>(storage)->npu_desc_;
  desc.base_sizes_ = owner.sizes().vec();
  desc.base_strides_ = owner.strides().vec();
  desc.storage_sizes_ = owner.sizes().vec();
  desc.origin_format_ = ACL_FORMAT_ND;
  desc.npu_format_ = ACL_FORMAT_ND;
  desc.data_type_ = view.dtype();
  return view;
}

int64_t validate_registered_dram_view(const at::Tensor &view) {
  TORCH_CHECK(view.device().is_privateuseone() &&
                  view.scalar_type() == at::kChar,
              "Expected an NPU INT8 header over registered DRAM");
  aclTensor *converted = ConvertType(view);
  TORCH_CHECK(converted != nullptr, "Original ACLNN tensor conversion failed");
  Release(converted);
  return reinterpret_cast<int64_t>(view.data_ptr());
}

TORCH_LIBRARY(a5_original_copysfa, ops) {
  ops.def("validate_registered_dram_view(Tensor view) -> int",
          &validate_registered_dram_view);
  ops.def("resolve_libraries() -> Dict(str, str)", &resolve_libraries);
  ops.def("registered_dram_view(Tensor owner, Tensor query, int "
          "device_address) -> Tensor",
          &registered_dram_view);
  ops.def("run(Tensor query, Tensor(a!) hbm_kv_bytes, Tensor dram_kv_bytes, "
          "Tensor hbm_block_table, "
          "Tensor dram_block_table, Tensor topk_dst_slots, Tensor "
          "topk_src_ids, Tensor topk_miss_counts, "
          "Tensor actual_seq_lengths_query, Tensor actual_seq_lengths_kv, "
          "Tensor num_cache_tokens, "
          "Tensor miss_src_ids, Tensor miss_dst_slots, Tensor miss_counts, "
          "float scale_value, "
          "Tensor(b!) attention_out) -> ()");
  ops.impl("run", torch::kPrivateUse1,
           &a5_original_copysfa_vllm::
               npu_fused_quant_scatter_copy_sparse_flash_attention);
}
