# A5 packed-C8 P/D companion branch

Paired framework: `JinyanYi/sglang`, branch
`feat/a5-pd-split-kv-20261104`, `benchmark/hisparse/a5_pd/README.md`.
Kernel base: `138319bddb54bec3333e92fb6d072a74c540293a`.

This branch retains the tested A5 original LIM, our native CopySFA performance
changes, and the existing packed 656-byte C8 host-pointer ABI. No operator source
or binding is changed for the P/D candidate. The A3 split-BF16 two-pointer ABI
from McZyWu `b47332ce` is deliberately not substituted for the A5 packed ABI.

The framework receives P's packed HBM pages in a page-aligned MF SHARED host
allocation and passes its GVA through the existing CopySFA DRAM pointer. Before
loading the model, run its `a5_pd/check.sh local` gate: actual MF byte readback
through AICore and the existing Q1/Q4 eager/graph LIM+CopySFA numerical oracle
with MF-backed storage. Then run the TP shared-GVA gate and same-input native
versus offload P/D smoke tests.

An already installed wheel built from the base above can be reused in a matching
CANN/torch runtime. A changed runtime ABI can require rebuilding. This checkout
does not supply an A5-compatible MemFabric/HDK/LingQu bundle. No A5 device or
inter-node validation was available when preparing the candidate; a local
source check is not a P/D or performance acceptance result.
