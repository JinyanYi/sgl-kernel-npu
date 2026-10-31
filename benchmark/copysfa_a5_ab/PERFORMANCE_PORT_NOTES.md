# Relationship to the A3 MTP performance fixes

The requested kernel base is `db77f6fd2db26b2e60a02c63f821121045482835`,
"Overlap steady MTP CopySFA after resident refresh". Its A3 implementation is
preserved on this branch. The earlier probe branch
`perf/copysfa-mtp-refresh-probe-20261031` ends at `837e5e0` and does not contain
that later production change.

The A5 C8 path uses a separate arch35 kernel. Retaining the A3 commit in Git
history does not mean the arch35 kernel executes its new
`resident_refresh_tail` branch. The A5 contract instead addresses the same
stale-current-KV problem through caller-owned tail storage:

- Draft has a complete native HBM KV pool and does not instantiate an offload
  manager. It does not incur target CopySFA or host history retrieval.
- Target current Q1/Q4 KV is staged from the current device tensor directly
  into its private circular HBM tail. Missing old tail bytes are restored
  separately; the current candidates do not need a host round trip.
- External C8 LIM must emit `C + source - prefix` tail slots as hits. Tail rows
  are excluded from the historical prefix pool and its miss-copy union. The
  framework rotates two HBM table pages to map those logical slots onto the
  physical ring.
- The pre/post metadata fusions remain available. Invalid/padding rows have
  zero copy counts and cannot force a cold-copy path for active requests.
- The A5 small-miss path retains PR18490's source-aware DRAM gather, owner
  tracking across queries, prefetch during hit work, and deduplicated HBM
  persistence. Those payload files retain their recorded upstream hashes.
- Native launch explicitly supplies batch scheduling, matching the original
  OPP's `SetScheduleMode(1)`. CANN's legacy generated stub otherwise uses a
  null launch configuration and loses an attribute attached to the inline
  implementation. Only CopySFA's host stub is redirected; the owned-launch
  submission and Tensor/workspace lifetimes are preserved.

There is an important retained upstream limitation: any request with more
than 512 unique misses selects the original batch-wide union-copy stage
before attention. The A3 production patch's more general hybrid refresh
dispatch was not transplanted into this C8 kernel. Removing that A5 fallback
would be a separate algorithm/performance change, requiring direct A5 parity
and replay tests. It is not necessary to match the selected vLLM baseline,
which has the same 512-miss gate.

The new matrix explicitly covers 512/513, cold C fills, disjoint and heavily
overlapping Q4 selections. It compares the unchanged original operator with
the port under identical plans. These measurements do not include framework
tail staging or metadata time, and do not establish end-to-end model speed.

The current priority is A5 accuracy. Performance ratios and raw samples are
retained, with a configurable warning instead of a default hard speed gate.
