"""Deterministic CopySFA plans with explicit selected-prefix C-pool hit rates."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass

import torch

K, ROW_BYTES, PAGE, COPY_CAPACITY = 2048, 656, 128, 32768


@dataclass(frozen=True)
class Config:
    capacity: int = 16384
    query_width: int = 4
    hit_percent: int = 95
    batch: int = 16
    heads: int = 8
    prefix: int = 65536
    dtype: str = "bfloat16"
    graph: bool = False
    shared_dram: bool = False
    padding: bool = False
    forced_misses: int | None = None
    query_overlap_percent: int = 0

    @property
    def name(self):
        suffix = ("_shared" if self.shared_dram else "") + (
            "_pad" if self.padding else ""
        )
        suffix += f"_m{self.forced_misses}" if self.forced_misses is not None else ""
        suffix += (
            f"_overlap{self.query_overlap_percent}"
            if self.query_overlap_percent
            else ""
        )
        return (
            f"c{self.capacity}_q{self.query_width}_hit{self.hit_percent}_"
            f"{'graph' if self.graph else 'eager'}_b{self.batch}_h{self.heads}_"
            f"p{self.prefix}_{self.dtype}{suffix}"
        )


def matrix(capacities=(8192, 16384, 32384), batch=16, heads=8):
    return [
        Config(c, q, hit, batch=batch, heads=heads, graph=graph)
        for c in capacities
        for q in (1, 4)
        for hit in (95, 91)
        for graph in (False, True)
    ]


def packed_rows(count):
    """Valid FP8/BF16/FP32 records; no NPU quantization during fixture creation."""
    positions = torch.arange(count, dtype=torch.int64)[:, None]
    nope = ((positions * 17 + torch.arange(512)[None] * 7) % 15 - 7).float() / 4 + 0.5
    nope = nope.to(torch.float8_e4m3fn).view(torch.uint8)
    rope = ((positions * 13 + torch.arange(64)[None] * 3) % 17 - 8).float() / 32
    # Exact BF16 base-256 digits identify physical rows beyond payload periods.
    for digit in range(3):
        rope[:, digit] = (((positions[:, 0] >> (8 * digit)) & 255).float() - 128) / 128
    rope = rope.to(torch.bfloat16).view(torch.uint8)
    scales = ((positions % 7 + torch.arange(4)[None] + 1).float() / 8).contiguous()
    return torch.cat((nope, rope, scales.view(torch.uint8)), dim=-1).view(torch.int8)


def unpack_rows(rows):
    raw = rows.contiguous().view(torch.uint8)
    nope = raw[:, :512].contiguous().view(torch.float8_e4m3fn).float()
    rope = raw[:, 512:640].contiguous().view(torch.bfloat16).float()
    scale = raw[:, 640:].contiguous().view(torch.float32).repeat_interleave(128, -1)
    value = (nope * scale).to(torch.bfloat16).float()
    return torch.cat((value, rope), -1), value


def physical(table, request, ids):
    ids = ids.long()
    return table[request, ids // PAGE].long() * PAGE + ids % PAGE


def fingerprint(tensors):
    result = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        result.update(name.encode())
        result.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        result.update(tensor.contiguous().view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def make_fixture(config: Config):
    c, q, b, h, prefix = (
        config.capacity,
        config.query_width,
        config.batch,
        config.heads,
        config.prefix,
    )
    if q not in (1, 4) or c < q * K or c > 32384 or c % PAGE:
        raise ValueError("Q1/Q4 needs aligned capacity >= Q*2048 and <=32384")
    if prefix % PAGE or prefix <= c or prefix + 128 > 262144:
        raise ValueError("Prefix must be page aligned, exceed C and fit source tables")
    if config.hit_percent not in (91, 95) or h not in (8, 16, 32, 64, 128) or b < 1:
        raise ValueError("Unsupported hit label, heads or batch")
    if config.dtype not in ("bfloat16", "float16"):
        raise ValueError("Compute dtype must be bfloat16 or float16")
    dtype = getattr(torch, config.dtype)
    generator = torch.Generator().manual_seed(18490 + c + q)
    tail_end = 129 if q == 4 else 127
    length, source_rows = prefix + tail_end, prefix + 256
    dram_pages, hbm_pages = source_rows // PAGE, (c + 256) // PAGE
    banks = 1 if config.shared_dram else b
    dram = packed_rows(banks * source_rows).view(-1, PAGE, 1, ROW_BYTES)
    dram_table = torch.empty(b, dram_pages, dtype=torch.int32)
    hbm_table = torch.empty(b, hbm_pages, dtype=torch.int32)
    for row in range(b):
        bank = 0 if config.shared_dram else row
        dram_table[row] = bank * dram_pages + torch.randperm(
            dram_pages, generator=generator
        )
        if config.shared_dram and row:
            dram_table[row].copy_(dram_table[0])
        hbm_table[row] = row * hbm_pages + torch.randperm(
            hbm_pages, generator=generator
        )
    query = torch.randn(b * q, h, 576, generator=generator).to(dtype) / 4
    initial = torch.full((b * hbm_pages, PAGE, 1, ROW_BYTES), -91, dtype=torch.int8)
    expected = initial.clone()
    src = torch.full((b * q, 1, K), -1, dtype=torch.int32)
    dst = torch.full_like(src, -1)
    route_misses = torch.zeros(b * q, dtype=torch.int32)
    miss_src = torch.full((b, 1, COPY_CAPACITY), -1, dtype=torch.int32)
    miss_dst = torch.full_like(miss_src, -1)
    misses = torch.zeros(b, dtype=torch.int32)
    lengths = torch.full((b,), length, dtype=torch.int32)
    cache_lengths = torch.full((b,), c + tail_end, dtype=torch.int32)
    golden = torch.zeros(b * q, h, 512)
    metrics = []
    dram_flat, init_flat, expected_flat = (
        dram.view(-1, ROW_BYTES),
        initial.view(-1, ROW_BYTES),
        expected.view(-1, ROW_BYTES),
    )
    for row in range(b):
        if config.padding and row == b // 2:
            lengths[row] = cache_lengths[row] = 0
            continue
        tail_counts = [tail_end - q + route + 1 for route in range(q)]
        common_count = (
            min(K - tail for tail in tail_counts) * config.query_overlap_percent // 100
        )
        if not 0 <= config.query_overlap_percent <= 100:
            raise ValueError("Query overlap percent must be in [0,100]")
        union_size = common_count + sum(K - tail - common_count for tail in tail_counts)
        miss_count = (
            config.forced_misses
            if config.forced_misses is not None
            else round(union_size * (100 - config.hit_percent) / 100)
        )
        cold = config.forced_misses == c
        if not cold and not 0 <= miss_count <= union_size:
            raise ValueError("Miss count exceeds selected-prefix union")
        chosen = torch.randperm(prefix, generator=generator)
        selected = chosen[:union_size]
        miss_ids = (
            chosen[:c]
            if cold
            else selected[torch.randperm(union_size, generator=generator)[:miss_count]]
        )
        hit_ids = selected[:0] if cold else selected[~torch.isin(selected, miss_ids)]
        if not cold and prefix < c + miss_count:
            raise ValueError(
                "Prefix must fit C resident rows plus distinct selected misses"
            )
        fillers = chosen[union_size : union_size + c - len(hit_ids)]
        residents = chosen[:c] if cold else torch.cat((hit_ids, fillers))
        residents = residents[torch.randperm(c, generator=generator)]
        slot_map = torch.full((prefix,), -1, dtype=torch.int32)
        slot_map[residents] = torch.arange(c, dtype=torch.int32)
        victim_slots = (
            slot_map[miss_ids].long()
            if cold
            else torch.nonzero(
                torch.isin(residents, fillers), as_tuple=False
            ).flatten()[:miss_count]
        )
        slot_map[miss_ids] = victim_slots.to(torch.int32)
        slots = torch.arange(c)
        if not cold:
            init_flat[physical(hbm_table, row, slots)] = dram_flat[
                physical(dram_table, row, residents)
            ]
        tail = torch.arange(tail_end)
        init_flat[physical(hbm_table, row, c + tail)] = dram_flat[
            physical(dram_table, row, prefix + tail)
        ]
        expected_flat[physical(hbm_table, row, torch.arange(c + tail_end))] = init_flat[
            physical(hbm_table, row, torch.arange(c + tail_end))
        ]
        expected_flat[physical(hbm_table, row, victim_slots)] = dram_flat[
            physical(dram_table, row, miss_ids)
        ]
        misses[row] = miss_count
        miss_src[row, 0, :miss_count] = miss_ids.to(torch.int32)
        miss_dst[row, 0, :miss_count] = victim_slots.to(torch.int32)
        offset, route_counts = common_count, []
        for route, tail_count in enumerate(tail_counts):
            private_count = K - tail_count - common_count
            prefix_ids = torch.cat(
                (selected[:common_count], selected[offset : offset + private_count])
            )
            offset += private_count
            is_miss = torch.isin(prefix_ids, miss_ids)
            # The original fused gather reads only the miss prefix from DRAM.
            ordered = torch.cat((prefix_ids[is_miss], prefix_ids[~is_miss]))
            ids = torch.cat((ordered, prefix + torch.arange(tail_count))).long()
            slots = torch.cat((slot_map[ordered], c + torch.arange(tail_count))).to(
                torch.int32
            )
            index = row * q + route
            src[index, 0], dst[index, 0] = ids.to(torch.int32), slots
            route_misses[index] = int(is_miss.sum())
            route_counts.append(int(is_miss.sum()))
            key, value = unpack_rows(dram_flat[physical(dram_table, row, ids)])
            golden[index] = (
                torch.softmax(query[index].float() @ key.T / math.sqrt(576), -1) @ value
            )
        metrics.append(
            dict(
                request=row,
                cold_fill=cold,
                request_copy_count=miss_count,
                selected_prefix_union=union_size,
                unique_prefix_misses=union_size if cold else miss_count,
                selected_prefix_union_hit_rate=(
                    0.0 if cold else 1 - miss_count / union_size
                ),
                route_miss_counts=route_counts,
                prefix_route_hit_rates=[
                    1 - m / (K - t) for m, t in zip(route_counts, tail_counts)
                ],
                all_route_hit_rate=1 - sum(route_counts) / (q * K),
                legacy_qk_label_hit_rate=1 - miss_count / (q * K),
                expected_copy_path="fused" if miss_count <= 512 else "union_copy",
            )
        )
    tensors = dict(
        query=query,
        initial=initial,
        expected=expected,
        dram=dram,
        hbm_table=hbm_table,
        dram_table=dram_table,
        src=src,
        dst=dst,
        route_misses=route_misses,
        query_ends=torch.arange(q, (b + 1) * q, q, dtype=torch.int32),
        lengths=lengths,
        cache_lengths=cache_lengths,
        miss_src=miss_src,
        miss_dst=miss_dst,
        misses=misses,
        golden=golden,
    )
    return tensors, metrics


def remap_physical_pages(data):
    """Change physical addresses while preserving all logical KV and outputs."""
    result = dict(data)
    for bank, keys, shift in (
        ("hbm_table", ("initial", "expected"), 1),
        ("dram_table", ("dram",), 3),
    ):
        permutation = torch.arange(data[keys[0]].shape[0]).roll(shift)
        result[bank] = permutation[data[bank].long()].to(torch.int32)
        for key in keys:
            value = torch.empty_like(data[key])
            value[permutation] = data[key]
            result[key] = value
    return result
