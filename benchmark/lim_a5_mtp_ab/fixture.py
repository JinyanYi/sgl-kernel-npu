"""Deterministic CPU-only fixtures for A5 LIM MTP comparisons."""

from __future__ import annotations

import hashlib
import json
import random
from itertools import product
from typing import Sequence


def make_matrix() -> list[dict[str, object]]:
    """Return the 16 requested MTP, execution, filler, and hit-rate cases."""
    matrix = []
    for mtp, mode, policy, hit_percent in product(
        (0, 3), ("eager", "graph"), ("low", "random"), (91, 95)
    ):
        matrix.append(
            {
                "mtp": mtp,
                "q": mtp + 1,
                "mode": mode,
                "policy": policy,
                "hit_percent": hit_percent,
                "hit_rate_basis": "selected_prefix_union",
            }
        )
    return matrix


def target_union_misses(q: int, hit: int, k: int = 2048) -> int:
    """Return rounded per-request misses for a Q*K selected-union target."""
    if isinstance(q, bool) or not isinstance(q, int) or q <= 0:
        raise ValueError("q must be a positive integer")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer")
    if isinstance(hit, bool) or not isinstance(hit, int) or not 0 <= hit <= 100:
        raise ValueError("hit must be an integer percentage from 0 to 100")
    return round(q * k * (1 - hit / 100))


def _normalize_reference(reference: Sequence[Sequence[Sequence[int]]]):
    if not reference:
        raise ValueError("reference must contain at least one request")
    normalized = []
    route_count = None
    topk = None
    for request_index, request in enumerate(reference):
        if not request:
            raise ValueError(f"request {request_index} must contain at least one route")
        if route_count is None:
            route_count = len(request)
        elif len(request) != route_count:
            raise ValueError("all requests must have the same route count")
        routes = []
        for route_index, route in enumerate(request):
            values = list(route)
            if not values:
                raise ValueError(
                    f"route {request_index}:{route_index} must not be empty"
                )
            if topk is None:
                topk = len(values)
            elif len(values) != topk:
                raise ValueError("all routes must have the same TopK length")
            if any(
                isinstance(source, bool) or not isinstance(source, int) or source < 0
                for source in values
            ):
                raise ValueError("source IDs must be non-negative integers")
            if len(set(values)) != len(values):
                raise ValueError(
                    f"route {request_index}:{route_index} contains duplicate source IDs"
                )
            routes.append(values)
        normalized.append(routes)
    return normalized, route_count, topk


def validate_causal_boundary(reference, final_lengths: int | Sequence[int]) -> bool:
    """Validate Q-route references against per-route causal visibility.

    For final KV length S and route r in Q, the visible token count is
    S - Q + r + 1. ``reference`` is nested as [B][Q][K].
    """
    normalized, q, _ = _normalize_reference(reference)
    if isinstance(final_lengths, bool):
        raise ValueError("final_lengths must be a positive length or sequence")
    lengths = (
        [final_lengths] * len(normalized)
        if isinstance(final_lengths, int)
        else list(final_lengths)
    )
    if len(lengths) != len(normalized):
        raise ValueError("final_lengths must have one value per request")
    for request_index, (routes, final_length) in enumerate(zip(normalized, lengths)):
        if (
            isinstance(final_length, bool)
            or not isinstance(final_length, int)
            or final_length < q
        ):
            raise ValueError("each final length must be an integer >= Q")
        for route_index, route in enumerate(routes):
            visible = final_length - q + route_index + 1
            if any(source >= visible for source in route):
                raise ValueError(
                    f"request {request_index} route {route_index} contains a source outside its causal boundary"
                )
    return True


def _miss_counts(misses: int | Sequence[int], batch: int) -> list[int]:
    if isinstance(misses, bool):
        raise ValueError("misses must be an integer or sequence of integer counts")
    counts = [misses] * batch if isinstance(misses, int) else list(misses)
    if len(counts) != batch:
        raise ValueError("misses must be one count or one count per request")
    if any(
        isinstance(count, bool) or not isinstance(count, int) or count < 0
        for count in counts
    ):
        raise ValueError("miss counts must be non-negative integers")
    return counts


def build_residency(
    reference,
    prefix_tokens: int,
    capacity: int,
    misses: int | Sequence[int],
    policy: str,
    seed: int,
) -> dict:
    """Build stable selected-hit slots and policy-specific nonselected fillers.

    ``misses`` is the number of selected prefix sources to leave nonresident
    for each request. A seed-shuffled prefix union makes miss sets nested when
    callers use the same seed with a smaller and larger miss target.
    """
    normalized, q, topk = _normalize_reference(reference)
    if (
        isinstance(prefix_tokens, bool)
        or not isinstance(prefix_tokens, int)
        or prefix_tokens <= 0
    ):
        raise ValueError("prefix_tokens must be a positive integer")
    if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
        raise ValueError("capacity must be a positive integer")
    if capacity > prefix_tokens:
        raise ValueError("capacity cannot exceed the prefix token count")
    if policy not in ("low", "random"):
        raise ValueError("policy must be 'low' or 'random'")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    miss_counts = _miss_counts(misses, len(normalized))
    maps, missing, request_stats = [], [], []
    occurrence_total = 0
    occurrence_misses_total = 0
    digest_payload = []
    total_selected_union = 0
    for request_index, (routes, miss_count) in enumerate(zip(normalized, miss_counts)):
        prefix_union = sorted(
            {source for route in routes for source in route if source < prefix_tokens}
        )
        prefix_union_set = set(prefix_union)
        total_selected_union += len(prefix_union)
        tail_union = {
            source for route in routes for source in route if source >= prefix_tokens
        }
        if miss_count > len(prefix_union):
            raise ValueError(
                f"request {request_index} miss count exceeds its selected prefix union"
            )
        chooser = random.Random(seed + request_index)
        shuffled_union = prefix_union.copy()
        chooser.shuffle(shuffled_union)
        missed = sorted(shuffled_union[:miss_count])
        missed_set = set(missed)
        selected_hits = [source for source in prefix_union if source not in missed_set]
        if len(selected_hits) > capacity:
            raise ValueError(
                f"request {request_index} selected prefix hits exceed capacity"
            )

        remaining_slots = capacity - len(selected_hits)
        filler_candidates = [
            source for source in range(prefix_tokens) if source not in prefix_union_set
        ]
        if remaining_slots > len(filler_candidates):
            raise ValueError(
                f"request {request_index} has too few nonselected prefix filler tokens"
            )
        if policy == "low":
            fillers = filler_candidates[:remaining_slots]
        else:
            filler_rng = random.Random(seed ^ (0x5A17 + request_index))
            fillers = filler_rng.sample(filler_candidates, remaining_slots)

        source_to_slot = {
            source: slot for slot, source in enumerate(selected_hits + fillers)
        }
        if len(source_to_slot) != capacity or set(source_to_slot.values()) != set(
            range(capacity)
        ):
            raise AssertionError("residency map is not a C-slot bijection")
        if missed_set.intersection(source_to_slot):
            raise AssertionError("a selected miss was assigned a resident slot")
        if any(source >= prefix_tokens for source in source_to_slot):
            raise AssertionError(
                "a tail source was inserted into the prefix residency map"
            )

        route_occurrences = sum(len(route) for route in routes)
        request_occurrence_misses = sum(
            source < prefix_tokens and source in missed_set
            for route in routes
            for source in route
        )
        occurrence_total += route_occurrences
        occurrence_misses_total += request_occurrence_misses
        maps.append(source_to_slot)
        missing.append(missed)
        request_stats.append(
            {
                "selected_prefix_union": len(prefix_union),
                "selected_prefix_misses": len(missed),
                "selected_prefix_hits": len(selected_hits),
                "selected_tail_union": len(tail_union),
                "capacity": capacity,
                "filler_count": len(fillers),
                "route_occurrences": route_occurrences,
                "route_occurrence_misses": request_occurrence_misses,
                "route_occurrence_hit_rate": (
                    1 - request_occurrence_misses / route_occurrences
                    if route_occurrences
                    else 1.0
                ),
            }
        )
        digest_payload.append(
            {"map": sorted(source_to_slot.items()), "missing": missed}
        )

    digest = hashlib.sha256(
        json.dumps(digest_payload, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
    ).hexdigest()
    return {
        "maps": maps,
        "missing": missing,
        "stats": {
            "hit_rate_basis": "selected_prefix_union",
            "requests": request_stats,
            "selected_prefix_union": total_selected_union,
            "selected_prefix_misses": sum(miss_counts),
            "selected_prefix_union_target_hit_rate": (
                1 - sum(miss_counts) / (q * topk * len(normalized))
            ),
            "unique_selected_prefix_hit_rate": (
                1 - sum(miss_counts) / total_selected_union
                if total_selected_union
                else 1.0
            ),
            "route_occurrences": occurrence_total,
            "route_occurrence_misses": occurrence_misses_total,
            "route_occurrence_hit_rate": (
                1 - occurrence_misses_total / occurrence_total
                if occurrence_total
                else 1.0
            ),
            "tail_sources_are_hits_and_not_resident": True,
        },
        "fixture_sha256": digest,
    }
