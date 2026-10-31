"""CPU checks for the A5 operator A/B inputs and hit-rate accounting."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

path = Path(__file__).parents[1] / "benchmark/copysfa_a5_ab/fixture.py"
spec = importlib.util.spec_from_file_location("copysfa_ab_fixture", path)
fixture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fixture
spec.loader.exec_module(fixture)


def test_matrix_is_complete_and_unique():
    configs = fixture.matrix()
    assert len(configs) == len({c.name for c in configs}) == 24
    assert {(c.query_width, c.hit_percent, c.graph) for c in configs} == {
        (q, hit, graph) for q in (1, 4) for hit in (91, 95) for graph in (False, True)
    }


@pytest.mark.parametrize("width", [1, 4])
@pytest.mark.parametrize("hit", [91, 95])
def test_hit_rates_payloads_and_causality(width, hit):
    config = fixture.Config(
        capacity=8192,
        query_width=width,
        hit_percent=hit,
        batch=2,
        heads=8,
        prefix=16384,
        padding=True,
        shared_dram=True,
    )
    data, metrics = fixture.make_fixture(config)
    assert len(metrics) == 1
    metric = metrics[0]
    assert (
        abs(metric["selected_prefix_union_hit_rate"] - hit / 100)
        <= 0.5 / metric["selected_prefix_union"]
    )
    count = int(data["misses"][0])
    assert count == metric["unique_prefix_misses"]
    assert count == int(data["route_misses"][:width].sum())
    src = data["miss_src"][0, 0, :count].long()
    dst = data["miss_dst"][0, 0, :count].long()
    assert len(src.unique()) == len(dst.unique()) == count
    source_bytes = data["dram"].view(-1, 656)[
        fixture.physical(data["dram_table"], 0, src)
    ]
    destination = fixture.physical(data["hbm_table"], 0, dst)
    torch.testing.assert_close(
        data["expected"].view(-1, 656)[destination], source_bytes, rtol=0, atol=0
    )
    touched = torch.zeros(data["expected"].shape[0] * 128, dtype=torch.bool)
    touched[destination] = True
    torch.testing.assert_close(
        data["initial"].view(-1, 656)[~touched],
        data["expected"].view(-1, 656)[~touched],
        rtol=0,
        atol=0,
    )
    for route in range(width):
        ids = data["src"][route, 0].long()
        slots = data["dst"][route, 0].long()
        visible = int(data["lengths"][0]) - width + route + 1
        assert bool(((ids >= 0) & (ids < visible)).all())
        tail = ids >= config.prefix
        assert torch.equal(slots[tail], config.capacity + ids[tail] - config.prefix)
        assert bool((slots[~tail] < config.capacity).all())
        torch.testing.assert_close(
            data["expected"].view(-1, 656)[
                fixture.physical(data["hbm_table"], 0, slots)
            ],
            data["dram"].view(-1, 656)[fixture.physical(data["dram_table"], 0, ids)],
            rtol=0,
            atol=0,
        )
    assert bool((data["src"][width:] == -1).all())
    assert bool((data["dst"][width:] == -1).all())
    assert bool((data["golden"][width:] == 0).all())
    assert float(data["golden"][:width].abs().mean()) > 0.1


def test_512_513_boundary_and_minimum_prefix_rejection():
    for count in (512, 513):
        data, metrics = fixture.make_fixture(
            fixture.Config(
                capacity=8192, query_width=4, batch=1, prefix=16384, forced_misses=count
            )
        )
        assert int(data["misses"][0]) == count
        assert metrics[0]["expected_copy_path"] == (
            "fused" if count == 512 else "union_copy"
        )
    with pytest.raises(ValueError, match="resident rows"):
        fixture.make_fixture(
            fixture.Config(
                capacity=8192, query_width=4, batch=1, prefix=8320, forced_misses=513
            )
        )


def test_cold_fill_copies_c_rows_and_overlapping_queries_count_unique_misses():
    data, metrics = fixture.make_fixture(
        fixture.Config(
            capacity=8192, query_width=4, batch=1, prefix=16384, forced_misses=8192
        )
    )
    assert int(data["misses"][0]) == 8192
    assert metrics[0]["cold_fill"] and metrics[0]["selected_prefix_union_hit_rate"] == 0
    destinations = fixture.physical(data["hbm_table"], 0, torch.arange(8192))
    assert bool((data["initial"].view(-1, 656)[destinations] == -91).all())
    for hit in (91, 95):
        data, metrics = fixture.make_fixture(
            fixture.Config(
                capacity=8192,
                query_width=4,
                hit_percent=hit,
                batch=1,
                prefix=16384,
                query_overlap_percent=90,
            )
        )
        metric = metrics[0]
        assert metric["unique_prefix_misses"] < sum(metric["route_miss_counts"])
        assert (
            abs(metric["selected_prefix_union_hit_rate"] - hit / 100)
            <= 0.5 / metric["selected_prefix_union"]
        )
        assert metric["unique_prefix_misses"] == int(data["misses"][0])


def test_physical_page_remapping_preserves_every_selected_payload():
    data, _ = fixture.make_fixture(
        fixture.Config(
            capacity=8192, query_width=4, batch=2, prefix=16384, shared_dram=True
        )
    )
    changed = fixture.remap_physical_pages(data)
    assert torch.equal(data["dram_table"][0], data["dram_table"][1])
    assert not torch.equal(changed["hbm_table"], data["hbm_table"])
    assert not torch.equal(changed["dram_table"], data["dram_table"])
    for row in range(2):
        sources = data["src"][row * 4 : (row + 1) * 4].long().flatten()
        slots = data["dst"][row * 4 : (row + 1) * 4].long().flatten()
        for cache, table, indices in (
            ("dram", "dram_table", sources),
            ("expected", "hbm_table", slots),
        ):
            torch.testing.assert_close(
                data[cache].view(-1, 656)[fixture.physical(data[table], row, indices)],
                changed[cache].view(-1, 656)[
                    fixture.physical(changed[table], row, indices)
                ],
                rtol=0,
                atol=0,
            )
