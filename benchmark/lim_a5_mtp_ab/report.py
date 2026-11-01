#!/usr/bin/env python3
"""Build a Chinese A5 LIM MTP report from per-case runner JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any

CASES = [
    f"q{q}_{mode}_{policy}_hit{hit}"
    for q in (1, 4)
    for mode in ("eager", "graph")
    for policy in ("low", "random")
    for hit in (91, 95)
]
BACKENDS = ("li", "original", "sglang")


def percentile(values: list[float], p: float) -> float:
    """Return the linearly interpolated percentile used by NumPy default."""
    if not values:
        raise ValueError("percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    index = (len(ordered) - 1) * p
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty sample list")
    return {
        "count": len(values),
        "mean_us": statistics.fmean(values),
        "median_us": statistics.median(values),
        "p95_us": percentile(values, 0.95),
    }


def _case_dict(payload: dict[str, Any]) -> dict[str, Any]:
    case = payload.get("case")
    if isinstance(case, dict):
        return case
    return {"case_id": payload.get("case_id", case)}


def _fixture_identity(payload: dict[str, Any]) -> tuple[Any, ...] | None:
    hashes = payload.get("hashes", {})
    fixture = payload.get("fixture", {})
    if not isinstance(hashes, dict) or not isinstance(fixture, dict):
        return None
    values = (
        fixture.get("fixture_sha256", hashes.get("fixture_sha256")),
        fixture.get("input_sha256", hashes.get("input_sha256")),
        fixture.get("reference_sha256", hashes.get("reference_sha256")),
        fixture.get("residency_sha256", hashes.get("residency_sha256")),
    )
    return values if all(value for value in values) else None


def _graph_replay_passed(value: Any) -> bool:
    """Accept the runner's validation detail or the legacy boolean schema."""
    if value is True:
        return True
    return isinstance(value, dict) and value.get("passed") is True


def load_case(case_dir: Path) -> tuple[dict[str, Any] | None, str]:
    """Load one runner result and validate its basic schema/status."""
    status_path = case_dir / "status"
    if status_path.is_file() and status_path.read_text(encoding="utf-8").strip() != "0":
        return None, "failed"
    result_path = case_dir / "runner" / "case.json"
    if not result_path.is_file():
        return None, "missing"
    if not status_path.is_file():
        return None, "failed"
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, "failed"
    if payload.get("schema_version") != 1:
        return payload, "failed"
    if payload.get("diagnostic_only") is True:
        return payload, "failed"
    if payload.get("status") != "ok":
        return payload, "failed"
    if not isinstance(payload.get("rounds"), list) or len(payload["rounds"]) < 3:
        return payload, "failed"
    correctness = payload.get("correctness")
    if not isinstance(correctness, dict) or any(
        not isinstance(correctness.get(backend), dict)
        or correctness[backend].get("passed") is not True
        for backend in BACKENDS
    ):
        return payload, "failed"
    if _case_dict(payload).get("mode") == "graph" and any(
        not _graph_replay_passed(correctness[backend].get("graph_replay"))
        for backend in ("original", "sglang")
    ):
        return payload, "failed"
    try:
        _round_samples(payload)
    except (KeyError, TypeError, ValueError):
        return payload, "failed"
    return payload, "measured"


def _round_samples(payload: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    result: dict[str, dict[str, list[float]]] = {}
    for round_item in payload["rounds"]:
        round_id = int(round_item["round"])
        if str(round_id) in result:
            raise ValueError("round IDs must be unique")
        samples = round_item["samples_us"]
        if not isinstance(samples, dict):
            raise ValueError("round samples_us must be a backend mapping")
        round_correctness = round_item.get("correctness")
        if not isinstance(round_correctness, dict) or any(
            not isinstance(round_correctness.get(backend), dict)
            or round_correctness[backend].get("passed") is not True
            for backend in ("original", "sglang")
        ):
            raise ValueError("each round must pass correctness for both LIM backends")
        result[str(round_id)] = {}
        for backend in BACKENDS:
            values = samples.get(backend)
            if not isinstance(values, list) or not values:
                raise ValueError(f"round {round_id} missing samples for {backend}")
            numbers = [float(value) for value in values]
            if any(not math.isfinite(value) or value <= 0 for value in numbers):
                raise ValueError("samples must be finite, positive microseconds")
            result[str(round_id)][backend] = numbers
    for round_samples in result.values():
        if len({len(round_samples[backend]) for backend in BACKENDS}) != 1:
            raise ValueError("all backends need equal nonzero sample counts per round")
    return result


def paired_ratio(
    payload: dict[str, Any], numerator: str, denominator: str
) -> list[float]:
    """Compute paired per-round ratios from recomputed round means."""
    if _fixture_identity(payload) is None:
        raise ValueError("fixture/hash mismatch or incomplete fixture hashes")
    backend_hashes = payload.get("hashes", {}).get("backends")
    if isinstance(backend_hashes, dict):
        identities = [
            backend_hashes.get(backend) for backend in (numerator, denominator)
        ]
        if (
            any(identity is None for identity in identities)
            or identities[0] != identities[1]
        ):
            raise ValueError("backend fixture/hash mismatch")
    rounds = _round_samples(payload)
    ratios = []
    for values in rounds.values():
        lhs = statistics.fmean(values[numerator])
        rhs = statistics.fmean(values[denominator])
        if rhs <= 0:
            raise ValueError("ratio denominator must be positive")
        ratios.append(lhs / rhs)
    return ratios


def _fmt(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def _metric(payload: dict[str, Any], backend: str) -> dict[str, float | int]:
    samples = [
        value
        for round_item in payload["rounds"]
        for value in round_item["samples_us"][backend]
    ]
    return summarize([float(value) for value in samples])


def _ratio_note(payload: dict[str, Any], threshold_pct: float) -> str:
    try:
        ratios = paired_ratio(payload, "sglang", "original")
    except (KeyError, TypeError, ValueError):
        return "无法比较：fixture/hash不匹配或配对数据缺失"
    mean_pct = (statistics.fmean(ratios) - 1) * 100
    if mean_pct > threshold_pct:
        flag = f"⚠ 超过{threshold_pct:g}%告警阈值"
    else:
        flag = f"未超过{threshold_pct:g}%告警阈值"
    return f"迁移/原版配对轮均值比 {_fmt(statistics.fmean(ratios))}（{mean_pct:+.2f}%）；{flag}"


def _write_plot(payloads: dict[str, dict[str, Any]], plots_dir: Path) -> str:
    measured = [(case_id, payload) for case_id, payload in payloads.items() if payload]
    if not measured:
        return "暂无有效实测数据，未生成性能图。"
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return "未安装 matplotlib，报告保留表格与CSV；性能图待安装依赖后生成。"

    plots_dir.mkdir(parents=True, exist_ok=True)
    for mode in ("eager", "graph"):
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
        fig.suptitle(f"A5 LIM MTP {mode}：原始样本与均值趋势")
        for ax, q in zip(axes, (1, 4)):
            subset = [
                (case_id, p)
                for case_id, p in measured
                if _case_dict(p).get("q") == q and _case_dict(p).get("mode") == mode
            ]
            subset.sort(key=lambda item: item[0])
            x_positions = list(range(len(subset)))
            for backend_index, backend in enumerate(BACKENDS):
                means = []
                for x, (case_id, payload) in zip(x_positions, subset):
                    values = [
                        float(v)
                        for rnd in payload["rounds"]
                        for v in rnd["samples_us"][backend]
                    ]
                    means.append(statistics.fmean(values))
                    jitter = (backend_index - 1) * 0.12
                    ax.scatter([x + jitter] * len(values), values, s=12, alpha=0.28)
                ax.plot(
                    [x + (backend_index - 1) * 0.12 for x in x_positions],
                    means,
                    marker="o",
                    linewidth=1.4,
                    label=backend,
                )
            ax.set_title(f"Q{q}")
            ax.set_xticks(
                x_positions,
                [
                    case_id.split("_")[2] + "_" + case_id.split("_")[-1]
                    for case_id, _ in subset
                ],
                rotation=45,
                ha="right",
            )
            ax.set_ylabel("延迟（µs）")
            ax.grid(True, alpha=0.25)
            if subset:
                ax.legend()
        fig.tight_layout()
        fig.savefig(plots_dir / f"latency_{mode}.png", dpi=160)
        plt.close(fig)
    return "已生成 eager/graph 分面图，包含原始样本散点与均值趋势。"


def build_report(output_root: Path, threshold_pct: float = 5.0) -> dict[str, Any]:
    if not math.isfinite(threshold_pct) or threshold_pct < 0:
        raise ValueError("threshold percentage must be finite and non-negative")
    payloads: dict[str, dict[str, Any]] = {}
    statuses: dict[str, str] = {}
    details: list[dict[str, Any]] = []
    for case_id in CASES:
        payload, status = load_case(output_root / case_id)
        statuses[case_id] = status
        if status == "measured" and payload is not None:
            case = _case_dict(payload)
            if payload.get("case_id", case_id) != case_id:
                status = statuses[case_id] = "failed"
                payload = None
            elif _fixture_identity(payload) is None:
                status = statuses[case_id] = "failed"
                payload = None
            else:
                try:
                    _round_samples(payload)
                except (KeyError, TypeError, ValueError):
                    status = statuses[case_id] = "failed"
                    payload = None
        payloads[case_id] = payload if status == "measured" else None
        details.append(
            {
                "case_id": case_id,
                "status": status,
                "case": _case_dict(payload) if payload else None,
                "log_index": {
                    "run": f"{case_id}/run.log",
                    "command": f"{case_id}/command.sh",
                    "status": f"{case_id}/status",
                },
            }
        )

    q_hashes: dict[int, tuple[Any, Any]] = {}
    mismatched_q: set[int] = set()
    for case_id, payload in payloads.items():
        if not payload:
            continue
        q = _case_dict(payload).get("q")
        identity = _fixture_identity(payload)
        if not isinstance(q, int) or identity is None:
            continue
        pair = (identity[1], identity[2])
        if q in q_hashes and q_hashes[q] != pair:
            mismatched_q.add(q)
        else:
            q_hashes[q] = pair

    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "report_data.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "case_id",
                "status",
                "q",
                "mode",
                "policy",
                "hit_percent",
                "backend",
                "round",
                "sample_index",
                "latency_us",
            ],
        )
        writer.writeheader()
        for case_id, payload in payloads.items():
            if not payload:
                continue
            case = _case_dict(payload)
            for round_item in payload["rounds"]:
                for backend in BACKENDS:
                    for sample_index, latency in enumerate(
                        round_item["samples_us"][backend], 1
                    ):
                        writer.writerow(
                            {
                                "case_id": case_id,
                                "status": "measured",
                                "q": case.get("q"),
                                "mode": case.get("mode"),
                                "policy": case.get("policy"),
                                "hit_percent": case.get("hit_percent"),
                                "backend": backend,
                                "round": round_item["round"],
                                "sample_index": sample_index,
                                "latency_us": latency,
                            }
                        )

    summary_lines = []
    for case_id in CASES:
        status = statuses[case_id]
        icon = {"measured": "✅ 已测", "failed": "❌ 失败", "missing": "⏳ 缺失"}[
            status
        ]
        payload = payloads[case_id]
        if payload:
            measures = "；".join(
                f"{backend}: 均值 {_fmt(float(_metric(payload, backend)['mean_us']))} µs，"
                f"中位数 {_fmt(float(_metric(payload, backend)['median_us']))} µs，"
                f"p95 {_fmt(float(_metric(payload, backend)['p95_us']))} µs"
                for backend in BACKENDS
            )
            ratios = []
            for numerator, denominator, label in (
                ("original", "li", "原版/LI"),
                ("sglang", "li", "迁移/LI"),
                ("sglang", "original", "迁移/原版"),
            ):
                if _case_dict(payload).get("q") in mismatched_q:
                    ratios.append(f"{label} 无法比较（同Q输入/reference哈希不一致）")
                else:
                    try:
                        ratios.append(
                            f"{label} {_fmt(statistics.fmean(paired_ratio(payload, numerator, denominator)))}"
                        )
                    except (KeyError, TypeError, ValueError):
                        ratios.append(f"{label} 无法比较（fixture/hash不匹配）")
            ratio_note = (
                "同 Q 输入/reference 哈希不一致，配对比值已禁用"
                if _case_dict(payload).get("q") in mismatched_q
                else _ratio_note(payload, threshold_pct)
            )
            summary_lines.append(
                f"| `{case_id}` | {icon} | {measures} | {'；'.join(ratios)} | {ratio_note} |"
            )
            case = _case_dict(payload)
            correctness = payload.get("correctness", {})
            detail = next(item for item in details if item["case_id"] == case_id)
            detail.update(
                {
                    "correctness": correctness,
                    "hashes": payload.get("hashes"),
                    "backend_versions": payload.get("backend_versions"),
                    "raw_rounds": payload.get("rounds"),
                    "fixture": payload.get("fixture"),
                    "timing_boundary": payload.get("timing_boundary"),
                    "log_index": {
                        "run": f"{case_id}/run.log",
                        "command": f"{case_id}/command.sh",
                        "status": f"{case_id}/status",
                    },
                }
            )
        else:
            err = (
                "状态文件非零或结果 JSON 不合规"
                if status == "failed"
                else "尚无本 case 结果"
            )
            summary_lines.append(f"| `{case_id}` | {icon} | {err} | — | — |")

    plot_note = _write_plot(payloads, output_root / "plots")
    table = "\n".join(summary_lines)
    config_lines = []
    for case_id in CASES:
        q = 1 if case_id.startswith("q1_") else 4
        hit = int(case_id.split("_")[-1][3:])
        policy = case_id.split("_")[2]
        payload = payloads[case_id]
        detail = "实测缺失或失败，配置按计划值列出。"
        if payload:
            fixture_stats = payload.get("fixture", {}).get("stats", {})
            request_rows = fixture_stats.get("requests", [])
            request_misses = sorted(
                {int(row.get("selected_prefix_misses", 0)) for row in request_rows}
            )
            detail = (
                f"每请求目标miss={round(q * 2048 * (1 - hit / 100))}，"
                f"实测union miss={request_misses}；"
                "目标标签hit="
                f"{_fmt(fixture_stats.get('selected_prefix_union_target_hit_rate'))}，"
                "唯一选中prefix并集hit="
                f"{_fmt(fixture_stats.get('unique_selected_prefix_hit_rate'))}，"
                "逐route访问hit="
                f"{_fmt(fixture_stats.get('route_occurrence_hit_rate'))}。"
            )
        config_lines.append(
            f"- `{case_id}`：Q{q}，{case_id.split('_')[1]}，{policy} filler，"
            f"hit{hit}%；LI、vLLM原版LIM、SGLang LIM；{detail}"
        )
    configs = "\n".join(config_lines)
    correctness_rows = []
    for case_id in CASES:
        payload = payloads[case_id]
        if payload:
            correctness_rows.append(
                f"- `{case_id}`：`{json.dumps(payload.get('correctness', {}), ensure_ascii=False, sort_keys=True)}`"
            )
    if not correctness_rows:
        correctness_rows = ["- 暂无实测正确性记录。"]
    version_rows = []
    for case_id in CASES:
        payload = payloads[case_id]
        if payload:
            version_rows.append(
                f"- `{case_id}`：`{json.dumps(payload.get('backend_versions', {}), ensure_ascii=False, sort_keys=True)}`"
            )
    if not version_rows:
        version_rows = ["- 暂无实测版本与库哈希；检查运行后的 preflight 和 manifest。"]
    raw_path = output_root / "report_data.json"
    raw_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "threshold_pct": threshold_pct,
                "q_hash_mismatch": sorted(mismatched_q),
                "cases": details,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    report = f"""# A5 LIM MTP 手动对照报告

## 运行状态

性能样本来自 A5 runner JSON；缺失 case 保持为缺失，不用 0 补齐。回归阈值为 {threshold_pct:g}%，仅作告警，不代表统计显著性。

| 配置 | 状态 | 性能统计 | 配对比值 | 告警 |
|---|---|---|---|---|
{table}

## 配置矩阵与差异

默认共同配置：BS16、K2048、C12288、32个索引头、head dim128、page128；
prefix65536加128个tail，实际KV65664；17行pool，请求使用1–16行，默认source步长66944；
Q/K为FP8 E4M3FN、weights为BF16、dequant scale为FP32、稳态request_state=-1。
实际池形状、输入、版本和任何覆盖值以各case的fixture和来源记录为准。

{configs}

命中标签定义为 `1 - unique_prefix_union_misses / (Q*2048)`：Q1 的 91%/95% 对应 184/102 个 miss；Q4 对应 737/410 个 miss。它不等于以实际唯一 selected-prefix union 为分母的命中率。逐 route occurrence 命中率、唯一 union miss 数及 tail slot 规则以 case fixture manifest 为准。低 ID/random 只改变非选中 filler；不同 Q、执行模式、filler 与 hit 目标是独立矩阵因素。

共享工作负载：B16、K2048、C12288、32个index head、head dim 128、page 128，稳态 `request_state=-1`；source pool宽度默认66944，稳定prefix为65536，实际KV长度65664。tail按 `C + source - prefix` 直接映射，不驻留在prefix pool。

本轮是部署配置对照。实际路径以每个case的 `backend_versions.sglang_backend`、编译变体、OPP manifest和已解析动态库为准，不能从分支名推断。历史direct版本使用vector后处理；新original版本通过正常SGLang接口调用固定原版LIM OPP。原版库的实际构建变体未知时保留unknown；配置未对齐时不能把差额全部归于迁移算法。两个原版OPP vendor在同一进程中还可能共享device-op路由；已解析host DSO不等于已证明每项使用独立device binary。

## 正确性与输入哈希

{chr(10).join(correctness_rows)}

## 版本、构建配置与动态库

{chr(10).join(version_rows)}

哈希或 fixture 不一致的 case 不进行配对比值计算。原版与迁移版均慢于 LI 时，不能据此把差异归因于迁移版；LI 只作为索引器 latency/reference backend。

正确性覆盖范围：setup 及 graph 捕获/回放阶段执行完整输出与 pool 校验；每轮结束执行完整校验；逐 timed sample 仅检查 miss counts，不表示逐样本完整验证所有输出张量。

## 图表

{plot_note}

## 日志与版本索引

- 每个 case 的合并 stdout/stderr：`<case>/run.log`
- 每个 case 的完整命令：`<case>/command.sh`
- shell 退出码：`<case>/status`
- runner 结果与正确性：`<case>/runner/case.json`、`manifest.json`、`error.json`
- 预检：`preflight/`
- 原始样本 CSV：`report_data.csv`；汇总索引：`report_data.json`

## 调用链

```mermaid
flowchart TD
  A[冻结量化输入与哈希]
  A --> B[计时外: 官方 LI 参考 TopK]
  B --> C[计时外: 固定 miss 集合与 low/random 缓存快照]
  C --> R[每次调用前恢复各自缓存池, 计时外]
  A --> D[正式计时: 官方 LI, 不读 resident 缓存池]
  A --> E[正式计时: PR 原版 LIM, 自行计算 TopK 与缓存管理]
  A --> F[正式计时: SGLang LIM, 自行计算 TopK 与缓存管理]
  R --> E
  R --> F
  D --> G[Eager 调用或 Graph 回放的分轮原始样本]
  E --> G
  F --> G
  B -. 参考校验 .-> H
  G --> H[计时外: 正确性, 状态与哈希检查]
  H --> I[合格结果生成 CSV/JSON/中文报告/图]
```
"""
    (output_root / "REPORT.md").write_text(report, encoding="utf-8")
    return {
        "statuses": statuses,
        "report": str(output_root / "REPORT.md"),
        "csv": str(csv_path),
        "json": str(raw_path),
        "plot_note": plot_note,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--regression-threshold-pct", type=float, default=5.0)
    args = parser.parse_args()
    result = build_report(args.output_root, args.regression_threshold_pct)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
