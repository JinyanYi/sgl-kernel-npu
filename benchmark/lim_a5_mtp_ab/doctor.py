#!/usr/bin/env python3
"""Read-only A5 build environment doctor; device access is opt-in."""

from __future__ import annotations

import argparse
import ctypes.util
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

OPTIONAL_BUILD_DISTRIBUTIONS = {
    "setuptools": "setuptools",
    "wheel": "wheel",
    "packaging": "packaging",
    "numpy": "numpy",
    "pyyaml": "PyYAML",
    "nlohmann_json": "nlohmann-json",
}
OPTIONAL_BUILD_COMMANDS = ("cmake", "ninja", "c++", "bisheng", "npu-smi")
CANN_CMAKE_RELATIVE_PATHS = (
    "tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake",
    "compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake",
    "ascendc_devkit/tikcpp/samples/cmake/ascendc.cmake",
)


def _distribution(name: str) -> dict[str, Any]:
    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"found": False, "name": name, "version": None, "location": None}
    return {
        "found": True,
        "name": dist.metadata.get("Name", name),
        "version": dist.version,
        "location": str(Path(dist.locate_file("")).resolve()),
    }


def _module(name: str) -> dict[str, Any]:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ModuleNotFoundError, ValueError) as error:
        return {"found": False, "name": name, "origin": None, "error": str(error)}
    if spec is None:
        return {"found": False, "name": name, "origin": None}
    locations = list(spec.submodule_search_locations or [])
    return {
        "found": True,
        "name": name,
        "origin": spec.origin,
        "locations": locations,
    }


def _command(name: str, *, version_args: tuple[str, ...] = ()) -> dict[str, Any]:
    executable = shutil.which(name)
    result: dict[str, Any] = {
        "found": executable is not None,
        "name": name,
        "path": executable,
    }
    if executable and version_args:
        try:
            completed = subprocess.run(
                [executable, *version_args],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            output = (completed.stdout or completed.stderr).strip().splitlines()
            result["version_output"] = output[0] if output else None
            result["version_exit_code"] = completed.returncode
        except (OSError, subprocess.TimeoutExpired) as error:
            result["version_error"] = str(error)
    return result


def _cmake_meets_project_floor(cmake: dict[str, Any]) -> bool | None:
    output = cmake.get("version_output")
    if not isinstance(output, str):
        return None
    parts = output.split()
    if len(parts) < 3:
        return None
    try:
        version = tuple(int(part) for part in parts[-1].split(".")[:3])
    except ValueError:
        return None
    version = (*version, 0, 0)[:3]
    return version >= (3, 20, 0)


def _root_checks(env: Mapping[str, str]) -> dict[str, Any]:
    configured = env.get("ASCEND_HOME_PATH", "").strip()
    root = Path(configured).expanduser() if configured else None
    if root and root.is_file() and root.name in {"set_env.sh", "setenv.bash"}:
        root = root.parent.parent if root.parent.name == "bin" else root.parent
    exists = bool(root and root.is_dir())
    resolved = root.resolve() if exists else None
    set_env_candidates = []
    if resolved:
        set_env_candidates = [
            resolved / "set_env.sh",
            resolved.parent / "set_env.sh",
            resolved / "bin/setenv.bash",
        ]
    set_env_script = next((path for path in set_env_candidates if path.is_file()), None)
    checks: dict[str, Any] = {
        "configured": configured or None,
        "found": exists,
        "resolved": str(resolved) if resolved else None,
        "set_env_script": str(set_env_script) if set_env_script else None,
        "ascendc_cmake": None,
        "acl_header": bool(resolved and (resolved / "include/acl/acl.h").is_file()),
        "aclnn_headers": {},
        "libraries": {},
        "version_files": [],
    }
    if resolved:
        cmake_files = [resolved / relative for relative in CANN_CMAKE_RELATIVE_PATHS]
        system_prefix = env.get("SYSTEM_PREFIX", "").strip().lstrip("/")
        if system_prefix:
            cmake_files.append(
                resolved / system_prefix / "tikcpp/ascendc_kernel_cmake/ascendc.cmake"
            )
        cmake_files.extend(
            [
                resolved / "ascendc_devkit/arch35/ASCConfig.cmake",
                resolved / "ascendc_devkit/arch35/cmake/ASCConfig.cmake",
                resolved / "arch35/ASCConfig.cmake",
            ]
        )
        checks["ascendc_cmake"] = next(
            (str(path) for path in cmake_files if path.is_file()), None
        )
        for header in ("aclnn/acl_meta.h", "aclnn/aclnn_base.h"):
            checks["aclnn_headers"][header] = (resolved / "include" / header).is_file()
        library_roots = [resolved / "lib64", resolved / "lib"]
        library_roots.extend(
            Path(path).expanduser()
            for path in env.get("LD_LIBRARY_PATH", "").split(":")
            if path
        )
        for name in ("libascendcl.so", "libopapi.so"):
            paths = [
                str(path)
                for root_path in library_roots
                for path in root_path.glob(name + "*")
                if path.is_file()
            ]
            if name == "libascendcl.so":
                resolved_system_lib = ctypes.util.find_library("ascendcl")
                if resolved_system_lib:
                    paths.append("ldconfig:" + resolved_system_lib)
            checks["libraries"][name] = {
                "found": bool(paths),
                "candidates": sorted(set(paths)),
            }
        checks["libraries"]["opsbase_candidates"] = sorted(
            {
                str(path)
                for root_path in library_roots
                for pattern in ("*opsbase*.so*", "*nnopbase*.so*")
                for path in root_path.glob(pattern)
                if path.is_file()
            }
        )
        for relative in ("version.cfg", "version.info", "compiler/version.info"):
            path = resolved / relative
            if path.is_file():
                try:
                    contents = path.read_text(encoding="utf-8", errors="replace")[:4096]
                except OSError:
                    contents = None
                checks["version_files"].append(
                    {"path": str(path), "contents": contents}
                )
    return checks


def _opp_checks(env: Mapping[str, str], cann_root: Path | None) -> dict[str, Any]:
    candidates: list[str] = []
    for key in ("ASCEND_CUSTOM_OPP_PATH", "ASCEND_OPP_PATH"):
        candidates.extend(path for path in env.get(key, "").split(":") if path)
    if cann_root:
        candidates.append(str(cann_root / "opp"))
    existing = sorted(
        {
            str(Path(path).expanduser().resolve())
            for path in candidates
            if Path(path).is_dir()
        }
    )
    invalid_configured = [
        path
        for key in ("ASCEND_CUSTOM_OPP_PATH", "ASCEND_OPP_PATH")
        for path in env.get(key, "").split(":")
        if path and not Path(path).is_dir()
    ]
    return {
        "found_candidate": bool(existing),
        "configured_custom_path": env.get("ASCEND_CUSTOM_OPP_PATH") or None,
        "existing_candidates": existing,
        "invalid_configured_paths": invalid_configured,
        "operator_api_verified": False,
        "note": "目录存在不等于原版 A5 LIM OPP/API 已安装；需后续 device-check 或构建清单确认。",
    }


def collect_environment(env: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Collect host software evidence without importing NPU packages or creating a context."""
    environment = os.environ if env is None else env
    distributions = {
        "torch": _distribution("torch"),
        "torch_npu": _distribution("torch-npu"),
    }
    modules = {name: _module(name) for name in ("torch", "torch_npu", "acl")}
    commands = {name: _command(name) for name in OPTIONAL_BUILD_COMMANDS}
    commands["python"] = {
        "found": True,
        "name": sys.executable,
        "version": platform.python_version(),
    }
    commands["cmake"] = _command("cmake", version_args=("--version",))
    cmake_floor = _cmake_meets_project_floor(commands["cmake"])
    root_checks = _root_checks(environment)
    configured_root = root_checks["resolved"]
    cann_root = Path(configured_root) if configured_root else None
    opp = _opp_checks(environment, cann_root)
    build_distributions = {
        key: _distribution(package)
        for key, package in OPTIONAL_BUILD_DISTRIBUTIONS.items()
    }

    required_missing: list[str] = []
    warnings: list[str] = []
    if not distributions["torch"]["found"] or not modules["torch"]["found"]:
        required_missing.append("Python 环境中找不到 torch distribution/module")
    if not distributions["torch_npu"]["found"] or not modules["torch_npu"]["found"]:
        required_missing.append("Python 环境中找不到 torch-npu distribution/module")
    if not root_checks["found"]:
        required_missing.append("ASCEND_HOME_PATH 未指向可访问的 CANN 安装目录")
    else:
        if not root_checks["set_env_script"]:
            required_missing.append(
                "CANN 根目录、父目录或 bin/ 下未发现 set_env.sh/setenv.bash"
            )
        if not root_checks["ascendc_cmake"]:
            required_missing.append("CANN 安装目录缺少 AscendC CMake 模块")
        if not root_checks["acl_header"]:
            required_missing.append("CANN 安装目录缺少 include/acl/acl.h")
        missing_aclnn_headers = [
            name for name, found in root_checks["aclnn_headers"].items() if not found
        ]
        if missing_aclnn_headers:
            required_missing.append(
                "CANN 安装目录缺少 ACLNN 基础头文件："
                + ", ".join(missing_aclnn_headers)
            )
        if not root_checks["libraries"]["libascendcl.so"]["found"]:
            required_missing.append(
                "未在 CANN 根目录、LD_LIBRARY_PATH 或 ldconfig 中发现 libascendcl.so"
            )
        if not root_checks["libraries"]["libopapi.so"]["found"]:
            warnings.append(
                "未发现 libopapi.so 候选；具体动态 API 提供者需结合 CANN/torch_npu 安装清单确认"
            )
        if not root_checks["libraries"]["opsbase_candidates"]:
            warnings.append(
                "未发现名称含 opsbase/nnopbase 的库；只记录候选，不据此判断 ABI 兼容"
            )
    if not commands["cmake"]["found"]:
        required_missing.append("找不到 cmake 命令")
    elif cmake_floor is False:
        required_missing.append("CMake 版本低于当前仓库声明的 3.20")
    elif cmake_floor is None:
        warnings.append("无法解析 CMake 版本；请人工确认满足 3.20 要求")
    if not commands["c++"]["found"]:
        required_missing.append("找不到 C++ 编译器命令 c++")
    optional_missing = [
        key for key, record in build_distributions.items() if not record["found"]
    ]
    optional_commands_missing = [
        name for name in ("ninja", "bisheng", "npu-smi") if not commands[name]["found"]
    ]
    if not commands["ninja"]["found"]:
        required_missing.append("找不到 ninja 命令（A5 构建 wrapper 需要）")
    if optional_missing:
        warnings.append(
            "可选构建 Python 依赖未发现（仓库/具体算子是否需要需按构建目标确认）："
            + ", ".join(optional_missing)
        )
    if optional_commands_missing:
        warnings.append(
            "未发现可选构建/设备工具：" + ", ".join(optional_commands_missing)
        )
    if not opp["found_candidate"]:
        warnings.append(
            "当前未发现 OPP 目录；原版 OPP 可在后续准备阶段安装，因此此项不阻止 prebuild"
        )
    if opp["invalid_configured_paths"]:
        warnings.append(
            "部分 OPP 环境路径不存在：" + ", ".join(opp["invalid_configured_paths"])
        )

    return {
        "schema_version": 1,
        "mode": "environment_only",
        "device_context_created": False,
        "host": {"system": platform.platform(), "machine": platform.machine()},
        "python": commands["python"],
        "distributions": distributions,
        "modules": modules,
        "build_distributions": build_distributions,
        "commands": commands,
        "cann": root_checks,
        "opp": opp,
        "required_missing": required_missing,
        "warnings": warnings,
        "status": "ok" if not required_missing else "missing_requirements",
    }


def _device_check(device: str) -> dict[str, Any]:
    """Explicitly initialize the requested NPU device and check its SoC identity."""
    if not (device.startswith("npu:") and device[4:].isdigit()):
        raise ValueError("--device-check requires --device npu:N")
    index = int(device[4:])
    import acl
    import torch
    import torch_npu  # noqa: F401

    torch.npu.set_device(index)
    soc_info = tuple(acl.rt.get_device_info(index, 601))
    is_a5 = soc_info == (3510, 0)
    return {
        "mode": "device_check",
        "device_context_created": True,
        "device": device,
        "device_name": torch.npu.get_device_name(index),
        "soc_info": list(soc_info),
        "is_ascend_950": is_a5,
        "official_a5_li_available": callable(
            getattr(torch_npu, "npu_quant_lightning_indexer", None)
        ),
        "dynamic_quant_available": callable(
            getattr(torch_npu, "npu_dynamic_quant", None)
        ),
    }


def _human_report(report: dict[str, Any]) -> str:
    lines = ["A5 环境 Doctor", f"状态：{report['status']}"]
    lines.append(
        "设备上下文："
        + ("已显式创建" if report.get("device_context_created") else "未创建")
    )
    for name, package in report.get("distributions", {}).items():
        status = (
            f"已发现 {package['version']} ({package['location']})"
            if package["found"]
            else "缺失"
        )
        lines.append(f"Python 包 {name}: {status}")
    cann = report.get("cann", {})
    lines.append(f"CANN 根目录：{cann.get('resolved') or '未配置/不可访问'}")
    lines.append(
        f"OPP 候选：{', '.join(report.get('opp', {}).get('existing_candidates', [])) or '未发现'}"
    )
    device_result = report.get("device_result", {})
    if isinstance(device_result, dict) and isinstance(device_result.get("npu"), dict):
        check = device_result["npu"]
        lines.append(
            "设备检查："
            + (
                "Ascend 950 已确认"
                if check.get("is_ascend_950")
                else "未确认 Ascend 950"
            )
        )
    elif isinstance(device_result, dict) and device_result.get("error"):
        lines.append(f"设备检查失败：{device_result['error']}")
    if report["required_missing"]:
        lines.append("缺少项：")
        lines.extend(f"  - {item}" for item in report["required_missing"])
    if report["warnings"]:
        lines.append("提示：")
        lines.extend(f"  - {item}" for item in report["warnings"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Write JSON to a new path")
    parser.add_argument(
        "--device-check",
        action="store_true",
        help="Explicitly create context on the requested NPU device",
    )
    parser.add_argument(
        "--device", help="Explicit device identifier, for example npu:0"
    )
    args = parser.parse_args(argv)
    if args.device_check and not args.device:
        parser.error("--device is required with --device-check")
    if args.device and not args.device_check:
        parser.error("--device requires the explicit --device-check flag")
    if args.output and args.output.exists():
        parser.error(f"refusing to overwrite JSON output: {args.output}")

    report = collect_environment()
    if args.device_check:
        try:
            report["device_result"] = {"npu": _device_check(args.device)}
            if not report["device_result"]["npu"]["is_ascend_950"]:
                report["required_missing"].append(
                    "显式 device-check 发现设备不是 Ascend 950"
                )
            if not report["device_result"]["npu"]["official_a5_li_available"]:
                report["required_missing"].append(
                    "torch_npu 未提供 A5 npu_quant_lightning_indexer"
                )
            if not report["device_result"]["npu"]["dynamic_quant_available"]:
                report["required_missing"].append("torch_npu 未提供 npu_dynamic_quant")
        except Exception as error:  # Device drivers may fail with vendor exceptions.
            report["device_result"] = {"error": f"{type(error).__name__}: {error}"}
            report["required_missing"].append(
                "显式 device-check 失败，详情见 device_result.error"
            )
        report["mode"] = "device_check"
        report["status"] = (
            "ok" if not report["required_missing"] else "missing_requirements"
        )

    print(_human_report(report))
    encoded = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = args.output.expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(encoded + "\n", encoding="utf-8")
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
