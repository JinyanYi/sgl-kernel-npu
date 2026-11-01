#!/usr/bin/env bash
set -uo pipefail

OUTPUT=""
DEVICE=""
while (($#)); do
  case "$1" in
    --output) OUTPUT="${2:?missing value for --output}"; shift 2 ;;
    --device) DEVICE="${2:?missing value for --device}"; shift 2 ;;
    -h|--help)
      printf 'Usage: preflight.sh --output NEW_DIR --device npu:N\n'
      exit 0
      ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

if [[ -z "$OUTPUT" || -z "$DEVICE" ]]; then
  printf 'Both --output and --device are required.\n' >&2
  exit 2
fi
if [[ "$OUTPUT" != /* ]]; then OUTPUT="$PWD/$OUTPUT"; fi
if [[ -e "$OUTPUT" ]]; then
  printf 'Refusing to overwrite preflight output: %s\n' "$OUTPUT" >&2
  exit 2
fi
mkdir -p "$OUTPUT"
OUTPUT="$(cd "$OUTPUT" && pwd)"
PREFLIGHT_LOG="$OUTPUT/preflight.log"
: > "$PREFLIGHT_LOG"

report_failure() {
  local check="$1" rc="$2" log_path="$3"
  printf 'FAILED check=%s exit_code=%s log=%s\n' "$check" "$rc" "$log_path" \
    | tee -a "$PREFLIGHT_LOG" >&2 || true
  if [[ -f "$log_path" ]]; then
    printf '%s\n' "--- last 60 lines: $log_path ---" | tee -a "$PREFLIGHT_LOG" >&2 || true
    tail -n 60 "$log_path" | tee -a "$PREFLIGHT_LOG" >&2 || true
    printf '%s\n' '--- end log tail ---' | tee -a "$PREFLIGHT_LOG" >&2 || true
  fi
}

failures=0
for required in ASCEND_HOME_PATH ASCEND_RT_VISIBLE_DEVICES KERNEL_PYTHON_ROOT; do
  if [[ -z "${!required:-}" ]]; then
    printf 'Missing required environment variable: %s\n' "$required" | tee -a "$PREFLIGHT_LOG" >&2
    failures=1
  fi
done
if [[ -n "${ORIGINAL_LIM_BINDING:-}" && ! -f "$ORIGINAL_LIM_BINDING" ]]; then
  printf 'ORIGINAL_LIM_BINDING does not exist: %s\n' "$ORIGINAL_LIM_BINDING" | tee -a "$PREFLIGHT_LOG" >&2
  failures=1
fi
if [[ -z "${ORIGINAL_LIM_BINDING:-}" && ( -z "${ORIGINAL_MODULE:-}" || -z "${ORIGINAL_OP:-}" ) ]]; then
  printf 'Set ORIGINAL_LIM_BINDING or both ORIGINAL_MODULE and ORIGINAL_OP.\n' | tee -a "$PREFLIGHT_LOG" >&2
  failures=1
fi
if [[ -n "${KERNEL_PYTHON_ROOT:-}" && ! -d "$KERNEL_PYTHON_ROOT" ]]; then
  printf 'KERNEL_PYTHON_ROOT does not exist: %s\n' "$KERNEL_PYTHON_ROOT" | tee -a "$PREFLIGHT_LOG" >&2
  failures=1
fi
if [[ -n "${ASCEND_CUSTOM_OPP_PATH:-}" ]]; then
  IFS=':' read -r -a opp_paths <<< "$ASCEND_CUSTOM_OPP_PATH"
  for opp_path in "${opp_paths[@]}"; do
    if [[ -n "$opp_path" && ! -d "$opp_path" ]]; then
      printf 'ASCEND_CUSTOM_OPP_PATH entry does not exist: %s\n' "$opp_path" | tee -a "$PREFLIGHT_LOG" >&2
      failures=1
    fi
  done
fi

if [[ -z "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
  printf 'ASCEND_RT_VISIBLE_DEVICES must be set by the operator; no device is selected automatically.\n' \
    | tee -a "$PREFLIGHT_LOG" >&2
  failures=1
fi
if [[ ! "$DEVICE" =~ ^npu:[0-9]+$ ]]; then
  printf 'Device argument must be an explicit npu:N value.\n' | tee -a "$PREFLIGHT_LOG" >&2
  failures=1
fi

# Record only a fixed allowlist; never serialize the process environment.
{
  printf 'device=%s\n' "$DEVICE"
  for key in ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_CUSTOM_OPP_PATH \
      ASCEND_RT_VISIBLE_DEVICES KERNEL_PYTHON_ROOT PYTHONPATH LD_LIBRARY_PATH \
      ORIGINAL_LIM_BINDING ORIGINAL_MODULE ORIGINAL_OP ORIGINAL_BINDING_MANIFEST \
      ORIGINAL_SOURCE_MANIFEST \
      ORIGINAL_SOURCE_REVISION ORIGINAL_VARIANT ORIGINAL_OPP_ROOT ASCEND_ENV_SCRIPT; do
    if [[ -n "${!key:-}" ]]; then printf '%s=%s\n' "$key" "${!key}"; fi
  done
} > "$OUTPUT/environment.txt"

if ! command -v npu-smi >/dev/null 2>&1; then
  printf 'npu-smi is unavailable.\n' | tee -a "$PREFLIGHT_LOG" >&2
  failures=1
else
  for entry in 'npu-smi info' 'npu-smi info -m'; do
    file_name="$(printf '%s' "$entry" | tr ' /' '__').log"
    printf '$ %s\n' "$entry" | tee -a "$OUTPUT/preflight.log"
    set +e
    bash -c "$entry" > "$OUTPUT/$file_name" 2>&1
    rc=$?
    set -e
    printf 'exit_code=%s\n' "$rc" >> "$OUTPUT/$file_name"
    if ((rc != 0)); then
      failures=1
      report_failure "$entry" "$rc" "$OUTPUT/$file_name"
    fi
  done

  process_log="$OUTPUT/npu-smi_info_proc.log"
  process_command='timeout 5s npu-smi info proc'
  if command -v timeout >/dev/null 2>&1; then
    printf '$ %s\n' "$process_command" | tee -a "$PREFLIGHT_LOG"
    printf '$ %s\n' "$process_command" > "$process_log"
    set +e
    timeout 5s npu-smi info proc >> "$process_log" 2>&1
    rc=$?
    set -e
    printf 'exit_code=%s\n' "$rc" >> "$process_log"
  else
    printf 'command=%s\nunavailable=timeout command not found\nexit_code=unavailable\n' \
      "$process_command" > "$process_log"
    printf 'Nonfatal: timeout unavailable; skipped npu-smi info proc. Log: %s\n' \
      "$process_log" | tee -a "$PREFLIGHT_LOG" >&2
  fi
fi

if command -v ldconfig >/dev/null 2>&1; then
  ldconfig -p 2>&1 | grep -E 'libascendcl|libhccl|libtorch_npu' > "$OUTPUT/runtime-libraries.txt" || true
else
  printf 'ldconfig unavailable\n' > "$OUTPUT/runtime-libraries.txt"
fi

if [[ -n "${PYTHON_BIN:-}" ]]; then python_bin="$PYTHON_BIN"; else python_bin=python3; fi
if ! command -v "$python_bin" >/dev/null 2>&1; then
  printf 'Python executable unavailable: %s\n' "$python_bin" | tee -a "$PREFLIGHT_LOG" >&2
  failures=1
else
  set +e
  "$python_bin" - "$DEVICE" "$OUTPUT/runtime.json" <<'PY' > "$OUTPUT/python-probe.log" 2>&1
import hashlib
import json
import os
import pathlib
import sys

device = sys.argv[1]
output = pathlib.Path(sys.argv[2])
record = {"device_argument": device, "python": sys.version}
try:
    import torch
    import torch_npu
    import acl

    record["torch"] = torch.__version__
    record["torch_npu"] = torch_npu.__version__
    index = int(device.split(":", 1)[1]) if ":" in device else int(device)
    torch.npu.set_device(index)
    record["device_name"] = torch.npu.get_device_name(index)
    record["soc_info"] = acl.rt.get_device_info(index, 601)
    if tuple(record["soc_info"])[0] != 3510:
        raise RuntimeError(f"expected Ascend 950 device tuple, got {record['soc_info']}")
    lib_roots = [pathlib.Path(torch_npu.__file__).resolve().parent / "lib"]
    roots = [root for root in lib_roots if root.is_dir()]
    libraries = []
    for root in roots:
        for pattern in ("libtorch_npu.so*", "libhccl.so*", "libascendcl.so*"):
            for path in sorted(root.glob(pattern)):
                if path.is_file():
                    digest_builder = hashlib.sha256()
                    with path.open("rb") as library_file:
                        for chunk in iter(lambda: library_file.read(1024 * 1024), b""):
                            digest_builder.update(chunk)
                    digest = digest_builder.hexdigest()
                    libraries.append({"path": str(path), "sha256": digest})
    if not any(pathlib.Path(item["path"]).name.startswith("libtorch_npu") for item in libraries):
        raise RuntimeError("runtime libtorch_npu library not found")
    record["runtime_libraries"] = libraries
except Exception as error:
    record["probe_error"] = f"{type(error).__name__}: {error}"
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    raise
output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
print(json.dumps(record, indent=2))
PY
  rc=$?
  set -e
  if ((rc != 0)); then
    failures=1
    report_failure "python runtime probe" "$rc" "$OUTPUT/python-probe.log"
  fi
fi

if ((failures)); then
  printf '1\n' > "$OUTPUT/status"
  printf 'Preflight failed; inspect %s\n' "$OUTPUT" >&2
  exit 1
fi
printf '0\n' > "$OUTPUT/status"
printf 'PREFLIGHT_OK %s\n' "$OUTPUT"
