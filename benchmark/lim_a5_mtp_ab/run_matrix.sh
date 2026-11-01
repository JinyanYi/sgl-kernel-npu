#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
OUTPUT_ROOT=""
DEVICE=""
DRY_RUN=0
WARMUP=20
ROUNDS=3
ITERATIONS=50
REGRESSION_THRESHOLD_PCT=5
PYTHON_BIN="${PYTHON_BIN:-python3}"

usage() {
  cat <<'EOF'
Usage: run_matrix.sh --output-root DIR --device npu:N [--dry-run]
       [--warmup N] [--rounds N] [--iterations N]
       [--regression-threshold-pct N]

The caller must source the matching Ascend environment before execution.
Required environment: KERNEL_PYTHON_ROOT, ASCEND_HOME_PATH,
ASCEND_RT_VISIBLE_DEVICES. Select the visible device explicitly before use.
Required original backend: ORIGINAL_LIM_BINDING or both ORIGINAL_MODULE and
ORIGINAL_OP. build_private.sh prepares the pinned original operator privately;
no full vLLM-Ascend checkout or Python package is required.
EOF
}

while (($#)); do
  case "$1" in
    --output-root) OUTPUT_ROOT="${2:?missing value for --output-root}"; shift 2 ;;
    --device) DEVICE="${2:?missing value for --device}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --warmup) WARMUP="${2:?missing value for --warmup}"; shift 2 ;;
    --rounds) ROUNDS="${2:?missing value for --rounds}"; shift 2 ;;
    --iterations) ITERATIONS="${2:?missing value for --iterations}"; shift 2 ;;
    --regression-threshold-pct) REGRESSION_THRESHOLD_PCT="${2:?missing value}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$OUTPUT_ROOT" || -z "$DEVICE" ]]; then
  usage >&2
  exit 2
fi
if [[ "$OUTPUT_ROOT" != /* ]]; then
  OUTPUT_ROOT="$PWD/$OUTPUT_ROOT"
fi

CASE_IDS=()
for q in 1 4; do
  for mode in eager graph; do
    for policy in low random; do
      for hit in 91 95; do
        CASE_IDS+=("q${q}_${mode}_${policy}_hit${hit}")
      done
    done
  done
done

declare -A Q_INPUT_READY=()
build_args() {
  local case_id="$1"
  local output_dir="$2"
  local q="$3"
  local input_file="$OUTPUT_ROOT/inputs/q${q}.pt"
  RUN_ARGS=("$PYTHON_BIN" "$SCRIPT_DIR/bench.py" --case-id "$case_id"
    --output "$output_dir" --device "$DEVICE"
    --kernel-python-root "${KERNEL_PYTHON_ROOT:-<KERNEL_PYTHON_ROOT>}"
    --warmup "$WARMUP" --rounds "$ROUNDS" --iterations "$ITERATIONS")
  if [[ -n "${Q_INPUT_READY[$q]:-}" ]]; then
    RUN_ARGS+=(--inputs-file "$input_file")
  else
    RUN_ARGS+=(--save-inputs "$input_file")
    Q_INPUT_READY[$q]=1
  fi
  if [[ -n "${ORIGINAL_SOURCE_REVISION:-}" ]]; then
    RUN_ARGS+=(--original-source-revision "$ORIGINAL_SOURCE_REVISION")
  fi
  RUN_ARGS+=(--original-variant "${ORIGINAL_VARIANT:-unknown}")
  if [[ -n "${ORIGINAL_OPP_ROOT:-}" ]]; then
    RUN_ARGS+=(--original-opp-root "$ORIGINAL_OPP_ROOT")
  fi
  if [[ -n "${ORIGINAL_LIM_BINDING:-}" ]]; then
    RUN_ARGS+=(--original-library "$ORIGINAL_LIM_BINDING")
  fi
  if [[ -n "${ORIGINAL_MODULE:-}" ]]; then
    RUN_ARGS+=(--original-module "$ORIGINAL_MODULE")
  fi
  if [[ -n "${ORIGINAL_OP:-}" ]]; then
    RUN_ARGS+=(--original-op "$ORIGINAL_OP")
  fi
}

if ((DRY_RUN)); then
  for case_id in "${CASE_IDS[@]}"; do
    q="${case_id#q}"; q="${q%%_*}"
    build_args "$case_id" "$OUTPUT_ROOT/$case_id/runner" "$q"
    printf 'case=%s output=%q\n  ' "$case_id" "$OUTPUT_ROOT/$case_id"
    printf '%q ' "${RUN_ARGS[@]}"
    printf '\n'
  done
  exit 0
fi

for required in KERNEL_PYTHON_ROOT ASCEND_HOME_PATH ASCEND_RT_VISIBLE_DEVICES; do
  if [[ -z "${!required:-}" ]]; then
    printf 'Missing required environment variable: %s\n' "$required" >&2
    exit 2
  fi
done
if [[ ! -d "$KERNEL_PYTHON_ROOT" ]]; then
  printf 'KERNEL_PYTHON_ROOT does not exist: %s\n' "$KERNEL_PYTHON_ROOT" >&2
  exit 2
fi
if [[ -e "$OUTPUT_ROOT" ]]; then
  printf 'Refusing to overwrite existing output root: %s\n' "$OUTPUT_ROOT" >&2
  exit 2
fi

mkdir -p "$OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT/inputs"
export PYTHONPATH="${KERNEL_PYTHON_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

bash "$SCRIPT_DIR/preflight.sh" --output "$OUTPUT_ROOT/preflight" --device "$DEVICE" || {
  rc=$?
  printf '%s\n' "$rc" > "$OUTPUT_ROOT/preflight.status"
  "$PYTHON_BIN" "$SCRIPT_DIR/report.py" --output-root "$OUTPUT_ROOT" \
    --regression-threshold-pct "$REGRESSION_THRESHOLD_PCT" || true
  exit "$rc"
}
printf '0\n' > "$OUTPUT_ROOT/preflight.status"

overall=0
for case_id in "${CASE_IDS[@]}"; do
  case_dir="$OUTPUT_ROOT/$case_id"
  mkdir -p "$case_dir"
  runner_output="$case_dir/runner"
  q="${case_id#q}"; q="${q%%_*}"
  build_args "$case_id" "$runner_output" "$q"
  {
    printf '#!/usr/bin/env bash\nset -euo pipefail\n'
    if [[ -n "${ASCEND_ENV_SCRIPT:-}" ]]; then
      printf 'source %q\n' "$ASCEND_ENV_SCRIPT"
    fi
    for key in ASCEND_HOME_PATH ASCEND_TOOLKIT_HOME ASCEND_CUSTOM_OPP_PATH \
        ASCEND_RT_VISIBLE_DEVICES LD_LIBRARY_PATH KERNEL_PYTHON_ROOT \
        ORIGINAL_LIM_BINDING ORIGINAL_MODULE ORIGINAL_OP ORIGINAL_SOURCE_REVISION \
        ORIGINAL_VARIANT ORIGINAL_OPP_ROOT ORIGINAL_BINDING_MANIFEST \
        ORIGINAL_SOURCE_MANIFEST; do
      if [[ -n "${!key:-}" ]]; then printf 'export %s=%q\n' "$key" "${!key}"; fi
    done
    printf 'export PYTHONPATH=%q\n' "$PYTHONPATH"
    printf 'exec '
    printf '%q ' "${RUN_ARGS[@]}"
    printf '\n'
  } > "$case_dir/command.sh"
  chmod +x "$case_dir/command.sh"
  printf 'Starting %s on explicitly configured device %s\n' "$case_id" "$DEVICE"
  set +e
  "${RUN_ARGS[@]}" 2>&1 | tee "$case_dir/run.log"
  rc=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$rc" > "$case_dir/status"
  if ((rc != 0)); then
    overall="$rc"
    printf 'Case %s failed with exit code %s; stopping matrix.\n' "$case_id" "$rc" >&2
    break
  fi
done

"$PYTHON_BIN" "$SCRIPT_DIR/report.py" --output-root "$OUTPUT_ROOT" \
  --regression-threshold-pct "$REGRESSION_THRESHOLD_PCT" || overall=1
exit "$overall"
