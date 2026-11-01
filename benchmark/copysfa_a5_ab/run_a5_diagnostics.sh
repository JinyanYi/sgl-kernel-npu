#!/usr/bin/env bash
# Run same-input dispatch diagnosis and optional full CopySFA performance matrix.
set -eo pipefail

if [[ $# -lt 1 || $# -gt 2 || ${1:-} == --help ]]; then
    echo 'Usage: bash run_a5_diagnostics.sh NEW_OUTPUT_ROOT [--full-matrix]'
    echo 'Optional: PYTHON_BIN, ASCEND_ENV, ORIGINAL_PATHS_ENV, BUILD_JOBS.'
    exit 2
fi
output_arg=$1
full_matrix=${2:-}
[[ -z "$full_matrix" || "$full_matrix" == --full-matrix ]] || exit 2
[[ ! -e "$output_arg" ]] || { echo "Preserve existing output: $output_arg" >&2; exit 2; }
python_bin=${PYTHON_BIN:-$(command -v python3)}
python_bin=$("$python_bin" -c 'import sys; print(sys.executable)')
source "${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
fi
export PYTHON_BIN="$python_bin"
export PATH="$(dirname -- "$python_bin"):$PATH"
set -u
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo=$(cd -- "$script_dir/../.." && pwd -P)
mkdir -p -- "$(dirname -- "$output_arg")"
mkdir -- "$output_arg"
output=$(cd -- "$output_arg" && pwd -P)
exec > >(tee "$output/run.log") 2>&1
finish() {
    rc=$?
    trap - EXIT
    printf '%s\n' "$rc" > "$output/exit_code"
    printf 'Final exit code: %s\nEvidence: %s\n' "$rc" "$output"
    exit "$rc"
}
trap finish EXIT
cd -- "$repo"
git rev-parse HEAD > "$output/source_head.txt"
git status --short > "$output/source_status.txt"
npu-smi info > "$output/npu-smi.txt"
printf 'Python: %s\nVisible devices: %s\n' "$PYTHON_BIN" "${ASCEND_RT_VISIBLE_DEVICES:-unset}"

if [[ -n ${ORIGINAL_PATHS_ENV:-} ]]; then
    [[ -f "$ORIGINAL_PATHS_ENV" ]] || { echo 'ORIGINAL_PATHS_ENV does not exist' >&2; exit 2; }
    reference_env=$ORIGINAL_PATHS_ENV
else
    # Shared filesystem clock skew can break CANN/Ninja incremental builds.
    reference_parent=$(mktemp -d /tmp/copysfa-reference-XXXXXXXX)
    printf '%s\n' "$reference_parent" > "$output/reference_build_root.txt"
    bash "$script_dir/reference/build_original.sh" "$reference_parent/build" \
        --ascend-env "${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}" \
        --jobs "${BUILD_JOBS:-8}"
    reference_env="$reference_parent/build/paths.env"
fi
set +u
source "$reference_env"
set -u
cp -- "$reference_env" "$output/reference_paths.env"

# The original OPP must be active before this first NPU initialization.
"$PYTHON_BIN" -m pytest -sv "$repo/tests/test_copysfa_reference_transport.py" \
    --junitxml="$output/reference_transport.xml" 2>&1 | tee "$output/reference_transport.log"

formal_rc=0
"$PYTHON_BIN" -m pytest -sv \
    "$repo/tests/test_fused_quant_scatter_copy_sparse_flash_attention.py" \
    "$repo/tests/test_fused_quant_scatter_copy_sparse_flash_attention_a5.py" \
    --junitxml="$output/formal.xml" 2>&1 | tee "$output/formal.log" || formal_rc=$?
printf '%s\n' "$formal_rc" > "$output/formal.exitcode"

diagnostic_rc=0
"$PYTHON_BIN" "$script_dir/dispatch_ab.py" --output-root "$output/dispatch" \
    2>&1 | tee "$output/dispatch.log" || diagnostic_rc=$?
printf '%s\n' "$diagnostic_rc" > "$output/dispatch.exitcode"

# An FP16 failure must not suppress the BF16 Q1/Q4 and C-pool performance cases.
matrix_rc=0
if [[ "$full_matrix" == --full-matrix ]]; then
    "$PYTHON_BIN" "$script_dir/run_matrix.py" --output-root "$output/matrix" \
        2>&1 | tee "$output/matrix.log" || matrix_rc=$?
    printf '%s\n' "$matrix_rc" > "$output/matrix.exitcode"
fi
report_rc=0
"$PYTHON_BIN" "$script_dir/report.py" "$output" \
    2>&1 | tee "$output/performance_report.log" || report_rc=$?
printf '%s\n' "$report_rc" > "$output/report.exitcode"
if (( formal_rc != 0 || diagnostic_rc != 0 || matrix_rc != 0 || report_rc != 0 )); then
    exit 1
fi
