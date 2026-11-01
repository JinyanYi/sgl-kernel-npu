#!/usr/bin/env bash
# Recover a private install from completed build artifacts without rebuilding them.
set -eo pipefail
usage() { echo 'Usage: recover_private.sh EXISTING_BUILD_ROOT NEW_OUTPUT_ROOT [--ascend-env SCRIPT] [--python PYTHON]'; }
if [[ $# -eq 0 || ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
[[ $# -ge 2 ]] || { usage >&2; exit 2; }
old_arg=$1; output_arg=$2; shift 2
ascend_env=""; python=python3
while [[ $# -gt 0 ]]; do
    case "$1" in
        --ascend-env) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; ascend_env=$2; shift 2 ;;
        --python) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; python=$2; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ "$output_arg" == /* && ! -e "$output_arg" && ! -L "$output_arg" ]] || { echo 'Output root must be absolute and not exist' >&2; exit 2; }
old=$(cd -- "$old_arg" && pwd -P)
parent=$(cd -- "$(dirname -- "$output_arg")" && pwd -P)
base=$(basename -- "$output_arg")
[[ "$base" != . && "$base" != .. ]] || { echo 'Invalid output root' >&2; exit 2; }
output="$parent/$base"
case "$output/" in "$old/"*) echo 'Output root cannot be inside existing build root' >&2; exit 2;; esac
[[ "$output" != *[[:space:]]* ]] || { echo 'Output path must not contain whitespace for the upstream installer' >&2; exit 2; }
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
if [[ -z "$ascend_env" ]]; then
    if [[ -n "${ASCEND_HOME_PATH:-}" && -f "${ASCEND_HOME_PATH}/bin/setenv.bash" ]]; then ascend_env="${ASCEND_HOME_PATH}/bin/setenv.bash"; else ascend_env=/usr/local/Ascend/ascend-toolkit/set_env.sh; fi
fi
[[ -f "$ascend_env" ]] || { echo "Ascend setup not found: $ascend_env" >&2; exit 2; }
# Vendor scripts may refer to unset variables.
source "$ascend_env"
set -u
python_cmd=$(command -v -- "$python") || { echo "Python executable not found: $python" >&2; exit 2; }
[[ -f "$python_cmd" && -x "$python_cmd" ]] || { echo "Python is not executable: $python_cmd" >&2; exit 2; }
python_dir=$(cd -- "$(dirname -- "$python_cmd")" && pwd -P)
python="$python_dir/$(basename -- "$python_cmd")"
export PYTHON_BIN="$python" PATH="$python_dir:$PATH" PYTHONDONTWRITEBYTECODE=1
source_root="$old/original/prepared/source"
manifest="$old/original/prepared/SOURCE_MANIFEST.json"
for file in "$old/kernel_runtime/sgl_kernel_npu/lib/libsgl_kernel_npu.so" "$old/kernel_runtime/sgl_kernel_npu/__init__.py" "$manifest" \
    "$source_root/csrc/attention/fused_quant_lightning_indexer_manage/fused_quant_lightning_indexer_manage_torch_adpt.h" \
    "$source_root/csrc/attention/fused_quant_lightning_indexer_manage/fused_quant_lightning_indexer_manage_ordered.h" \
    "$source_root/csrc/aclnn_torch_adapter/op_api_common.h" "$source_root/csrc/aclnn_torch_adapter/NPUBridge.cpp" "$source_root/csrc/aclnn_torch_adapter/NPUStorageImpl.cpp"; do
    [[ -f "$file" ]] || { echo "Missing required input: $file" >&2; exit 2; }
done
revision=$("$python" -c 'import json,re,sys; m=json.load(open(sys.argv[1],encoding="utf-8")); r=m["upstream"]["revision"]; assert m["operator"] == "lim" and re.fullmatch(r"[0-9a-f]{40}",r) and r == "7d5a4cd9d7e8c4e2e4f0f6415a71f0e80edcff8e", "unexpected original source revision or operator"; print(r)' "$manifest")
shopt -s nullglob
installers=("$source_root"/csrc/build/cann-ops-transformer*.run)
shopt -u nullglob
[[ ${#installers[@]} -eq 1 && -f "${installers[0]}" ]] || { echo "Expected exactly one original installer, got ${#installers[@]}" >&2; exit 2; }
installer=${installers[0]}
mkdir -- "$output"
exec > >(tee "$output/recover.log") 2>&1
finish() { rc=$?; trap - EXIT; printf '%s\n' "$rc" > "$output/recover.exitcode"; printf 'Recovery exit code: %s\n' "$rc"; exit "$rc"; }
trap finish EXIT
printf 'Recovering from %s to %s\nPython: %s\n' "$old" "$output" "$python"
"$python" - "$old" "$installer" "$output/recovery_manifest.json" <<'PY'
import hashlib, json, sys
from pathlib import Path
old, installer, dest = map(Path, sys.argv[1:])
inputs = [installer, old / 'kernel_runtime/sgl_kernel_npu/lib/libsgl_kernel_npu.so',
          old / 'kernel_runtime/sgl_kernel_npu/__init__.py', old / 'original/prepared/SOURCE_MANIFEST.json']
inputs += [old / name for name in ('wheel.json', 'source_provenance.json', 'doctor.json') if (old / name).is_file()]
inputs += sorted(old.glob('sgl_kernel_npu*.whl'))
csrc = old / 'original/prepared/source/csrc'
inputs += [csrc / name for name in (
    'attention/fused_quant_lightning_indexer_manage/fused_quant_lightning_indexer_manage_torch_adpt.h',
    'attention/fused_quant_lightning_indexer_manage/fused_quant_lightning_indexer_manage_ordered.h',
    'aclnn_torch_adapter/op_api_common.h', 'aclnn_torch_adapter/NPUBridge.cpp',
    'aclnn_torch_adapter/NPUStorageImpl.cpp')]
config = csrc / 'attention/fused_quant_lightning_indexer_manage/op_kernel/arch35/lim_simt_config.h'
if config.is_file():
    inputs.append(config)
files = {}
for path in inputs:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    files[str(path)] = digest.hexdigest()
dest.write_text(json.dumps({'existing_build_root': str(old), 'sha256': files}, indent=2) + '\n', encoding='utf-8')
PY
mkdir -- "$output/opp"
bash "$installer" -- "--install-path=$output/opp"
vendor="$output/opp/vendors/a5limbaseline_transformer"
[[ -f "$vendor/bin/set_env.bash" ]] || { echo 'Missing installed vendor environment' >&2; exit 1; }
set +u
source "$vendor/bin/set_env.bash"
set -u
"$python" "$script_dir/build_original_binding.py" --source-root "$source_root" --source-manifest "$manifest" --build-dir "$output/binding"
shopt -s nullglob
bindings=("$output"/binding/a5_original_lim_binding*.so)
shopt -u nullglob
[[ ${#bindings[@]} -eq 1 && -f "$output/binding/binding_manifest.json" ]] || { echo 'Expected exactly one binding and its manifest' >&2; exit 1; }
config="$source_root/csrc/attention/fused_quant_lightning_indexer_manage/op_kernel/arch35/lim_simt_config.h"
variant='source-default; verify source config; binary not independently inspected'
if [[ -f "$config" ]] && grep -Eq '^#define[[:space:]]+LIM_SIMT_POSTPROCESS[[:space:]]+1' "$config" && grep -Eq '^#define[[:space:]]+LIM_SIMT_VICTIM_UNPACK[[:space:]]+1' "$config"; then
    variant='source-default SIMT_POSTPROCESS=1 SIMT_VICTIM_UNPACK=1; binary not independently inspected'
fi
{
    printf 'export PYTHON_BIN=%q\n' "$python"
    printf 'export PATH=%q:$PATH\n' "$python_dir"
    printf 'export KERNEL_PYTHON_ROOT=%q\n' "$old/kernel_runtime"
    printf 'export ORIGINAL_LIM_BINDING=%q\n' "${bindings[0]}"
    printf 'export ORIGINAL_LIBRARY=%q\n' "${bindings[0]}"
    printf 'export ORIGINAL_OPP_ROOT=%q\n' "$vendor"
    printf 'export ORIGINAL_BINDING_MANIFEST=%q\n' "$output/binding/binding_manifest.json"
    printf 'export ORIGINAL_SOURCE_ROOT=%q\n' "$source_root"
    printf 'export ORIGINAL_SOURCE_REVISION=%q\n' "$revision"
    printf 'export ORIGINAL_SOURCE_MANIFEST=%q\n' "$manifest"
    printf 'export ORIGINAL_VARIANT=%q\n' "$variant"
    printf 'export ASCEND_CUSTOM_OPP_PATH=%q${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}\n' "$vendor"
    printf 'export LD_LIBRARY_PATH=%q${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' "$vendor/op_api/lib"
} > "$output/paths.env"
printf 'Recovery complete: %s/paths.env\n' "$output"
