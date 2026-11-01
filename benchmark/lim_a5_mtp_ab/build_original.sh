#!/usr/bin/env bash
# Build and install the donor A5 LIM OPP into a fresh private directory.
set -eo pipefail

usage() {
    cat <<'EOF'
Usage: build_original.sh OUTPUT_ROOT [--source-root DONOR_ROOT] [--ascend-env SET_ENV] [--jobs N]
OUTPUT_ROOT must not exist; its parent must exist. The default donor comes from
prepare_original.py. An explicit donor is copied before building. No system OPP
is installed or overwritten. Source the emitted paths.env after Ascend setup.
EOF
}
if [[ $# -eq 0 || ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
output_arg=$1; shift
source_arg=""; ascend_env=""; jobs=8
while [[ $# -gt 0 ]]; do
    case "$1" in
        --source-root) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; source_arg=$2; shift 2 ;;
        --ascend-env) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; ascend_env=$2; shift 2 ;;
        --jobs) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; jobs=$2; shift 2 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done
[[ "$jobs" =~ ^[1-9][0-9]*$ ]] || { echo "--jobs must be a positive integer" >&2; exit 2; }
[[ ! -e "$output_arg" ]] || { echo "Output root already exists: $output_arg" >&2; exit 2; }
parent=$(cd -- "$(dirname -- "$output_arg")" && pwd -P)
base=$(basename -- "$output_arg")
[[ "$base" != . && "$base" != .. ]] || { echo "Invalid output root" >&2; exit 2; }
output="$parent/$base"
# Upstream installer uses some unquoted paths internally.
[[ "$output" != *[[:space:]]* ]] || { echo "Output path must not contain whitespace for the upstream installer" >&2; exit 2; }
if [[ -n "$source_arg" ]]; then
    checked_source=$(cd -- "$source_arg" && pwd -P)
    case "$output/" in "$checked_source/"*) echo "Output cannot be inside the source tree" >&2; exit 2;; esac
fi
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
if [[ -z "$ascend_env" ]]; then
    if [[ -n "${ASCEND_HOME_PATH:-}" && -f "${ASCEND_HOME_PATH}/bin/setenv.bash" ]]; then
        ascend_env="${ASCEND_HOME_PATH}/bin/setenv.bash"
    else
        ascend_env=/usr/local/Ascend/ascend-toolkit/set_env.sh
    fi
fi
[[ -f "$ascend_env" ]] || { echo "Ascend setup not found: $ascend_env (use --ascend-env)" >&2; exit 2; }
# Vendor setup scripts may reference unset variables, so nounset is enabled afterwards.
source "$ascend_env"
set -u
: "${ASCEND_HOME_PATH:?Ascend setup did not set ASCEND_HOME_PATH}"
mkdir -- "$output"
exec > >(tee "$output/build.log") 2>&1
finish() {
    rc=$?
    trap - EXIT
    printf '%s\n' "$rc" > "$output/exit_code"
    printf 'build_original exit code: %s\n' "$rc"
    exit "$rc"
}
trap finish EXIT
printf 'Started %s\n' "$(date -u +%FT%TZ)"
printf 'Ascend setup: %s\n' "$ascend_env"
printf 'Output root: %s\n' "$output"
source_revision=unknown
if [[ -n "$source_arg" ]]; then
    donor=$(cd -- "$source_arg" && pwd -P)
    git_top=$(git -C "$donor" rev-parse --show-toplevel 2>/dev/null || true)
    if [[ -n "$git_top" && "$(cd -- "$git_top" && pwd -P)" == "$donor" ]] && [[ -z "$(git -C "$donor" status --porcelain)" ]]; then
        source_revision=$(git -C "$donor" rev-parse HEAD)
    fi
    [[ -f "$donor/csrc/build.sh" ]] || { echo "Explicit donor lacks csrc/build.sh" >&2; exit 1; }
    mkdir -- "$output/prepared"
    cp -a -- "$donor" "$output/prepared/source"
    if [[ -f "$donor/SOURCE_MANIFEST.json" ]]; then
        cp -- "$donor/SOURCE_MANIFEST.json" "$output/prepared/SOURCE_MANIFEST.json"
    elif [[ -f "$(dirname -- "$donor")/SOURCE_MANIFEST.json" ]]; then
        cp -- "$(dirname -- "$donor")/SOURCE_MANIFEST.json" "$output/prepared/SOURCE_MANIFEST.json"
    fi
else
    python3 "$script_dir/prepare_original.py" --output-root "$output/prepared" --operator lim
    source_revision=$(python3 -c 'import json, re, sys; m=json.load(open(sys.argv[1], encoding="utf-8")); r=m["upstream"]["revision"]; assert m["operator"] == "lim" and re.fullmatch(r"[0-9a-f]{40}", r); print(r)' "$output/prepared/SOURCE_MANIFEST.json")
fi
source_root="$output/prepared/source"
csrc="$source_root/csrc"
[[ -f "$csrc/build.sh" ]] || { echo "Prepared source lacks csrc/build.sh" >&2; exit 1; }
for generated in build output build_out; do
    [[ ! -e "$csrc/$generated" ]] || { echo "Donor copy contains old csrc/$generated; refusing ambiguous build" >&2; exit 1; }
done
(cd "$source_root" && find csrc -type f -print0 | sort -z | xargs -0 sha256sum) > "$output/source_files.sha256"
printf 'Building donor OPP: bash build.sh --pkg --soc=ascend950 --ops=fused_quant_lightning_indexer_manage --vendor_name=a5limbaseline -j%s\n' "$jobs"
(cd "$csrc" && bash build.sh --pkg --soc=ascend950 --ops=fused_quant_lightning_indexer_manage --vendor_name=a5limbaseline "-j$jobs")
shopt -s nullglob
installers=("$csrc"/build/*.run)
shopt -u nullglob
[[ ${#installers[@]} -eq 1 ]] || { echo "Expected exactly one built .run installer, got ${#installers[@]}" >&2; exit 1; }
install_root="$output/opp"
mkdir -- "$install_root"
installer=${installers[0]}
printf 'Installing %s into private %s\n' "$installer" "$install_root"
# Makeself consumes options before `--`; keep installer-specific flags after it.
bash "$installer" -- "--install-path=$install_root"
vendor="$install_root/vendors/a5limbaseline_transformer"
vendor_env="$vendor/bin/set_env.bash"
[[ -f "$vendor_env" ]] || { echo "Missing installed donor vendor environment: $vendor_env" >&2; exit 1; }
(cd "$install_root" && find vendors -type f -print0 | sort -z | xargs -0 sha256sum) > "$output/installed_files.sha256"
sha256sum "$installer" > "$output/installer.sha256"
set +u
source "$vendor_env"
set -u
binding_args=(--source-root "$source_root" --build-dir "$output/binding")
if [[ -f "$output/prepared/SOURCE_MANIFEST.json" ]]; then
    binding_args+=(--source-manifest "$output/prepared/SOURCE_MANIFEST.json")
fi
python3 "$script_dir/build_original_binding.py" "${binding_args[@]}"
shopt -s nullglob
bindings=("$output"/binding/a5_original_lim_binding*.so)
shopt -u nullglob
[[ ${#bindings[@]} -eq 1 ]] || { echo "Expected exactly one binding .so, got ${#bindings[@]}" >&2; exit 1; }
config="$csrc/attention/fused_quant_lightning_indexer_manage/op_kernel/arch35/lim_simt_config.h"
variant='source-default; verify build log and source config'
if [[ -f "$config" ]] && grep -Eq '^#define[[:space:]]+LIM_SIMT_POSTPROCESS[[:space:]]+1' "$config" && grep -Eq '^#define[[:space:]]+LIM_SIMT_VICTIM_UNPACK[[:space:]]+1' "$config"; then
    variant='source-default SIMT_POSTPROCESS=1 SIMT_VICTIM_UNPACK=1; binary not independently inspected'
fi
{
    printf 'export ORIGINAL_LIM_BINDING=%q\n' "${bindings[0]}"
    printf 'export ORIGINAL_LIBRARY=%q\n' "${bindings[0]}"
    printf 'export ORIGINAL_OPP_ROOT=%q\n' "$vendor"
    printf 'export ORIGINAL_BINDING_MANIFEST=%q\n' "$output/binding/binding_manifest.json"
    printf 'export ORIGINAL_SOURCE_ROOT=%q\n' "$source_root"
    printf 'export ORIGINAL_SOURCE_REVISION=%q\n' "$source_revision"
    if [[ -f "$output/prepared/SOURCE_MANIFEST.json" ]]; then
        printf 'export ORIGINAL_SOURCE_MANIFEST=%q\n' "$output/prepared/SOURCE_MANIFEST.json"
    else
        printf 'export ORIGINAL_SOURCE_MANIFEST=%q\n' ""
    fi
    printf 'export ORIGINAL_SOURCE_HASHES=%q\n' "$output/source_files.sha256"
    printf 'export ORIGINAL_INSTALLED_HASHES=%q\n' "$output/installed_files.sha256"
    printf 'export ORIGINAL_VARIANT=%q\n' "$variant"
    printf 'export ASCEND_CUSTOM_OPP_PATH=%q${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}\n' "$vendor"
    printf 'export LD_LIBRARY_PATH=%q${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' "$vendor/op_api/lib"
} > "$output/paths.env"
printf 'Paths: %s\n' "$output/paths.env"
printf 'Build and private install completed. Device execution and binary dispatch remain unverified.\n'
