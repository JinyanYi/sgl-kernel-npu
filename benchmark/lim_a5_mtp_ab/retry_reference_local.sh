#!/usr/bin/env bash
# Reuse a completed private kernel wheel and rebuild only the donor reference OPP.
set -eo pipefail

usage() {
  cat <<'EOF'
Usage: retry_reference_local.sh EXISTING_BUILD_ROOT NEW_LOCAL_ROOT
       [--ascend-env SCRIPT] [--python BASE_PYTHON]
       [--reference-root EXISTING_COMPLETE_REFERENCE]

NEW_LOCAL_ROOT must be a new absolute path on a local filesystem. The default
path rebuilds only original/ with local private build tools. With
--reference-root, a verified completed build_original.sh output is reused.
EOF
}
if [[ $# -eq 0 || ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
[[ $# -ge 2 ]] || { usage >&2; exit 2; }
old_arg=$1; new_arg=$2; shift 2
ascend_env=""; base_python=python3; reference_arg=""
while (($#)); do
  case "$1" in
    --ascend-env) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; ascend_env=$2; shift 2 ;;
    --python) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; base_python=$2; shift 2 ;;
    --reference-root) [[ $# -ge 2 ]] || { usage >&2; exit 2; }; reference_arg=$2; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ "$new_arg" == /* && ! -e "$new_arg" && ! -L "$new_arg" ]] || {
  echo 'NEW_LOCAL_ROOT must be an absolute path that does not exist (including as a symlink).' >&2; exit 2;
}
[[ -d "$old_arg" ]] || { echo "Existing build root not found: $old_arg" >&2; exit 2; }
old=$(cd -- "$old_arg" && pwd -P)
new_parent=$(cd -- "$(dirname -- "$new_arg")" 2>/dev/null && pwd -P) || {
  echo 'NEW_LOCAL_ROOT parent must already exist.' >&2; exit 2;
}
new="$new_parent/$(basename -- "$new_arg")"
case "$new/" in "$old/source/"*) echo 'NEW_LOCAL_ROOT must be outside EXISTING_BUILD_ROOT/source.' >&2; exit 2;; esac
case "$new/" in "$old/"*) echo 'NEW_LOCAL_ROOT must be outside EXISTING_BUILD_ROOT.' >&2; exit 2;; esac
[[ "$new" != *[[:space:]]* ]] || { echo 'Build paths must not contain whitespace.' >&2; exit 2; }

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
source_root=$(cd -- "$script_dir/../.." && pwd -P)
if [[ -z "$ascend_env" ]]; then
  if [[ -n "${ASCEND_HOME_PATH:-}" && -f "${ASCEND_HOME_PATH}/bin/setenv.bash" ]]; then
    ascend_env="${ASCEND_HOME_PATH}/bin/setenv.bash"
  else
    ascend_env=/usr/local/Ascend/ascend-toolkit/set_env.sh
  fi
fi
[[ -f "$ascend_env" ]] || { echo "Ascend setup not found: $ascend_env (use --ascend-env)." >&2; exit 2; }

mkdir -- "$new"
exec > >(tee -a "$new/retry-reference.log") 2>&1
finish() {
  rc=$?
  trap - EXIT
  printf '%s\n' "$rc" > "$new/retry-reference.exitcode"
  printf 'Reference retry exit code: %s\n' "$rc"
  exit "$rc"
}
trap finish EXIT

echo "Existing build root: $old"
echo "New local root: $new"
echo "Ascend setup: $ascend_env"

# Ascend setup scripts may reference unset variables; enable nounset afterwards.
source "$ascend_env"
set -u
: "${ASCEND_HOME_PATH:?Ascend setup did not set ASCEND_HOME_PATH}"

python_cmd=$(command -v -- "$base_python") || { echo "Python executable not found: $base_python" >&2; exit 2; }
[[ -f "$python_cmd" && -x "$python_cmd" ]] || { echo "Python must be an executable file: $python_cmd" >&2; exit 2; }
python_dir=$(cd -- "$(dirname -- "$python_cmd")" && pwd -P)
base_python="$python_dir/$(basename -- "$python_cmd")"
selected_python="$base_python"
export PYTHONPATH="${old}/kernel_runtime${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1

# Reject common network filesystems before creating build content, then test the
# actual mount's timestamp behavior with a real write. Never adjust system time.
mount_type=$(findmnt -T "$new_parent" -n -o FSTYPE 2>/dev/null | head -n 1 || true)
case "${mount_type,,}" in
  nfs|nfs4|cifs|smb3|smbfs|fuse.sshfs|sshfs|9p|ceph|glusterfs)
    echo "Refusing network filesystem for NEW_LOCAL_ROOT: $mount_type ($new_parent)" >&2; exit 2 ;;
  "") echo "Cannot determine filesystem type for NEW_LOCAL_ROOT parent: $new_parent" >&2; exit 2 ;;
esac
"$base_python" - "$new" <<'PY'
import os, sys, time
from pathlib import Path
root = Path(sys.argv[1])
probe = root / ".mtime-probe"
probe.write_text("mtime probe\n", encoding="utf-8")
delta = abs(probe.stat().st_mtime - time.time())
print(f"Filesystem write/mtime probe: delta={delta:.3f}s path={probe}")
probe.unlink()
if delta > 2:
    raise SystemExit(f"Filesystem timestamp differs from current time by {delta:.3f}s (>2s)")
PY

old_package="$old/kernel_runtime/sgl_kernel_npu"
[[ -f "$old/sglang-build.exitcode" && $(<"$old/sglang-build.exitcode") == 0 ]] || {
  echo 'Existing sglang build is not recorded as successful (sglang-build.exitcode must be 0).' >&2; exit 2;
}
[[ -d "$old_package" && -f "$old_package/__init__.py" ]] || { echo "Missing installed package: $old_package" >&2; exit 2; }
[[ -f "$old_package/lib/libsgl_kernel_npu.so" ]] || { echo "Missing native library: $old_package/lib/libsgl_kernel_npu.so" >&2; exit 2; }
[[ -f "$old/wheel.json" ]] || { echo "Missing wheel receipt: $old/wheel.json" >&2; exit 2; }

# Validate the exact wheel and installed package tree without importing torch.
"$base_python" - "$old" "$old_package" "$new/wheel-install-proof.json" <<'PY'
import hashlib, json, sys, zipfile
from pathlib import Path, PurePosixPath
old, installed, proof_path = map(Path, sys.argv[1:])
receipt = json.loads((old / "wheel.json").read_text(encoding="utf-8"))
wheel = Path(receipt["path"]).resolve(strict=True)
if not wheel.is_relative_to(old.resolve()):
    raise SystemExit("wheel.json path escapes EXISTING_BUILD_ROOT")
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()
actual_wheel_sha = sha(wheel)
if actual_wheel_sha != receipt.get("sha256"):
    raise SystemExit("wheel.json SHA256 does not match the wheel")
prefix = "sgl_kernel_npu/"
wheel_files = {}
with zipfile.ZipFile(wheel) as archive:
    for info in archive.infolist():
        name = info.filename
        if not name.startswith(prefix) or info.is_dir():
            continue
        rel = PurePosixPath(name[len(prefix):])
        if rel.is_absolute() or not rel.parts or any(p in ("", ".", "..") for p in rel.parts):
            raise SystemExit(f"unsafe wheel package path: {name}")
        wheel_files[rel.as_posix()] = hashlib.sha256(archive.read(info)).hexdigest()
if not wheel_files:
    raise SystemExit("wheel contains no regular sgl_kernel_npu package files")
installed_files = {
    p.relative_to(installed).as_posix(): sha(p)
    for p in installed.rglob("*")
    if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"
}
if wheel_files.keys() != installed_files.keys():
    missing = sorted(wheel_files.keys() - installed_files.keys())
    extra = sorted(installed_files.keys() - wheel_files.keys())
    raise SystemExit(f"installed package file set differs from wheel (missing={missing[:5]}, extra={extra[:5]})")
bad = [name for name, digest in wheel_files.items() if installed_files[name] != digest]
if bad:
    raise SystemExit("installed package SHA256 differs from wheel: " + ", ".join(bad[:8]))
proof = {"wheel": str(wheel), "wheel_sha256": actual_wheel_sha,
         "package_root": str(installed.resolve()), "package_file_count": len(wheel_files),
         "package_files_sha256": wheel_files, "status": "verified"}
Path(proof_path).write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"Verified wheel and installed package: {len(wheel_files)} files; SHA256={actual_wheel_sha}")
PY

# Validate the bundled production OPP by asking its own stdlib-only exporter to
# activate and verify the manifest/file hashes. This does not import torch.
"$base_python" "$source_root/tools/a5_c8/env.py" --package-root "$old_package" \
  > "$new/sglang-opp.env"
grep -q '^export ASCEND_CUSTOM_OPP_PATH=' "$new/sglang-opp.env" || { echo 'Production OPP exporter returned no bundle path.' >&2; exit 2; }
grep -q '^export LD_LIBRARY_PATH=' "$new/sglang-opp.env" || { echo 'Production OPP exporter returned no library path.' >&2; exit 2; }
echo "Verified production OPP bundle with $source_root/tools/a5_c8/env.py"

reference=""
if [[ -n "$reference_arg" ]]; then
  [[ -d "$reference_arg" ]] || { echo "Reference build root not found: $reference_arg" >&2; exit 2; }
  reference=$(cd -- "$reference_arg" && pwd -P)
  case "$new/" in "$reference/"*) echo 'NEW_LOCAL_ROOT must not be inside the reused reference root.' >&2; exit 2;; esac
  "$base_python" - "$reference" "$new/reference-proof.json" <<'PY'
import hashlib, json, re, sys
from importlib import metadata
from pathlib import Path
root, proof_path = map(Path, sys.argv[1:])
expected_revision = "7d5a4cd9d7e8c4e2e4f0f6415a71f0e80edcff8e"
def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    return h.hexdigest()
def verify_list(path, base, complete_subtree=None):
    if not path.is_file(): raise SystemExit(f"missing required hash list: {path}")
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip(): continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match: raise SystemExit(f"malformed hash row in {path}: {line!r}")
        digest, rel = match.groups()
        target = (base / rel).resolve(strict=True)
        if not target.is_relative_to(base.resolve()): raise SystemExit(f"hash path escapes root: {rel}")
        if not target.is_file() or sha(target) != digest: raise SystemExit(f"hash mismatch: {target}")
        rows[rel] = digest
    if not rows: raise SystemExit(f"empty hash list: {path}")
    scope = complete_subtree or base
    actual = {p.relative_to(base).as_posix() for p in scope.rglob("*") if p.is_file()}
    if actual != rows.keys():
        raise SystemExit(f"hash list does not cover the complete tree: {path}")
    return rows
manifest_path = root / "prepared/SOURCE_MANIFEST.json"
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
revision = manifest.get("upstream", {}).get("revision")
if manifest.get("operator") != "lim" or revision != expected_revision:
    raise SystemExit("reference source manifest is not the pinned LIM revision")
if not (root / "exit_code").is_file() or (root / "exit_code").read_text().strip() != "0":
    raise SystemExit("reference build exit_code must be 0")
source_root = root / "prepared/source"
source_rows = verify_list(root / "source_files.sha256", source_root, source_root / "csrc")
installed_base = root / "opp"
installed_rows = verify_list(root / "installed_files.sha256", installed_base)
vendor = installed_base / "vendors/a5limbaseline_transformer"
if not (vendor / "bin/set_env.bash").is_file():
    raise SystemExit("verified reference lacks the fixed a5limbaseline_transformer vendor")
if not list((vendor / "op_api/lib").glob("*.so")):
    raise SystemExit("verified reference vendor lacks op_api/lib/*.so")
binding_manifest_path = root / "binding/binding_manifest.json"
binding_manifest = json.loads(binding_manifest_path.read_text(encoding="utf-8"))
bindings = list((root / "binding").glob("a5_original_lim_binding*.so"))
if len(bindings) != 1: raise SystemExit(f"expected one reference binding .so, found {len(bindings)}")
binding = bindings[0].resolve(strict=True)
if binding_manifest.get("kind") != "donor_adapter_wrapper_not_a_rebuilt_device_kernel":
    raise SystemExit("reference binding manifest has an unexpected kind")
if Path(binding_manifest.get("library", "")).resolve() != binding:
    raise SystemExit("binding manifest library path does not identify the unique binding")
if sha(binding) != binding_manifest.get("library_sha256"):
    raise SystemExit("reference binding library SHA256 mismatch")
manifest_sha = sha(root / "prepared/SOURCE_MANIFEST.json")
if binding_manifest.get("source_manifest_sha256") != manifest_sha:
    raise SystemExit("binding manifest source_manifest_sha256 does not match the pinned source manifest")
for distribution in ("torch", "torch-npu"):
    try:
        current = metadata.version(distribution)
    except metadata.PackageNotFoundError:
        raise SystemExit(f"current Python environment lacks {distribution}; cannot safely reuse this reference")
    if binding_manifest.get(distribution.replace("-", "_")) != current:
        raise SystemExit(f"reference {distribution} version does not match current Python metadata: "
                         f"reference={binding_manifest.get(distribution.replace('-', '_'))!r}, current={current!r}")
proof = {"root": str(root.resolve()), "revision": revision, "binding": str(binding),
         "binding_sha256": sha(binding), "source_file_count": len(source_rows),
         "installed_file_count": len(installed_rows), "status": "verified-standard-build_original-output",
         "vendor": str(vendor.resolve()), "manifest_sha256": manifest_sha}
Path(proof_path).write_text(json.dumps(proof, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(f"Verified complete reference build: {root} ({len(source_rows)} source, {len(installed_rows)} installed files)")
PY
  # Recreate paths from verified artifact locations. Validate the legacy paths
  # file values first; never propagate unchecked paths into the new session.
  set +u
  source "$reference/paths.env"
  set -u
  shopt -s nullglob
  verified_bindings=("$reference"/binding/a5_original_lim_binding*.so)
  shopt -u nullglob
  [[ ${#verified_bindings[@]} -eq 1 && "${ORIGINAL_LIM_BINDING:-}" == "${verified_bindings[0]}" ]] || {
    echo 'Reference paths.env binding does not match the verified root.' >&2; exit 2;
  }
  [[ "${ORIGINAL_OPP_ROOT:-}" == "$reference/opp/vendors/a5limbaseline_transformer" && \
     "${ORIGINAL_BINDING_MANIFEST:-}" == "$reference/binding/binding_manifest.json" && \
     "${ORIGINAL_SOURCE_ROOT:-}" == "$reference/prepared/source" && \
     "${ORIGINAL_SOURCE_REVISION:-}" == 7d5a4cd9d7e8c4e2e4f0f6415a71f0e80edcff8e && \
     "${ORIGINAL_SOURCE_MANIFEST:-}" == "$reference/prepared/SOURCE_MANIFEST.json" && \
     "${ORIGINAL_SOURCE_HASHES:-}" == "$reference/source_files.sha256" && \
     "${ORIGINAL_INSTALLED_HASHES:-}" == "$reference/installed_files.sha256" ]] || {
    echo 'Reference paths.env exports do not match the verified artifacts.' >&2; exit 2;
  }
  {
    printf 'export ORIGINAL_LIM_BINDING=%q\n' "$reference/binding/$(basename -- "$ORIGINAL_LIM_BINDING")"
    printf 'export ORIGINAL_LIBRARY=%q\n' "$reference/binding/$(basename -- "$ORIGINAL_LIM_BINDING")"
    printf 'export ORIGINAL_OPP_ROOT=%q\n' "$reference/opp/vendors/a5limbaseline_transformer"
    printf 'export ORIGINAL_BINDING_MANIFEST=%q\n' "$reference/binding/binding_manifest.json"
    printf 'export ORIGINAL_SOURCE_ROOT=%q\n' "$reference/prepared/source"
    printf 'export ORIGINAL_SOURCE_REVISION=%q\n' "$ORIGINAL_SOURCE_REVISION"
    printf 'export ORIGINAL_SOURCE_MANIFEST=%q\n' "$reference/prepared/SOURCE_MANIFEST.json"
    printf 'export ORIGINAL_SOURCE_HASHES=%q\n' "$reference/source_files.sha256"
    printf 'export ORIGINAL_INSTALLED_HASHES=%q\n' "$reference/installed_files.sha256"
    printf 'export ORIGINAL_VARIANT=%q\n' "${ORIGINAL_VARIANT:-unknown}"
    printf 'export ASCEND_CUSTOM_OPP_PATH=%q${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}\n' "$reference/opp/vendors/a5limbaseline_transformer"
    printf 'export LD_LIBRARY_PATH=%q${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}\n' "$reference/opp/vendors/a5limbaseline_transformer/op_api/lib"
  } > "$new/reference-paths.env"
  source "$new/reference-paths.env"
else
  echo 'Creating private local build tools; only build-only dependencies are installed.'
  bash "$script_dir/bootstrap_deps.sh" "$new/tools" --python "$base_python"
  [[ -f "$new/tools/paths.env" ]] || { echo 'bootstrap_deps.sh did not create tools/paths.env.' >&2; exit 2; }
  source "$new/tools/paths.env"
  [[ -x "${A5_BUILD_PYTHON:-}" ]] || { echo 'Private build Python missing after bootstrap.' >&2; exit 2; }
  selected_python="$A5_BUILD_PYTHON"
  export PATH="$(dirname -- "$A5_BUILD_PYTHON"):$PATH"
  echo 'Building only the independent reference OPP and binding (jobs=4).'
  bash "$script_dir/build_original.sh" "$new/original" --ascend-env "$ascend_env" --jobs 4
  reference="$new/original"
fi
if [[ ! -f "$new/reference-paths.env" ]]; then
  [[ -f "$reference/paths.env" ]] || { echo 'Reference paths.env missing after build.' >&2; exit 2; }
  cp -- "$reference/paths.env" "$new/reference-paths.env"
fi

# Compose a fresh paths.env around the preserved wheel and verified reference.
{
  printf 'export KERNEL_PYTHON_ROOT=%q\n' "$old/kernel_runtime"
  printf 'source %q\n' "$new/reference-paths.env"
} > "$new/paths.env"
{
  printf '# Source this file in a Bash session before running the A5 smoke matrix.\n'
  printf 'source %q\n' "$ascend_env"
  printf 'export ASCEND_ENV_SCRIPT=%q\n' "$ascend_env"
  if [[ -f "$new/tools/paths.env" ]]; then printf 'source %q\n' "$new/tools/paths.env"; fi
  printf 'source %q\n' "$new/paths.env"
  printf 'export PYTHONPATH=%q${PYTHONPATH:+:$PYTHONPATH}\n' "$old/kernel_runtime"
  printf 'source %q\n' "$new/sglang-opp.env"
  printf 'export PYTHON_BIN=%q\n' "$selected_python"
  printf 'export PATH=%q:$PATH\n' "$(dirname -- "$selected_python")"
  printf 'export A5_BUILD=%q\n' "$old"
  printf 'export A5_REFERENCE=%q\n' "$reference"
  printf 'export A5_KERNEL=%q\n' "$source_root"
  printf 'export A5_ROOT=%q\n' "$(dirname -- "$old")"
} > "$new/session.env"
echo "Reference retry complete. Source: $new/session.env"
echo "Logs: $new/retry-reference.log"
