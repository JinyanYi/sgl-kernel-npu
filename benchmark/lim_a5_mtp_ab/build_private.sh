#!/usr/bin/env bash
# Build into a task-private package root; never replace the installed wheel.
set -eo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SOURCE_ROOT=$(cd -- "$HERE/../.." && pwd -P)
BUILD_ARTIFACT_ROOT=${1:?Usage: bash build_private.sh /new/absolute/build-artifacts}
PYTHON=${PYTHON:-python3}
SOC_VERSION=${SOC_VERSION:-Ascend950}
if [[ "$BUILD_ARTIFACT_ROOT" != /* || -e "$BUILD_ARTIFACT_ROOT" ]]; then
  echo 'Choose a new absolute artifact directory; existing results are preserved.' >&2
  exit 2
fi
# Resolve existing path components before rejecting descendants, including symlinks.
BUILD_ARTIFACT_ROOT=$(realpath -m -- "$BUILD_ARTIFACT_ROOT")
case "$BUILD_ARTIFACT_ROOT/" in
  "$SOURCE_ROOT/"*) echo 'Artifact directory must be outside the source tree.' >&2; exit 2 ;;
esac
mkdir -p -- "$BUILD_ARTIFACT_ROOT/logs"
trap 'rc=$?; printf "%s\n" "$rc" > "$BUILD_ARTIFACT_ROOT/build.exitcode"' EXIT
ASCEND_ENV=${ASCEND_ENV:-/usr/local/Ascend/ascend-toolkit/set_env.sh}
source "$ASCEND_ENV"
set -u
python_cmd=$(command -v -- "$PYTHON") || { echo "Python executable not found: $PYTHON" >&2; exit 2; }
[[ -f "$python_cmd" && -x "$python_cmd" ]] || { echo "Python must be an executable file: $python_cmd" >&2; exit 2; }
python_dir=$(cd -- "$(dirname -- "$python_cmd")" && pwd -P)
# Preserve the executable symlink: resolving it can escape a virtual environment.
PYTHON="$python_dir/$(basename -- "$python_cmd")"
PYTHON_BIN="$PYTHON"
shim_dir="$BUILD_ARTIFACT_ROOT/python_shims"
mkdir -- "$shim_dir"
printf '#!/usr/bin/env bash\nexec %q "$@"\n' "$PYTHON" > "$shim_dir/python3"
printf '#!/usr/bin/env bash\nexec %q -m pip "$@"\n' "$PYTHON" > "$shim_dir/pip3"
chmod +x "$shim_dir/python3" "$shim_dir/pip3"
export PATH="$shim_dir:$python_dir:$PATH" PYTHON PYTHON_BIN
export BUILD_ARTIFACT_ROOT SOURCE_ROOT SOC_VERSION
"$PYTHON" "$HERE/doctor.py" --output "$BUILD_ARTIFACT_ROOT/doctor.json" \
  2>&1 | tee "$BUILD_ARTIFACT_ROOT/logs/doctor.log"
"$PYTHON" - <<'PY' > "$BUILD_ARTIFACT_ROOT/source_provenance.json"
import hashlib
import json
import os
import subprocess
from pathlib import Path

root = Path(os.environ["SOURCE_ROOT"])
def git(*args):
    try:
        result = subprocess.run(["git", "-C", str(root), *args], capture_output=True)
    except FileNotFoundError:
        return None
    return result.stdout.decode(errors="replace") if result.returncode == 0 else None
found_top = git("rev-parse", "--show-toplevel")
is_root = bool(found_top and Path(found_top.strip()).resolve() == root.resolve())
diff = git("diff", "HEAD", "--binary") if is_root else None
print(json.dumps({"source_root": str(root), "revision": git("rev-parse", "HEAD") if is_root else None,
                  "status": git("status", "--short") if is_root else None,
                  "diff_sha256": hashlib.sha256(diff.encode()).hexdigest() if diff is not None else None,
                  "ascend_home": os.environ.get("ASCEND_HOME_PATH"),
                  "target": os.environ.get("SOC_VERSION", "Ascend950")}, indent=2))
PY
# Keep source Git metadata private, retaining HEAD for the CMake commit stamp.
# The working tree below is a copy of current files, including user changes.
PRIVATE_SOURCE="$BUILD_ARTIFACT_ROOT/sglang_source"
source_git_top=$(git -C "$SOURCE_ROOT" rev-parse --show-toplevel 2>/dev/null || true)
if [[ -n "$source_git_top" && "$(cd -- "$source_git_top" && pwd -P)" == "$SOURCE_ROOT" ]]; then
  git clone --shared --no-checkout -- "$SOURCE_ROOT" "$PRIVATE_SOURCE" \
    2>&1 | tee "$BUILD_ARTIFACT_ROOT/logs/source-clone.log"
else
  mkdir -- "$PRIVATE_SOURCE"
  printf 'Source has no matching Git root; copying without a Git stamp.\n' > "$BUILD_ARTIFACT_ROOT/logs/source-clone.log"
fi
"$PYTHON" - "$SOURCE_ROOT" "$PRIVATE_SOURCE" <<'PY'
import shutil
import sys
from pathlib import Path

source, destination = map(Path, sys.argv[1:])

def ignored(directory, names):
    relative = Path(directory).relative_to(source)
    excluded = {name for name in names if name in {".git", "__pycache__", ".pytest_cache", ".mypy_cache"} or name.endswith(".egg-info")}
    if relative == Path("."):
        excluded.update({"build", "output"} & set(names))
    if relative == Path("python/sgl_kernel_npu"):
        excluded.update({"build", "dist"} & set(names))
    if relative == Path("python/sgl_kernel_npu/sgl_kernel_npu"):
        excluded.update({"lib"} & set(names))
    return excluded

shutil.copytree(source, destination, dirs_exist_ok=True, symlinks=True, ignore=ignored)
PY
cd -- "$PRIVATE_SOURCE"
# A source tarball must not inherit an unrelated enclosing Git repository.
export GIT_CEILING_DIRECTORIES="$BUILD_ARTIFACT_ROOT${GIT_CEILING_DIRECTORIES:+:$GIT_CEILING_DIRECTORIES}"
set +e
bash build.sh -a kernels "$SOC_VERSION" 2>&1 | tee "$BUILD_ARTIFACT_ROOT/logs/sglang-build.log"
build_rc=${PIPESTATUS[0]}
set -e
printf '%s\n' "$build_rc" > "$BUILD_ARTIFACT_ROOT/sglang-build.exitcode"
if (( build_rc != 0 )); then exit "$build_rc"; fi
"$PYTHON" - "$PRIVATE_SOURCE" "$BUILD_ARTIFACT_ROOT" <<'PY'
import hashlib
import json
import shutil
import sys
from pathlib import Path

source, target = map(Path, sys.argv[1:])
wheels = list((source / "output").glob("sgl_kernel_npu*.whl"))
if len(wheels) != 1:
    raise SystemExit("Expected one fresh kernel wheel; inspect output/ before retrying")
destination = target / wheels[0].name
shutil.copy2(wheels[0], destination)
with destination.open("rb") as stream:
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    checksum = digest.hexdigest()
(target / "wheel.json").write_text(json.dumps({"path": str(destination),
                                             "sha256": checksum}, indent=2) + "\n")
PY
wheel=("$BUILD_ARTIFACT_ROOT"/sgl_kernel_npu*.whl)
"$PYTHON" -m pip install --no-deps --target "$BUILD_ARTIFACT_ROOT/kernel_runtime" "${wheel[0]}" \
  2>&1 | tee "$BUILD_ARTIFACT_ROOT/logs/private-install.log"
bash "$HERE/build_original.sh" "$BUILD_ARTIFACT_ROOT/original" --ascend-env "$ASCEND_ENV" \
  2>&1 | tee "$BUILD_ARTIFACT_ROOT/logs/original-build.log"
printf 'export KERNEL_PYTHON_ROOT=%q\n' "$BUILD_ARTIFACT_ROOT/kernel_runtime" \
  > "$BUILD_ARTIFACT_ROOT/paths.env"
printf 'source %q\n' "$BUILD_ARTIFACT_ROOT/original/paths.env" \
  >> "$BUILD_ARTIFACT_ROOT/paths.env"
printf 'Private build complete. Read %s/paths.env before running.\n' "$BUILD_ARTIFACT_ROOT"
