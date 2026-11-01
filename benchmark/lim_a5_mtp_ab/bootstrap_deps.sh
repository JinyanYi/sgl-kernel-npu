#!/usr/bin/env bash
# Create a private build-tool virtualenv while exposing existing site packages.
set -euo pipefail

PYTHON_BIN="${PYTHON:-python3}"
VENV_ROOT=""
REQUIREMENTS_FILE=""
WHEELHOUSE=""
DRY_RUN=0

to_unix_path() {
  local input="$1"
  if [[ "$input" =~ ^[A-Za-z]:[/\\] ]]; then
    if command -v cygpath >/dev/null 2>&1; then
      cygpath -u "$input"
    else
      local drive="${input:0:1}"
      local tail="${input:2}"
      tail="${tail//\\//}"
      printf '/%s%s\n' "${drive,,}" "$tail"
    fi
  else
    printf '%s\n' "$input"
  fi
}

usage() {
  cat <<'EOF'
Usage: bootstrap_deps.sh NEW_ABSOLUTE_ROOT [--python PATH]
       [--requirements PINNED_BUILD_REQUIREMENTS.txt]
       [--wheelhouse DIR] [--dry-run]

Creates NEW_ABSOLUTE_ROOT/venv using --system-site-packages and writes
NEW_ABSOLUTE_ROOT/paths.env. The default installs private build tools only;
their pins are tool choices, not an A5/CANN compatibility matrix. Additional
build-only packages may be supplied in an exact-version requirements file.
Torch, torch-npu, CANN, device drivers, and operator packages are never installed.
EOF
}

if (($# == 0)); then usage >&2; exit 2; fi
if [[ "$1" == -h || "$1" == --help ]]; then usage; exit 0; fi
VENV_ROOT="$1"
shift
while (($#)); do
  case "$1" in
    --python) PYTHON_BIN="${2:?missing value for --python}"; shift 2 ;;
    --requirements) REQUIREMENTS_FILE="${2:?missing value for --requirements}"; shift 2 ;;
    --wheelhouse) WHEELHOUSE="${2:?missing value for --wheelhouse}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
  esac
done

VENV_ROOT="$(to_unix_path "$VENV_ROOT")"
if [[ -n "$REQUIREMENTS_FILE" ]]; then
  REQUIREMENTS_FILE="$(to_unix_path "$REQUIREMENTS_FILE")"
fi
if [[ -n "$WHEELHOUSE" ]]; then
  WHEELHOUSE="$(to_unix_path "$WHEELHOUSE")"
fi

if [[ "$VENV_ROOT" != /* ]]; then
  printf 'The new venv root must be an absolute path.\n' >&2
  exit 2
fi
if [[ -e "$VENV_ROOT" ]]; then
  printf 'Refusing to overwrite existing venv root: %s\n' "$VENV_ROOT" >&2
  exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  printf 'Python executable not found: %s\n' "$PYTHON_BIN" >&2
  exit 2
fi
if [[ -n "$WHEELHOUSE" && ! -d "$WHEELHOUSE" ]]; then
  printf 'Wheelhouse does not exist: %s\n' "$WHEELHOUSE" >&2
  exit 2
fi
if [[ -n "$REQUIREMENTS_FILE" && ! -f "$REQUIREMENTS_FILE" ]]; then
  printf 'Requirements file does not exist: %s\n' "$REQUIREMENTS_FILE" >&2
  exit 2
fi

INSTALL_SPECS=("cmake==3.30.5" "ninja==1.11.1" "wheel==0.45.1")
if [[ -n "$REQUIREMENTS_FILE" ]]; then
  while IFS= read -r line || [[ -n "$line" ]]; do
    line="${line%%#*}"
    line="${line//[$'\r\t ']/}"
    [[ -z "$line" ]] && continue
    if [[ ! "$line" =~ ^([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)$ ]]; then
      printf 'Only exact pinned package requirements are accepted: %s\n' "$line" >&2
      exit 2
    fi
    package="${BASH_REMATCH[1],,}"
    normalized="${package//_/-}"
    normalized="${normalized//./-}"
    case "$normalized" in
      cmake|ninja|pyyaml|numpy|setuptools|wheel|packaging|nlohmann-json) ;;
      *)
        printf 'Package is not on the build-only allowlist: %s\n' "$package" >&2
        exit 2
        ;;
    esac
    if [[ "$normalized" == wheel && "$line" != 'wheel==0.45.1' ]]; then
      printf 'This repository pins wheel==0.45.1; conflicting pin rejected: %s\n' "$line" >&2
      exit 2
    fi
    if [[ "$normalized" == cmake && "$line" != 'cmake==3.30.5' ]]; then
      printf 'This bootstrap pins its private CMake tool to 3.30.5: %s\n' "$line" >&2
      exit 2
    fi
    if [[ "$normalized" == ninja && "$line" != 'ninja==1.11.1' ]]; then
      printf 'This bootstrap pins its private Ninja tool to 1.11.1: %s\n' "$line" >&2
      exit 2
    fi
    INSTALL_SPECS+=("$line")
  done < "$REQUIREMENTS_FILE"
fi

if ((DRY_RUN)); then
  printf 'Would create: %s/venv\n' "$VENV_ROOT"
  printf 'Python: %s\n' "$PYTHON_BIN"
  printf 'Isolation: --system-site-packages\n'
  printf 'Build-only installs: %s\n' "${INSTALL_SPECS[*]}"
  printf 'Wheelhouse: %s\n' "${WHEELHOUSE:-<configured pip index>}"
  printf 'Protected distributions: torch, torch-npu, torchvision, CANN, driver, custom OPP\n'
  exit 0
fi

mkdir -p "$VENV_ROOT"
exec > >(tee "$VENV_ROOT/bootstrap.log") 2>&1
trap 'rc=$?; printf "%s\n" "$rc" > "$VENV_ROOT/bootstrap.exitcode"' EXIT
"$PYTHON_BIN" -m venv --system-site-packages "$VENV_ROOT/venv"
VENV_PYTHON="$VENV_ROOT/venv/bin/python"

"$VENV_PYTHON" - "$VENV_ROOT/site-packages-before.json" <<'PY'
import importlib.metadata
import json
import sys
from pathlib import Path

result = {}
for name in ("torch", "torch-npu"):
    try:
        distribution = importlib.metadata.distribution(name)
        result[name] = {"version": distribution.version,
                        "location": str(Path(distribution.locate_file("")).resolve())}
    except importlib.metadata.PackageNotFoundError:
        result[name] = None
Path(sys.argv[1]).write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
PY

PIP_ARGS=(--disable-pip-version-check --no-deps)
if [[ -n "$WHEELHOUSE" ]]; then
  PIP_ARGS+=(--no-index --find-links "$WHEELHOUSE")
fi
"$VENV_ROOT/venv/bin/python" -m pip install "${PIP_ARGS[@]}" "${INSTALL_SPECS[@]}"

"$VENV_PYTHON" - "$VENV_ROOT/site-packages-before.json" "$VENV_ROOT/site-packages-after.json" <<'PY'
import importlib.metadata
import json
import sys
from pathlib import Path

before = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
after = {}
for name in ("torch", "torch-npu"):
    try:
        distribution = importlib.metadata.distribution(name)
        after[name] = {"version": distribution.version,
                       "location": str(Path(distribution.locate_file("")).resolve())}
    except importlib.metadata.PackageNotFoundError:
        after[name] = None
Path(sys.argv[2]).write_text(json.dumps(after, indent=2) + "\n", encoding="utf-8")
if before != after:
    raise SystemExit("Refusing bootstrap: torch or torch-npu metadata changed in the private environment")
PY

cat > "$VENV_ROOT/paths.env" <<EOF
source $(printf '%q' "$VENV_ROOT/venv/bin/activate")
export A5_BUILD_PYTHON=$(printf '%q' "$VENV_PYTHON")
EOF
printf 'Created private build-tools environment at %s/venv\n' "$VENV_ROOT"
printf 'Source %s/paths.env to activate it.\n' "$VENV_ROOT"
