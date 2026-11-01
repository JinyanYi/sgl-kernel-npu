"""CPU-only recovery checks using a fake installer and binding compiler."""

import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "benchmark/lim_a5_mtp_ab/recover_private.sh"
BASH = shutil.which("bash") or (
    "C:/Program Files/Git/bin/bash.exe" if os.name == "nt" else None
)
PIN = "7d5a4cd9d7e8c4e2e4f0f6415a71f0e80edcff8e"


@unittest.skipUnless(BASH, "bash required")
class Recovery(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="a5recover")
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.old = self.base / "old"
        self.out = self.base / "new"
        self.env = self.base / "env.sh"
        self.env.write_text("export ASCEND_HOME_PATH=/fake\n", encoding="utf-8")
        self.python = self.base / "mockpython"
        self.python.write_text(
            """#!/bin/bash
set -e
if [[ $1 == -c ]]; then echo 7d5a4cd9d7e8c4e2e4f0f6415a71f0e80edcff8e; exit; fi
if [[ $1 == - ]]; then printf '{}\\n' > "$4"; exit; fi
shift
while [[ $# -gt 0 ]]; do
  if [[ $1 == --build-dir ]]; then out=$2; shift 2; else shift; fi
done
mkdir "$out"
printf wrapper > "$out/a5_original_lim_binding.so"
printf '{}' > "$out/binding_manifest.json"
""",
            encoding="utf-8",
        )
        self.python.chmod(0o755)
        files = [
            "kernel_runtime/sgl_kernel_npu/lib/libsgl_kernel_npu.so",
            "kernel_runtime/sgl_kernel_npu/__init__.py",
            "original/prepared/SOURCE_MANIFEST.json",
            "original/prepared/source/csrc/attention/fused_quant_lightning_indexer_manage/fused_quant_lightning_indexer_manage_torch_adpt.h",
            "original/prepared/source/csrc/attention/fused_quant_lightning_indexer_manage/fused_quant_lightning_indexer_manage_ordered.h",
            "original/prepared/source/csrc/aclnn_torch_adapter/op_api_common.h",
            "original/prepared/source/csrc/aclnn_torch_adapter/NPUBridge.cpp",
            "original/prepared/source/csrc/aclnn_torch_adapter/NPUStorageImpl.cpp",
        ]
        for name in files:
            p = self.old / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("fixture", encoding="utf-8")
        installer = (
            self.old
            / "original/prepared/source/csrc/build/cann-ops-transformer-mock.run"
        )
        installer.parent.mkdir(parents=True)
        installer.write_text(
            """#!/bin/bash
set -e
[[ $# == 2 && $1 == -- ]] || exit 32
case "$2" in --install-path=*) root=${2#--install-path=} ;; *) exit 33 ;; esac
vendor="$root/vendors/a5limbaseline_transformer"
mkdir -p "$vendor/bin" "$vendor/op_api/lib"
printf 'export MOCK_VENDOR=1\\n' > "$vendor/bin/set_env.bash"
""",
            encoding="utf-8",
        )
        self.installer = installer

    def shell_path(self, path):
        value = path.as_posix()
        return "/" + value[0].lower() + value[2:] if os.name == "nt" else value

    def run_recover(self, output=None, python=None):
        return subprocess.run(
            [
                BASH,
                str(SCRIPT),
                self.shell_path(self.old),
                self.shell_path(output or self.out),
                "--ascend-env",
                self.shell_path(self.env),
                "--python",
                self.shell_path(python or self.python),
            ],
            capture_output=True,
            text=True,
        )

    def snapshot(self):
        return {
            str(p.relative_to(self.old)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in self.old.rglob("*")
            if p.is_file()
        }

    def test_complete_preserves_inputs_and_exports_paths(self):
        before = self.snapshot()
        result = self.run_recover()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(before, self.snapshot())
        self.assertEqual((self.out / "recover.exitcode").read_text().strip(), "0")
        paths = (self.out / "paths.env").read_text()
        for key in (
            "PYTHON_BIN",
            "KERNEL_PYTHON_ROOT",
            "ORIGINAL_LIM_BINDING",
            "ORIGINAL_SOURCE_REVISION",
            "ASCEND_CUSTOM_OPP_PATH",
            "LD_LIBRARY_PATH",
        ):
            self.assertIn("export " + key + "=", paths)
        self.assertIn(PIN, paths)
        self.assertTrue((self.out / "recovery_manifest.json").is_file())

    def test_real_python_validates_manifest_and_hashes_inputs(self):
        manifest = self.old / "original/prepared/SOURCE_MANIFEST.json"
        manifest.write_text(
            json.dumps({"operator": "lim", "upstream": {"revision": PIN}})
        )
        shim = self.base / "realpython-shim"
        shim.write_text(
            "#!/bin/bash\n"
            "if [[ $1 == */build_original_binding.py ]]; then\n"
            "  shift\n"
            "  while [[ $# -gt 0 ]]; do\n"
            "    if [[ $1 == --build-dir ]]; then out=$2; shift 2; else shift; fi\n"
            "  done\n"
            '  mkdir "$out"\n'
            '  printf wrapper > "$out/a5_original_lim_binding.so"\n'
            '  printf "{}" > "$out/binding_manifest.json"\n'
            "else\n"
            f'  exec {shlex.quote(self.shell_path(Path(sys.executable)))} "$@"\n'
            "fi\n",
            encoding="utf-8",
        )
        shim.chmod(0o755)
        before = self.snapshot()
        result = self.run_recover(python=shim)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(before, self.snapshot())
        recovery = json.loads((self.out / "recovery_manifest.json").read_text())
        hashes = {
            Path(key).resolve(): value for key, value in recovery["sha256"].items()
        }
        library = self.old / "kernel_runtime/sgl_kernel_npu/lib/libsgl_kernel_npu.so"
        for path in (library, self.installer):
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(hashes[path.resolve()], expected)
        self.assertIn(PIN, (self.out / "paths.env").read_text())

    def test_installer_failure_preserves_logs_and_old_tree(self):
        self.installer.write_text("#!/bin/bash\necho intentional-failure\nexit 42\n")
        before = self.snapshot()
        result = self.run_recover()
        self.assertEqual(result.returncode, 42, result.stdout + result.stderr)
        self.assertEqual(before, self.snapshot())
        self.assertEqual((self.out / "recover.exitcode").read_text().strip(), "42")
        self.assertIn("intentional-failure", (self.out / "recover.log").read_text())
        self.assertFalse((self.out / "paths.env").exists())

    def test_rejects_existing_and_nested_outputs(self):
        before = self.snapshot()
        self.out.mkdir()
        (self.out / "marker").write_text("keep")
        self.assertNotEqual(self.run_recover().returncode, 0)
        self.assertEqual((self.out / "marker").read_text(), "keep")
        self.assertNotEqual(self.run_recover(self.old / "nested").returncode, 0)
        self.assertFalse((self.old / "nested").exists())
        self.assertEqual(before, self.snapshot())

    def test_invalid_source_manifest_fails_with_real_python(self):
        result = subprocess.run(
            [
                BASH,
                str(SCRIPT),
                self.shell_path(self.old),
                self.shell_path(self.out),
                "--ascend-env",
                self.shell_path(self.env),
                "--python",
                self.shell_path(Path(sys.executable)),
            ],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.out.exists())

    def test_missing_input_fails_before_creation(self):
        self.installer.unlink()
        result = self.run_recover()
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()
