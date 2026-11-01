"""CPU-only checks for private donor build entry points."""

import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / "benchmark" / "lim_a5_mtp_ab"
SHELL = BASE / "build_original.sh"
BINDING = BASE / "build_original_binding.py"
MAKESELF_ARCHIVE = BASE / "vendor" / "makeself-upstream-2.5.0.tar.gz"
BASH = shutil.which("bash")
if not BASH and os.name == "nt":
    candidate = Path("C:/Program Files/Git/bin/bash.exe")
    if candidate.is_file():
        BASH = str(candidate)


class BindingPreflight(unittest.TestCase):
    def test_help_is_independent_of_torch(self):
        result = subprocess.run(
            [sys.executable, str(BINDING), "--help"],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--source-root", result.stdout)
        self.assertIn("--source-manifest", result.stdout)

    def test_existing_build_dir_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, str(BINDING), "--build-dir", directory],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("build directory exists", result.stderr)


@unittest.skipUnless(BASH, "bash is required")
class ShellPreflight(unittest.TestCase):
    def run_bash(self, *args):
        return subprocess.run(
            [BASH, str(SHELL), *args], capture_output=True, text=True, check=False
        )

    def test_syntax_and_help(self):
        result = subprocess.run(
            [BASH, "-n", str(SHELL)], capture_output=True, text=True, check=False
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.run_bash("--help")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--source-root", result.stdout)

    def test_existing_directory_is_not_modified(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            marker = path / "keep.txt"
            marker.write_text("keep", encoding="utf-8")
            result = self.run_bash(path.as_posix())
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
            self.assertIn("already exists", result.stderr)

    def test_failed_donor_build_preserves_log_and_exit_code(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            donor = base / "donor"
            (donor / "csrc").mkdir(parents=True)
            (donor / "csrc" / "build.sh").write_text(
                "#!/bin/bash\necho intentional-build-failure\nexit 42\n",
                encoding="utf-8",
            )
            env_script = base / "set_env.sh"
            env_script.write_text(
                "export ASCEND_HOME_PATH=/fake/ascend\n", encoding="utf-8"
            )
            output = base / "output"
            result = self.run_bash(
                output.as_posix(),
                "--source-root",
                donor.as_posix(),
                "--ascend-env",
                env_script.as_posix(),
            )
            self.assertEqual(result.returncode, 42, result.stderr + result.stdout)
            self.assertEqual((output / "exit_code").read_text().strip(), "42")
            self.assertIn(
                "intentional-build-failure", (output / "build.log").read_text()
            )
            self.assertFalse((output / "paths.env").exists())
            self.assertFalse((donor / "csrc" / "build").exists())

    def test_upstream_makeself_requires_separator_for_installer_arguments(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            makeself_root = base / "makeself"
            with tarfile.open(MAKESELF_ARCHIVE, "r:gz") as archive:
                archive.extractall(makeself_root, filter="data")
            makeself = makeself_root / "makeself-release-2.5.0" / "makeself.sh"
            payload = base / "payload"
            payload.mkdir()
            received = base / "received-args.txt"
            startup = payload / "install.sh"
            startup.write_text(
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > "
                + repr(received.as_posix())
                + "\n",
                encoding="utf-8",
            )
            chmod = subprocess.run(
                [BASH, "-c", 'chmod +x "$1"', "bash", str(startup)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(chmod.returncode, 0, chmod.stderr)
            installer = base / "installer.run"
            created = subprocess.run(
                [
                    BASH,
                    str(makeself),
                    str(payload),
                    str(installer),
                    "test",
                    "./install.sh",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(created.returncode, 0, created.stderr + created.stdout)

            install_root = base / "private-opp"
            without_separator = subprocess.run(
                [BASH, str(installer), f"--install-path={install_root}"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(without_separator.returncode, 0)
            self.assertIn(
                "Unrecognized flag", without_separator.stderr + without_separator.stdout
            )
            self.assertFalse(received.exists())

            with_separator = subprocess.run(
                [BASH, str(installer), "--", f"--install-path={install_root}"],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(
                with_separator.returncode,
                0,
                with_separator.stderr + with_separator.stdout,
            )
            self.assertEqual(
                received.read_text(encoding="utf-8").strip(),
                f"--install-path={install_root}",
            )

    def test_mock_build_installs_only_into_private_root(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            donor = base / "donor"
            (donor / "csrc").mkdir(parents=True)
            build_script = """#!/bin/bash
set -e
printf '%s\\n' "$@" > arguments.txt
mkdir -p build
cat > build/fake.run <<'INSTALLER'
#!/bin/bash
set -e
[[ $1 == -- ]] || exit 32
shift
case "$1" in --install-path=*) root=${1#--install-path=} ;; *) exit 33 ;; esac
vendor="$root/vendors/a5limbaseline_transformer"
mkdir -p "$vendor/bin" "$vendor/op_api/lib"
printf 'export ASCEND_CUSTOM_OPP_PATH=%s\\n' "$vendor" > "$vendor/bin/set_env.bash"
printf library > "$vendor/op_api/lib/libcust_opapi.so"
INSTALLER
"""
            (donor / "csrc" / "build.sh").write_text(build_script, encoding="utf-8")
            bin_dir = base / "bin"
            bin_dir.mkdir()
            fake_python = bin_dir / "python3"
            fake_python.write_text(
                """#!/bin/bash
set -e
shift
while [[ $# -gt 0 ]]; do
    if [[ $1 == --build-dir ]]; then out=$2; shift 2; else shift; fi
done
mkdir -p "$out"
printf wrapper > "$out/a5_original_lim_binding.so"
printf '{}' > "$out/binding_manifest.json"
""",
                encoding="utf-8",
            )
            fake_python.chmod(0o755)
            env_script = base / "set_env.sh"
            env_script.write_text(
                f"export ASCEND_HOME_PATH=/fake/ascend\n"
                f"mock_bin='{bin_dir.as_posix()}'\n"
                'if command -v cygpath >/dev/null; then mock_bin=$(cygpath -u "$mock_bin"); fi\n'
                'export PATH="$mock_bin:$PATH"\n',
                encoding="utf-8",
            )
            output = base / "output"
            result = self.run_bash(
                output.as_posix(),
                "--source-root",
                donor.as_posix(),
                "--ascend-env",
                env_script.as_posix(),
                "--jobs",
                "3",
            )
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertEqual((output / "exit_code").read_text().strip(), "0")
            paths = (output / "paths.env").read_text(encoding="utf-8")
            self.assertIn("ORIGINAL_LIM_BINDING=", paths)
            self.assertIn("ORIGINAL_SOURCE_REVISION=unknown", paths)
            self.assertIn("a5limbaseline_transformer", paths)
            args = (
                output / "prepared" / "source" / "csrc" / "arguments.txt"
            ).read_text()
            self.assertIn("--soc=ascend950", args)
            self.assertIn("--ops=fused_quant_lightning_indexer_manage", args)
            self.assertIn("--vendor_name=a5limbaseline", args)
            self.assertIn("-j3", args)
            self.assertFalse((donor / "csrc" / "build").exists())
            self.assertTrue((output / "installed_files.sha256").is_file())

    def test_invalid_jobs_fail_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new"
            result = self.run_bash(path.as_posix(), "--jobs", "0")
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse(path.exists())
            self.assertIn("positive integer", result.stderr)


if __name__ == "__main__":
    unittest.main()
