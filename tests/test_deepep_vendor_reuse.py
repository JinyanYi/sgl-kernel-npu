"""CPU checks for preserving vendor bytes during a host-only DeepEP rebuild."""

import importlib.util
import json
import py_compile
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "tools/deepep_vendor.py"
SPEC = importlib.util.spec_from_file_location("deepep_vendor", SCRIPT)
VENDOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VENDOR)


class VendorReuseTests(unittest.TestCase):
    def make_vendor(self, root):
        library = root / "op_api/lib/libcust_opapi.so"
        library.parent.mkdir(parents=True)
        library.write_bytes(b"existing API bytes")
        payload = root / "op_impl/device.bin"
        payload.parent.mkdir()
        payload.write_bytes(b"existing device bytes")

    def test_stage_replaces_stale_destination_and_preserves_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "installed"
            self.make_vendor(source)
            before = VENDOR.manifest(source)
            target = root / "package/vendors/hwcomputing"
            target.mkdir(parents=True)
            (target / "stale.bin").write_bytes(b"stale")
            result = VENDOR.stage(source, root / "package")
            self.assertEqual(result["files"], before)
            self.assertEqual(VENDOR.manifest(target), before)
            self.assertEqual(VENDOR.manifest(source), before)

    def test_overlap_rejected_before_removing_source(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            source = package / "vendors/hwcomputing"
            self.make_vendor(source)
            before = VENDOR.manifest(source)
            with self.assertRaisesRegex(ValueError, "overlap"):
                VENDOR.stage(source, package)
            self.assertEqual(VENDOR.manifest(source), before)

    def test_changed_device_payload_fails_cli_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "installed"
            self.make_vendor(source)
            VENDOR.stage(source, root / "new")
            target = root / "new/vendors/hwcomputing"
            (target / "op_impl/device.bin").write_bytes(b"changed device bytes")
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "--source",
                    str(source),
                    "--compare",
                    str(target),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("DeepEP vendor bytes changed", result.stderr)

    def test_generated_bytecode_requires_identical_source_on_both_sides(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "installed"
            self.make_vendor(source)
            (source / "op_impl/probe.py").write_text("VALUE = 1\n")
            VENDOR.stage(source, root / "new")
            target = root / "new/vendors/hwcomputing"
            py_compile.compile(str(target / "op_impl/probe.py"), doraise=True)
            report = VENDOR.compare(source, target)
            self.assertEqual(report["status"], "identical_payload")
            self.assertEqual(len(report["ignored_generated_bytecode"]), 1)
            (target / "op_impl/probe.py").write_text("VALUE = 2\n")
            report = VENDOR.compare(source, target)
            self.assertEqual(report["status"], "payload_changed")
            self.assertIn(
                "op_impl/probe.py", [row["path"] for row in report["changed"]]
            )

    def test_orphan_bytecode_is_protected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "installed"
            self.make_vendor(source)
            VENDOR.stage(source, root / "new")
            target = root / "new/vendors/hwcomputing"
            cache = target / "op_impl/__pycache__/orphan.cpython-312.pyc"
            cache.parent.mkdir()
            cache.write_bytes(b"no corresponding source")
            report = VENDOR.compare(source, target)
            self.assertEqual(report["status"], "payload_changed")
            self.assertEqual(report["added"], [cache.relative_to(target).as_posix()])

    def test_original_backup_must_match_saved_receipt(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "installed"
            self.make_vendor(source)
            saved = root / "before.json"
            saved.write_text(json.dumps(dict(files=VENDOR.manifest(source))))
            VENDOR.stage(source, root / "new")
            (source / "op_impl/device.bin").write_bytes(b"modified original backup")
            with self.assertRaisesRegex(ValueError, "Original vendor backup differs"):
                VENDOR.compare(source, root / "new/vendors/hwcomputing", saved)

    def test_real_wheel_install_bytecode_does_not_mask_library_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            project.mkdir()
            shutil.copyfile(
                SCRIPT.parents[1] / "python/deep_ep/setup.py", project / "setup.py"
            )
            package = project / "deep_ep"
            package.mkdir()
            (package / "__init__.py").write_text("")
            source = package / "vendors/hwcomputing"
            self.make_vendor(source)
            (source / "op_impl/probe.py").write_text("VALUE = 1\n")
            wheels = root / "wheels"
            wheels.mkdir()
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "wheel",
                    "--no-deps",
                    "--no-build-isolation",
                    "--disable-pip-version-check",
                    "-w",
                    str(wheels),
                    str(project),
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            installed = root / "installed"
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "--no-deps",
                    "--disable-pip-version-check",
                    "--target",
                    str(installed),
                    str(next(wheels.glob("*.whl"))),
                ],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            target = installed / "deep_ep/vendors/hwcomputing"
            self.assertNotEqual(VENDOR.manifest(source), VENDOR.manifest(target))
            report = VENDOR.compare(source, target)
            self.assertEqual(report["status"], "identical_payload")
            self.assertTrue(report["ignored_generated_bytecode"])
            (target / "op_api/lib/libcust_opapi.so").write_bytes(b"changed API library")
            report = VENDOR.compare(source, target)
            self.assertEqual(report["status"], "payload_changed")
            self.assertIn(
                "op_api/lib/libcust_opapi.so",
                [row["path"] for row in report["changed"]],
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
