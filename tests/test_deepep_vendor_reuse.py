"""CPU checks for preserving vendor bytes during a host-only DeepEP rebuild."""

import importlib.util
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
