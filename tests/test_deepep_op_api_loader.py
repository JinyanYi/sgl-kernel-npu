"""Native Linux loader regression; no Torch, CANN or NPU is required."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(
    sys.platform == "linux" and shutil.which("g++"), "Requires Linux g++"
)
class DeepEpLibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix="deepep-library-test-")
        cls.root = Path(cls.temp.name)
        for name, source, soname in (
            ("lim", 'extern "C" int common_marker() {return 10;}', "libcust_opapi.so"),
            (
                "deep",
                'extern "C" int common_marker() {return 20;} extern "C" int aclnnMoeLowLatencyDispatchV2() {return 20;}',
                "libcust_opapi.so",
            ),
            ("standard", 'extern "C" int builtin_helper() {return 30;}', "libopapi.so"),
        ):
            folder = cls.root / name
            folder.mkdir()
            code = folder / "library.cpp"
            code.write_text(source)
            subprocess.run(
                [
                    "g++",
                    "-shared",
                    "-fPIC",
                    str(code),
                    "-Wl,-soname," + soname,
                    "-o",
                    str(folder / soname),
                ],
                check=True,
            )
        code = cls.root / "probe.cpp"
        code.write_text(
            r"""
#include "op_api_library.hpp"
#include <iostream>
using Call = int (*)();
int main(int argc, char **argv) {
    try {
        if (!dlopen(argv[1], RTLD_NOW | RTLD_LOCAL) || !dlopen(argv[2], RTLD_NOW | RTLD_LOCAL)) return 5;
        auto bare = dlopen("libcust_opapi.so", RTLD_NOW | RTLD_LOCAL);
        auto marker = reinterpret_cast<Call>(dlsym(bare, "common_marker"));
        auto api = reinterpret_cast<Call>(deep_ep::op_api::resolve("aclnnMoeLowLatencyDispatchV2"));
        auto builtin = reinterpret_cast<Call>(deep_ep::op_api::resolve("builtin_helper"));
        if (!api || !builtin) return 4;
        std::cout << "bare=" << marker() << " pinned=" << api() << " builtin=" << builtin() << std::endl;
        return api() == 20 && builtin() == 30 ? 0 : 6;
    } catch (const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 3;
    }
}
"""
        )
        cls.program = cls.root / "probe"
        subprocess.run(
            [
                "g++",
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-Werror",
                "-Wno-unused-parameter",
                "-I",
                str(ROOT / "csrc/deepep"),
                str(code),
                "-ldl",
                "-o",
                str(cls.program),
            ],
            check=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def run_probe(self, order, pin):
        env = dict(
            os.environ,
            LD_LIBRARY_PATH=":".join(
                str(self.root / d) for d in ("lim", "deep", "standard")
            ),
        )
        env.pop("SGLANG_DEEPEP_OP_API_LIBRARY", None)
        if pin is not None:
            env["SGLANG_DEEPEP_OP_API_LIBRARY"] = pin
        libraries = [str(self.root / d / "libcust_opapi.so") for d in order]
        return subprocess.run(
            [str(self.program), *libraries], env=env, text=True, capture_output=True
        )

    def test_lim_loaded_first_still_resolves_deepep_and_builtin(self):
        result = self.run_probe(
            ("lim", "deep"), str(self.root / "deep/libcust_opapi.so")
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("bare=10 pinned=20 builtin=30", result.stdout)

    def test_deepep_loaded_first_also_works(self):
        result = self.run_probe(
            ("deep", "lim"), str(self.root / "deep/libcust_opapi.so")
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("bare=20 pinned=20 builtin=30", result.stdout)

    def test_original_unpinned_lookup_reproduces_collision(self):
        self.assertEqual(self.run_probe(("lim", "deep"), None).returncode, 4)

    def test_missing_pinned_library_does_not_fall_back_to_bare(self):
        result = self.run_probe(("deep", "lim"), str(self.root / "missing.so"))
        self.assertEqual(result.returncode, 3)
        self.assertIn("Cannot load pinned DeepEP API library", result.stderr)

    def test_relative_pin_is_rejected(self):
        result = self.run_probe(("deep", "lim"), "libcust_opapi.so")
        self.assertEqual(result.returncode, 3)
        self.assertIn("must be an absolute path", result.stderr)

    def test_wrong_pin_does_not_search_other_custom_library(self):
        self.assertEqual(
            self.run_probe(
                ("deep", "lim"), str(self.root / "lim/libcust_opapi.so")
            ).returncode,
            4,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
