# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run native dependency discovery and a real CBLAS link/execution probe."""

from pathlib import Path
import subprocess
import shutil
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


class ReferenceBlasDiscoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        executable = shutil.which("cmake")
        if executable is None:
            raise RuntimeError("CMake is required for the native CBLAS probe")
        cls.cmake = str(Path(executable).resolve())

    def test_installer_finds_installed_cblas_header(self):
        result = subprocess.run(
            [str(ROOT / "tests/host/test_installer_cblas_header.sh")],
            capture_output=True, text=True, check=False,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        header = Path(result.stdout.strip())
        self.assertEqual(header.name, "cblas.h")
        self.assertTrue(header.is_file(), header)

    def run_probe(self, directory, options):
        directory.mkdir()
        (directory / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.16)\n'
            'project(CblasDiscoveryProbe C)\n'
            f'include("{ROOT / "cmake/test.cmake"}")\n'
            '_ops_blas_ensure_refblas_found()\n'
            'file(WRITE "${CMAKE_BINARY_DIR}/found.txt" '
            '"${REFBLAS_INCLUDE_DIR}\n${REFBLAS_LIB}\n")\n'
            'add_executable(cblas_probe probe.c)\n'
            'target_include_directories(cblas_probe PRIVATE ${REFBLAS_INCLUDE_DIR})\n'
            'target_link_libraries(cblas_probe PRIVATE ${REFBLAS_LIB})\n'
        )
        (directory / "probe.c").write_text(
            '#include <cblas.h>\n'
            'int main(void) {\n'
            '    const float values[] = {1.0f, 2.0f, 3.0f};\n'
            '    return cblas_sdot(3, values, 1, values, 1) == 14.0f ? 0 : 1;\n'
            '}\n'
        )
        build = directory / "build"
        commands = [
            [self.cmake, "-S", str(directory), "-B", str(build), *options],
            [self.cmake, "--build", str(build), "--parallel", "2"],
            [str(build / "cblas_probe")],
        ]
        for command in commands:
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        include, library = (build / "found.txt").read_text().splitlines()
        self.assertTrue((Path(include) / "cblas.h").is_file())
        self.assertTrue(Path(library).is_file())
        return include, library

    def test_native_cmake_discovery_and_cache_overrides(self):
        with tempfile.TemporaryDirectory(prefix="ops-blas-cblas-") as tmp:
            directory = Path(tmp)
            include, library = self.run_probe(directory / "automatic", [])
            options = [
                ("cached-library", [f"-DREFBLAS_LIB={library}"]),
                ("cached-missing-header", [
                    f"-DREFBLAS_LIB={library}",
                    "-DREFBLAS_INCLUDE_DIR:PATH=REFBLAS_INCLUDE_DIR-NOTFOUND",
                ]),
                ("explicit-paths", [
                    f"-DREFBLAS_LIB={library}", f"-DREFBLAS_INCLUDE_DIR={include}",
                ]),
            ]
            for name, arguments in options:
                with self.subTest(configuration=name):
                    actual = self.run_probe(directory / name, arguments)
                    self.assertEqual(actual, (include, library))


if __name__ == "__main__":
    unittest.main(verbosity=2)
