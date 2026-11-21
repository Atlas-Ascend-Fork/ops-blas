# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
"""Validate the build entry point with a real, device-free CMake project."""

import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

BUILD_SCRIPT = Path(os.environ.get("BUILD_SCRIPT", Path(__file__).resolve().parents[2] / "build.sh"))
PROJECT = r"""cmake_minimum_required(VERSION 3.20)
project(BuildOptionsCheck LANGUAGES NONE)
set(CMAKE_FIND_USE_CMAKE_ENVIRONMENT_PATH FALSE)
set(CMAKE_FIND_USE_PACKAGE_REGISTRY FALSE)
find_package(ASC CONFIG REQUIRED)
file(WRITE "${CMAKE_BINARY_DIR}/options.txt"
    "${ASCEND_CANN_PACKAGE_PATH}\n${ASC_DIR}\n${CMAKE_PREFIX_PATH}\n"
    "${SOC_VERSION}\n${TEST_NAMES}\n${TEST_DEVICE_ID}\n${BUILD_TEST}\n${ENABLE_PACKAGE}\n"
    "${CHEMV_NETLIB_CONFIG}\n")
add_custom_target(package COMMAND ${CMAKE_COMMAND} -E touch "${CMAKE_BINARY_DIR}/packaged")
install(CODE "file(WRITE \"${CMAKE_BINARY_DIR}/installed\" \"ok\")")
"""


class BuildCmakeOptionsTests(unittest.TestCase):
    def check_configuration(self, names, package_location, use_fallback=False, options=(), chemv_config=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / names[0]
            root.mkdir()
            shutil.copyfile(BUILD_SCRIPT, root / "build.sh")
            (root / "CMakeLists.txt").write_text(PROJECT, encoding="utf-8")
            toolkit = Path(temporary) / names[1]
            package = toolkit / package_location
            package.mkdir(parents=True)
            (package / "ASCConfig.cmake").write_text("set(ASC_FOUND TRUE)\n", encoding="utf-8")
            env = dict(os.environ)
            for key in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME", "ASC_DIR", "SOC_VERSION", "CHEMV_NETLIB_CONFIG"):
                env.pop(key, None)
            env["ASCEND_TOOLKIT_HOME" if use_fallback else "ASCEND_HOME_PATH"] = str(toolkit)
            if package_location == "explicit package/ASC":
                env["ASC_DIR"] = str(package)
            reference = root / "reference config.cmake"
            if chemv_config:
                reference.write_text("# Reference configuration\n", encoding="utf-8")
                env["CHEMV_NETLIB_CONFIG"] = str(reference)
            result = subprocess.run(
                ["/bin/bash", str(root / "build.sh"), *options],
                cwd=Path(temporary), env=env, text=True, capture_output=True, timeout=30, check=False,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            values = (root / "build/options.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(values[0], str(toolkit))
            self.assertEqual(values[1], str(package))
            expected_prefix = "" if "ASC_DIR" in env else str(package.parent)
            self.assertEqual(values[2], expected_prefix)
            self.assertEqual(values[8], str(reference.resolve()) if chemv_config else "")
            if options:
                self.assertEqual(values[3:8], ["ascend950", "scopy;caxpy", "2", "ON", "ON"])
                self.assertTrue((root / "build/packaged").is_file())
            else:
                self.assertEqual(values[3:8], ["ascend910b3", "", "0", "", ""])
                self.assertTrue((root / "build/installed").is_file())

    def test_default_build(self):
        self.check_configuration(("project", "toolkit"), "lib64/cmake/ASC")

    def test_repository_path_with_spaces(self):
        self.check_configuration(("source tree", "toolkit"), "lib64/cmake/ASC")

    def test_toolkit_path_with_spaces(self):
        self.check_configuration(("project", "CANN toolkit"), "lib64/cmake/ASC")

    def test_toolkit_fallback_with_spaces(self):
        self.check_configuration(("project", "CANN toolkit"), "lib64/cmake/ASC", use_fallback=True)

    def test_explicit_asc_package_with_spaces(self):
        self.check_configuration(("project", "toolkit"), "explicit package/ASC")

    def test_compiler_package_layout_with_spaces(self):
        self.check_configuration(("project", "CANN toolkit"), "compiler/latest/lib64/cmake/ASC")

    def test_chemv_reference_path_with_spaces(self):
        self.check_configuration(
            ("source tree", "CANN toolkit"), "lib64/cmake/ASC", chemv_config=True,
        )

    def test_packaging_and_multiple_operators(self):
        self.check_configuration(
            ("source tree", "CANN toolkit"), "lib64/cmake/ASC",
            options=("--soc=ascend950", "--ops=scopy,caxpy", "--device=2", "--pkg"),
        )


if __name__ == "__main__":
    unittest.main()
