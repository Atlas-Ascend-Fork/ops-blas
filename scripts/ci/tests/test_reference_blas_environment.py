# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check independent CPU preflight, fixed binding and unchanged GTest registration."""
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
POLICY = "OPS_BLAS_REFBLAS_NONFINITE"

DECLARATIONS = """
extern "C" void cblas_cscal(int, const void*, void*, int);
extern "C" void cblas_caxpy(int, const void*, const void*, int, void*, int);
"""
REFERENCE_SOURCE = r"""
#include <cmath>
#include <complex>
static std::complex<float> multiply(const float* a, const float* x) {
#ifdef BROKEN_FINITE
    if (std::isfinite(x[0])) return {0.0f, 0.0f};
#endif
#ifdef OTHER_RESULT
    if (!std::isfinite(x[0])) return {0.0f, 0.0f};
    return {a[0] * x[0] - a[1] * x[1], a[0] * x[1] + a[1] * x[0]};
#elif defined(RECOVER_INF)
    return std::complex<float>(a[0], a[1]) * std::complex<float>(x[0], x[1]);
#else
    return {a[0] * x[0] - a[1] * x[1], a[0] * x[1] + a[1] * x[0]};
#endif
}
extern "C" void cblas_cscal(int n, const void* alpha, void* data, int inc) {
    float* x = static_cast<float*>(data);
    for (int i = 0; i < n; ++i) {
        auto value = multiply(static_cast<const float*>(alpha), x + 2*i*inc);
        x[2*i*inc] = value.real(); x[2*i*inc+1] = value.imag();
    }
}
extern "C" void cblas_caxpy(int n, const void* alpha, const void* input, int incx, void* output, int incy) {
    const float* x = static_cast<const float*>(input);
    float* y = static_cast<float*>(output);
    for (int i = 0; i < n; ++i) {
        auto value = multiply(static_cast<const float*>(alpha), x + 2*i*incx);
        y[2*i*incy] += value.real(); y[2*i*incy+1] += value.imag();
    }
}
"""
TEST_SOURCE = r"""
#include <cfenv>
#include <iostream>
#include <gtest/gtest.h>
struct InitialExceptions {
    InitialExceptions() { std::feraiseexcept(FE_DIVBYZERO); }
} initialExceptions;
TEST(ReferenceEnvironment, TestBody) {
    EXPECT_NE(std::fetestexcept(FE_DIVBYZERO), 0);
    EXPECT_EQ(std::fetestexcept(FE_INVALID), 0);
    std::cout << "TEST_BODY_RAN\n";
}
"""

REGISTERED_PROJECT = """
cmake_minimum_required(VERSION 3.20)
project(RegisteredReferenceTargets CXX)
include("${OPS_BLAS_SOURCE}/cmake/test.cmake")
set(BLAS_ASCENDCL_LIB "")
set(OPS_BLAS test_operator)
foreach(library test_operator tiling_api platform register c_sec unified_dlog)
    add_library(${library} INTERFACE)
endforeach()
add_subdirectory(test/cgemm_strided_batched)
_ops_blas_register_all_runner("${OPS_BLAS}")
"""


def executable_path(name):
    path = shutil.which(name)
    if path is None:
        raise unittest.SkipTest(name + " is required")
    return str(Path(path).resolve(strict=True))


class TestReferenceBlasEnvironment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = executable_path("g++")
        cls.archiver = executable_path("ar")
        cls.cmake = executable_path("cmake")
        cls.temporary = tempfile.TemporaryDirectory(prefix="reference_blas_")
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.folder = Path(cls.temporary.name)
        (cls.folder / "cblas.h").write_text(DECLARATIONS)
        (cls.folder / "reference.cpp").write_text(REFERENCE_SOURCE)
        (cls.folder / "runner.cpp").write_text(TEST_SOURCE)
        cls.build_shared_references(compiler)
        cls.build_runners(compiler)

    @classmethod
    def build_shared_references(cls, compiler):
        for mode in ("nan", "recover-inf", "other", "broken-finite"):
            directory = cls.folder / mode
            directory.mkdir()
            command = [compiler, "-std=c++17", "-O2", "-ffp-contract=off", "-shared", "-fPIC",
                       "-Wl,-soname,libreference.so", str(cls.folder / "reference.cpp"),
                       "-o", str(directory / "libreference.so")]
            if mode == "recover-inf":
                command.append("-DRECOVER_INF")
            elif mode == "other":
                command.append("-DOTHER_RESULT")
            elif mode == "broken-finite":
                command.append("-DBROKEN_FINITE")
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)

    @classmethod
    def build_runners(cls, compiler):
        cls.executable = cls.folder / "reference_test"
        command = [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-I" + str(cls.folder),
                   str(ROOT / "scripts/reference_blas/probe.cpp"), "-L" + str(cls.folder / "nan"),
                   "-Wl,--enable-new-dtags,-rpath," + str(cls.folder / "nan"),
                   "-lreference", "-ldl",
                   "-o", str(cls.executable)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        reference_object = cls.folder / "reference.o"
        subprocess.run([compiler, "-std=c++17", "-O2", "-ffp-contract=off", "-c",
                        str(cls.folder / "reference.cpp"), "-o", str(reference_object)],
                       check=True, capture_output=True, text=True, timeout=60)
        archive = cls.folder / "libreference.a"
        subprocess.run([cls.archiver, "rcs", str(archive), str(reference_object)], check=True, timeout=30)
        cls.fixed_executable = cls.folder / "fixed_reference_test"
        command = [str(archive) if arg == "-lreference" else arg for arg in command]
        command[-1] = str(cls.fixed_executable)
        command.append('-DOPS_BLAS_REFERENCE_PROFILE="test-static-profile"')
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=60)

    def run_reference(self, actual, expected=None, extra=(), *, fixed=False):
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GTEST_") and key not in (POLICY, "BLAS_TEST_CASES", "LD_PRELOAD")
        }
        env["LD_LIBRARY_PATH"] = str(self.folder / actual)
        if expected is not None:
            env[POLICY] = expected
        executable = self.fixed_executable if fixed else self.executable
        if fixed:
            env["LD_PRELOAD"] = str(self.folder / actual / "libreference.so")
        return subprocess.run([str(executable), *extra], env=env, capture_output=True, text=True, timeout=30)

    def test_records_runtime_library(self):
        for mode in ("nan", "recover-inf"):
            with self.subTest(mode=mode):
                result = self.run_reference(mode)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn(str(self.folder / mode / "libreference.so"), result.stdout)
                self.assertIn("nonfinite=" + mode, result.stdout)
                self.assertIn("finite=pass", result.stdout)

    def test_expected_reference_runs(self):
        for mode in ("nan", "recover-inf"):
            with self.subTest(mode=mode):
                result = self.run_reference(mode, mode)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("finite=pass", result.stdout)

    def test_runtime_override_is_rejected(self):
        result = self.run_reference("recover-inf", "nan")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Reference environment mismatch", result.stderr)
        self.assertIn(str(self.folder / "recover-inf/libreference.so"), result.stdout)
        self.assertNotIn("finite=pass", result.stdout)

    def test_invalid_policy_is_rejected(self):
        result = self.run_reference("nan", "typo")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be nan or recover-inf", result.stderr)
        self.assertNotIn("finite=pass", result.stdout)

    def test_empty_policy_is_unset(self):
        result = self.run_reference("recover-inf", "")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("finite=pass", result.stdout)

    def test_other_behavior_prints_raw_components(self):
        result = self.run_reference("other")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("nonfinite=other", result.stdout)
        self.assertIn("alpha=(1,0) cblas_cscal=(0,0) cblas_caxpy=(1,-2)", result.stdout)
        self.assertIn("alpha=(0,1) cblas_cscal=(0,0) cblas_caxpy=(1,-2)", result.stdout)

    def test_finite_sanity_failure_is_rejected(self):
        result = self.run_reference("broken-finite", "nan")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("nonfinite=nan", result.stdout)
        self.assertIn("Finite sanity check failed", result.stderr)

    def test_static_reference_ignores_runtime_library_override(self):
        result = self.run_reference("recover-inf", fixed=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("linkage=static", result.stdout)
        self.assertIn("nonfinite=nan", result.stdout)
        self.assertIn("finite=pass", result.stdout)

    def test_static_reference_rejects_conflicting_policy(self):
        result = self.run_reference("recover-inf", "recover-inf", fixed=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("compiled source profile requires nan", result.stderr)
        self.assertNotIn("finite=pass", result.stdout)

    def test_static_reference_accepts_empty_policy(self):
        result = self.run_reference("recover-inf", "", fixed=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("nonfinite=nan", result.stdout)
        self.assertIn("finite=pass", result.stdout)

    def prepare_registered_project(self, project):
        test_dir = project / "test/cgemm_strided_batched"
        test_dir.mkdir(parents=True)
        (project / "test/frame").symlink_to(ROOT / "test/frame", target_is_directory=True)
        (test_dir / "CMakeLists.txt").symlink_to(ROOT / "test/cgemm_strided_batched/CMakeLists.txt")
        (test_dir / "cgemm_strided_batched_test.cpp").write_text(TEST_SOURCE)
        (project / "CMakeLists.txt").write_text(REGISTERED_PROJECT)
        self.write_bundle(project)

    def write_bundle(self, project, reference=None):
        reference = reference or self.folder / "libreference.a"
        (project / "reference.cmake").write_text(
            'set(OPS_BLAS_REFERENCE_ID "netlib-3.10.0-flang18.1.8-gcc9.4.0-strict-v1")\n'
            f'set(OPS_BLAS_REFERENCE_ARCH "{platform.machine()}")\n'
            f'set(REFBLAS_INCLUDE_DIR "{self.folder}")\n'
            f'set(REFBLAS_LIB "{reference}")\n'
        )
        (project / "reference.json").write_text(json.dumps({
            "source_url": "test-fixture", "source_sha256": "test-fixture",
            "cc": "test C compiler", "fc": "test Fortran compiler",
            "flags": ["-O2", "-fno-fast-math", "-ffp-contract=off"],
        }))

    def check_registered_targets(self, mode):
        with tempfile.TemporaryDirectory(prefix="reference_registered_") as directory:
            project = Path(directory)
            self.prepare_registered_project(project)
            reference = self.folder / ("libreference.a" if mode == "source" else "nan/libreference.so")
            command = [self.cmake, "-S", str(project), "-B", str(project / "build"),
                       "-DOPS_BLAS_SOURCE=" + str(ROOT), "-DOPS_BLAS_REFBLAS_MODE=" + mode,
                       "-DOPS_BLAS_REFERENCE_ROOT=" + str(project),
                       "-DREFBLAS_INCLUDE_DIR=" + str(self.folder),
                       "-DREFBLAS_LIB=" + str(reference), "-DREFLAPACK_LIB=",
                       "-DCGEMM_SB_NETLIB_LIBRARY=" + str(self.folder / "recover-inf/libreference.so")]
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            expected_count = 1 if mode == "source" else 0
            self.assertEqual(result.stdout.count("Reference BLAS CPU preflight passed:"), expected_count)
            result = subprocess.run([self.cmake, "--build", str(project / "build"), "--parallel", "2"],
                                    capture_output=True, text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.run_registered_targets(project, mode)

    def run_registered_targets(self, project, mode):
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GTEST_") and key not in (POLICY, "BLAS_TEST_CASES", "LD_PRELOAD")
        }
        for target in ("test/cgemm_strided_batched/cgemm_strided_batched_test", "blas_all_test"):
            # An invalid diagnostic policy must not affect operator test entry points.
            env[POLICY] = "typo"
            result = subprocess.run([str(project / "build" / target)], env=env,
                                    capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertNotIn("[ReferenceBLAS]", result.stdout)
            self.assertIn("TEST_BODY_RAN", result.stdout)

    def test_registered_targets_link_system_reference(self):
        self.check_registered_targets("system")

    def test_registered_targets_link_source_reference(self):
        self.check_registered_targets("source")

    def configure_reference(self, mode):
        with tempfile.TemporaryDirectory(prefix="reference_config_") as directory:
            project = Path(directory)
            (project / "CMakeLists.txt").write_text(
                "cmake_minimum_required(VERSION 3.20)\n"
                "project(ReferenceConfiguration CXX)\n"
                f'include("{ROOT / "cmake/test.cmake"}")\n'
                "_ops_blas_ensure_refblas_found()\n"
            )
            command = [self.cmake, "-S", str(project), "-B", str(project / "build"),
                       "-DOPS_BLAS_REFBLAS_MODE=" + mode,
                       "-DOPS_BLAS_REFERENCE_ROOT=" + str(project / "absent"),
                       "-DREFBLAS_INCLUDE_DIR=" + str(self.folder),
                       "-DREFBLAS_LIB=" + str(self.folder / "recover-inf/libreference.so")]
            return subprocess.run(command, capture_output=True, text=True, timeout=60)

    def test_source_mode_does_not_fall_back_to_system_library(self):
        result = self.configure_reference("source")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("fixed source-built BLAS reference is missing", result.stderr)
        self.assertIn("does not fall back", result.stderr)

    def test_system_mode_remains_available_for_diagnosis(self):
        result = self.configure_reference("system")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Reference BLAS mode=system", result.stdout)

    def test_unknown_reference_mode_is_rejected(self):
        result = self.configure_reference("typo")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("must be source or system", result.stderr)

    def preflight_command(self, project):
        return [self.cmake, "-S", str(ROOT / "scripts/reference_blas"),
                "-B", str(project / "build"), "-DOPS_BLAS_REFERENCE_ROOT=" + str(project)]

    def run_configuration(self, command, **overrides):
        env = {
            key: value
            for key, value in os.environ.items()
            if key not in (POLICY, "LD_PRELOAD")
        }
        env.update(overrides)
        return subprocess.run(command, env=env, capture_output=True, text=True, timeout=60)

    def test_independent_preflight_reports_environment_and_rechecks_on_configure(self):
        with tempfile.TemporaryDirectory(prefix="reference_preflight_") as directory:
            project = Path(directory)
            self.write_bundle(project)
            command = self.preflight_command(project)
            for _ in range(2):
                result = self.run_configuration(command)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertEqual(result.stdout.count("Reference BLAS CPU preflight passed:"), 1)
            report = (project / "build/reference-blas-preflight/report.txt").read_text()
            for field in ("profile=netlib-3.10.0", "cc=test C compiler", "fc=test Fortran compiler",
                          "-ffp-contract=off", "nonfinite=nan", "finite=pass", "preflight_exit=0"):
                self.assertIn(field, report)
            self.write_bundle(project, self.folder / "recover-inf/libreference.so")
            result = self.run_configuration(command)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Reference environment mismatch: expected nan, observed recover-inf",
                          " ".join(result.stderr.split()))
            report = (project / "build/reference-blas-preflight/report.txt").read_text()
            self.assertIn("nonfinite=recover-inf", report)
            self.assertNotIn("preflight_exit=0", report)

    def test_preflight_static_reference_ignores_preload(self):
        with tempfile.TemporaryDirectory(prefix="reference_preload_") as directory:
            project = Path(directory)
            self.write_bundle(project)
            result = self.run_configuration(self.preflight_command(project),
                                            LD_PRELOAD=str(self.folder / "recover-inf/libreference.so"))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("nonfinite=nan", result.stdout)

    def test_preflight_missing_symbol_stops_configuration(self):
        with tempfile.TemporaryDirectory(prefix="reference_missing_symbol_") as directory:
            project = Path(directory)
            archive = project / "empty.a"
            subprocess.run([self.archiver, "rcs", str(archive)], check=True, timeout=30)
            self.write_bundle(project, archive)
            result = self.run_configuration(self.preflight_command(project))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("CPU preflight failed to build", result.stderr)
            self.assertIn("cblas_cscal", result.stderr)

    def test_preflight_cannot_change_source_expectation(self):
        with tempfile.TemporaryDirectory(prefix="reference_policy_") as directory:
            project = Path(directory)
            self.write_bundle(project)
            result = self.run_configuration(self.preflight_command(project),
                                            OPS_BLAS_REFBLAS_NONFINITE="recover-inf")
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("compiled source profile requires nan", result.stderr)


if __name__ == "__main__":
    unittest.main()
