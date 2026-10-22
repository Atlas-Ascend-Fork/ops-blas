# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Run the complete installer only against private real package directories."""

import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pytest

ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class InstallResultFixture:
    version: Path
    directories: Sequence[Path]
    libraries: Path
    header: Path


def command_path(name):
    path = shutil.which(name)
    assert path is not None, f"Required Host command not found: {name}"
    assert Path(path).is_absolute(), path
    return path


def create_package(tmp_path, scenario):
    package = tmp_path / "package"
    script = package / "share/info/ops_blas/install.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts/package/install.sh", script)
    include = package / "include"
    libraries = package / "lib64"
    include.mkdir()
    libraries.mkdir()
    header = include / "cann_ops_blas_fixture.h"
    header.write_text("#pragma once\nint permission_payload(void);\n")
    if scenario != "missing_library":
        source = tmp_path / "payload.c"
        source.write_text("int permission_payload(void) { return 7; }\n")
        # Real Host shared-library bytes are copied only, never loaded or used
        # as an ACL/CANN runtime, compiler or device API replacement.
        command = [
            command_path("clang"),
            "-shared",
            "-fPIC",
            str(source),
            "-o",
            str(libraries / "libops_blas_fixture.so"),
        ]
        compiled = subprocess.run(command, capture_output=True, text=True, check=False)
        assert compiled.returncode == 0, compiled.stdout + compiled.stderr
    return script, libraries, header


def create_readonly_destination(tmp_path, scenario, header_name):
    prefix = tmp_path / "destination"
    version = prefix / "cann"
    directories = [
        version,
        version / "lib64",
        version / "include",
        version / "share",
        version / "share/info",
        version / "share/info/ops_blas",
    ]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
    if scenario == "copy_failure":
        # Real cp cannot replace an existing directory by the source header file.
        (version / "include" / header_name).mkdir()
    for directory in directories:
        directory.chmod(0o500)
    return prefix, version, directories


def assert_install_result(run, scenario, fixture):
    assert run.returncode == (0 if scenario == "success" else 1), (
        run.stdout + run.stderr
    )
    if scenario == "copy_failure":
        assert "directory" in run.stderr
    for directory in fixture.directories:
        assert stat.S_IMODE(directory.stat().st_mode) == 0o500, (
            scenario,
            directory,
            run.stdout + run.stderr,
        )
    if scenario == "success":
        assert (fixture.version / "lib64/libops_blas_fixture.so").read_bytes() == (
            fixture.libraries / "libops_blas_fixture.so"
        ).read_bytes()
        assert (fixture.version / "include" / fixture.header.name).read_bytes() == (
            fixture.header.read_bytes()
        )


@pytest.mark.parametrize("scenario", ["copy_failure", "success", "missing_library"])
def test_complete_install_cli_restores_owner_permission_on_every_exit(
    tmp_path, scenario
):
    script, libraries, header = create_package(tmp_path, scenario)
    prefix, version, directories = create_readonly_destination(
        tmp_path, scenario, header.name
    )
    fixture = InstallResultFixture(
        version=version,
        directories=directories,
        libraries=libraries,
        header=header,
    )
    command = [
        command_path("bash"),
        str(script),
        "--install",
        "--quiet",
        f"--install-path={prefix}",
    ]
    try:
        run = subprocess.run(command, capture_output=True, text=True, check=False)
        assert_install_result(run, scenario, fixture)
    finally:
        # Writable modes are restored only on this test's own private directories
        # so pytest can remove artifacts even when the original baseline fails.
        for directory in directories:
            directory.chmod(0o700)
