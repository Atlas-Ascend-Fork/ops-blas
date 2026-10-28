# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Read actual uninstall path diagnostics without any installation or deletion."""

import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def command_path(name):
    path = shutil.which(name)
    assert path is not None, f"Required Host command not found: {name}"
    assert Path(path).is_absolute(), path
    return path


@pytest.mark.parametrize("form", ["tilde", "absolute", "relative", "empty", "space"])
def test_complete_cli_resolves_current_user_tilde_before_working_directory(
    tmp_path, form
):
    script = tmp_path / "package/share/info/ops_blas/install.sh"
    script.parent.mkdir(parents=True)
    shutil.copyfile(ROOT / "scripts/package/install.sh", script)
    # A unique nonexistent prefix lets the real do_uninstall return before any
    # mkdir, chmod, file copy or deletion. HOME is read unchanged from the process.
    token = "__cann_uninstalled_path_" + uuid.uuid4().hex
    if form == "tilde":
        argument = "~/" + token
        expected = Path.home() / token / "cann"
    elif form == "absolute":
        argument = str(tmp_path / token)
        expected = tmp_path / token / "cann"
    elif form == "relative":
        argument = token
        expected = tmp_path / token / "cann"
    else:
        argument = "" if form == "empty" else token + " invalid"
        expected = tmp_path / token / "cann"
    assert not expected.parent.exists()
    run = subprocess.run(
        [
            command_path("bash"),
            str(script),
            "--uninstall",
            "--quiet",
            "--install-path=" + argument,
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if form in ["empty", "space"]:
        assert run.returncode == 1, run.stdout + run.stderr
        if form == "empty":
            assert "cannot be empty" in run.stdout
        else:
            assert "cannot contain spaces" in run.stdout
    else:
        assert run.returncode == 0, run.stdout + run.stderr
        assert "No installation found at " + str(expected.resolve()) in run.stdout, (
            run.stdout + run.stderr
        )
    assert not expected.parent.exists()
    assert not (tmp_path / "~").exists()
