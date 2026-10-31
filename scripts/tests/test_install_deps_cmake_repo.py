# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Exercise the Ubuntu 18.04 CMake repository setup without root or network."""

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(os.environ.get("INSTALL_DEPS_SCRIPT", Path(__file__).resolve().parents[2] / "install_deps.sh"))
HARNESS = r'''
source "$SCRIPT_COPY"
OS=debian
PKG_MANAGER=apt
grep() { if [ "$*" = '-q Ubuntu 18.04 /etc/os-release' ]; then return 0; fi; command grep "$@"; }
cmake() { if [ -e "$CASE_ROOT/installed" ]; then echo 'cmake version 3.20.0'; else echo 'cmake version 3.10.0'; fi; }
wget() { if [ "$FAIL_DOWNLOAD" = 1 ]; then return 7; fi; printf 'MOCK-PUBLIC-KEY\n'; }
gpg() { IFS= read -r key || return 1; if [ "$PERMISSIVE_GPG" != 1 ]; then [ "$key" = MOCK-PUBLIC-KEY ] || return 1; fi; printf 'MOCK-KEYRING\n'; }
sudo() {
    if [ "$1" = tee ]; then
        case "$2" in *keyrings*) cat > "$CASE_ROOT/keyring" ;; *) cat > "$CASE_ROOT/repo" ;; esac
    elif [ "$1" = apt ] && [ "$2" = install ]; then touch "$CASE_ROOT/installed"; fi
}
install_cmake
'''


class CmakeRepositoryTests(unittest.TestCase):
    def run_case(self, root, fail_download=False, permissive_gpg=False):
        # Normalise the Windows checkout's line endings for the Linux shell.
        script_copy = root / "install_deps.sh"
        script_copy.write_text(SCRIPT.read_text(encoding="utf-8"), encoding="utf-8")
        return subprocess.run(
            ["/bin/bash", "-c", HARNESS],
            env={**os.environ, "SCRIPT_COPY": str(script_copy), "CASE_ROOT": str(root),
                 "FAIL_DOWNLOAD": "1" if fail_download else "0",
                 "PERMISSIVE_GPG": "1" if permissive_gpg else "0"},
            capture_output=True, text=True, timeout=30, check=False,
        )

    def test_key_and_repository_are_forwarded_to_pipeline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.run_case(root)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertEqual((root / "keyring").read_text(), "MOCK-KEYRING\n")
            self.assertEqual((root / "repo").read_text(),
                             "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] "
                             "https://apt.kitware.com/ubuntu/ bionic main\n")
            self.assertTrue((root / "installed").exists())

    def test_failed_download_stops_before_apt_install(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.run_case(root, fail_download=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertFalse((root / "installed").exists())

    def test_repository_line_contains_no_wrapper_log(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.run_case(root, permissive_gpg=True)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
            self.assertEqual((root / "repo").read_text(),
                             "deb [signed-by=/usr/share/keyrings/kitware-archive-keyring.gpg] "
                             "https://apt.kitware.com/ubuntu/ bionic main\n")


if __name__ == "__main__":
    unittest.main()
