# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""真实冒烟 CLI 和从 csv_loader.h 提取的 C++ 模板读取同一批 CSV（仅 Host）。"""

import csv
import io
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[3]
NAMES = ["positive_%03d" % value for value in range(8)] + ["negative_000", "negative_001"]
FORMATS = ("plain", "bom", "comments", "padded", "combined")


class TestCsvInputFormats(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temporary.cleanup)
        cls.root = Path(cls.temporary.name)
        cls.script = cls.root / "scripts/ci/smoke_sample.py"
        cls.script.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / "scripts/ci/smoke_sample.py", cls.script)
        # 编译源码中的真实纯 CSV 模板；不包含 ACL/GTest 依赖，不模拟 NPU。
        header = (ROOT / "test/frame/csv_loader.h").read_text(encoding="utf-8")
        start = header.index("using csv_map =")
        end = header.index("// ── ReplaceFileExtension2Csv")
        source = (
            "#include <fstream>\n#include <iostream>\n#include <stdexcept>\n"
            "#include <string>\n#include <unordered_map>\n#include <vector>\n"
        )
        source += header[start:end]
        source += r'''
struct Row { csv_map fields; explicit Row(const csv_map& data) : fields(data) {} };
int main(int argc, char** argv) {
    try {
        for (const auto& row : GetCasesFromCsv<Row>(argv[1]))
            std::cout << ReadMap(row.fields, std::string("case_name"), std::string("<missing>")) << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what(); return 2;
    }
}
'''
        cpp = cls.root / "csv_reader.cpp"
        cpp.write_text(source, encoding="utf-8")
        cls.reader = cls.root / "csv_reader"
        compiler = shutil.which("c++")
        if compiler is None:
            raise RuntimeError("C++ compiler required for the native CSV Host regression")
        build = subprocess.run(
            [compiler, "-std=c++17", str(cpp), "-o", str(cls.reader)],
            capture_output=True, text=True, check=False,
        )
        if build.returncode:
            raise RuntimeError(build.stderr)

    def write_cases(self, format_name):
        path = self.root / "test/sample/arch35/sample_test.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        content = io.StringIO(newline="")
        padded = format_name in ("padded", "combined")
        writer = csv.writer(content, lineterminator="\r\n" if format_name == "combined" else "\n")
        writer.writerow([" case_name ", " expect_result ", " description "] if padded else
                        ["case_name", "expect_result", "description"])
        for index, name in enumerate(NAMES):
            writer.writerow([name, "SUCCESS" if index < 8 else "INVALID_VALUE", "中文,边界"])
        text = content.getvalue()
        if format_name in ("comments", "combined"):
            text = "# generated cases\n\n" + text + "\n  # end of cases\n"
        if format_name in ("bom", "combined"):
            text = "\ufeff" + text
        path.write_bytes(text.encode("utf-8"))
        return path

    def run_sampler(self):
        return subprocess.run([sys.executable, str(self.script), "arch35", "sample"],
                              capture_output=True, text=True, timeout=10)

    def test_smoke_cli_formats(self):
        for format_name in FORMATS:
            with self.subTest(format=format_name):
                self.write_cases(format_name)
                result = self.run_sampler()
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout, "sample:" + ",".join(sorted(NAMES)) + ";")

    def test_native_csv_formats(self):
        for format_name in FORMATS:
            with self.subTest(format=format_name):
                path = self.write_cases(format_name)
                result = subprocess.run([str(self.reader), str(path)], capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(result.stdout.splitlines(), NAMES)

    def test_missing_and_empty_controls(self):
        path = self.write_cases("plain")
        for empty in (True, False):
            with self.subTest(empty=empty):
                if empty:
                    path.write_bytes(b"")
                else:
                    path.unlink()
                result = self.run_sampler()
                self.assertEqual((result.returncode, result.stdout), (0, ""))
                native = subprocess.run([str(self.reader), str(path)], capture_output=True, text=True, timeout=10)
                self.assertEqual(native.returncode, 2)
                self.assertTrue(native.stderr)


if __name__ == "__main__":
    unittest.main()
