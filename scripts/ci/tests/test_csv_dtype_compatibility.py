# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compile the shared dtype parser with SDK enums that include or omit HIFLOAT4."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]


class TestCsvDtypeCompatibility(unittest.TestCase):
    def compile_parser(self, folder, supported):
        compiler = shutil.which("g++")
        if compiler is None:
            self.skipTest("g++ is required for CSV dtype compatibility checks")
        header = (ROOT / "test/frame/csv_loader.h").read_text(encoding="utf-8")
        start = header.index("using T_DtypeLocal =")
        end = header.index("// issue #838:", start)
        declarations = header[start:end]
        start = header.index("inline constexpr std::pair<const char*, T_DtypeLocal> kDtypeTokenTable")
        end = header.index("inline std::vector<std::string> splitBySemicolon", start)
        parser = header[start:end]
        tokens = set(re.findall(r"static_cast<T_DtypeLocal>\((ACL_\w+)\)", declarations + parser))
        tokens.difference_update(("ACL_DT_UNDEFINED", "ACL_HIFLOAT4"))
        enum = "enum aclDataType {ACL_DT_UNDEFINED=-1," + ",".join(sorted(tokens))
        enum += (",ACL_HIFLOAT4" if supported else "") + "};\n"
        source = "\n".join(("#include <cstdint>", "#include <string>", "#include <utility>",
                            "#include <stdexcept>", "#include <iostream>", "#include <cassert>"))
        source += "\n" + enum + declarations + parser + "\nint main(){"
        source += 'assert(parseDataType("FP32")==ACL_FLOAT);'
        if supported:
            source += 'assert(parseDataType("HIFLOAT4")==ACL_HIFLOAT4);'
            source += 'assert(parseDataType("ACL_HIFLOAT4")==ACL_HIFLOAT4);'
        else:
            source += 'for(const auto* token : {"HIFLOAT4","ACL_HIFLOAT4"}) {'
            source += 'bool rejected=false;try{parseDataType(token);}catch(const std::runtime_error&){rejected=true;}'
            source += 'assert(rejected);}'
        source += 'assert(parseDataType("-999")==-999);return 0;}'
        cpp = folder / "dtype.cpp"
        cpp.write_text(source, encoding="utf-8")
        executable = folder / ("dtype.exe" if os.name == "nt" else "dtype")
        result = subprocess.run([str(Path(compiler).resolve()), "-std=c++17", "-Wall", "-Wextra", "-Werror",
                                 str(cpp), "-o", str(executable)],
                                capture_output=True, text=True, check=False, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return executable

    def check_sdk(self, supported):
        with tempfile.TemporaryDirectory(prefix="csv_dtype_") as temporary:
            executable = self.compile_parser(Path(temporary), supported)
            result = subprocess.run([str(executable)], capture_output=True, text=True, check=False, timeout=30)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_sdk_without_hifloat4(self):
        self.check_sdk(False)

    def test_sdk_with_hifloat4(self):
        self.check_sdk(True)


if __name__ == "__main__":
    unittest.main()
