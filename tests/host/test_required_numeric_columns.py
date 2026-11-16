# Copyright (c) 2026 balckwell
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Compile the actual pure Host numeric-column parsers without ACL/GTest stubs."""

import csv
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]


CPP_INCLUDES = (
    "#include <iomanip>\n#include <iostream>\n#include <stdexcept>\n"
    "#include <string>\n#include <unordered_map>\n#include <vector>\n#include <fstream>\n"
    "#include <gtest/gtest.h>\n"
)
CPP_DRIVER = r'''
std::string csvPath;
struct NumericRow {
    std::string name;
    int m;
    float alpha;
    explicit NumericRow(const csv_map& row)
        : name(ReadMap(row, "case_name")), m(ParseIntColumn(row, "m")),
          alpha(ReadMap(row, "alpha_real") == "null" ? 0.0f : ParseFloatColumn(row, "alpha_real")) {}
};
TEST(RequiredColumnsCsv, ConstructRows)
{
    std::cout << std::setprecision(9);
    for (const auto& row : GetCasesFromCsv<NumericRow>(csvPath)) {
        std::cout << "ROW " << row.name << " m=" << row.m << " alpha=" << row.alpha << '\n';
    }
}
int main(int argc, char** argv)
{
    if (std::string(argv[1]) == "csv") {
        csvPath = argv[2];
        ::testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    }
    csv_map row;
    if (argc > 2 && std::string(argv[2]) != "--missing") row["value"] = argv[2];
    try {
        std::cout << std::setprecision(9);
        if (std::string(argv[1]) == "int") {
            std::cout << ParseIntColumn(row, "value", argc > 3 ? argv[3] : "");
        } else {
            std::cout << ParseFloatColumn(row, "value", argc > 3 ? argv[3] : "");
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what();
        return 1;
    }
}
'''


def write_parser_source(directory):
    header = (ROOT / "test/frame/csv_loader.h").read_text(encoding="utf-8")
    loader = header[header.index("using csv_map ="):header.index("// ── ReplaceFileExtension2Csv")]
    parsers = header[header.index("inline int ParseIntColumn"):header.index("// ── Safe numeric parsers")]
    source = directory / "required_columns.cpp"
    source.write_text(CPP_INCLUDES + loader + parsers + CPP_DRIVER, encoding="utf-8")
    return source


def compile_parser(source, binary):
    compiler = shutil.which("c++")
    if compiler is None:
        raise RuntimeError("A C++ compiler is required for the native parser regression")
    result = subprocess.run(
        [compiler, "-std=c++17", "-Wall", "-Wextra", "-Werror", str(source),
         "-o", str(binary), "-lgtest", "-pthread"],
        capture_output=True, text=True, check=False,
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)


class RequiredNumericColumnsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.workspace = tempfile.TemporaryDirectory(prefix="ops-blas-required-columns-")
        cls.addClassCleanup(cls.workspace.cleanup)
        directory = Path(cls.workspace.name)
        source = write_parser_source(directory)
        cls.binary = directory / "required_columns"
        compile_parser(source, cls.binary)

    def parse(self, kind, *values):
        return subprocess.run([str(self.binary), kind, *values], capture_output=True, text=True, check=False)

    def assert_rejected(self, kind, value):
        result = self.parse(kind, value)
        self.assertEqual(result.returncode, 1, f"{kind} accepted {value!r} as {result.stdout!r}")
        self.assertIn("column 'value'", result.stderr)
        self.assertIn(value, result.stderr)

    def test_integer_cells_reject_numeric_prefixes(self):
        for value in ("0oops", "16x", "1.5", "1e3", "0x10", "42 24", "3\t4", "1\n2"):
            with self.subTest(value=value):
                self.assert_rejected("int", value)

    def test_float_cells_reject_numeric_prefixes(self):
        for value in ("0oops", "1.25junk", "1e", "1e+", "1.0f", "3.5 4", "nan_typo", "infjunk"):
            with self.subTest(value=value):
                self.assert_rejected("float", value)

    def test_integer_boundaries_and_whitespace(self):
        for value in ("0", "+16", "-16", "2147483647", "-2147483648", " \t42\r\n"):
            with self.subTest(value=value):
                result = self.parse("int", value)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(int(result.stdout), int(value))

    def test_float_values_and_special_value_injection(self):
        for value in ("0", "-0", "+1.25", "-2.5e-3", ".5", "1.", " \t3.5\r\n", "nan", "inf", "-inf"):
            with self.subTest(value=value):
                result = self.parse("float", value)
                self.assertEqual(result.returncode, 0, result.stderr)
                actual, expected = float(result.stdout), float(value)
                if math.isnan(expected):
                    self.assertTrue(math.isnan(actual))
                else:
                    self.assertAlmostEqual(actual, expected)
        for value, expected in (("0x1p+2", 4.0), ("nan(payload)", float("nan"))):
            with self.subTest(value=value):
                result = self.parse("float", value)
                self.assertEqual(result.returncode, 0, result.stderr)
                if math.isnan(expected):
                    self.assertTrue(math.isnan(float(result.stdout)))
                else:
                    self.assertEqual(float(result.stdout), expected)

    def test_existing_missing_nonnumeric_and_range_errors(self):
        for kind, values in (("int", ("", "oops", "2147483648", "-2147483649")),
                             ("float", ("", "oops", "1e1000", "-1e1000"))):
            for value in values:
                with self.subTest(kind=kind, value=value):
                    self.assert_rejected(kind, value)
            result = self.parse(kind)
            self.assertEqual(result.returncode, 1, result.stdout)
            self.assertIn("column 'value'", result.stderr)

    def test_default_value_also_requires_a_complete_number(self):
        for kind, value in (("int", "16"), ("float", "1.25")):
            with self.subTest(kind=kind):
                result = self.parse(kind, "--missing", value)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(float(result.stdout), float(value))
                result = self.parse(kind, "--missing", value + "oops")
                self.assertEqual(result.returncode, 1, result.stdout)
                self.assertIn("column 'value'", result.stderr)

    def test_complete_csv_loader_rejects_bad_rows_and_keeps_valid_rows(self):
        official = ROOT / "test/trsmbatched/ctrsmbatched/arch22/ctrsmbatched_test.csv"
        with official.open(newline="", encoding="utf-8") as source:
            originals = list(csv.DictReader(source))[:2]
        for column, value in (("m", "0oops"), ("m", "1.5"), ("alpha_real", "1.25oops")):
            with self.subTest(column=column, value=value):
                rows = [dict(row) for row in originals]
                bad_name = rows[0]["case_name"]
                rows[0][column] = value
                path = Path(self.workspace.name) / "cases.csv"
                with path.open("w", newline="", encoding="utf-8") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
                result = self.parse("csv", str(path))
                self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
                self.assertIn(f"column '{column}'", result.stdout)
                self.assertIn(value, result.stdout)
                self.assertIn("csv row at line 1 failed to construct a test case", result.stdout)
                self.assertNotIn("ROW " + bad_name + " ", result.stdout)
                self.assertIn("ROW " + rows[1]["case_name"] + " m=8 alpha=1", result.stdout)

    def assert_numeric_row_equal(self, fields, expected):
        self.assertEqual(fields[1], expected["case_name"])
        self.assertEqual(int(fields[2].split("=", 1)[1]), int(expected["m"]))
        alpha = 0.0 if expected["alpha_real"] == "null" else float(expected["alpha_real"])
        actual_alpha = float(fields[3].split("=", 1)[1])
        if math.isnan(alpha):
            self.assertTrue(math.isnan(actual_alpha))
        else:
            self.assertAlmostEqual(actual_alpha, alpha, places=6)

    def test_complete_official_csv_preserves_required_numeric_values(self):
        for arch in ("arch22", "arch35"):
            with self.subTest(arch=arch):
                path = ROOT / "test/trsmbatched/ctrsmbatched" / arch / "ctrsmbatched_test.csv"
                with path.open(newline="", encoding="utf-8") as source:
                    rows = list(csv.DictReader(source))
                result = self.parse("csv", str(path))
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                actual = [line.split() for line in result.stdout.splitlines() if line.startswith("ROW ")]
                self.assertEqual(len(actual), len(rows))
                for fields, expected in zip(actual, rows):
                    self.assert_numeric_row_equal(fields, expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
