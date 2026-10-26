# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
import importlib.util
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[2] / 'test/nrm2/scnrm2/verify_performance.py'
SPEC = importlib.util.spec_from_file_location('performance_under_test', SOURCE)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_duplicate_timing_records_fail_the_host_log_gate(tmp_path):
    binary = tmp_path / 'timing_program.sh'
    binary.write_text(
        "#!/bin/sh\nprintf '%s\n' '[STRICT_PERF] case=TC_PF_1001 n=16 incx=1"
        " warmup=20 samples=100 average_us=1.0' '[STRICT_PERF] case=TC_PF_"
        "1001 n=16 incx=1 warmup=20 samples=100 average_us=1.0'\n"
    )
    binary.chmod(0o755)
    returncode, timings = MODULE.run_gtest(binary, device=None, timeout=10, strict=True)
    assert returncode != 0
    assert 'TC_PF_1001' in timings
