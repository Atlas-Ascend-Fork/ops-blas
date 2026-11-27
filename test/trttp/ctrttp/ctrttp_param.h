/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <algorithm>
#include <stdexcept>
#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

// One CSV row of test/trttp/ctrttp/arch35/ctrttp_test.csv. Every projected column is read
// explicitly and an absent/empty column aborts the case instead of silently falling back to
// a default, so a broken CSV cannot shrink coverage without notice.
struct CtrttpParam : public BlasTestParamBase {
    aclblasFillMode_t uplo = ACLBLAS_LOWER;
    int n = 0;
    BlasFillMode a = parseFill("RANDOM_NORM_5_5");
    int lda = 1;
    BlasFillMode ap = parseFill("VALUE_NORM_0");

    CtrttpParam(const csv_map& map) : BlasTestParamBase(map)
    {
        auto require = [&](const char* key) -> std::string {
            auto value = ReadMap(map, key);
            if (value.empty()) {
                throw std::runtime_error(std::string("missing or empty CSV column: ") + key);
            }
            return value;
        };

        caseName = require("case_name");
        description = require("description");
        uplo = parseFillMode(require("uplo"));
        n = parseInt(require("n"));
        a = parseFill(require("a"));
        lda = parseInt(require("lda"));
        ap = parseFill(require("ap"));
        expectResult = parseStatus(require("expect_result"));
        randomSeed = parseUint(require("random_seed"));
        // This operator only moves bytes, so the task book pins rtol = atol = 0. The two
        // columns are still read to keep the CSV contract identical to the rest of the family.
        mereThreshold = parseDouble(require("mere_threshold"));
        mareMultiplier = parseDouble(require("mare_multiplier"));
    }
};

