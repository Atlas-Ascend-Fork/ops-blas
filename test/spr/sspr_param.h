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
#include <climits>
#include <string>

#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

/**
 * SsprParam — CSV 驱动参数。列格式：
 *   case_name,description,uplo,n,alpha,x_fill,incx,ap_fill,
 *   expect_result,mere_threshold,mare_multiplier,random_seed[,null_handle]
 * 列名兼容：优先读取 x_fill / ap_fill，缺失时回退 x / ap；
 * 附加列 null_handle 可选（缺省 0）；alpha 空指针写 "null" 或 "NULLPTR"。
 */
struct SsprParam : public BlasTestParamBase {
    aclblasFillMode_t uplo = ACLBLAS_UPPER;
    int n = 0;
    float alpha = 1.0f;
    bool alphaNull = false;
    BlasFillMode x = parseFill("RANDOM");
    int incx = 1;
    BlasFillMode ap = parseFill("RANDOM");
    bool nullHandle = false;

    SsprParam(const csv_map& map) : BlasTestParamBase(map)
    {
        // CSV 使用全枚举名 ACLBLAS_FILL_MODE_UPPER/LOWER，frame parseFillMode
        // 表仅含 ACLBLAS_UPPER/UPPER/ACLBLAS_LOWER/LOWER（数字回退仅覆盖 999 负向），
        // 此处归一化；非法枚举（999）仍走 parseFillMode 数字回退。
        uplo = parseFillMode(normalizeUplo(ReadMap(map, "uplo", "UPPER")));
        n = parseInt(ReadMap(map, "n", "0"));
        std::string alphaStr = ReadMap(map, "alpha", "1.0");
        alphaNull = (alphaStr == "null" || alphaStr == "NULLPTR");
        if (!alphaNull) {
            alpha = parseFloat(alphaStr, 1.0f);
        }
        x = parseFill(normalizeFillToken(ReadMap(map, "x_fill", ReadMap(map, "x", "RANDOM"))));
        incx = parseInt(ReadMap(map, "incx", "1"));
        ap = parseFill(normalizeFillToken(ReadMap(map, "ap_fill", ReadMap(map, "ap", "RANDOM"))));
        nullHandle = (ReadMap(map, "null_handle", "0") == "1");
    }

    // VALUE_ALL_<V> -> VALUE_NORM_<V>（V 形如 M5：M 前缀映射为 N 表示负值）
    static std::string normalizeFillToken(const std::string& s)
    {
        const std::string prefix = "VALUE_ALL_";
        if (s.size() > prefix.size() && s.compare(0, prefix.size(), prefix) == 0) {
            std::string tail = s.substr(prefix.size());
            if (!tail.empty() && tail[0] == 'M') {
                tail[0] = 'N';
            }
            return "VALUE_NORM_" + tail;
        }
        return s;
    }

    // 全枚举名 -> frame 短枚举名
    static std::string normalizeUplo(const std::string& s)
    {
        if (s == "ACLBLAS_FILL_MODE_UPPER") {
            return "ACLBLAS_UPPER";
        }
        if (s == "ACLBLAS_FILL_MODE_LOWER") {
            return "ACLBLAS_LOWER";
        }
        return s;
    }
};
