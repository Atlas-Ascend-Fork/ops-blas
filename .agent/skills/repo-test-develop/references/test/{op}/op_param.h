/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// TEMPLATE: 测试参数结构体（与芯片无关）。文件落地为 test/<family>/{{op}}/{{op}}_param.h
// - 继承 BlasTestParamBase；字段按 API 参数顺序排列
// - 数组参数类型为 BlasFillMode（填充方式见 SKILL 的 BlasFillMode 命名规则）
// - 若按用例控制精度，额外加 mereThreshold / mareMultiplier 字段并从 CSV 读取

#ifndef{{OP } } _PARAM_H
#define{{OP } } _PARAM_H

#include <string>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"

struct {
    {
        Op
    }
} Param : public BlasTestParamBase
{
    // TEMPLATE: 按 API 参数顺序声明字段。以下为 tpttr 形态示例，按算子替换：
    aclblasFillMode_t uplo = ACLBLAS_LOWER;
    int n = 0;
    BlasFillMode ap = BlasFillMode("INDEX");          // 顺序正整数 1, 2, 3, ...
    BlasFillMode a = BlasFillMode("VALUE_NORM_N999"); // 哨兵值 -999
    int lda = 0;

    {
        {
            Op
        }
    }
    Param(const csv_map& m) : BlasTestParamBase(m)
    {
        // TEMPLATE: 每个字段从 CSV 读取，键名 = CSV 列名，第三参为缺省值
        uplo = parseFillMode(ReadMap(m, "uplo", "LOWER"));
        n = parseInt(ReadMap(m, "n", "0"));
        ap = BlasFillMode(ReadMap(m, "ap", "INDEX"));
        a = BlasFillMode(ReadMap(m, "a", "VALUE_NORM_N999"));
        lda = parseInt(ReadMap(m, "lda", std::to_string(std::max(1, n))));
    }
};

#endif // {{OP}}_PARAM_H
