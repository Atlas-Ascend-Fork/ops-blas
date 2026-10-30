/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file cgeru_perf.cpp
 * \brief arch35 cgeru performance tool (shared plumbing in ../cgeru_perf_common.h)
 */

#include "cgeru_perf_common.h"

namespace {
constexpr int PERF_WARMUP = 10;
constexpr int PERF_ITERATIONS = 100;
constexpr double PERF_REQUIRED_RATIO = 0.4;

} // namespace

int main(int argc, char** argv)
{
    return cgeru_perf::CgeruPerfMain(
        argc, argv, PERF_WARMUP, PERF_ITERATIONS, PERF_REQUIRED_RATIO, cgeru_perf::LaunchCgeruCase);
}
