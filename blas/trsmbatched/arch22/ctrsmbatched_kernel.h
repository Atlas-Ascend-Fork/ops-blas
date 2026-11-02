/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once

void LaunchCtrsmBatchedSmallInverse(void*, void*, void*, int, int, int, int, int, int, void*, int);
void LaunchCtrsmBatchedZero(void*, int, int, int, int, int, void*);
void LaunchCtrsmBatchedVectorFallback(
    void*, void*, int, int, int, int, int, int, int, int, int, float, float, int, void*);
void LaunchCtrsmBatchedDirectRhs(void*, void*, int, int, int, int, int, int, int, int, int, int, void*);
void LaunchCtrsmBatchedInverseCube(void*, void*, void*, int, int, int, int, int, int, int, int, int, int, void*);
void LaunchCtrsmBatchedBlockInverse(void*, void*, void*, int, int, int, int, int, int, int, int, int, int, void*);
void LaunchCtrsmBatchedPlanarSquare(void*, void*, int, int, int, int, void*);
void LaunchCtrsmBatchedHalfDiagonalSmall(void*, void*, void*, int, int, int, void*);
