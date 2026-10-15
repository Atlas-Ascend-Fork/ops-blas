/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Compile-time defaults; the standard build does not expose CMake options for these values.
// An explicit -D override must reach both host and device compilation, and needs revalidation.
// The values below are initial choices, not empirically tuned performance settings.
#pragma once
#ifndef CSCAL_ENABLE_VECTOR
#define CSCAL_ENABLE_VECTOR 1
#endif
// 0: auto, 1: always SIMT, 2: vector for contiguous input (SIMT for strides).
#ifndef CSCAL_KERNEL_POLICY
#define CSCAL_KERNEL_POLICY 0
#endif
#ifndef CSCAL_TILE_COMPLEX
#define CSCAL_TILE_COMPLEX 4096
#endif
#ifndef CSCAL_QUEUE_DEPTH
#define CSCAL_QUEUE_DEPTH 1
#endif
#ifndef CSCAL_VECTOR_THRESHOLD
#define CSCAL_VECTOR_THRESHOLD 256
#endif
#ifndef CSCAL_MIN_COMPLEX_PER_CORE
#define CSCAL_MIN_COMPLEX_PER_CORE 2048
#endif
// 1: task wording, alpha=(0,0) writes +0 to selected elements, even for Inf/NaN.
// 0: explicit four-product complex multiplication, including zero alpha.
#ifndef CSCAL_ZERO_WRITES_ZERO
#define CSCAL_ZERO_WRITES_ZERO 1
#endif
static_assert(
    CSCAL_TILE_COMPLEX > 0 && CSCAL_TILE_COMPLEX % 8 == 0, "tile must be a positive multiple of 8 complex elements");
static_assert(CSCAL_QUEUE_DEPTH == 1 || CSCAL_QUEUE_DEPTH == 2, "queue depth: 1 or 2");
static_assert(CSCAL_KERNEL_POLICY >= 0 && CSCAL_KERNEL_POLICY <= 2, "invalid policy");
static_assert(CSCAL_MIN_COMPLEX_PER_CORE > 0 && CSCAL_VECTOR_THRESHOLD > 0, "positive thresholds");
static_assert(CSCAL_ZERO_WRITES_ZERO == 0 || CSCAL_ZERO_WRITES_ZERO == 1, "zero policy: 0 or 1");
// RegBase uses two AoS queues, each rounded to 64 complex elements; no UB scratch.
// Reserve 8 KiB within the DAV-3510 UB capacity of 248 KiB.
static_assert(
    16ULL * CSCAL_QUEUE_DEPTH * ((CSCAL_TILE_COMPLEX + 63ULL) / 64ULL * 64ULL) + 8192ULL <= 248ULL * 1024ULL,
    "RegBase UB budget exceeded; decrease tile/depth or re-audit memory budget");
