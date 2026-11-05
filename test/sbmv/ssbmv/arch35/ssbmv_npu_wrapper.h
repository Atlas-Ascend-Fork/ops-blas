/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file ssbmv_npu_wrapper.h
 * \brief Ascend 950（arch35） 的测试封装入口。
 *
 * 封装体只依赖算子的 C 接口，与架构无关，实现放在 family 级 ssbmv_npu_wrapper_impl.h，
 * 两个架构共用同一份，避免同一段代码在两个 arch 目录下各存一份。
 */

#pragma once

#include "ssbmv_npu_wrapper_impl.h"
