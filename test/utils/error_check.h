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

#include <cstdlib>
#include <cstdio>
#include <iostream>

#include "acl/acl.h"
#include "cann_ops_blasLt.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

#define CHECK_ACLRT(func)                                                               \
  {                                                                                     \
    aclError status = (func);                                                           \
    if (status != ACL_SUCCESS) {                                                        \
      std::cerr << "ACL Runtime Error at " << __FILE__ << ":" << __LINE__ << " (error code: " << status << ")" << std::endl; \
      exit(EXIT_FAILURE);                                                               \
    }                                                                                   \
  }

#define CHECK_ACLBLAS(func)                                                             \
  {                                                                                     \
    aclblasStatus_t status = (func);                                                    \
    if (status != ACLBLAS_STATUS_SUCCESS) {                                             \
      std::cerr << "BLASLT Error at " << __FILE__ << ":" << __LINE__ << " (error code: " << status << ")" << std::endl; \
      exit(EXIT_FAILURE);                                                               \
    }                                                                                   \
  }

