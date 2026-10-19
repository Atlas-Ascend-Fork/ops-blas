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
 * \file devkit_version_compat.h
 * \brief Shared C preprocessor macro for asc-devkit version checks.
 */

#pragma once

/* Evaluate the version check at preprocessing time so that the macro expands
 * to a plain 0/1 constant; using defined() inside a macro expansion is
 * undefined behavior and triggers -Wexpansion-to-defined warnings. */
#if defined(ASC_DEVKIT_MAJOR) && defined(ASC_DEVKIT_MINOR) && \
    (ASC_DEVKIT_MAJOR > 9 || (ASC_DEVKIT_MAJOR == 9 && ASC_DEVKIT_MINOR > 0))
#define ASC_DEVKIT_GE_9_1 1
#else
#define ASC_DEVKIT_GE_9_1 0
#endif
