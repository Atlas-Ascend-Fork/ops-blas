/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file ctpsv_tiling_data.h
 * \brief Tiling data for complex64 packed triangular solver (arch22, GM 传递).
 *
 * DESIGN.md §3.3：ops-blas 直调范式 TilingDataDevice——host 组装定长结构体 →
 * aclrtMemcpyAsync H2D 一次（per-stream 槽免同步复用 + 内容去重）→ kernel 侧
 * reinterpret_cast<__gm__*> 解析；不使用 BEGIN_TILING_DATA_DEF / TILING_KEY_IS
 * 废弃宏，也不引入 registry 两段式 workspace。
 *
 * 文件命名对齐上游 blas/tpsv/ 现版（arch35 stpsv_tiling_data.h，遗留决策点 6
 * 迭代一核对结论）；结构体名加 Ctpsv 前缀防跨算子头文件同名冲突（stpsv arch22
 * StpsvTilingDataDevice 同款防撞口径）。
 *
 * 不设运行期遍历方向字段：前代/回代方向由编译期模板参数（UPLO/TRANS/DIAG）定死，
 * 运行期再传方向参数存在与模板方向不一致的误用风险（stpsv MDE_REVIEW #7 裁剪）。
 * incx=1 连续快路径 / incx≠1 跨度路径为运行期参数选择，不增编译分支。
 */

#pragma once
#include <cstdint>

struct CtpsvTilingDataDevice { // 定长 POD，host 组装后 aclrtMemcpyAsync H2D 一次
    uint64_t apPtr;            // AP Device 地址（complex64 packed，只读）
    uint64_t xPtr;             // x Device 地址（complex64，原地覆写）
    uint32_t n;                // 阶数（n=0 不进 kernel，host 短路）
    int32_t incx;              // 步长（非零，负值合法）
    uint32_t useCoreNum;       // 实际启用核数（禁写死）；串行链单 AIV 核闭合 = 1
    uint32_t chunk;            // AP 列段分块宽度（复数元素数，host 决策）
    uint32_t negInc;           // |incx| 预计算值（0 表示 incx=1 快路径，DESIGN §3.3）
    uint32_t reserved;         // 对齐垫片（定长 POD 尾部填充）
};

// opt2 P2 小 n 标量快路径门槛（host/kernel 共享单一真源，UT 镜像同源）：n≤8 且
// incx=1 时 host 直调 ctpsv_kernel_do_small（独立 kernel 符号，12 模板 _small 实例）
// ——主 kernel 入口零触碰（结构隔离：内联/入口分支形态均实测移位 T7 档 n>8 未改
// 路径热循环布局 +3.2~3.7% scalar，msprof 定谳，见 ctpsv_kernel.cpp「opt2 P2」节）。
constexpr uint32_t CTPSV_SMALL_N_MAX = 8;
