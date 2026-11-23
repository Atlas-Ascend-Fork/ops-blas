/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// TEMPLATE: Ex 变体 kernel 侧实现骨架（arch35 / DAV_3510）
// ============================================================================
// ex 变体 kernel 侧的核心增量：按 DTypeCase 派发到不同模板实例。
// 计算模型随算子形态差异极大，本模板不写死任何一种，按形态分支说明：
//
//   ┌─────────────────────┬──────────────────────────────────────────────────────┐
//   │ 形态                 │ kernel 实现特征                                       │
//   ├─────────────────────┼──────────────────────────────────────────────────────┤
//   │ 矩阵类 (gemm_ex 系列)│ Cube BlockMmad 宏批量实例化 + Vector alpha/beta 后处理│
//   │ 向量类 (scalex)      │ Vector SIMD 模板类 + SIMT VF 函数，按 xType 派发      │
//   │ 分组类 (gemm_grouped)│ Cube kernel + Epilogue kernel，tiling 从 GM 读取      │
//   └─────────────────────┴──────────────────────────────────────────────────────┘
//
// 共性骨架（所有 ex 变体）：
//   1. 每个 DTypeCase 对应一个 extern "C" __global__ kernel 实例（模板参数不同）
//   2. kernel_do 内 switch(dtypeCase) 派发到对应实例
//   3. kernel 实例间用宏/模板消除重复（矩阵类用宏，向量类用模板类）
// ============================================================================

#include "kernel_operator.h"
// VARIANT: 向量类追加 SIMT 头 #include "simt_api/asc_simt.h"
// VARIANT: 矩阵类追加 #define ASCENDC_CUBE_ONLY 与 #include "common/arch/hardware.h"
#include "{{op}}_tiling_data.h"
#include "{{op}}_kernel.h"

// ============================================================================
// VARIANT: 矩阵类（gemm_ex 系列）—— Cube BlockMmad 宏 + 派发
// ============================================================================
// 设计要点（参考 gemm_ex_kernel.cpp）：
//   - 用宏 GEMM_CUBE_KERNEL(FUNC, A_TYPE, B_TYPE, C_GM_TYPE, BM, BK, BN, C0, QUANT)
//     一次性生成所有 cube kernel 变体，消除重复
//   - arch35 限制：Cube kernel 必须为 standalone __aicore__ 函数，禁止类模板方法（Mmad hang）
//   - 每个 DTypeCase 一个实例，baseM/baseN/baseK/c0Size 随 dtype 硬编码为模板非类型参数
//   - kernel 内手动管理 L1/A1/B1/A2/B2 LocalTensor，K-outer + 2D blocking
//
// #define ASCENDC_CUBE_ONLY
// #define {{OP}}_CUBE_KERNEL(FUNC_NAME, A_TYPE, B_TYPE, C_GM_TYPE, BM, BK, BN, C0_VAL, QUANT_MODE)  \
//     extern "C" __global__ __cube__ void FUNC_NAME(                                                 \
//         __gm__ uint8_t* a, __gm__ uint8_t* b, __gm__ uint8_t* c, {{Op}}TilingData tiling)          \
//     {                                                                                               \
//         AscendC::InitSocState();                                                                    \
//         /* InitState: 解析 mBlockIdx/nBlockIdx, 计算 actualM/N, baseMCount/tailM, 2D blocking */     \
//         /* K-outer loop: LoadATile -> ProcessNTile -> WriteFixpipeBlock */                           \
//     }
//
// {{OP}}_CUBE_KERNEL({{op}}_kernel_fp16,     half,         half,         half, 128, 16, 128, 16, QuantMode_t::F322F16)
// {{OP}}_CUBE_KERNEL({{op}}_kernel_bf16,     bfloat16_t,   bfloat16_t,   bfloat16_t, 128, 16, 128, 16,
// QuantMode_t::F322BF16)
// {{OP}}_CUBE_KERNEL({{op}}_kernel_fp32,     float,        float,        float,  32,  8,  16,  8, QuantMode_t::NoQuant)
// {{OP}}_CUBE_KERNEL({{op}}_kernel_fp8_e4m3, fp8_e4m3fn_t, fp8_e4m3fn_t, half,  32, 32,  16, 32, QuantMode_t::F322F16)
// {{OP}}_CUBE_KERNEL({{op}}_kernel_fp8_e5m2, fp8_e5m2_t,   fp8_e5m2_t,   half,  32, 32,  16, 32, QuantMode_t::F322F16)
// {{OP}}_CUBE_KERNEL({{op}}_kernel_fp16_out_f32, half, half, float, 128, 16, 128, 16, QuantMode_t::NoQuant)
// {{OP}}_CUBE_KERNEL({{op}}_kernel_bf16_out_f32, bfloat16_t, bfloat16_t, float, 128, 16, 128, 16, QuantMode_t::NoQuant)
// ...（按 DTypeCase 枚举一一实例化）

// ============================================================================
// VARIANT: 向量类（scalex 类）—— Vector SIMD 模板类 + SIMT VF 函数
// ============================================================================
// 设计要点（参考 scalex_kernel.cpp）：
//   - 模板类 template<typename XType> class {{Op}}AIV，按 xType 实例化 float/half/bfloat16_t
//   - incx==1 走 SIMD 连续路径（DataCopy + Muls + DataCopy），incx!=1 走 SIMT stride 路径
//   - 混合精度：FP16/BF16 输入先 Cast->FP32，Muls 后 Cast 回原类型（midBuf 中转）
//   - 两个 kernel entry：{{op}}_aiv_kernel（SIMD）+ {{op}}_simt_kernel（SIMT），kernel_do 按 incx 分支
//
// template<typename XType>
// class {{Op}}AIV {
//     __aicore__ inline void Init(GM_ADDR x, GM_ADDR alpha, const {{Op}}TilingData& t, TPipe* pipe);
//     __aicore__ inline void Process();
//     __aicore__ inline void SingleIteration(uint32_t off, uint32_t cnt);
//     __aicore__ inline void ProcessFp32(...);   // XType=float 直乘
//     __aicore__ inline void ProcessMixed(...);  // XType=half/bf16 经 FP32 中转
// };
// template<typename XType>
// __simt_vf__ __aicore__ LAUNCH_BOUND(SIMT_MAX_THREAD_NUM) inline void {{Op}}SimtCompute(...);
//
// extern "C" __global__ __aicore__ void {{op}}_aiv_kernel(GM_ADDR x, GM_ADDR alpha, {{Op}}TilingData t) {
//     KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY); TPipe pipe;
//     if (t.xType == ACL_FLOAT)      { {{Op}}AIV<float> op; op.Init(...); op.Process(); }
//     else if (t.xType == ACL_FLOAT16) { {{Op}}AIV<half> op; ... }
//     else                            { {{Op}}AIV<bfloat16_t> op; ... }
// }

// ============================================================================
// VARIANT: 矩阵类 alpha/beta 后处理 Vector kernel（可选）
// ============================================================================
// 设计要点（参考 gemm_ex ab_kernel）：
//   - template<typename TEMP_TYPE, typename C_TYPE, RoundMode OUTPUT_ROUND> class AlphaBetaKernel
//   - C = alpha * tempAB + beta * C_orig；tempAB 可能是 FP32 中间态或原 dtype
//   - 按 (dtypeCase, useFP32Temp) 派发到不同模板实例（f32_to_f16 / f32_to_bf16 / fp16 / bf16 / fp32）
//   - 列方向切分多核：colsPerCore = (n + blockNum - 1) / blockNum

// ============================================================================
// TEMPLATE: kernel_do 派发函数（所有 ex 变体共性的出口）
// host 侧调用此函数，内部 switch(dtypeCase) 派发到上述 kernel 实例
// ============================================================================
void
{
    {
        op
    }
}
_kernel_do(
    /* GM_ADDR 数据指针, */
    uint32_t numBlocks, void* stream,
    const {
        {
            Op
        }
    } TilingData& tiling,
    {
        {
            Op
        }
    } DTypeCase dtypeCase)
{
    auto aclStream = static_cast<aclrtStream>(stream);
    // GM_ADDR 数据指针转 __gm__ uint8_t*
    // VARIANT: 矩阵类 —— switch 派发 cube kernel
    switch (dtypeCase) {
        case { { OP } } _DTYPE_FP16:
            // {{op}}_kernel_fp16<<<numBlocks, nullptr, aclStream>>>(a, b, c, tiling);
            break;
        case { { OP } } _DTYPE_BF16:
            // {{op}}_kernel_bf16<<<numBlocks, nullptr, aclStream>>>(a, b, c, tiling);
            break;
        case { { OP } } _DTYPE_FP8_E4M3:
            // {{op}}_kernel_fp8_e4m3<<<numBlocks, nullptr, aclStream>>>(a, b, c, tiling);
            break;
        case { { OP } } _DTYPE_FP16_OUT_F32:
            // {{op}}_kernel_fp16_out_f32<<<numBlocks, nullptr, aclStream>>>(a, b, c, tiling);
            break;
        // ... 每个 DTypeCase 一个 case
        default:
            break;
    }
    // VARIANT: 向量类 —— 按 tiling.incx 分支 SIMD/SIMT
    // if (tiling.incx == 1) { {{op}}_aiv_kernel<<<numBlocks, nullptr, aclStream>>>(x, alpha, tiling); }
    // else                  { {{op}}_simt_kernel<<<numBlocks, nullptr, aclStream>>>(x, alpha, tiling); }
    // VARIANT: 分组类 —— tilingGm 从 GM 读取，cube + epilogue 两段 launch
}

// TEMPLATE: alpha/beta 后处理派发（矩阵类可选，向量类删除）
void
{
    {
        op
    }
}
_alpha_beta_do(
    uint32_t numBlocks, void* stream, uint8_t* tempAB, uint8_t* cOrig, uint8_t* cOut,
    const {
        {
            Op
        }
    } TilingData& tiling,
    {
        {
            Op
        }
    } DTypeCase dtypeCase,
    bool useFP32Temp)
{
    auto aclStream = static_cast<aclrtStream>(stream);
    // VARIANT: useFP32Temp -> f32_to_f16 / f32_to_bf16 实例；否则 -> fp16/bf16/fp32 实例
    // if (useFP32Temp) {
    //     switch (dtypeCase) {
    //         case {{OP}}_DTYPE_FP16: ab_kernel::{{op}}_alpha_beta_kernel_f32_to_f16<<<...>>>; break;
    //         case {{OP}}_DTYPE_BF16: ab_kernel::{{op}}_alpha_beta_kernel_f32_to_bf16<<<...>>>; break;
    //         default: break;
    //     }
    //     return;
    // }
    // switch (dtypeCase) {
    //     case {{OP}}_DTYPE_FP16: {{op}}_alpha_beta_kernel_fp16<<<...>>>; break;
    //     case {{OP}}_DTYPE_BF16: {{op}}_alpha_beta_kernel_bf16<<<...>>>; break;
    //     case {{OP}}_DTYPE_FP32: {{op}}_alpha_beta_kernel_fp32<<<...>>>; break;
    //     default: break;
    // }
    (void)numBlocks;
    (void)tempAB;
    (void)cOrig;
    (void)cOut;
    (void)tiling;
    (void)dtypeCase;
    (void)useFP32Temp;
}
