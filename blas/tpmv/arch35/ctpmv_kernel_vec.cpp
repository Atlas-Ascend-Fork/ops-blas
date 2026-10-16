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
 * \file ctpmv_kernel_vec.cpp
 * \brief v4.7: trans=T/C + UPPER Vector fast path (arch35) + ROTATION 诊断映射.
 * 数据流: preswap(SIMT预计算x符号变体→GM scratch) → 全MTE2装载(数据头+零尾) →
 * 向量计算(plain Mul + cadd_v) → GM直写输出。跨面板 PIPE_ALL 屏障。
 * ROTATION: 槽local装载列(local==0?末列:local-1)——诊断"槽位0路径 vs 列数据通路"。
 */

#include <cstdint>
#include "cann_ops_blas_common.h"
#include "kernel_operator.h"
#include "common/arch/hardware.h"
#include "common/compute/simd.h"
#include "common/memory/mem.h"
#include "ctpmv_tiling_data.h"

using namespace AscendC;

#pragma clang fp contract(off)

namespace {

constexpr uint32_t CTPMV_VEC_PANEL = 256U;
constexpr uint32_t CTPMV_VEC_GROUP = 16U;
constexpr uint32_t CTPMV_VEC_MATRIX = CTPMV_VEC_PANEL * CTPMV_VEC_GROUP;
constexpr uint32_t CTPMV_COL_REPEATS = CTPMV_VEC_PANEL / 64U * 2U;

__aicore__ inline uint32_t VecMin(uint32_t a, uint32_t b) { return a < b ? a : b; }

__aicore__ inline uint64_t VecUpperIndex(uint32_t row, uint32_t col)
{
    uint64_t c = static_cast<uint64_t>(col);
    return static_cast<uint64_t>(row) + c * (c + 1ULL) / 2ULL;
}

// v5.1: 下三角打包偏移——列 j 起点为 j*(2n-j+1)/2, 行 r 元素在其后 (r-j) 处 (r>=j)
__aicore__ inline uint64_t VecLowerIndex(uint32_t row, uint32_t col, uint32_t n)
{
    uint64_t c = static_cast<uint64_t>(col);
    uint64_t nn = static_cast<uint64_t>(n);
    uint64_t colStart = c * (2ULL * nn - c + 1ULL) / 2ULL;
    return colStart + (static_cast<uint64_t>(row) - c);
}

__aicore__ inline void VecCopyAosPad(LocalTensor<float> dst, GlobalTensor<float> src, uint32_t floatCount)
{
    const uint32_t rightPadding = (-floatCount) & 7U;
    DataCopyExtParams cp{1U, static_cast<uint32_t>(floatCount * sizeof(float)), 0U, 0U, 0U};
    DataCopyPadExtParams<float> pp{true, 0U, static_cast<uint8_t>(rightPadding), 0.0F};
    DataCopyPad(dst, src, cp, pp);
}

class CtpmvTcVecAiv {
public:
    __aicore__ inline void Init(
        GM_ADDR aPacked, GM_ADDR x, GM_ADDR snapshot, GM_ADDR scrA, GM_ADDR scrB, GM_ADDR zeros,
        const CtpmvTilingData& tiling)
    {
        n_ = tiling.n;
        unit_ = tiling.diag == ACLBLAS_UNIT;
        upper_ = tiling.uplo == ACLBLAS_UPPER;
        conj_ = tiling.trans == ACLBLAS_OP_C;
        groupCount_ = (n_ + CTPMV_VEC_GROUP - 1U) / CTPMV_VEC_GROUP;
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aPacked), static_cast<uint64_t>(n_) * (n_ + 1ULL));
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), static_cast<uint64_t>(n_) * 2ULL);
        snapGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(snapshot), static_cast<uint64_t>(n_) * 2ULL);
        scrAGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scrA), static_cast<uint64_t>(n_) * 2ULL);
        scrBGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scrB), static_cast<uint64_t>(n_) * 2ULL);
        zeroGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(zeros), 8192U);
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t coreNum = GetBlockNum();
        if (blockIdx >= coreNum) {
            return;
        }

        constexpr uint32_t kMatrixAos = 0U;
        constexpr uint32_t kVarA = kMatrixAos + CTPMV_VEC_MATRIX * 2U;
        constexpr uint32_t kVarB = kVarA + CTPMV_VEC_PANEL * 2U;
        constexpr uint32_t kProd = kVarB + CTPMV_VEC_PANEL * 2U;
        constexpr uint32_t kPart = kProd + CTPMV_VEC_PANEL * 2U;

        AsdopsBuffer<ArchType::ASCEND_V350> buffer;
        LocalTensor<float> matrixAos = buffer.GetBuffer<BufferType::ASCEND_UB, float>(kMatrixAos * sizeof(float));
        LocalTensor<float> varA = buffer.GetBuffer<BufferType::ASCEND_UB, float>(kVarA * sizeof(float));
        LocalTensor<float> varB = buffer.GetBuffer<BufferType::ASCEND_UB, float>(kVarB * sizeof(float));
        LocalTensor<float> prod = buffer.GetBuffer<BufferType::ASCEND_UB, float>(kProd * sizeof(float));
        LocalTensor<float> part = buffer.GetBuffer<BufferType::ASCEND_UB, float>(kPart * sizeof(float));

        for (uint32_t group = blockIdx; group < groupCount_; group += coreNum) {
            const uint32_t columnBegin = group * CTPMV_VEC_GROUP;
            const uint32_t validColumns = VecMin(CTPMV_VEC_GROUP, n_ - columnBegin);
            float yR[CTPMV_VEC_GROUP];
            float yI[CTPMV_VEC_GROUP];
            for (uint32_t c = 0U; c < validColumns; ++c) {
                yR[c] = 0.0F;
                yI[c] = 0.0F;
            }

            if (unit_) {
                InitGroupDiagonal(columnBegin, validColumns, yR, yI);
            }

            // v5.1: 面板区间分 uplo——UPPER 仍为 [0, 组末列+1); LOWER 列数据在行 [j,n), 从列所在面板起到 n
            const uint32_t panelStart = upper_ ? 0U : (columnBegin / CTPMV_VEC_PANEL) * CTPMV_VEC_PANEL;
            const uint32_t panelLimit = upper_ ? VecMin(n_, columnBegin + validColumns) : n_;
            for (uint32_t panelBegin = panelStart; panelBegin < panelLimit; panelBegin += CTPMV_VEC_PANEL) {
                const uint32_t panelEnd = VecMin(n_, panelBegin + CTPMV_VEC_PANEL);
                const uint32_t validRows = panelEnd - panelBegin;

                for (uint32_t local = 0U; local < validColumns; ++local) {
                    LoadPanelColumn(
                        columnBegin, validColumns, local, panelBegin, panelEnd, matrixAos, yR[local], yI[local]);
                }
                VecCopyAosPad(varA, scrAGm_[static_cast<uint64_t>(panelBegin) * 2ULL], validRows * 2U);
                VecCopyAosPad(varB, scrBGm_[static_cast<uint64_t>(panelBegin) * 2ULL], validRows * 2U);
                PipeBarrier<PIPE_ALL>();

                for (uint32_t local = 0U; local < validColumns; ++local) {
                    AccumulateColumn(matrixAos, varA, varB, prod, part, local, yR[local], yI[local]);
                }
                PipeBarrier<PIPE_ALL>(); // 跨面板屏障: 计算/读数完成后再进下一面板装载
            }

            StoreGroupColumns(columnBegin, validColumns, yR, yI);
        }
    }

private:
    // v5.1 修复注记: 槽位↔列存在 ROTATION 映射(槽0=末列,槽c=列c-1),
    // 对角初始化必须取映射后那一列的 x 快照——此前直接取 columnBegin+c 是错位源。
    __aicore__ inline void InitGroupDiagonal(uint32_t columnBegin, uint32_t validColumns, float* yR, float* yI) const
    {
        for (uint32_t c = 0U; c < validColumns; ++c) {
            const uint32_t diagCol = columnBegin + ((c == 0U) ? (validColumns - 1U) : (c - 1U));
            yR[c] = snapGm_.GetValue(static_cast<uint64_t>(diagCol) * 2ULL);
            yI[c] = snapGm_.GetValue(static_cast<uint64_t>(diagCol) * 2ULL + 1ULL);
        }
    }

    // ROTATION: 槽 local 装载列 (local==0 ? 末列 : local-1); 标量补头累加进 yRLoc/yILoc
    __aicore__ inline void LoadPanelColumn(
        uint32_t columnBegin, uint32_t validColumns, uint32_t local, uint32_t panelBegin, uint32_t panelEnd,
        LocalTensor<float> matrixAos, float& yRLoc, float& yILoc)
    {
        const uint32_t column = columnBegin + ((local == 0U) ? (validColumns - 1U) : (local - 1U));
        uint32_t headFloats = 0U;
        uint32_t dataFloats = 0U;
        uint64_t packedOffset = 0ULL;
        if (upper_) {
            const uint32_t lastRow = VecMin(column + (unit_ ? 0U : 1U), panelEnd);
            dataFloats = (lastRow > panelBegin) ? (lastRow - panelBegin) * 2U : 0U;
            packedOffset = VecUpperIndex(panelBegin, column);
        } else {
            // v5.2 LOWER: 列 j 数据 = 行 [startRow, panelEnd) 连续打包段。
            // MTE 装载 dst 须 32B 对齐——数据起点向上对齐到 8 float 边界,
            // 对齐让出的 ≤3 行用标量补(仅该列首个面板出现错位头; 后续面板 startRow==panelBegin 天然对齐)。
            uint32_t startRow = unit_ ? (column + 1U) : column;
            if (startRow < panelBegin) {
                startRow = panelBegin;
            }
            if (panelEnd > startRow) {
                const uint32_t headExact = (startRow - panelBegin) * 2U;
                const uint32_t dataOff = (headExact + 7U) & ~7U;
                const uint32_t dataStartRow = startRow + ((dataOff - headExact) >> 1U);
                if (dataStartRow < panelEnd) {
                    dataFloats = (panelEnd - dataStartRow) * 2U;
                    headFloats = dataOff;
                    packedOffset = VecLowerIndex(dataStartRow, column, n_);
                }
                AccumLowerScalarHead(startRow, VecMin(dataStartRow, panelEnd), column, yRLoc, yILoc);
            }
        }
        const uint32_t dataAligned = headFloats + ((dataFloats + 7U) & ~7U);
        if (headFloats > 0U) {
            VecCopyAosPad(matrixAos[local * CTPMV_VEC_PANEL * 2U], zeroGm_, headFloats);
        }
        if (dataAligned < CTPMV_VEC_PANEL * 2U) {
            VecCopyAosPad(
                matrixAos[local * CTPMV_VEC_PANEL * 2U + dataAligned], zeroGm_, CTPMV_VEC_PANEL * 2U - dataAligned);
        }
        if (dataFloats > 0U) {
            VecCopyAosPad(matrixAos[local * CTPMV_VEC_PANEL * 2U + headFloats], aGm_[packedOffset * 2ULL], dataFloats);
        }
    }

    // 对齐让出的错位头(≤3 行)用标量补乘, 走 GM 逐元素读
    __aicore__ inline void AccumLowerScalarHead(
        uint32_t startRow, uint32_t scalarEnd, uint32_t column, float& yRLoc, float& yILoc)
    {
        for (uint32_t r = startRow; r < scalarEnd; ++r) {
            const uint64_t soff = VecLowerIndex(r, column, n_) * 2ULL;
            float aR = aGm_.GetValue(soff);
            float aI = aGm_.GetValue(soff + 1ULL);
            if (conj_) {
                aI = -aI;
            }
            const uint64_t xoff = static_cast<uint64_t>(r) * 2ULL;
            const float xR = snapGm_.GetValue(xoff);
            const float xI = snapGm_.GetValue(xoff + 1ULL);
            yRLoc += aR * xR - aI * xI;
            yILoc += aR * xI + aI * xR;
        }
    }

    __aicore__ inline void AccumulateColumn(
        LocalTensor<float> matrixAos, LocalTensor<float> varA, LocalTensor<float> varB, LocalTensor<float> prod,
        LocalTensor<float> part, uint32_t local, float& yRLoc, float& yILoc)
    {
        Mul(prod, matrixAos[local * CTPMV_VEC_PANEL * 2U], varA, CTPMV_VEC_PANEL * 2U);
        PipeBarrier<PIPE_V>();
        cadd_v<ArchType::ASCEND_V350, float>(part, prod, CTPMV_COL_REPEATS, 1U, 1U, 8U);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t k = 0U; k < CTPMV_COL_REPEATS; ++k) {
            yRLoc += part.GetValue(k);
        }
        Mul(prod, matrixAos[local * CTPMV_VEC_PANEL * 2U], varB, CTPMV_VEC_PANEL * 2U);
        PipeBarrier<PIPE_V>();
        cadd_v<ArchType::ASCEND_V350, float>(part, prod, CTPMV_COL_REPEATS, 1U, 1U, 8U);
        PipeBarrier<PIPE_ALL>();
        for (uint32_t k = 0U; k < CTPMV_COL_REPEATS; ++k) {
            yILoc += part.GetValue(k);
        }
    }

    // 输出: GM 直写(ROTATION 反向映射)
    __aicore__ inline void StoreGroupColumns(
        uint32_t columnBegin, uint32_t validColumns, const float* yR, const float* yI)
    {
        for (uint32_t c = 0U; c < validColumns; ++c) {
            const uint32_t col = columnBegin + ((c == 0U) ? (validColumns - 1U) : (c - 1U));
            const uint64_t off = static_cast<uint64_t>(col) * 2ULL;
            xGm_.SetValue(off, yR[c]);
            xGm_.SetValue(off + 1ULL, yI[c]);
        }
    }

    GlobalTensor<float> aGm_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> snapGm_;
    GlobalTensor<float> scrAGm_;
    GlobalTensor<float> scrBGm_;
    GlobalTensor<float> zeroGm_;
    uint32_t n_ = 0U;
    uint32_t groupCount_ = 1U;
    bool unit_ = false;
    bool upper_ = true;
    bool conj_ = false;
};

} // namespace

// ==========================================================================
//  v5.6: 单遍 vec 归约——y[c] = Σ_{p<parts} yPart[p][c]。
//  N 路径主 kernel 产出 parts(=56) 行部分和, 原两遍 SIMT 归约逐线程跨行跳读
//  延迟死路(实测 20-39us)；本 kernel 按列段分 block, 每块 DMA 分块装载各行的
//  列段(连续 2KB 流)+向量 Add 累加+整段写回, 一遍出终值(无 mid 中间区)。
// ==========================================================================

namespace {

constexpr uint32_t CTPMV_RVEC_BLOCKS = 16U;
constexpr uint32_t CTPMV_RVEC_CHUNK = 256U; // 每块列段内的装载块(complex)

class CtpmvReduceAllVecAiv {
public:
    __aicore__ inline void Init(GM_ADDR src, GM_ADDR dst, const CtpmvTilingData& tiling, uint32_t parts)
    {
        n_ = tiling.n;
        parts_ = parts;
        srcGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ float*>(src), static_cast<uint64_t>(parts_) * static_cast<uint64_t>(n_) * 2ULL);
        dstGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dst), static_cast<uint64_t>(n_) * 2ULL);
    }

    __aicore__ inline void Process()
    {
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t coreNum = GetBlockNum();
        if (blockIdx >= coreNum) {
            return;
        }
        AsdopsBuffer<ArchType::ASCEND_V350> buffer;
        LocalTensor<float> acc = buffer.GetBuffer<BufferType::ASCEND_UB, float>(0U);
        LocalTensor<float> tmp = buffer.GetBuffer<BufferType::ASCEND_UB, float>(2U * CTPMV_RVEC_CHUNK * sizeof(float));
        // v5.8: 批深 8——tmp 区按 8 个 chunk 槽使用(后续偏移 b*CHUNK*2), 预留 8 槽空间
        LocalTensor<float> tmpB =
            buffer.GetBuffer<BufferType::ASCEND_UB, float>(2U * CTPMV_RVEC_CHUNK * (1U + 8U) * sizeof(float));

        const uint32_t colsPer = (n_ + CTPMV_RVEC_BLOCKS - 1U) / CTPMV_RVEC_BLOCKS;
        const uint32_t c0 = blockIdx * colsPer;
        const uint32_t c1 = VecMin(c0 + colsPer, n_);
        if (c1 <= c0) {
            return;
        }
        for (uint32_t base = c0; base < c1; base += CTPMV_RVEC_CHUNK) {
            const uint32_t valid = VecMin(CTPMV_RVEC_CHUNK, c1 - base);
            const uint32_t validFloats = valid * 2U;
            // 矢量 Add 的 count 须 256B 对齐(64 float); 尾部垃圾不落盘(写回按精确字节)
            const uint32_t addCount = (validFloats + 63U) & ~63U;
            VecCopyAosPad(acc, srcGm_[2ULL * static_cast<uint64_t>(base)], validFloats);
            PipeBarrier<PIPE_ALL>();
            // v5.7: 4 深装载批——4 个 DMA 连发走 MTE2 流水+单屏障, 消掉逐行 load->barrier 的
            // 延迟串行(v5.6 实测 15-20us 的根因); Add 链间用轻量 PIPE_V
            uint32_t p = 1U;
            while (p < parts_) {
                const uint32_t batch = VecMin(8U, parts_ - p);
                for (uint32_t b = 0U; b < batch; ++b) {
                    VecCopyAosPad(
                        tmpB[b * CTPMV_RVEC_CHUNK * 2U],
                        srcGm_[2ULL * (static_cast<uint64_t>(p + b) * static_cast<uint64_t>(n_) + base)], validFloats);
                }
                PipeBarrier<PIPE_ALL>();
                for (uint32_t b = 0U; b < batch; ++b) {
                    Add(acc, acc, tmpB[b * CTPMV_RVEC_CHUNK * 2U], addCount);
                    PipeBarrier<PIPE_V>();
                }
                PipeBarrier<PIPE_ALL>();
                p += batch;
            }
            DataCopyExtParams cp{1U, static_cast<uint32_t>(validFloats * sizeof(float)), 0U, 0U, 0U};
            DataCopyPad(dstGm_[2ULL * static_cast<uint64_t>(base)], acc, cp);
            PipeBarrier<PIPE_ALL>();
        }
    }

private:
    GlobalTensor<float> srcGm_;
    GlobalTensor<float> dstGm_;
    uint32_t n_ = 0U;
    uint32_t parts_ = 1U;
};

} // namespace

__global__ __aicore__ void ctpmv_reduce_all_vec_kernel(GM_ADDR src, GM_ADDR dst, CtpmvTilingData tiling, uint32_t parts)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CtpmvReduceAllVecAiv op;
    op.Init(src, dst, tiling, parts);
    op.Process();
}

void ctpmv_arch35_reduceall_vec_do(
    GM_ADDR src, GM_ADDR dst, const CtpmvTilingData& tiling, uint32_t parts, void* stream)
{
    ctpmv_reduce_all_vec_kernel<<<CTPMV_RVEC_BLOCKS, nullptr, stream>>>(src, dst, tiling, parts);
}

// ==========================================================================
//  v5.12: N-AXPY 向量 kernel (8<n<=128, incx=1)——SoA 双槽 + 完整复乘。
//  v5.10/11 教训: (a)AsdopsBuffer 仅构造一次; (b)复数乘必须四项——只广播实部
//  xR 是 92 挂的真根因。本版: strided 装载把列实/虚拆进 slotR/slotI(cgeam
//  先例), 每列 Muls x4 (xR, -xI, xI, xR) + Add x4; 槽尾零区用降序(UPPER)/
//  升序(LOWER)遍历+8对齐边界摊销; 残余行归 prep kernel(纯 GM 标量)。
// ==========================================================================

namespace {

class CtpmvNAxpyAiv {
public:
    __aicore__ inline void Init(
        GM_ADDR aPacked, GM_ADDR xIn, GM_ADDR yOut, GM_ADDR zeros, GM_ADDR gap, const CtpmvTilingData& tiling)
    {
        n_ = tiling.n;
        upper_ = tiling.uplo == ACLBLAS_UPPER;
        unit_ = tiling.diag == ACLBLAS_UNIT;
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(aPacked), static_cast<uint64_t>(n_) * (n_ + 1ULL));
        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(xIn), static_cast<uint64_t>(n_) * 2ULL);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(yOut), static_cast<uint64_t>(n_) * 2ULL);
        zeroGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(zeros), 8192U);
        gapGm_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(gap), static_cast<uint64_t>(n_) * 2ULL);
    }

    __aicore__ inline void Process()
    {
        if (GetBlockIdx() != 0u) {
            return;
        }
        AsdopsBuffer<ArchType::ASCEND_V350> buffer; // 全程唯一构造
        yR_ = buffer.GetBuffer<BufferType::ASCEND_UB, float>(kYR * sizeof(float));
        yI_ = buffer.GetBuffer<BufferType::ASCEND_UB, float>(kYI * sizeof(float));
        p1_ = buffer.GetBuffer<BufferType::ASCEND_UB, float>(kP1 * sizeof(float));
        p2_ = buffer.GetBuffer<BufferType::ASCEND_UB, float>(kP2 * sizeof(float));
        sR_ = buffer.GetBuffer<BufferType::ASCEND_UB, float>(kSR * sizeof(float));
        sI_ = buffer.GetBuffer<BufferType::ASCEND_UB, float>(kSI * sizeof(float));
        cnt_ = (n_ + 63U) & ~63U; // 实数域计数, 256B 对齐

        Duplicate(yR_, 0.0F, cnt_);
        Duplicate(yI_, 0.0F, cnt_);
        Duplicate(sR_, 0.0F, cnt_);
        Duplicate(sI_, 0.0F, cnt_);
        PipeBarrier<PIPE_ALL>();

        if (upper_) {
            // 降序: [A8(dR_c), cnt) 恒为零(装载只写 [0, A8)); 列间补零 [A8_next, A8_cur)
            for (uint32_t c = n_; c > 0U; --c) {
                const uint32_t cc = c - 1U;
                const uint32_t dR = unit_ ? cc : (cc + 1U);
                const uint32_t a8 = dR & ~7U;
                LoadColumnUpper(cc, a8);
                const float xR = xGm_.GetValue(static_cast<uint64_t>(cc) * 2ULL);
                const float xI = xGm_.GetValue(static_cast<uint64_t>(cc) * 2ULL + 1ULL);
                PipeBarrier<PIPE_ALL>();
                AccumColumn(xR, xI);
                PipeBarrier<PIPE_ALL>(); // V 读 sR_ 与下一列 MTE2 覆写之间必须全序
                FillUpperGap(cc, a8);
            }
        } else {
            // 升序: 头零区 [0, DS) 增长, 列间补零 [DS_prev, DS_cur); 尾零区天然保持
            for (uint32_t c = 0U; c < n_; ++c) {
                const uint32_t startRow = unit_ ? (c + 1U) : c;
                const uint32_t ds = (startRow + 7U) & ~7U;
                const uint32_t tailCnt = (ds < n_) ? ((n_ - ds) & ~7U) : 0U;
                LoadColumnLower(c, ds, tailCnt);
                const float xR = xGm_.GetValue(static_cast<uint64_t>(c) * 2ULL);
                const float xI = xGm_.GetValue(static_cast<uint64_t>(c) * 2ULL + 1ULL);
                PipeBarrier<PIPE_ALL>();
                AccumColumn(xR, xI);
                PipeBarrier<PIPE_ALL>(); // V 读 sR_ 与下一列 MTE2 覆写之间必须全序
                FillLowerGap(c, ds);
            }
        }
        PipeBarrier<PIPE_ALL>();
        FinalizeY();
    }

    // 收尾: UNIT 对角累加 + prep kernel 残余行贡献 + SoA 交错写回
    __aicore__ inline void FinalizeY()
    {
        if (unit_) {
            // UNIT 对角: y[j] += x[j] (SoA: 实部/虚部分别整段加)
            LoadXSoa();
            PipeBarrier<PIPE_ALL>();
            Add(yR_, yR_, sR_, cnt_);
            Add(yI_, yI_, sI_, cnt_);
            PipeBarrier<PIPE_V>();
        }
        // prep kernel 预置的残余行贡献(GM gap, SoA 拆装)
        LoadGapSoa();
        PipeBarrier<PIPE_ALL>();
        Add(yR_, yR_, sR_, cnt_);
        Add(yI_, yI_, sI_, cnt_);
        PipeBarrier<PIPE_ALL>();
        // 交错写回(cgeam store 先例: blockSize=4, dstStride=4)
        DataCopyExtParams cpR{static_cast<uint16_t>(n_), sizeof(float), 0, sizeof(float), 0};
        DataCopyPad(yGm_, yR_, cpR);
        DataCopyPad(yGm_[1], yI_, cpR);
        PipeBarrier<PIPE_ALL>();
    }

private:
    // UB 布局(实数域, 各 cnt_<=256 float): yR | yI | p1 | p2 | sR | sI
    static constexpr uint32_t kYR = 0U;
    static constexpr uint32_t kYI = 256U;
    static constexpr uint32_t kP1 = 512U;
    static constexpr uint32_t kP2 = 768U;
    static constexpr uint32_t kSR = 1024U;
    static constexpr uint32_t kSI = 1280U;

    // UPPER 降序遍历: 列间补零 [A8_next, A8_cur), 使 sR_/sI_ 尾部零区衔接下一列
    __aicore__ inline void FillUpperGap(uint32_t cc, uint32_t a8)
    {
        if (cc > 0U) {
            const uint32_t dRn = unit_ ? (cc - 1U) : cc;
            const uint32_t a8n = dRn & ~7U;
            if (a8n < a8) {
                VecCopyAosPad(sR_[a8n], zeroGm_, a8 - a8n);
                VecCopyAosPad(sI_[a8n], zeroGm_, a8 - a8n);
                PipeBarrier<PIPE_ALL>();
            }
        }
    }

    // LOWER 升序遍历: 头零区 [0, DS) 增长, 列间补零 [DS_prev, DS_cur)
    __aicore__ inline void FillLowerGap(uint32_t c, uint32_t ds)
    {
        if (c + 1U < n_) {
            const uint32_t startNext = unit_ ? (c + 2U) : (c + 1U);
            const uint32_t dsNext = (startNext + 7U) & ~7U;
            if (dsNext > ds && dsNext <= cnt_) {
                const uint32_t fillEnd = (dsNext < cnt_) ? dsNext : cnt_;
                if (fillEnd > ds) {
                    VecCopyAosPad(sR_[ds], zeroGm_, fillEnd - ds);
                    VecCopyAosPad(sI_[ds], zeroGm_, fillEnd - ds);
                    PipeBarrier<PIPE_ALL>();
                }
            }
        }
    }

    __aicore__ inline void AccumColumn(float xR, float xI)
    {
        // yR += sR*xR - sI*xI ; yI += sR*xI + sI*xR
        Muls(p1_, sR_, xR, cnt_);
        Muls(p2_, sI_, -xI, cnt_);
        PipeBarrier<PIPE_V>();
        Add(yR_, yR_, p1_, cnt_);
        Add(yR_, yR_, p2_, cnt_);
        PipeBarrier<PIPE_V>(); // p1/p2 WAR: 下一组 Muls 覆写前等 Add 退役
        Muls(p1_, sR_, xI, cnt_);
        Muls(p2_, sI_, xR, cnt_);
        PipeBarrier<PIPE_V>();
        Add(yI_, yI_, p1_, cnt_);
        Add(yI_, yI_, p2_, cnt_);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void LoadColumnUpper(uint32_t c, uint32_t a8)
    {
        if (a8 == 0U) {
            return;
        }
        const uint64_t cs = VecUpperIndex(0, c);
        DataCopyExtParams cp{static_cast<uint16_t>(a8), sizeof(float), sizeof(float), 0, 0};
        DataCopyPadExtParams<float> np;
        DataCopyPad<float, PaddingMode::Compact>(sR_, aGm_[cs * 2ULL], cp, np);
        DataCopyPad<float, PaddingMode::Compact>(sI_, aGm_[cs * 2ULL + 1ULL], cp, np);
    }

    __aicore__ inline void LoadColumnLower(uint32_t c, uint32_t ds, uint32_t tailCnt)
    {
        if (ds > 0U) {
            VecCopyAosPad(sR_, zeroGm_, ds);
            VecCopyAosPad(sI_, zeroGm_, ds);
        }
        if (tailCnt > 0U) {
            const uint64_t cs = VecLowerIndex(ds, c, n_);
            DataCopyExtParams cp{static_cast<uint16_t>(tailCnt), sizeof(float), sizeof(float), 0, 0};
            DataCopyPadExtParams<float> np;
            DataCopyPad<float, PaddingMode::Compact>(sR_[ds], aGm_[cs * 2ULL], cp, np);
            DataCopyPad<float, PaddingMode::Compact>(sI_[ds], aGm_[cs * 2ULL + 1ULL], cp, np);
        }
    }

    __aicore__ inline void LoadXSoa()
    {
        DataCopyExtParams cp{static_cast<uint16_t>(n_), sizeof(float), sizeof(float), 0, 0};
        DataCopyPadExtParams<float> np;
        DataCopyPad<float, PaddingMode::Compact>(sR_, xGm_, cp, np);
        DataCopyPad<float, PaddingMode::Compact>(sI_, xGm_[1], cp, np);
        const uint32_t aligned = (n_ + 7U) & ~7U;
        if (aligned < cnt_) {
            VecCopyAosPad(sR_[aligned], zeroGm_, cnt_ - aligned);
            VecCopyAosPad(sI_[aligned], zeroGm_, cnt_ - aligned);
        }
    }

    __aicore__ inline void LoadGapSoa()
    {
        DataCopyExtParams cp{static_cast<uint16_t>(n_), sizeof(float), sizeof(float), 0, 0};
        DataCopyPadExtParams<float> np;
        DataCopyPad<float, PaddingMode::Compact>(sR_, gapGm_, cp, np);
        DataCopyPad<float, PaddingMode::Compact>(sI_, gapGm_[1], cp, np);
        const uint32_t aligned = (n_ + 7U) & ~7U;
        if (aligned < cnt_) {
            VecCopyAosPad(sR_[aligned], zeroGm_, cnt_ - aligned);
            VecCopyAosPad(sI_[aligned], zeroGm_, cnt_ - aligned);
        }
    }

    GlobalTensor<float> aGm_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> yGm_;
    GlobalTensor<float> zeroGm_;
    GlobalTensor<float> gapGm_;
    LocalTensor<float> yR_;
    LocalTensor<float> yI_;
    LocalTensor<float> p1_;
    LocalTensor<float> p2_;
    LocalTensor<float> sR_;
    LocalTensor<float> sI_;
    uint32_t cnt_ = 0U;
    uint32_t n_ = 0U;
    bool unit_ = false;
    bool upper_ = true;
};

} // namespace

__global__ __aicore__ void ctpmv_naxpy_kernel(
    GM_ADDR aP, GM_ADDR x, GM_ADDR y, GM_ADDR zeros, GM_ADDR gap, CtpmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CtpmvNAxpyAiv op;
    op.Init(aP, x, y, zeros, gap, tiling);
    op.Process();
}

void ctpmv_arch35_naxpy_do(
    GM_ADDR aP, GM_ADDR x, GM_ADDR y, GM_ADDR zeros, GM_ADDR gap, const CtpmvTilingData& tiling, void* stream)
{
    ctpmv_naxpy_kernel<<<1, nullptr, stream>>>(aP, x, y, zeros, gap, tiling);
}

__global__ __aicore__ void ctpmv_tc_vec_kernel(
    GM_ADDR aPacked, GM_ADDR x, GM_ADDR snapshot, GM_ADDR scrA, GM_ADDR scrB, GM_ADDR zeros, CtpmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    CtpmvTcVecAiv op;
    op.Init(aPacked, x, snapshot, scrA, scrB, zeros, tiling);
    op.Process();
}

void ctpmv_arch35_vec_tc_do(
    GM_ADDR aPacked, GM_ADDR x, GM_ADDR snapshot, GM_ADDR scrA, GM_ADDR scrB, GM_ADDR zeros,
    const CtpmvTilingData& tiling, uint32_t numBlocks, void* stream)
{
    ctpmv_tc_vec_kernel<<<numBlocks, nullptr, stream>>>(aPacked, x, snapshot, scrA, scrB, zeros, tiling);
}
