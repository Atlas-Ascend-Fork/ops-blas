/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include "kernel_operator.h"
#include "cgeru_tiling_data.h"

using namespace AscendC;

constexpr uint32_t CGERU_BYTES_PER_FLOAT = 4;
constexpr uint32_t CGERU_FLOATS_PER_COMPLEX = 2;
constexpr uint32_t CGERU_BYTES_PER_COMPLEX = 8;
constexpr uint32_t CGERU_TILE_BYTES = CGERU_TILE_FLOATS * CGERU_BYTES_PER_FLOAT;
constexpr uint32_t CGERU_ROW_FLOATS = CGERU_FLOATS_PER_COMPLEX * CGERU_ROW_TILE_MAX;
// 多列块中每列在 UB 内按 2*rowTile 个 float 对齐排布（32 字节的整数倍）。
constexpr uint32_t CGERU_COL_STRIDE_FLOATS = CGERU_ROW_FLOATS;
constexpr uint32_t CGERU_ALIGN_FLOATS = CGERU_ALIGN_COMPLEX * CGERU_FLOATS_PER_COMPLEX;

// 行块按 32 个复数对齐后的 float 数不得超过 UB 单块容量（越界会踩坏相邻缓冲）。
static_assert(
    CGERU_FLOATS_PER_COMPLEX * ((CGERU_ROW_TILE_MAX + CGERU_ALIGN_COMPLEX - 1U) / CGERU_ALIGN_COMPLEX) *
            CGERU_ALIGN_COMPLEX <=
        CGERU_TILE_FLOATS,
    "cgeru row tile exceeds UB buffer capacity");
static_assert(CGERU_ALIGN_FLOATS <= CGERU_ROW_FLOATS, "cgeru row buffers must hold one 32-complex aligned chunk");

namespace {
// 流水事件号：搬出标志按槽编号，搬入标志按遍号奇偶，向量搬出标志即时消费。
constexpr event_t kEvStore0 = static_cast<event_t>(0);
constexpr event_t kEvStore1 = static_cast<event_t>(1);
constexpr event_t kEvStore2 = static_cast<event_t>(2);
constexpr event_t kEvLoadEven = static_cast<event_t>(3);
constexpr event_t kEvLoadOdd = static_cast<event_t>(4);
constexpr event_t kEvVecStore = static_cast<event_t>(5);
constexpr uint32_t kNoPendingRow = 0xFFFFFFFFU;

// 多列块路径的几何参数：槽宽、是否整段连续、tile 分块是否可用。
struct ColumnSlabPlan {
    bool contiguousSlab;
    bool alignedStride;
    uint32_t colStride;
    bool tileEnabled;
};

// 逐列路径的分遍参数。
struct ColumnPassPlan {
    uint32_t colsPerPass;
    uint32_t ntiles;
};
}  // namespace

/**
 * @brief aclblasCgeru 的 arch22(Atlas 800I A3 / ascend910b) 向量核实现。
 *
 * 复数在 GM 中按 (re, im) 交错存放。把一行块内的 x 记成交错视图
 * X = (xr0, xi0, xr1, xi1, ...)，并取常量模式 sign = (-1, +1, -1, +1, ...)
 * 与成对交换表 swap[k] = 4 * (k ^ 1)（Gather 的偏移按字节计），则
 *     S = gather(X * sign, swap) = (xi0, -xr0, xi1, -xr1, ...)
 * 记 c = alpha * y(j) = (u, v)，由复数乘法有 c * x = u * X - v * S，于是
 *     A(:, j) += u * X - v * S
 * 只需在交错布局上做标量乘与向量累加，全程不拆分实/虚部，也不需要重交错。
 * 其中 u、v 与 Netlib cgeru 一样按 TEMP = ALPHA*Y(JY) 计算。
 *
 * 分块：列方向按核切 slab，每个 A 元素只由一个核负责读-改-写，因此不需要原子加；
 * 行方向按 rowTile 切成 32 字节对齐的块。X 与 S 在同一个行块内只算一次，供该行块的
 * 所有列复用（x 的 GM 流量因此与列数无关）。y(j) 在 kernel 入口一次性折算成 (u, v)
 * 缓存进 UB，主循环内不再访问 y 的 GM。alpha*y(j) 为 (0, 0) 的列按 Netlib 语义整列跳过。
 */
class CgeruKernel {
public:
    __aicore__ inline CgeruKernel() {}
    __aicore__ inline void Init(const CgeruTilingData& tiling, TPipe* pipeIn);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessPass();

    // ---- 初始化 ----
    __aicore__ inline void BindTiling(const CgeruTilingData& tiling);
    __aicore__ inline void BindGlobalBuffers(const CgeruTilingData& tiling);
    __aicore__ inline void ComputeColumnRange();
    __aicore__ inline uint32_t ComputeBufferSizes();
    __aicore__ inline uint32_t ComputeChunkMax() const;
    __aicore__ inline uint32_t ComputeTableLayout(uint32_t chunkMax);
    __aicore__ inline uint32_t AlignCoeffFloats(uint32_t cols) const;
    __aicore__ inline uint32_t ComputeYBufBytes(uint32_t cols) const;
    __aicore__ inline void PlanColumnBuffers(uint32_t fixedBytes, uint32_t chunkMax);
    __aicore__ inline void ComputeSlotPlan(uint32_t chunkMax);
    __aicore__ inline void AllocateBuffers();

    // ---- 行块与系数 ----
    __aicore__ inline void BuildConstTables();
    __aicore__ inline void PrepareColumnCoeffs();
    __aicore__ inline void IssueEarlyX();
    __aicore__ inline bool LoadYCoefficients();
    __aicore__ inline void PrepareCoeffsVectorized(uint32_t coeffFloats);
    __aicore__ inline void PrepareCoeffsScalar(uint32_t coeffFloats);
    __aicore__ inline void GetUV(uint32_t col, float& u, float& v);
    __aicore__ inline void LoadX(uint32_t rowOff, uint32_t chunk);
    __aicore__ inline void BuildSwapPattern(uint32_t calcNum);

    // ---- 列搬运与累加 ----
    __aicore__ inline void ProcessColumns(uint32_t rowOff, uint32_t chunk, bool prologueOnly);
__aicore__ inline void PrimeTileZero(uint32_t cols, uint32_t rowOff, uint32_t chunk,
                                                uint32_t colStride, LocalTensor<float>& all);
__aicore__ inline void ProcessTilePipeline(uint32_t rowOff, uint32_t chunk, const ColumnSlabPlan& plan);
    __aicore__ inline void ProcessColumnsByColumn(uint32_t rowOff, uint32_t chunk);
    __aicore__ inline ColumnSlabPlan ComputeColumnSlabPlan(uint32_t chunk) const;
    __aicore__ inline uint32_t ComputeTileCols(const ColumnSlabPlan& plan) const;
    __aicore__ inline ColumnPassPlan MakeColumnPassPlan(uint32_t paddedStride) const;
    __aicore__ inline void LoadColumnPass(LocalTensor<float> dst, uint32_t col0, uint32_t cols, uint32_t rowOff,
                                          uint32_t chunk, uint32_t colStride, bool singleBlock);
    __aicore__ inline void StoreColumnPass(LocalTensor<float> src, uint32_t col0, uint32_t cols, uint32_t rowOff,
                                           uint32_t chunk, uint32_t colStride, bool singleBlock);
    __aicore__ inline void LoadAColumnTileTo(uint32_t col0, uint32_t cols, uint32_t rowOff, uint32_t chunk,
                                             uint32_t colStride, LocalTensor<float> dst);
    __aicore__ inline void AccumulateColumnTileOn(LocalTensor<float> buf, uint32_t col0, uint32_t cols,
                                                  uint32_t floats, uint32_t colStride);
    __aicore__ inline void StoreAColumnTileFrom(uint32_t col0, uint32_t cols, uint32_t rowOff, uint32_t chunk,
                                                uint32_t colStride, LocalTensor<float> src);
    __aicore__ inline void AccumulateColumns(LocalTensor<float> buf, uint32_t col0, uint32_t cols, uint32_t floats,
                                             uint32_t colStride);

    // ---- 同步 ----
    __aicore__ inline void WaitLoadFlag(uint32_t pass);
    __aicore__ inline void SetLoadFlag(uint32_t pass);
    __aicore__ inline void WaitSlotFree(uint32_t pass, uint32_t nslot);
    __aicore__ inline void SetStoreFlag(uint32_t slot);
    __aicore__ inline void WaitPendingStores(uint32_t ntiles, uint32_t nslot);
    __aicore__ inline LocalTensor<float> SlotBuffer(LocalTensor<float> all, uint32_t slot, uint32_t slotFloats);

    __aicore__ inline bool IsContiguousSlab(uint32_t chunk, uint32_t cols, uint32_t colStride) const;
    __aicore__ inline uint64_t ColumnOffset(uint32_t col, uint32_t rowOff) const;

    TPipe* pipe;
    // 列块缓冲：nslot 槽同宽，双缓冲流水就地更新，读完即写回。
    TBuf<TPosition::VECCALC> aTileBuf;
    // 行块内的 x（交错）、S = gather(X*sign)、Gather 用的标量暂存。
    TBuf<TPosition::VECCALC> xBuf;
    TBuf<TPosition::VECCALC> swapPatternBuf;
    TBuf<TPosition::VECCALC> calcBuf;
    // 本核所有列的 (u, v) = ALPHA*Y(JY) 以及向量化准备用的两块临时缓冲。
    TBuf<TPosition::VECCALC> coeffBuf;
    TBuf<TPosition::VECCALC> coeffSyBuf;
    TBuf<TPosition::VECCALC> coeffTmpBuf;
    // |incy| == 1 时本核负责的 y 列片段（一次搬入，避免逐列 GM 标量读）。
    TBuf<TPosition::VECCALC> yBuf;
    // 尾部两张模式表（swap 字节偏移表 + sign 符号模式）由 kernel 在 UB 内现算并复用。
    TBuf<TPosition::VECCALC> constBuf;

    GlobalTensor<float> aGM;
    GlobalTensor<float> xGM;
    GlobalTensor<float> yGM;
    int32_t blockIdx;
    int32_t blockNum;
    int64_t m;
    int64_t n;
    int64_t lda;
    int incx;
    int incy;
    uint32_t absIncx;
    uint32_t absIncy;
    uint32_t useCoreNum;
    uint32_t colsPerCore;
    uint32_t colStart;
    uint32_t colCount;
    // 本核负责的列总数；colCount 在 Init 里被夹到单批上限，Process 里按批推进。
    uint32_t totalCols;
    uint32_t rowTile;
    float alphaReal;
    float alphaImag;
    // |incy| == 1 且 incy < 0 时，UB 里的 y 片（以及由它算出的 (u, v)）按物理列号递增存放，
    // 与逻辑列号相反，取 (u, v) 时需要反向映射。
    bool yReversed;
    // 动态 UB 划分：行缓冲 float 数、模式表定容、A 列块总容量与单槽宽度。
    uint32_t rowFloats;
    uint32_t patternFloats;
    uint32_t aTileFloats;
    uint32_t aTileSlotFloats;
    uint32_t aTileNslot;
    // 首行块的 x 搬运提前到系数准备阶段发起：等待由那次 MTE2_V 一并收口，在 LoadX 里直接返回。
    uint32_t pendXRowOff = kNoPendingRow;
    uint32_t pendXChunk = 0U;
    bool xEarlyIssued = false;
    // 首列块的 A 搬运早于 swap pattern 构建发起（prologue），主调用里不再重复搬。
    bool tile0Primed = false;
};

__aicore__ inline void CgeruKernel::Init(const CgeruTilingData& tiling, TPipe* pipeIn)
{
    this->pipe = pipeIn;
    BindTiling(tiling);
    ComputeColumnRange();
    BindGlobalBuffers(tiling);

    const uint32_t chunkMax = ComputeBufferSizes();
    ComputeSlotPlan(chunkMax);
    AllocateBuffers();
    BuildConstTables();
}

__aicore__ inline void CgeruKernel::BindTiling(const CgeruTilingData& tiling)
{
    this->blockNum = GetBlockNum();
    this->blockIdx = GetBlockIdx();
    this->m = tiling.m;
    this->n = tiling.n;
    this->lda = tiling.lda;
    this->useCoreNum = tiling.useCoreNum;
    this->colsPerCore = tiling.colsPerCore;
    this->rowTile = tiling.rowTile;
    this->incx = tiling.incx;
    this->incy = tiling.incy;
    this->absIncx = (this->incx < 0) ? static_cast<uint32_t>(-this->incx) : static_cast<uint32_t>(this->incx);
    this->absIncy = (this->incy < 0) ? static_cast<uint32_t>(-this->incy) : static_cast<uint32_t>(this->incy);
    this->alphaReal = tiling.alphaReal;
    this->alphaImag = tiling.alphaImag;
    this->yReversed = (this->absIncy == 1U) && (this->incy < 0);
}

__aicore__ inline void CgeruKernel::ComputeColumnRange()
{
    this->colStart = static_cast<uint32_t>(this->blockIdx) * this->colsPerCore;
    if (this->colStart >= static_cast<uint32_t>(this->n)) {
        this->colCount = 0U;
    } else if (static_cast<uint32_t>(this->blockIdx) == this->useCoreNum - 1U) {
        this->colCount = static_cast<uint32_t>(this->n) - this->colStart;
    } else {
        this->colCount = this->colsPerCore;
    }
    // 列数不能整分到每个核时，非末核的尾列可能越过 n，必须再夹一次。
    const uint32_t remainCols = static_cast<uint32_t>(this->n) - this->colStart;
    if (this->colCount > remainCols) {
        this->colCount = remainCols;
    }

    // UB 里的 y/(u,v) 缓冲与模式表都按 colCount 线性增长：本核列数过大时把列切成批次，
    // 每批不超过模式表定容（CGERU_PATTERN_FLOATS 项，每列 2 项）。
    this->totalCols = this->colCount;
    constexpr uint32_t maxPassCols = CGERU_PATTERN_FLOATS / CGERU_FLOATS_PER_COMPLEX;
    if (this->colCount > maxPassCols) {
        this->colCount = maxPassCols;
    }
}

__aicore__ inline void CgeruKernel::BindGlobalBuffers(const CgeruTilingData& tiling)
{
    aGM.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(tiling.A),
        static_cast<uint64_t>(this->n) * static_cast<uint64_t>(this->lda) * CGERU_FLOATS_PER_COMPLEX);
    xGM.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(tiling.x),
        (static_cast<uint64_t>(this->m) * this->absIncx + 1U) * CGERU_FLOATS_PER_COMPLEX);
    yGM.SetGlobalBuffer(
        reinterpret_cast<__gm__ float*>(tiling.y),
        (static_cast<uint64_t>(this->n) * this->absIncy + 1U) * CGERU_FLOATS_PER_COMPLEX);
}

// 本核行块的最大复数数，受 tiling 的 rowTile 与 m 夹取。
__aicore__ inline uint32_t CgeruKernel::ComputeChunkMax() const
{
    uint32_t chunkMax = this->rowTile;
    if (chunkMax > static_cast<uint32_t>(this->m)) {
        chunkMax = static_cast<uint32_t>(this->m);
    }
    if (chunkMax > CGERU_ROW_TILE_MAX) {
        chunkMax = CGERU_ROW_TILE_MAX;
    }
    return chunkMax;
}

// 每核列向量化 (u, v) 所需的 float 数（2*colCount，向上补齐到 8 的倍数，最小 8）。
__aicore__ inline uint32_t CgeruKernel::AlignCoeffFloats(uint32_t cols) const
{
    uint32_t coeffFloats = cols * CGERU_FLOATS_PER_COMPLEX;
    coeffFloats = ((coeffFloats + 7U) / 8U) * 8U;
    return (coeffFloats < 8U) ? 8U : coeffFloats;
}

// |incy| == 1 时 y 片在 UB 中的字节数（每列 2 个 float，按 32 字节向上取整）。
__aicore__ inline uint32_t CgeruKernel::ComputeYBufBytes(uint32_t cols) const
{
    if (this->absIncy != 1U) {
        return 0U;
    }
    uint32_t yBytes = cols * CGERU_BYTES_PER_COMPLEX;
    yBytes = ((yBytes + 31U) / 32U) * 32U;
    return (yBytes < 32U) ? 32U : yBytes;
}

// 行缓冲与模式表前缀的定容，返回行缓冲字节数。
__aicore__ inline uint32_t CgeruKernel::ComputeTableLayout(uint32_t chunkMax)
{
    uint32_t rowF = ((chunkMax + CGERU_ALIGN_COMPLEX - 1U) / CGERU_ALIGN_COMPLEX) * CGERU_ALIGN_COMPLEX *
                    CGERU_FLOATS_PER_COMPLEX;
    if (rowF < 64U) {
        rowF = 64U;
    }
    this->rowFloats = rowF;

    // 模式表定容：向量化 (u,v) 准备要用表覆盖 ((2*colCount+7)/8)*8 项，可能超过 rowF。
    uint32_t tableFloats = rowF;
    const uint32_t needTableFloats = AlignCoeffFloats(this->colCount);
    if (needTableFloats > tableFloats) {
        tableFloats = needTableFloats;
    }
    // 兜底：模式表定容 CGERU_PATTERN_FLOATS 项（colCount 已在 ComputeColumnRange 夹过）。
    if (tableFloats > CGERU_PATTERN_FLOATS) {
        tableFloats = CGERU_PATTERN_FLOATS;
    }
    this->patternFloats = tableFloats;
    return rowF * CGERU_BYTES_PER_FLOAT;
}

// 扣除固定占用后，给 A 列块分配总容量与单槽宽度。
__aicore__ inline void CgeruKernel::PlanColumnBuffers(uint32_t fixedBytes, uint32_t chunkMax)
{
    constexpr uint32_t ubBytes = 192U * 1024U;
    uint32_t aFloats = (ubBytes > fixedBytes) ? ((ubBytes - fixedBytes) / CGERU_BYTES_PER_FLOAT) : 0U;
    aFloats = (aFloats / 8U) * 8U;
    if (aFloats < 8192U) {
        aFloats = 8192U;  // 至少 32KB
    }
    uint32_t slotFloats = aFloats / 2U;
    if (slotFloats > 8192U) {
        slotFloats = 8192U;
    }
    slotFloats = (slotFloats / 8U) * 8U;

    const uint32_t colStrideEst = ((CGERU_FLOATS_PER_COMPLEX * chunkMax + 7U) / 8U) * 8U;
    const bool contiguousEst =
        (chunkMax == static_cast<uint32_t>(this->m)) && (this->lda == static_cast<int64_t>(this->m));
    const bool alignedEst =
        (!contiguousEst) || (((chunkMax % 4U) == 0U) && ((static_cast<uint32_t>(this->lda) % 4U) == 0U));
    const uint32_t blockNumEst = static_cast<uint32_t>(this->blockNum);
    const uint32_t colsPerCoreEst =
        (blockNumEst == 0U) ? 0U : ((static_cast<uint32_t>(this->n) + blockNumEst - 1U) / blockNumEst);
    const uint64_t workFloats = static_cast<uint64_t>(colsPerCoreEst) * static_cast<uint64_t>(colStrideEst);
    if ((!alignedEst) && (workFloats >= 4096U)) {
        // 逐列路径目标 ~10 遍：单槽只留 1/10 列宽的整 8 对齐量。
        uint32_t colsPerTile = (colsPerCoreEst + 9U) / 10U;
        if (colsPerTile == 0U) {
            colsPerTile = 1U;
        }
        uint32_t wantFloats = colsPerTile * colStrideEst;
        wantFloats = (wantFloats / 8U) * 8U;
        if (wantFloats < 512U) {
            wantFloats = 512U;
        }
        if (wantFloats < slotFloats) {
            slotFloats = wantFloats;
        }
    }
    this->aTileFloats = aFloats;
    this->aTileSlotFloats = slotFloats;
}

// 返回本核行块的最大复数数（chunkMax）。
__aicore__ inline uint32_t CgeruKernel::ComputeBufferSizes()
{
    const uint32_t chunkMax = ComputeChunkMax();
    const uint32_t rowBytes = ComputeTableLayout(chunkMax);
    const uint32_t constBytes = 2U * this->patternFloats * CGERU_BYTES_PER_FLOAT;
    const uint32_t yBytes = ComputeYBufBytes(this->colCount);
    const uint32_t coeffBytes = AlignCoeffFloats(this->colCount) * CGERU_BYTES_PER_FLOAT;
    const uint32_t fixedBytes = 3U * rowBytes + constBytes + yBytes + 3U * coeffBytes + 8192U;
    PlanColumnBuffers(fixedBytes, chunkMax);
    return chunkMax;
}

__aicore__ inline void CgeruKernel::ComputeSlotPlan(uint32_t chunkMax)
{
    uint32_t slotFloats = this->aTileSlotFloats;
    const uint32_t aFloats = this->aTileFloats;
    const uint32_t colStrideEst = ((CGERU_FLOATS_PER_COMPLEX * chunkMax + 7U) / 8U) * 8U;
    uint32_t tileColsEst = (colStrideEst == 0U) ? 1U : (slotFloats / colStrideEst);
    if (tileColsEst == 0U) {
        tileColsEst = 1U;
    }
    const uint32_t colsPerCoreEst =
        (static_cast<uint32_t>(this->n) + this->blockNum - 1U) / this->blockNum;
    const uint32_t ntilesEst = (colsPerCoreEst + tileColsEst - 1U) / tileColsEst;
    const uint32_t nslot = ((aFloats >= 3U * slotFloats) && (ntilesEst >= 3U)) ? 3U : 2U;
    this->aTileNslot = nslot;
    this->aTileSlotFloats = slotFloats;
    this->aTileFloats = nslot * slotFloats;
    (void)tileColsEst;
}

__aicore__ inline void CgeruKernel::AllocateBuffers()
{
    const uint32_t rowBytes = this->rowFloats * CGERU_BYTES_PER_FLOAT;
    const uint32_t constBytes = 2U * this->patternFloats * CGERU_BYTES_PER_FLOAT;
    const uint32_t coeffBytes = AlignCoeffFloats(this->colCount) * CGERU_BYTES_PER_FLOAT;

    this->pipe->InitBuffer(aTileBuf, this->aTileFloats * CGERU_BYTES_PER_FLOAT);
    this->pipe->InitBuffer(xBuf, rowBytes);
    this->pipe->InitBuffer(swapPatternBuf, rowBytes);
    this->pipe->InitBuffer(calcBuf, rowBytes);
    this->pipe->InitBuffer(constBuf, constBytes);
    this->pipe->InitBuffer(coeffBuf, coeffBytes);
    this->pipe->InitBuffer(coeffSyBuf, coeffBytes);
    this->pipe->InitBuffer(coeffTmpBuf, coeffBytes);

    // 列片段连续（|incy| == 1）时把本核的 y 一次搬进 UB：每列 2 个 float，按 32 字节向上取整。
    const uint32_t yBytes = ComputeYBufBytes(this->colCount);
    if (yBytes > 0U) {
        this->pipe->InitBuffer(yBuf, yBytes);
    }
}

// 在 UB 内现算两张模式表，逐位等价于原先 host 侧生成、经 GM 搬入的常量表：
//   swap[k] = 4 * (k ^ 1)（Gather 的字节偏移，把相邻两个 float 交换位置），
//   sign[k] = (k 为偶数 ? -1.0f : +1.0f)（交错正负号模式）。
// 两张表只与下标有关、与 m/n 无关，Init 里算一次供整个 kernel 复用。
__aicore__ inline void CgeruKernel::BuildConstTables()
{
    LocalTensor<uint32_t> swapTable = constBuf.Get<uint32_t>();
    // sign 区先当 (k >> 1) << 1 的暂存，最后由 Cast 整段覆盖成 ±1.0f。
    LocalTensor<uint32_t> scratch = constBuf.Get<uint32_t>()[this->patternFloats];
    LocalTensor<float> signTable = constBuf.Get<float>()[this->patternFloats];
    // 奇偶位临时量借 A 列块缓冲：Init 阶段尚未装载 A 数据，容量不小于模式表定容。
    LocalTensor<uint32_t> parity = aTileBuf.Get<uint32_t>();
    const uint32_t count = this->patternFloats;

    CreateVecIndex(swapTable.ReinterpretCast<int32_t>(), 0, count);
    PipeBarrier<PIPE_V>();
    ShiftRight(scratch, swapTable, 1U, count);
    PipeBarrier<PIPE_V>();
    ShiftLeft(scratch, scratch, 1U, count);
    PipeBarrier<PIPE_V>();
    Sub(parity.ReinterpretCast<int32_t>(), swapTable.ReinterpretCast<int32_t>(),
        scratch.ReinterpretCast<int32_t>(), count);
    PipeBarrier<PIPE_V>();
    // 1 - 2*(k & 1)：偶数 +1、奇数 -1，正好是 k ^ 1 相对 k 的增量。
    Muls(parity.ReinterpretCast<int32_t>(), parity.ReinterpretCast<int32_t>(), -2, count);
    PipeBarrier<PIPE_V>();
    Adds(parity.ReinterpretCast<int32_t>(), parity.ReinterpretCast<int32_t>(), 1, count);
    PipeBarrier<PIPE_V>();
    Add(swapTable.ReinterpretCast<int32_t>(), swapTable.ReinterpretCast<int32_t>(),
        parity.ReinterpretCast<int32_t>(), count);
    PipeBarrier<PIPE_V>();
    ShiftLeft(swapTable, swapTable, 2U, count);
    PipeBarrier<PIPE_V>();
    // sign = -(增量)：偶数 -1.0f、奇数 +1.0f。
    Cast(signTable, parity.ReinterpretCast<int32_t>(), RoundMode::CAST_NONE, count);
    PipeBarrier<PIPE_V>();
    Muls(signTable, signTable, -1.0f, count);
}

__aicore__ inline void CgeruKernel::PrepareColumnCoeffs()
{
    IssueEarlyX();
    const uint32_t coeffFloats =
        ((static_cast<uint32_t>(this->colCount) * CGERU_FLOATS_PER_COMPLEX + 7U) / 8U) * 8U;
    const bool yInUb = LoadYCoefficients();
    if (yInUb) {
        // 首行块的 x 与 y 片都排在 MTE2 流水上，进入向量计算前等一次 MTE2_V 收口。
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        PrepareCoeffsVectorized(coeffFloats);
    } else {
        PrepareCoeffsScalar(coeffFloats);
    }

    event_t eventIdVToS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    SetFlag<HardEvent::V_S>(eventIdVToS);
    WaitFlag<HardEvent::V_S>(eventIdVToS);
}

__aicore__ inline void CgeruKernel::IssueEarlyX()
{
    if ((this->pendXRowOff == kNoPendingRow) || (this->incx != 1) || (this->absIncy != 1U)) {
        return;
    }
    // 首行块的 x 在这里就发起（排在 y 片之前）：GM 时延与 y 片的搬运重叠，
    // 由下面 y 片那次 MTE2_V 一并收口，不额外占用事件号。
    LocalTensor<float> xLocal = xBuf.Get<float>();
    DataCopyExtParams xParams{1, static_cast<uint32_t>(this->pendXChunk * CGERU_BYTES_PER_COMPLEX), 0, 0, 0};
    DataCopyPadExtParams<float> noPadX{false, 0, 0, 0.0f};
    DataCopyPad(
        xLocal, xGM[static_cast<uint64_t>(this->pendXRowOff) * CGERU_FLOATS_PER_COMPLEX], xParams, noPadX);
    this->xEarlyIssued = true;
}

// 返回 true 表示本核的 y 已整段搬入 UB（|incy| == 1）。
__aicore__ inline bool CgeruKernel::LoadYCoefficients()
{
    if (this->absIncy != 1U || this->colCount == 0U) {
        return false;
    }
    // |incy| == 1 时本核的列在 GM 中连续：整段一次搬入 UB，避免逐列 2 次 GM 标量读。
    // 负步长时物理列号逆序，按 readCol 反向取即可。
    LocalTensor<float> yLocal = yBuf.Get<float>();
    const int64_t firstPhys = (this->incy > 0) ?
                                  static_cast<int64_t>(this->colStart) :
                                  (static_cast<int64_t>(this->n) - 1 - static_cast<int64_t>(this->colStart) -
                                   static_cast<int64_t>(this->colCount) + 1);
    DataCopyExtParams yParams{1, static_cast<uint32_t>(this->colCount * CGERU_BYTES_PER_COMPLEX), 0, 0, 0};
    DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
    DataCopyPad(yLocal, yGM[static_cast<uint64_t>(firstPhys) * CGERU_FLOATS_PER_COMPLEX], yParams, noPad);
    return true;
}

__aicore__ inline void CgeruKernel::PrepareCoeffsVectorized(uint32_t coeffFloats)
{
    // 一次向量化算出全部 (u, v)：S_y = gather(y*sign, swap) = (yi, -yr)，
    // coeff = ar*y - ai*S_y = (ar*yr - ai*yi, ar*yi + ai*yr)，与 Netlib 的 TEMP = ALPHA*Y(JY) 同算式。
    LocalTensor<float> yLocal = yBuf.Get<float>();
    LocalTensor<float> signLocal = constBuf.Get<float>()[this->patternFloats];
    LocalTensor<uint32_t> swapLocal = constBuf.Get<uint32_t>();
    LocalTensor<float> syLocal = coeffSyBuf.Get<float>();
    LocalTensor<float> tmpLocal = coeffTmpBuf.Get<float>();
    LocalTensor<float> coeff = coeffBuf.Get<float>();
    Mul(tmpLocal, yLocal, signLocal, coeffFloats);
    Gather(syLocal, tmpLocal, swapLocal, static_cast<uint32_t>(0), coeffFloats);
    Muls(coeff, yLocal, this->alphaReal, coeffFloats);
    Axpy(coeff, syLocal, -this->alphaImag, coeffFloats);
}

__aicore__ inline void CgeruKernel::PrepareCoeffsScalar(uint32_t coeffFloats)
{
    (void)coeffFloats;
    // 非连续列片段（|incy| != 1）逐列从 GM 标量读；本条路径不在验收性能用例上。
    LocalTensor<float> coeff = coeffBuf.Get<float>();
    for (uint32_t col = 0; col < this->colCount; ++col) {
        const int64_t logical = static_cast<int64_t>(this->colStart + col);
        const int64_t phys = (this->incy > 0) ?
                                 (logical * this->incy) :
                                 ((static_cast<int64_t>(this->n) - 1 - logical) * (-static_cast<int64_t>(this->incy)));
        const uint64_t yIdx = static_cast<uint64_t>(phys) * CGERU_FLOATS_PER_COMPLEX;
        const float yReal = yGM.GetValue(yIdx);
        const float yImag = yGM.GetValue(yIdx + 1U);
        coeff.SetValue(CGERU_FLOATS_PER_COMPLEX * col, this->alphaReal * yReal - this->alphaImag * yImag);
        coeff.SetValue(CGERU_FLOATS_PER_COMPLEX * col + 1U, this->alphaReal * yImag + this->alphaImag * yReal);
    }
}

__aicore__ inline void CgeruKernel::GetUV(uint32_t col, float& u, float& v)
{
    LocalTensor<float> coeff = coeffBuf.Get<float>();
    const uint32_t ti = this->yReversed ? (this->colCount - 1U - col) : col;
    u = coeff.GetValue(CGERU_FLOATS_PER_COMPLEX * ti);
    v = coeff.GetValue(CGERU_FLOATS_PER_COMPLEX * ti + 1U);
}

__aicore__ inline void CgeruKernel::LoadX(uint32_t rowOff, uint32_t chunk)
{
    LocalTensor<float> xLocal = xBuf.Get<float>();
    if (this->xEarlyIssued && (this->pendXRowOff == rowOff) && (this->pendXChunk == chunk) &&
        (this->incx == 1)) {
        // 已在 PrepareColumnCoeffs 阶段发起，且由首列块的 MTE2_V 标志一并收口。
        this->xEarlyIssued = false;
        this->pendXRowOff = kNoPendingRow;
        return;
    }
    if (this->incx == 1) {
        DataCopyExtParams xParams{1, static_cast<uint32_t>(chunk * CGERU_BYTES_PER_COMPLEX), 0, 0, 0};
        DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
        DataCopyPad(
            xLocal, xGM[static_cast<uint64_t>(rowOff) * CGERU_FLOATS_PER_COMPLEX], xParams, noPad);
        event_t eventIdMte2ToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        WaitFlag<HardEvent::MTE2_V>(eventIdMte2ToV);
        return;
    }

    // 步长不为 1 时按行逐个搬：负步长反序访问；复数逐元素搬写 xLocal 也更直接。
    for (uint32_t i = 0; i < chunk; ++i) {
        const int64_t logical = static_cast<int64_t>(rowOff + i);
        const int64_t phys = (this->incx > 0) ?
                                 (logical * this->incx) :
                                 ((static_cast<int64_t>(this->m) - 1 - logical) * (-static_cast<int64_t>(this->incx)));
        const uint64_t xIdx = static_cast<uint64_t>(phys) * CGERU_FLOATS_PER_COMPLEX;
        xLocal.SetValue(i * CGERU_FLOATS_PER_COMPLEX, xGM.GetValue(xIdx));
        xLocal.SetValue(i * CGERU_FLOATS_PER_COMPLEX + 1U, xGM.GetValue(xIdx + 1U));
    }
    event_t eventIdSToV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    SetFlag<HardEvent::S_V>(eventIdSToV);
    WaitFlag<HardEvent::S_V>(eventIdSToV);
}

__aicore__ inline void CgeruKernel::BuildSwapPattern(uint32_t calcNum)
{
    LocalTensor<float> xLocal = xBuf.Get<float>();
    LocalTensor<float> signLocal = constBuf.Get<float>()[this->patternFloats];
    LocalTensor<uint32_t> swapLocal = constBuf.Get<uint32_t>();
    LocalTensor<float> pattern = swapPatternBuf.Get<float>();
    LocalTensor<float> scratch = calcBuf.Get<float>();
    // calcNum 是复数个数，向量指令按 float 计数：交错布局下为 2*calcNum 个 float。
    const uint32_t calcFloats = CGERU_FLOATS_PER_COMPLEX * calcNum;
    // scratch = X * sign = (-xr, +xi)；pattern = gather(scratch, swap) = (xi, -xr)。
    Mul(scratch, xLocal, signLocal, calcFloats);
    Gather(pattern, scratch, swapLocal, static_cast<uint32_t>(0), calcFloats);
}

__aicore__ inline uint64_t CgeruKernel::ColumnOffset(uint32_t col, uint32_t rowOff) const
{
    return (static_cast<uint64_t>(this->colStart + col) * static_cast<uint64_t>(this->lda) +
            static_cast<uint64_t>(rowOff)) *
           CGERU_FLOATS_PER_COMPLEX;
}

__aicore__ inline LocalTensor<float> CgeruKernel::SlotBuffer(LocalTensor<float> all, uint32_t slot,
                                                             uint32_t slotFloats)
{
    return all[static_cast<int32_t>(slot * slotFloats)];
}

__aicore__ inline void CgeruKernel::WaitLoadFlag(uint32_t pass)
{
    if ((pass & 1U) == 0U) {
        WaitFlag<HardEvent::MTE2_V>(kEvLoadEven);
    } else {
        WaitFlag<HardEvent::MTE2_V>(kEvLoadOdd);
    }
}

__aicore__ inline void CgeruKernel::SetLoadFlag(uint32_t pass)
{
    if ((pass & 1U) == 0U) {
        SetFlag<HardEvent::MTE2_V>(kEvLoadEven);
    } else {
        SetFlag<HardEvent::MTE2_V>(kEvLoadOdd);
    }
}

__aicore__ inline void CgeruKernel::SetStoreFlag(uint32_t slot)
{
    if (slot == 0U) {
        SetFlag<HardEvent::MTE3_MTE2>(kEvStore0);
    } else if (slot == 1U) {
        SetFlag<HardEvent::MTE3_MTE2>(kEvStore1);
    } else {
        SetFlag<HardEvent::MTE3_MTE2>(kEvStore2);
    }
}

__aicore__ inline void CgeruKernel::WaitSlotFree(uint32_t pass, uint32_t nslot)
{
    // 目标槽（第 pass+1 遍要写的槽）上一次使用是第 pass+1-nslot 遍，等它的搬出结束。
    const uint32_t safeNslot = (nslot == 0U) ? 1U : nslot;
    if (pass + 1U < safeNslot) {
        return;
    }
    const uint32_t slotIdx = (pass + 1U) % safeNslot;
    if (slotIdx == 0U) {
        WaitFlag<HardEvent::MTE3_MTE2>(kEvStore0);
    } else if (slotIdx == 1U) {
        WaitFlag<HardEvent::MTE3_MTE2>(kEvStore1);
    } else {
        WaitFlag<HardEvent::MTE3_MTE2>(kEvStore2);
    }
}

__aicore__ inline void CgeruKernel::WaitPendingStores(uint32_t ntiles, uint32_t nslot)
{
    // 循环里消费了前 ntiles-(nslot-1) 个搬出标志，剩最后 nslot-1 个（不足时就是全部）在这里消费。
    const uint32_t safeNslot = (nslot == 0U) ? 1U : nslot;
    const uint32_t pendStores = (ntiles >= safeNslot) ? (safeNslot - 1U) : ntiles;
    for (uint32_t k = 0U; k < pendStores; ++k) {
        const uint32_t slotIdx = (ntiles - 1U - k) % safeNslot;
        if (slotIdx == 0U) {
            WaitFlag<HardEvent::MTE3_MTE2>(kEvStore0);
        } else if (slotIdx == 1U) {
            WaitFlag<HardEvent::MTE3_MTE2>(kEvStore1);
        } else {
            WaitFlag<HardEvent::MTE3_MTE2>(kEvStore2);
        }
    }
}

__aicore__ inline bool CgeruKernel::IsContiguousSlab(uint32_t chunk, uint32_t cols, uint32_t colStride) const
{
    return (chunk == static_cast<uint32_t>(this->m)) && (this->lda == static_cast<int64_t>(this->m)) &&
           (colStride == chunk * CGERU_FLOATS_PER_COMPLEX) &&
           (((chunk * CGERU_FLOATS_PER_COMPLEX * cols) % 8U) == 0U);
}

__aicore__ inline void CgeruKernel::LoadAColumnTileTo(uint32_t col0, uint32_t cols, uint32_t rowOff, uint32_t chunk,
                                                      uint32_t colStride, LocalTensor<float> dst)
{
    const uint64_t aOff = ColumnOffset(col0, rowOff);
    if (IsContiguousSlab(chunk, cols, colStride)) {
        // GM 与 UB 两侧都是连续流：一条 1D 指令搬完，长度不必是 32B 整倍。
        const uint32_t oneDimFloats = cols * chunk * CGERU_FLOATS_PER_COMPLEX;
        if ((oneDimFloats % 8U) == 0U) {
            DataCopy(dst, aGM[aOff], oneDimFloats);
        } else {
            DataCopyExtParams oneDimParams{1U, static_cast<uint32_t>(oneDimFloats * CGERU_BYTES_PER_FLOAT), 0U, 0U,
                                           0U};
            DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
            DataCopyPad(dst, aGM[aOff], oneDimParams, noPad);
        }
        return;
    }
    // UB 侧每 burst 由硬件按 ceil32(blockLen) 补齐，槽宽已上调到 32B 整倍 → dstStride 恒 0。
    DataCopyExtParams aParams{
        static_cast<uint16_t>(cols), static_cast<uint32_t>(chunk * CGERU_BYTES_PER_COMPLEX),
        static_cast<uint32_t>((this->lda - static_cast<int64_t>(chunk)) * CGERU_BYTES_PER_COMPLEX), 0, 0};
    DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
    DataCopyPad(dst, aGM[aOff], aParams, noPad);
}

__aicore__ inline void CgeruKernel::StoreAColumnTileFrom(uint32_t col0, uint32_t cols, uint32_t rowOff,
                                                         uint32_t chunk, uint32_t colStride,
                                                         LocalTensor<float> src)
{
    const uint64_t aOff = ColumnOffset(col0, rowOff);
    if (IsContiguousSlab(chunk, cols, colStride)) {
        const uint32_t oneDimFloats = cols * chunk * CGERU_FLOATS_PER_COMPLEX;
        if ((oneDimFloats % 8U) == 0U) {
            DataCopy(aGM[aOff], src, oneDimFloats);
        } else {
            DataCopyExtParams oneDimParams{1U, static_cast<uint32_t>(oneDimFloats * CGERU_BYTES_PER_FLOAT), 0U, 0U,
                                           0U};
            DataCopyPad(aGM[aOff], src, oneDimParams);
        }
        return;
    }
    DataCopyExtParams outParams{
        static_cast<uint16_t>(cols), static_cast<uint32_t>(chunk * CGERU_BYTES_PER_COMPLEX), 0,
        static_cast<uint32_t>((this->lda - static_cast<int64_t>(chunk)) * CGERU_BYTES_PER_COMPLEX), 0};
    DataCopyPad(aGM[aOff], src, outParams);
}

// 每列两次就地 Axpy：A(col) += u*X + (-v)*S。
// 零 y 列（alpha 有限时 u == v == 0）整列跳过，与 Netlib cgeru 的 `IF (Y(JY).EQ.ZERO) GO TO`
// 以及 arch35 的 yFlagUb 判断保持一致：零列不参与累加，x 含 Inf/NaN 时 A 也保持原值。
// alpha 非有限（Inf/NaN）时 u/v 非有限，不会误命中；alpha 有限而 y 极小时乘积下溢为 0
// 同样按零列处理（A 不变，等价于 Netlib 的 TEMP == 0 分支）。
__aicore__ inline void CgeruKernel::AccumulateColumns(LocalTensor<float> buf, uint32_t col0, uint32_t cols,
                                                      uint32_t floats, uint32_t colStride)
{
    LocalTensor<float> pattern = swapPatternBuf.Get<float>();
    LocalTensor<float> xLocal = xBuf.Get<float>();
    for (uint32_t c = 0; c < cols; ++c) {
        float u = 0.0f;
        float v = 0.0f;
        GetUV(col0 + c, u, v);
        if (u == 0.0f && v == 0.0f) {
            continue;
        }
        const int32_t off = static_cast<int32_t>(c * colStride);
        Axpy(buf[off], xLocal, u, floats);
        Axpy(buf[off], pattern, -v, floats);
    }
}

__aicore__ inline void CgeruKernel::AccumulateColumnTileOn(LocalTensor<float> buf, uint32_t col0, uint32_t cols,
                                                           uint32_t floats, uint32_t colStride)
{
    AccumulateColumns(buf, col0, cols, floats, colStride);
}
__aicore__ inline void CgeruKernel::LoadColumnPass(LocalTensor<float> dst, uint32_t col0, uint32_t cols,
                                                   uint32_t rowOff, uint32_t chunk, uint32_t colStride,
                                                   bool singleBlock)
{
    if (singleBlock) {
        const uint64_t aOff = ColumnOffset(col0, rowOff);
        DataCopyExtParams aParams{1, cols * chunk * CGERU_BYTES_PER_COMPLEX, 0, 0, 0};
        DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
        DataCopyPad(dst, aGM[aOff], aParams, noPad);
        return;
    }
    DataCopyExtParams oneColParams{1, static_cast<uint32_t>(chunk * CGERU_BYTES_PER_COMPLEX), 0, 0, 0};
    DataCopyPadExtParams<float> noPad{false, 0, 0, 0.0f};
    for (uint32_t c = 0U; c < cols; ++c) {
        const uint64_t aOff = ColumnOffset(col0 + c, rowOff);
        DataCopyPad(dst[static_cast<int32_t>(c * colStride)], aGM[aOff], oneColParams, noPad);
    }
}

__aicore__ inline void CgeruKernel::StoreColumnPass(LocalTensor<float> src, uint32_t col0, uint32_t cols,
                                                    uint32_t rowOff, uint32_t chunk, uint32_t colStride,
                                                    bool singleBlock)
{
    if (singleBlock) {
        const uint64_t aOff = ColumnOffset(col0, rowOff);
        DataCopyExtParams outParams{1, cols * chunk * CGERU_BYTES_PER_COMPLEX, 0, 0, 0};
        DataCopyPad(aGM[aOff], src, outParams);
        return;
    }
    DataCopyExtParams oneColParams{1, static_cast<uint32_t>(chunk * CGERU_BYTES_PER_COMPLEX), 0, 0, 0};
    for (uint32_t c = 0U; c < cols; ++c) {
        const uint64_t aOff = ColumnOffset(col0 + c, rowOff);
        DataCopyPad(aGM[aOff], src[static_cast<int32_t>(c * colStride)], oneColParams);
    }
}

__aicore__ inline ColumnPassPlan CgeruKernel::MakeColumnPassPlan(uint32_t paddedStride) const
{
    ColumnPassPlan plan;
    plan.colsPerPass = (paddedStride == 0U) ? 1U : (this->aTileSlotFloats / paddedStride);
    if (plan.colsPerPass == 0U) {
        plan.colsPerPass = 1U;
    }
    if (plan.colsPerPass > this->colCount) {
        plan.colsPerPass = this->colCount;
    }
    plan.ntiles = (this->colCount + plan.colsPerPass - 1U) / plan.colsPerPass;
    return plan;
}

/**
 * @brief 逐列 1D 搬运、整批消费、多槽流水：chunk/lda 不满足 32B 对齐时使用。
 *
 * 搬入第 t+1 遍的同时，计算第 t 遍、搬出第 t-1 遍。每列在 UB 内的槽宽按 32B 向上取整，
 * 保证向量指令与 DMA 目的地址对齐。本核列 slab 在 GM 中整段连续且列宽是 8 的倍数时，
 * 一遍列的搬入/搬出各只发一次单块 DataCopyPad（块数 K -> 1）。
 */
__aicore__ inline void CgeruKernel::ProcessColumnsByColumn(uint32_t rowOff, uint32_t chunk)
{
    const uint32_t floatsPerCol = CGERU_FLOATS_PER_COMPLEX * chunk;
    const uint32_t paddedStride = ((floatsPerCol + 7U) / 8U) * 8U;  // 每列槽宽（float），32B 对齐
    const bool contiguousColumns =
        (this->lda == static_cast<int64_t>(this->m)) && (chunk == static_cast<uint32_t>(this->m));
    const bool singleBlock = contiguousColumns && ((floatsPerCol % 8U) == 0U);

    const uint32_t nslot = this->aTileNslot;
    const uint32_t halfFloats = this->aTileSlotFloats;
    const ColumnPassPlan plan = MakeColumnPassPlan(paddedStride);
    const uint32_t colsPerPass = plan.colsPerPass;
    const uint32_t ntiles = plan.ntiles;
    LocalTensor<float> all = aTileBuf.Get<float>();

    // 第 0 遍搬入。
    uint32_t cols0 = this->colCount;
    if (cols0 > colsPerPass) {
        cols0 = colsPerPass;
    }
    LoadColumnPass(all, 0U, cols0, rowOff, chunk, paddedStride, singleBlock);
    SetLoadFlag(0U);

    for (uint32_t t = 0U; t < ntiles; ++t) {
        const uint32_t cur = t % nslot;
        const uint32_t col0 = t * colsPerPass;
        uint32_t cols = this->colCount - col0;
        if (cols > colsPerPass) {
            cols = colsPerPass;
        }
        LocalTensor<float> curBuf = SlotBuffer(all, cur, halfFloats);
        WaitLoadFlag(t);
        WaitSlotFree(t, nslot);

        // 预取下一遍，与下面的向量计算重叠。
        if (t + 1U < ntiles) {
            const uint32_t ncol0 = col0 + cols;
            uint32_t ncols = this->colCount - ncol0;
            if (ncols > colsPerPass) {
                ncols = colsPerPass;
            }
            LocalTensor<float> dst = SlotBuffer(all, (t + 1U) % nslot, halfFloats);
            LoadColumnPass(dst, ncol0, ncols, rowOff, chunk, paddedStride, singleBlock);
            SetLoadFlag(t + 1U);
        }

        AccumulateColumns(curBuf, col0, cols, floatsPerCol, paddedStride);
        SetFlag<HardEvent::V_MTE3>(kEvVecStore);
        WaitFlag<HardEvent::V_MTE3>(kEvVecStore);
        StoreColumnPass(curBuf, col0, cols, rowOff, chunk, paddedStride, singleBlock);
        SetStoreFlag(cur);
    }
    WaitPendingStores(ntiles, nslot);
}

__aicore__ inline ColumnSlabPlan CgeruKernel::ComputeColumnSlabPlan(uint32_t chunk) const
{
    ColumnSlabPlan plan;
    plan.contiguousSlab =
        (chunk == static_cast<uint32_t>(this->m)) && (this->lda == static_cast<int64_t>(this->m));
    const uint32_t rawStride = CGERU_FLOATS_PER_COMPLEX * chunk;
    plan.colStride = plan.contiguousSlab ? rawStride : (((rawStride + 7U) / 8U) * 8U);
    plan.alignedStride =
        (!plan.contiguousSlab) || (((chunk % 4U) == 0U) && ((static_cast<uint32_t>(this->lda) % 4U) == 0U));
    plan.tileEnabled = (this->colCount > 1U) && plan.alignedStride && (plan.colStride <= CGERU_COL_STRIDE_FLOATS);
    return plan;
}

__aicore__ inline uint32_t CgeruKernel::ComputeTileCols(const ColumnSlabPlan& plan) const
{
    const uint32_t slotFloats = this->aTileSlotFloats;
    uint32_t tileCols = slotFloats / plan.colStride;
    if (tileCols == 0U) {
        tileCols = 1U;
    }
    if (tileCols > this->colCount) {
        tileCols = this->colCount;
    }
    if (plan.contiguousSlab && ((plan.colStride * tileCols) % 8U) != 0U) {
        // 1D 搬入的尾块由硬件补到 32B：给本遍尾部留 8 float，避免补位写过界。
        const uint32_t maxCols = (slotFloats > 8U) ? ((slotFloats - 8U) / plan.colStride) : 1U;
        if (maxCols > 0U && tileCols > maxCols) {
            tileCols = maxCols;
        }
    }
    return tileCols;
}

__aicore__ inline void CgeruKernel::PrimeTileZero(uint32_t cols, uint32_t rowOff, uint32_t chunk,
                                                  uint32_t colStride, LocalTensor<float>& all)
{
    LoadAColumnTileTo(0U, cols, rowOff, chunk, colStride, all);
    SetFlag<HardEvent::MTE2_V>(kEvLoadEven);
}

/**
 * @brief 处理一个行块上的本核全部列。
 *
 * chunk 按 32 个复数对齐（每列在 UB 内正好占 2*chunk 个 float）时走多列块流水；
 * chunk 未对齐（或只有 1 列）时退回逐列路径。
 */
__aicore__ inline void CgeruKernel::ProcessColumns(uint32_t rowOff, uint32_t chunk, bool prologueOnly)
{
    // 整列都在 UB 里（chunk == m 且 lda == m）时本核的 A slab 在 GM/UB 两侧都是连续流：
    // 保持紧排槽宽、走一条 1D 指令。其余形状把槽宽按 32B 上调，走多 burst 2D 描述符。
    const ColumnSlabPlan plan = ComputeColumnSlabPlan(chunk);
    if (!plan.tileEnabled) {
        if (!prologueOnly) {
            ProcessColumnsByColumn(rowOff, chunk);
        }
        return;
    }

    const uint32_t tileCols = ComputeTileCols(plan);
    const uint32_t cols0 = (this->colCount < tileCols) ? this->colCount : tileCols;
    LocalTensor<float> all = aTileBuf.Get<float>();
    if (prologueOnly) {
        if (!this->tile0Primed) {
            // 只发首列块的 A 搬运就返回：让它的时延与本行块的 swap pattern 构建重叠。
            PrimeTileZero(cols0, rowOff, chunk, plan.colStride, all);
            this->tile0Primed = true;
        }
        return;
    }
    if (!this->tile0Primed) {
        PrimeTileZero(cols0, rowOff, chunk, plan.colStride, all);
    }
    this->tile0Primed = false;  // 首列块已在 prologue 阶段发出（含 SetFlag<MTE2_V>(kEvLoadEven)）
    ProcessTilePipeline(rowOff, chunk, plan);
}

// 多列块流水：`plan` 决定槽宽，首列块由 ProcessColumns 的 prologue 阶段先行发出。
__aicore__ inline void CgeruKernel::ProcessTilePipeline(uint32_t rowOff, uint32_t chunk,
                                                        const ColumnSlabPlan& plan)
{
    const uint32_t nslot = this->aTileNslot;
    const uint32_t slotFloats = this->aTileSlotFloats;
    const uint32_t tileCols = ComputeTileCols(plan);
    const uint32_t safeTileCols = (tileCols == 0U) ? 1U : tileCols;
    LocalTensor<float> all = aTileBuf.Get<float>();
    const uint32_t ntiles = (this->colCount + safeTileCols - 1U) / safeTileCols;

    for (uint32_t t = 0U; t < ntiles; ++t) {
        const uint32_t cur = t % nslot;
        const uint32_t col0 = t * tileCols;
        uint32_t cols = this->colCount - col0;
        if (cols > tileCols) {
            cols = tileCols;
        }
        LocalTensor<float> curBuf = SlotBuffer(all, cur, slotFloats);
        WaitLoadFlag(t);
        WaitSlotFree(t, nslot);

        if (t + 1U < ntiles) {
            const uint32_t ncol0 = col0 + cols;
            uint32_t ncols = this->colCount - ncol0;
            if (ncols > tileCols) {
                ncols = tileCols;
            }
            LocalTensor<float> dst = SlotBuffer(all, (t + 1U) % nslot, slotFloats);
            LoadAColumnTileTo(ncol0, ncols, rowOff, chunk, plan.colStride, dst);
            SetLoadFlag(t + 1U);
        }

        AccumulateColumns(curBuf, col0, cols, CGERU_FLOATS_PER_COMPLEX * chunk, plan.colStride);
        SetFlag<HardEvent::V_MTE3>(kEvVecStore);
        WaitFlag<HardEvent::V_MTE3>(kEvVecStore);
        StoreAColumnTileFrom(col0, cols, rowOff, chunk, plan.colStride, curBuf);
        SetStoreFlag(cur);
    }
    WaitPendingStores(ntiles, nslot);
}

__aicore__ inline void CgeruKernel::Process()
{
    const uint32_t total = this->totalCols;
    if (total == 0U || this->m == 0U) {
        return;
    }
    const uint32_t coreStart = this->colStart;
    // Init 已把 colCount 夹到单批上限；正常情况下 total <= colCount，只跑一批。
    const uint32_t batch = this->colCount;
    if (batch == 0U) {
        return;
    }
    for (uint32_t off = 0U; off < total; off += batch) {
        this->colStart = coreStart + off;
        this->colCount = ((total - off) > batch) ? batch : (total - off);
        // 每批都从干净的预取状态开始，避免上一批的 x/首列块标志跨批生效。
        this->pendXRowOff = kNoPendingRow;
        this->xEarlyIssued = false;
        this->tile0Primed = false;
        ProcessPass();
    }
}

__aicore__ inline void CgeruKernel::ProcessPass()
{
    const uint32_t rowNum = static_cast<uint32_t>(this->m);
    if (rowNum == 0 || this->colCount == 0) {
        return;
    }
    uint32_t tile = (this->rowTile < rowNum) ? this->rowTile : rowNum;
    if (tile > CGERU_ROW_TILE_MAX) {
        tile = CGERU_ROW_TILE_MAX;
    }
    // 首行块的 x 预取：Init 阶段发起，由 PrepareColumnCoeffs 的 MTE2_V 收口。
    this->pendXRowOff = 0U;
    this->pendXChunk = (tile < rowNum) ? tile : rowNum;

    PrepareColumnCoeffs();

    for (uint32_t rowOff = 0; rowOff < rowNum; rowOff += tile) {
        const uint32_t chunk = (rowOff + tile <= rowNum) ? tile : (rowNum - rowOff);
        // Gather/Mul 以 32 个复数（64 个 float）为对齐粒度向上取整；多算出来的部分只留在 UB 内。
        const uint32_t calcNum =
            ((chunk + CGERU_ALIGN_COMPLEX - 1U) / CGERU_ALIGN_COMPLEX) * CGERU_ALIGN_COMPLEX;
        LoadX(rowOff, chunk);
        ProcessColumns(rowOff, chunk, true);
        BuildSwapPattern(calcNum);
        ProcessColumns(rowOff, chunk, false);
    }
}

__global__ __aicore__ void cgeru_kernel(const CgeruTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CgeruKernel op;
    op.Init(tiling, &pipe);
    op.Process();
}

void cgeru_kernel_do(const CgeruTilingData& tiling, uint32_t numBlocks, void* stream)
{
    cgeru_kernel<<<numBlocks, nullptr, stream>>>(tiling);
}
