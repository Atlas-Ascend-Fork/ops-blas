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
 * \file ssbmv_kernel.cpp
 * \brief y = alpha * A * x + beta * y, A symmetric banded (n x n, half bandwidth k), column major.
 *
 * Only one triangle of A is stored, so every stored entry A(t, j) is used twice. With the band
 * convention of the reference implementation
 *   UPPER: A(t, j) = M(j - k + t, j),  t = 0..k   (diagonal of column j at t = k)
 *   LOWER: A(t, j) = M(j + t, j),      t = 0..k   (diagonal of column j at t = 0)
 * the two uses of A(t, j) are
 *   scatter:  y[row] += A(t, j) * x[j]
 *   mirror:   y[j]   += A(t, j) * x[row]
 * where row is j - k + t (UPPER) or j + t (LOWER). Both are computed by the core owning the output
 * row they feed, so a row is written by exactly one core: no atomics, no workspace.
 *
 * Scatter: for a fixed column the stored band is contiguous in GM and its intersection with the
 * row block is a contiguous segment, which DataCopyPad's leftPadding shifts onto the aligned row
 * accumulator.
 *
 * Mirror: for a fixed t the same entries are one per row of the block, i.e. a strided gather down
 * the stored array (stride lda). A strided DataCopyPad cannot express that - its UB destination
 * advances in whole 32B blocks, so a 4B element per block lands one per 32B and the values arrive
 * spread out. The span the mirror needs is therefore loaded once, contiguously, and the
 * per-t vectors are extracted from it with Gather, which does take byte offsets. The span is the
 * same memory the operator has to read anyway, so the extraction costs no extra bandwidth.
 *
 * The x operand of the mirror is a slice that moves by one element per t. Loading it per t would
 * be one small DMA per step, and a small DMA carries a fixed issue-and-latency cost that dominates
 * the small-k shapes. So x is pulled into UB once per chunk - a
 * window wide enough for every slice this chunk will ask for, zero filled outside [0, n) - and the
 * slices are cut out of it with Gather, which needs no DMA and no cross pipe synchronisation at
 * all. The scatter reads the same window, so no x element is fetched twice.
 *
 * UB footprint: a handful of row sized buffers, one x window and the A span, all derived by the
 * host from the UB size the platform reports, so any n within range fits.
 */

#include <cstdint>

#include "kernel_operator.h"
#include "cann_ops_blas_common.h"
#include "ssbmv_tiling_data.h"

using namespace AscendC;

namespace {
constexpr uint32_t SSBMV_BUFFER_NUM = 2;
/*!< Local spelling of the shared element size; the value itself lives in ssbmv_tiling_data.h so
 *   the host sizes its UB budget with the same number. */
constexpr uint32_t BYTES_PER_FLOAT32 = SSBMV_BYTES_PER_FLOAT32;
/*!< Upper bound, in elements, of one contiguous A load. The pieces are kept a multiple of the
 *   vector alignment so that every piece starts on a 32B boundary in UB. */
constexpr uint32_t SSBMV_SPAN_CHUNK = 4096;
/*!< 有限性探测的基准值：对任意 v，v - v 只在 v 为有限值时为 0，用它与探针结果比较即可识别
 *   Inf/NaN。命名出来是为了让「这里在检测特殊值」在调用处自明，而不是散落的 0.0f。 */
constexpr float SSBMV_FINITE_PROBE = 0.0f;

/*!
 * \brief 一列在本轮 seg slot 中的落位参数与乘数。
 *
 * 每列各自算好后存下来，等本轮所有列搬完再统一做 head 修正与 Axpy —— 这样一次队列往返
 * （四次旗标）可以服务多列，而旗标正是列成本的大头（见 mirror 的性能记录）。
 */
struct SsbmvScatterSlot {
    uint32_t offset = 0; /**< 该列在行累加器中的起始下标，恒为 SSBMV_ALIGN_ELEMS 的倍数 */
    uint32_t span = 0;   /**< Axpy 的元素个数：头部 padding + 有效长度 */
    uint8_t headPad = 0; /**< 头部 padding 长度，乘数非有限时这几 lane 需显式清零 */
    float mult = 0.0f;   /**< 该列的乘数 x[j] */
};

/*! \brief Round val up to the next multiple of SSBMV_ALIGN_ELEMS. */
__aicore__ inline uint32_t AlignUp(uint32_t val)
{
    return (val + SSBMV_ALIGN_ELEMS - 1U) / SSBMV_ALIGN_ELEMS * SSBMV_ALIGN_ELEMS;
}

/*! \brief 与 AlignUp 同义，独立命名是为了在 host/kernel 两侧的预算公式里逐字对应。 */
__aicore__ inline uint32_t CeilAlignU32(uint32_t val)
{
    return (val + SSBMV_ALIGN_ELEMS - 1U) / SSBMV_ALIGN_ELEMS * SSBMV_ALIGN_ELEMS;
}

/*!< 偏移表递推的步长（lanes）。取 64 是因为一次 Adds 正好覆盖一个 repeat（4 字节元素、
 *   8 个 block × 8 lane），既不让标量头部变长，也让大表只需要几十条指令。 */
constexpr uint32_t SSBMV_OFFSET_BLOCK = 64;

/*!
 * \brief 生成 Gather 用的字节偏移表：table[i] = i * strideBytes。
 *
 * 不用库里的 CreateVecIndex / ArithProgression —— 无 TPipe 的入口里它们必崩：两者的实现
 * （kernel_operator_vec_createvecindex_impl.h 的 CreateVecIndexOneBlk、arithprogression_
 * common_impl.h）内部都会调用 GetTPipePtr()->FetchEventID()，而没有构造过 TPipe 时那个
 * 全局指针是 null，设备侧解引用就是 507035（vector core exception）。
 *
 * 头部一个 64 lane 的块用标量写（随后补一次 S_V 同步，因为后面要用向量指令读它），
 * 其余每 64 lane 用一次 Adds 从前一块递推 —— 这样 accRows 再大也只有几十条向量指令。
 */
__aicore__ inline void BuildOffsetTable(const LocalTensor<uint32_t>& table, uint32_t strideBytes, uint32_t count)
{
    LocalTensor<int32_t> tbl = table.ReinterpretCast<int32_t>();
    const uint32_t head = (count < SSBMV_OFFSET_BLOCK) ? count : SSBMV_OFFSET_BLOCK;
    for (uint32_t i = 0; i < head; i++) {
        tbl.SetValue(i, static_cast<int32_t>(i * strideBytes));
    }
    SetFlag<HardEvent::S_V>(EVENT_ID4);
    WaitFlag<HardEvent::S_V>(EVENT_ID4);
    for (uint32_t off = head; off < count; off += SSBMV_OFFSET_BLOCK) {
        const uint32_t lanes = (count - off < SSBMV_OFFSET_BLOCK) ? (count - off) : SSBMV_OFFSET_BLOCK;
        Adds(
            tbl[off], tbl[off - SSBMV_OFFSET_BLOCK], static_cast<int32_t>(SSBMV_OFFSET_BLOCK * strideBytes),
            static_cast<int32_t>(lanes));
    }
}
} // namespace

class SsbmvKernel {
public:
    __aicore__ inline SsbmvKernel() = default;
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR x, GM_ADDR y, const SsbmvTilingData& tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline uint32_t XPhysicalPos(uint32_t logical) const
    {
        return (incx > 0) ? (logical * absIncx) : ((n - 1U - logical) * absIncx);
    }
    __aicore__ inline uint32_t YPhysicalPos(uint32_t logical) const
    {
        return (incy > 0) ? (logical * absIncy) : ((n - 1U - logical) * absIncy);
    }

    __aicore__ inline void InitBuffers(uint32_t accRows, uint32_t xBufCount, uint32_t spanCount);
    __aicore__ inline void InitBandFormBuffers();
    __aicore__ inline void InitBandFormStaging();
    __aicore__ inline uint32_t XBufFloats() const;
    __aicore__ inline void InitBandFormTables();
    __aicore__ inline LocalTensor<float> YAcc();
    __aicore__ inline LocalTensor<float> YOld();
    __aicore__ inline uint32_t DiagSpanFloats() const;
    __aicore__ inline void LoadDiagSpan(uint32_t c0, uint32_t c1);
    __aicore__ inline void LoadXWindowDiag(uint32_t c0, uint32_t c1);
    __aicore__ inline void DiagPass(uint32_t c0, uint32_t c1, int32_t e, bool accumulate);
    __aicore__ inline void ProcessDiag();
    __aicore__ inline void FetchEvents();
    __aicore__ inline void InitXGatherOffsets(uint32_t accRows);
    __aicore__ inline bool StageScatterColumn(
        uint32_t column, uint32_t c0, uint32_t c1, const LocalTensor<float>& seg, const LocalTensor<float>& xWin,
        uint32_t slot, uint32_t slotFloats, SsbmvScatterSlot& out);
    __aicore__ inline void ApplyScatterBatch(
        uint32_t staged, const LocalTensor<float>& seg, uint32_t slotFloats, const SsbmvScatterSlot* slots);
    __aicore__ inline bool ResolveScatterColumn(
        uint32_t j, uint32_t c0, uint32_t c1, uint32_t& iLo, uint32_t& iHi, uint64_t& srcOffset) const;
    __aicore__ inline uint32_t MaskBandHoles(
        const LocalTensor<float>& aVec, uint32_t c0, uint32_t t, uint32_t rowsAligned);
    __aicore__ inline void ClearBandHolePrefix(
        const LocalTensor<float>& aVec, uint32_t c0, uint32_t t, uint32_t rowsAligned);
    __aicore__ inline void BuildSpanOffsets(uint32_t rows, uint32_t rowsAligned);
    __aicore__ inline void LoadXWindow(uint32_t c0, uint32_t c1);
    __aicore__ inline void ScatterChunk(uint32_t c0, uint32_t c1);
    __aicore__ inline void LoadSpan(const LocalTensor<float>& span, uint32_t sc, uint32_t cols);
    __aicore__ inline void LoadChunkBand(uint32_t c0, uint32_t c1);
    __aicore__ inline void ScatterChunkFromSpan(uint32_t c0, uint32_t c1);
    __aicore__ inline void ScatterChunkFromSpanPerColumn(uint32_t c0, uint32_t c1);
    __aicore__ inline void MirrorChunk(uint32_t c0, uint32_t c1);
    __aicore__ inline void ReduceBandColumn(uint32_t j, uint32_t tLo, uint32_t tHi, uint32_t redLane);
    __aicore__ inline void MirrorChunkByColumn(uint32_t c0, uint32_t c1);
    __aicore__ inline void FinalizeChunk(uint32_t c0, uint32_t c1);
    __aicore__ inline void StoreStridedChunk(uint32_t c0, uint32_t c1, bool useAcc);
    __aicore__ inline void HandleAlphaZero();

    __aicore__ inline void SyncMte2ToScalar()
    {
        SetFlag<HardEvent::MTE2_S>(eventMte2S);
        WaitFlag<HardEvent::MTE2_S>(eventMte2S);
    }
    __aicore__ inline void SyncMte2ToVector()
    {
        SetFlag<HardEvent::MTE2_V>(eventMte2V);
        WaitFlag<HardEvent::MTE2_V>(eventMte2V);
    }
    __aicore__ inline void SyncVectorToMte3()
    {
        SetFlag<HardEvent::V_MTE3>(eventVToMte3);
        WaitFlag<HardEvent::V_MTE3>(eventVToMte3);
    }
    __aicore__ inline void SyncScalarToVector()
    {
        SetFlag<HardEvent::S_V>(eventStoV);
        WaitFlag<HardEvent::S_V>(eventStoV);
    }
    __aicore__ inline void SyncVectorToScalar()
    {
        SetFlag<HardEvent::V_S>(eventVtoS);
        WaitFlag<HardEvent::V_S>(eventVtoS);
    }
    __aicore__ inline void SyncVectorToMte2()
    {
        SetFlag<HardEvent::V_MTE2>(eventVtoMte2);
        WaitFlag<HardEvent::V_MTE2>(eventVtoMte2);
    }
    __aicore__ inline void SyncMte3ToVector()
    {
        SetFlag<HardEvent::MTE3_V>(eventMte3V);
        WaitFlag<HardEvent::MTE3_V>(eventMte3V);
    }
    __aicore__ inline void SyncScalarToMte3()
    {
        SetFlag<HardEvent::S_MTE3>(eventStoMte3);
        WaitFlag<HardEvent::S_MTE3>(eventStoMte3);
    }
    __aicore__ inline void SyncMte3ToMte2()
    {
        SetFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
        WaitFlag<HardEvent::MTE3_MTE2>(eventMte3ToMte2);
    }
    __aicore__ inline void SyncScalarToMte2()
    {
        SetFlag<HardEvent::S_MTE2>(eventStoMte2);
        WaitFlag<HardEvent::S_MTE2>(eventStoMte2);
    }
    /*!
     * \brief Orders the pipes between two chunks that reuse the same UB buffers.
     *
     * Every buffer here is a plain TBuf, so the framework inserts no synchronisation of its own and
     * each cross-pipe reuse has to be ordered by hand. A chunk ends with the scalar pipe reading
     * the accumulator (strided y) and x window, MTE3 storing the accumulator and MTE2 filling the
     * x window and the span, and the vector pipe reading all of them - so before the next chunk
     * overwrites anything, each of those readers has to have caught up.
     */
    __aicore__ inline void SyncPipesBetweenChunks()
    {
        SyncScalarToVector();
        SyncMte3ToVector();
        SyncVectorToMte2();
        SyncScalarToMte2();
    }

    TPipe pipe;
    GlobalTensor<float> aGM;
    GlobalTensor<float> xGM;
    GlobalTensor<float> yGM;

    TBuf<TPosition::VECCALC> yBuf;
    TBuf<TPosition::VECCALC> yOldBuf;
    TBuf<TPosition::VECCALC> aVecBuf;
    TBuf<TPosition::VECCALC> spanBuf;
    /* 宽带列形式专用（bandForm == 1）：A 段 / x 段 / 归约临时区 / 归约结果 / 分块归约累加 / 步长收集表 */
    TBuf<TPosition::VECCALC> prodBuf;
    TBuf<TPosition::VECCALC> xSlBuf;
    TBuf<TPosition::VECCALC> redTmpBuf;
    TBuf<TPosition::VECCALC> redBuf;
    TBuf<TPosition::VECCALC> redAccBuf;
    TBuf<TPosition::VECCALC> off8Buf;
    TBuf<TPosition::VECCALC> offBuf;
    TBuf<TPosition::VECCALC> offXBuf;
    TBuf<TPosition::VECCALC> xBuf;
    TBuf<TPosition::VECCALC> xVecBuf;
    TBuf<TPosition::VECCALC> yStageBuf;
    TQue<TPosition::VECIN, SSBMV_BUFFER_NUM> segQueue;
    /* 行-对角形式把全部 UB 需求合并成**一块** TBuf 手工切分：InitBuffer 的次数直接进每核的
     * 固定开销，核数多时会被逐核放大，所以这条路径只调一次。切分偏移都按 8 个 float（32B）对齐。 */
    TBuf<TPosition::VECCALC> ubBuf;
    LocalTensor<float> yT;
    LocalTensor<float> yOldT;
    LocalTensor<float> aVecT;
    LocalTensor<float> xvT;
    LocalTensor<float> xT;
    LocalTensor<float> spanT;
    LocalTensor<uint32_t> offAT;
    LocalTensor<uint32_t> offXT;

    event_t eventMte2S;
    event_t eventMte2V;
    event_t eventVToMte3;
    event_t eventStoV;
    event_t eventVtoS;
    event_t eventVtoMte2;
    event_t eventMte3V;
    event_t eventStoMte2;
    event_t eventStoMte3;
    event_t eventMte3ToMte2;

    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t lda = 0;
    uint32_t uplo = ACLBLAS_UPPER;
    uint32_t blockRows = 0;
    uint32_t chunkRows = 0;
    uint32_t spanExt = 0;
    uint32_t bandForm = 0;
    uint32_t diagForm = 0;
    uint32_t spanCol0 = 0;
    /*!< 行-对角形式下 band span 在缓冲区里的起点：前置一段零哨兵，吸收「对齐下取整」带来的
     *   至多 7*lda 个负偏移（见 DiagPass）。非行-对角形式恒为 0。 */
    uint32_t spanOff = 0;
    uint32_t yStageRows = 0;
    uint32_t segBatch = 1;
    uint32_t segSlot = SSBMV_ALIGN_ELEMS;
    /* First logical x index the chunk's window holds; negative when the band reaches before x. */
    int64_t winLo = 0;

    bool active = false;
    uint32_t r0 = 0;
    uint32_t r1 = 0;

    float alpha = 1.0f;
    float beta = 0.0f;
    int64_t incx = 1;
    int64_t incy = 1;
    uint32_t absIncx = 1;
    uint32_t absIncy = 1;
};

__aicore__ inline void SsbmvKernel::InitBuffers(uint32_t accRows, uint32_t xBufCount, uint32_t spanCount)
{
    if (diagForm != 0U) {
        /* 顺序：y / yOld / aVec / xv（各 accRows 个 float）→ 两张 uint32 偏移表（各 accRows 个元素，
         * 按 1 个 float 的额度计）→ x 窗口 → band span。与 host 的 SsbmvDiagUbFloats 逐项对应。 */
        uint64_t total = 6ULL * accRows + xBufCount + spanCount;
        uint32_t yStageFloats = 0;
        if (incy != 1) {
            yStageFloats = AlignUp((yStageRows - 1U) * absIncy + 1U) + SSBMV_UB_SLACK;
            total += yStageFloats;
        }
        pipe.InitBuffer(ubBuf, static_cast<uint32_t>(total) * BYTES_PER_FLOAT32);
        LocalTensor<float> base = ubBuf.Get<float>();
        yT = base[0];
        yOldT = base[accRows];
        aVecT = base[2U * accRows];
        xvT = base[3U * accRows];
        offAT = base[4U * accRows].ReinterpretCast<uint32_t>();
        offXT = base[5U * accRows].ReinterpretCast<uint32_t>();
        xT = base[6U * accRows];
        spanT = base[6U * accRows + xBufCount];
        if (incy != 1) {
            pipe.InitBuffer(yStageBuf, yStageFloats * BYTES_PER_FLOAT32);
        }
        return;
    }
    pipe.InitBuffer(yBuf, accRows * BYTES_PER_FLOAT32);
    pipe.InitBuffer(yOldBuf, accRows * BYTES_PER_FLOAT32);
    pipe.InitBuffer(aVecBuf, accRows * BYTES_PER_FLOAT32);
    pipe.InitBuffer(xVecBuf, accRows * BYTES_PER_FLOAT32);
    pipe.InitBuffer(offBuf, accRows * sizeof(uint32_t));
    pipe.InitBuffer(offXBuf, accRows * sizeof(uint32_t));
    pipe.InitBuffer(xBuf, xBufCount * BYTES_PER_FLOAT32);
    pipe.InitBuffer(spanBuf, spanCount * BYTES_PER_FLOAT32);
    if (incy != 1) {
        /* Only the strided path needs a staging run: (yStageRows - 1) * |incy| + 1 floats. */
        const uint32_t yStageFloats = AlignUp((yStageRows - 1U) * absIncy + 1U) + SSBMV_UB_SLACK;
        pipe.InitBuffer(yStageBuf, yStageFloats * BYTES_PER_FLOAT32);
    }
    /* The scatter stages segBatch columns per round, so each queue buffer holds that many slots.
     * A slot is align8(k + 8) wide, which covers the widest segment (k + 1 behind a head of up to
     * 7), so no row sized floor is needed and the host can charge the same amount. */
    const uint32_t segFloats = segBatch * segSlot;
    if (diagForm == 0U) {
        pipe.InitBuffer(segQueue, SSBMV_BUFFER_NUM, segFloats * BYTES_PER_FLOAT32);
    }
    if (bandForm != 0U) {
        InitBandFormBuffers();
    }
}

/*!
 * \brief 列形式的段缓冲：按 SSBMV_MIRROR_TILE 分块搬运与归约，故只需装一块（lda 小于一块时按 lda）。
 *
 * 归约结果按 8 lane 一组存放（32B 对齐）；分块归约先落到 redAcc 再累加进位。与 host 的
 * SsbmvBandUbFloats 逐项对应。
 */
__aicore__ inline void SsbmvKernel::InitBandFormBuffers()
{
    const uint32_t tile = (lda < SSBMV_MIRROR_TILE) ? lda : SSBMV_MIRROR_TILE;
    const uint32_t slab = AlignUp(tile) + SSBMV_UB_SLACK;
    pipe.InitBuffer(prodBuf, slab * BYTES_PER_FLOAT32);
    pipe.InitBuffer(xSlBuf, slab * BYTES_PER_FLOAT32);
    pipe.InitBuffer(redTmpBuf, slab * BYTES_PER_FLOAT32);
    pipe.InitBuffer(redBuf, 64U * BYTES_PER_FLOAT32);
    pipe.InitBuffer(redAccBuf, SSBMV_ALIGN_ELEMS * BYTES_PER_FLOAT32);
    pipe.InitBuffer(off8Buf, 8U * sizeof(uint32_t));
}

/*!
 * \brief 列形式的 seg 队列槽位：scatter 每列只暂存与本 chunk 行块相交的一段
 *   （<= chunkRows + 对齐头），队列按行数开槽而不是按 k —— 宽带下 k+1 个 float 的槽位
 *   本身就装不进 UB。
 */
__aicore__ inline void SsbmvKernel::InitBandFormStaging()
{
    if (bandForm == 0U) {
        return;
    }
    segBatch = 1U;
    segSlot = AlignUp(chunkRows + SSBMV_ALIGN_ELEMS);
}

/*!
 * \brief x 窗口占用的 float 数。bandForm 不建窗口（mirror 从 GM 按段读 x、scatter 标量读
 *   x[j]），只留最小占位；其余形式才需要「本 chunk 的列 + 两侧各 k 列」的窗口，越界部分以 0
 *   参与 mirror。与 host 的两个预算公式逐字对应。
 */
__aicore__ inline uint32_t SsbmvKernel::XBufFloats() const
{
    return (bandForm != 0U) ? (SSBMV_ALIGN_ELEMS + SSBMV_UB_SLACK) : (AlignUp(chunkRows + 2U * k) + SSBMV_UB_SLACK);
}

/*! \brief 列形式的收集表：8 列一组，每 lane 一个 float 的字节偏移。 */
__aicore__ inline void SsbmvKernel::InitBandFormTables()
{
    if (bandForm == 0U) {
        return;
    }
    LocalTensor<uint32_t> off8 = off8Buf.Get<uint32_t>();
    for (uint32_t i = 0; i < 8U; i++) {
        off8.SetValue(i, i * SSBMV_ALIGN_ELEMS * BYTES_PER_FLOAT32);
    }
    SyncScalarToVector();
}

__aicore__ inline void SsbmvKernel::FetchEvents()
{
    eventMte2S = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_S));
    eventMte2V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
    eventVToMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
    if (diagForm != 0U && blockRows <= chunkRows && incy == 1 && alpha != 0.0f) {
        /* 行-对角形式 + 每核只有一个 chunk 时全程只用得到这两个事件（搬入一次等待、写回一次等待）。
         * 其余八个是给多 chunk 复用同一批 UB 缓冲用的；每核的入口成本直接进总时长，能省则省。 */
        return;
    }
    eventStoV = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_V));
    eventVtoS = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_S));
    eventVtoMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
    eventMte3V = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    eventStoMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE2));
    eventStoMte3 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::S_MTE3));
    eventMte3ToMte2 = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
}

/*!
 * \brief x 侧 gather 偏移表：每行只差一个元素，所以恒为 0, 1, 2, ...
 *
 * 这张表与 t、chunk 都无关，整核只建一次；每个 t 的位移退化成一个标量基址。
 */
__aicore__ inline void SsbmvKernel::InitXGatherOffsets(uint32_t accRows)
{
    /* 行-对角形式同样需要一张等步长表（步长 lda），用矢量指令生成，避免逐 lane 标量直链。
     * 只有 [0, cnt) 会被使用（cnt <= 行数），所以不需要像旧路径那样把尾部夹到 rows-1。 */
    if (diagForm != 0U) {
        LocalTensor<int32_t> offAi = offAT.ReinterpretCast<int32_t>();
        LocalTensor<int32_t> offXi = offXT.ReinterpretCast<int32_t>();
        CreateVecIndex(offAi, static_cast<int32_t>(0), static_cast<int32_t>(accRows));
        Muls(offAi, offAi, static_cast<int32_t>(lda * BYTES_PER_FLOAT32), static_cast<int32_t>(accRows));
        CreateVecIndex(offXi, static_cast<int32_t>(0), static_cast<int32_t>(accRows));
        Muls(offXi, offXi, static_cast<int32_t>(BYTES_PER_FLOAT32), static_cast<int32_t>(accRows));
        spanOff = SSBMV_ALIGN_ELEMS * lda;
        Duplicate(spanT, 0.0f, spanOff);
    } else {
        /* 表就是 [0, 4, 8, ...]（每 lane 一个 float 的字节偏移，见 D-012(2)），用矢量指令生成：
         * 逐 lane SetValue 会在每个核的直链里排进 accRows 条标量指令外加一次 S_V 同步，
         * 这里换成两条矢量指令。 */
        LocalTensor<int32_t> offXi = offXBuf.Get<int32_t>();
        CreateVecIndex(offXi, static_cast<int32_t>(0), static_cast<int32_t>(accRows));
        Muls(offXi, offXi, static_cast<int32_t>(BYTES_PER_FLOAT32), static_cast<int32_t>(accRows));
    }
}

__aicore__ inline void SsbmvKernel::Init(GM_ADDR a, GM_ADDR x, GM_ADDR y, const SsbmvTilingData& tiling)
{
    n = tiling.n;
    k = tiling.k;
    lda = tiling.lda;
    uplo = tiling.uplo;
    blockRows = tiling.blockRows;
    chunkRows = tiling.chunkRows;
    spanExt = tiling.spanExt;
    bandForm = tiling.bandForm;
    diagForm = tiling.diagForm;
    yStageRows = tiling.yStageRows;
    segBatch = (tiling.segBatch == 0U) ? 1U : tiling.segBatch;
    segSlot = (tiling.segSlot == 0U) ? SSBMV_ALIGN_ELEMS : tiling.segSlot;
    InitBandFormStaging();
    alpha = tiling.alpha;
    beta = tiling.beta;
    incx = tiling.incx;
    incy = tiling.incy;
    absIncx = static_cast<uint32_t>(incx > 0 ? incx : -incx);
    absIncy = static_cast<uint32_t>(incy > 0 ? incy : -incy);

    const uint32_t accRows = AlignUp(chunkRows) + SSBMV_UB_SLACK;
    const uint32_t xBufCount = XBufFloats();
    /* The span the mirror gathers from: every entry of the rows the chunk owns, so lda per row. */
    /* 列形式（bandForm != 0）整条路径都不读 band span（mirror 逐列搬 A/x 段、scatter 走逐列 GM
     * 路径），这里就只留一个最小占位 —— 否则宽带下 lda*(chunkRows-1)+k+1 会撑爆 UB。 */
    uint32_t spanCount = SSBMV_ALIGN_ELEMS + SSBMV_UB_SLACK;
    if (diagForm != 0U) {
        spanCount = DiagSpanFloats();
    } else if (bandForm == 0U) {
        const uint32_t spanCols = chunkRows + ((spanExt != 0U) ? k : 0U);
        const uint64_t spanMax = static_cast<uint64_t>(lda) * (spanCols - 1U) + k + 1U;
        spanCount = AlignUp(static_cast<uint32_t>(spanMax)) + SSBMV_UB_SLACK;
    }

    InitBuffers(accRows, xBufCount, spanCount);
    FetchEvents();
    InitXGatherOffsets(accRows);

    r0 = GetBlockIdx() * blockRows;
    active = (r0 < n);
    if (!active) {
        return;
    }
    r1 = (r0 + blockRows < n) ? (r0 + blockRows) : n;

    InitBandFormTables();
    aGM.SetGlobalBuffer((__gm__ float*)a, static_cast<uint64_t>(n) * lda);
    xGM.SetGlobalBuffer((__gm__ float*)x, static_cast<uint64_t>(n - 1U) * absIncx + 1U);
    yGM.SetGlobalBuffer((__gm__ float*)y, static_cast<uint64_t>(n - 1U) * absIncy + 1U);
}

/*!
 * \brief x 窗口片段的 32B 对齐搬运（三个窗口装载路径共用的尾巴）。
 *
 * 目的地址落在「lead 向下取整到 8 的倍数」处（32B 对齐），错位 lead % 8 由 leftPadding 吸收；
 * 尾部不足一个 32B 块的部分由 rightPadding 清零。lead = vLo - winLo，为该片段相对窗口起点的元素数。
 */
__aicore__ inline void CopyXWindowAligned(
    const LocalTensor<float>& xWin, const GlobalTensor<float>& xGM, int64_t vLo, uint32_t lead, uint32_t payload)
{
    const uint32_t leadAligned = lead - (lead % SSBMV_ALIGN_ELEMS);
    const uint8_t leftPad = static_cast<uint8_t>(lead - leadAligned);
    const uint32_t used = leftPad + payload;
    const uint8_t rightPad = static_cast<uint8_t>(AlignUp(used) - used);
    DataCopyExtParams copyParams(1, payload * BYTES_PER_FLOAT32, 0, 0, 0);
    DataCopyPadExtParams<float> padParams(true, leftPad, rightPad, 0.0f);
    DataCopyPad(xWin[leadAligned], xGM[vLo], copyParams, padParams);
}

/*!
 * \brief Pulls the slice of x both halves of the chunk work from into UB once.
 *
 * The window spans [c0 - k, c1 + k): the scatter reaches k columns ahead of the block and the
 * mirror k columns back of it. Entries that fall outside [0, n) are part of the window too, as
 * zeros - the mirror's slices run off both ends of x on the first and last chunk, and reading a
 * zero there is what keeps those lanes out of the sum.
 *
 * The stride is arbitrary in which case the window is filled by scalar loads; that path is not
 * performance critical, so it is not worth a vector formulation.
 */
__aicore__ inline void SsbmvKernel::LoadXWindow(uint32_t c0, uint32_t c1)
{
    const uint32_t rows = c1 - c0;
    const uint32_t count = rows + 2U * k;
    LocalTensor<float> xWin = xBuf.Get<float>();

    if (incx == 1) {
        /* Zero first: only the part of [0, n) the window overlaps is written below, and the copy
         * lands the payload on a 32B boundary, so the leading and trailing lanes are left to the
         * zeroing. The tail is not written over afterwards, so it needs no second pass. */
        /* 多清 SSBMV_ALIGN_ELEMS 个 lane：mirror 的 x 侧 gather 用 rowsAligned = AlignUp(rows) 个
         * lane，当 rows 不是 8 的倍数时最多比 count = rows + 2k 多读 rowsAligned - rows 个元素
         * （LOWER 的索引是 m + t + k，t 取到 k 时最远）。这些 lane 若不清零就是未初始化的 UB
         * （首次）或上一个 chunk 的残留 x，乘上真实 A 值后变成伪贡献、污染结果。缓冲区另留了
         * SSBMV_UB_SLACK，放得下这几 lane。 */
        Duplicate(xWin, 0.0f, count + SSBMV_ALIGN_ELEMS);
        SyncVectorToMte2();

        const int64_t vLo = (winLo > 0) ? winLo : 0;
        const int64_t vHi = (static_cast<int64_t>(n) < static_cast<int64_t>(c1) + k) ? static_cast<int64_t>(n) :
                                                                                       (static_cast<int64_t>(c1) + k);
        if (vHi > vLo) {
            CopyXWindowAligned(xWin, xGM, vLo, static_cast<uint32_t>(vLo - winLo), static_cast<uint32_t>(vHi - vLo));
        }
        /* 窗口有两个消费者：scatter 用标量逐列取值（MTE2_S），mirror 用 Gather 走向量流水线
         * （MTE2_V）。两个等待都挪到调用方，与 band span 的搬运共用一次同步 —— 每次等待都要
         * 付一整条 HBM 往返，能合并的等待就合并。 */
    } else {
        for (uint32_t i = 0; i < count; i++) {
            const int64_t index = winLo + static_cast<int64_t>(i);
            const bool inRange = (index >= 0) && (index < static_cast<int64_t>(n));
            xWin.SetValue(i, inRange ? xGM.GetValue(XPhysicalPos(static_cast<uint32_t>(index))) : 0.0f);
        }
        for (uint32_t i = count; i < count + SSBMV_ALIGN_ELEMS; i++) {
            xWin.SetValue(i, 0.0f);
        }
        /* 标量填充的窗口，向量侧的 Gather 必须等到这些 SetValue 完成。 */
        SyncScalarToVector();
    }
}

/*!
 * \brief 求列 j 与本 chunk 行块的相交区间，以及该区间在 GM 上的起点。
 *
 * UPPER 列 j 存 M[i][j]（j-k <= i <= j），LOWER 存 M[i][j]（j <= i <= j+k）：与该 chunk 的
 * 行块取交集即可。列与行块不相交时返回 false。
 */
__aicore__ inline bool SsbmvKernel::ResolveScatterColumn(
    uint32_t j, uint32_t c0, uint32_t c1, uint32_t& iLo, uint32_t& iHi, uint64_t& srcOffset) const
{
    if (uplo == ACLBLAS_UPPER) {
        iLo = (j > k) ? (j - k) : 0U;
        if (iLo < c0) {
            iLo = c0;
        }
        iHi = (j < c1) ? j : (c1 - 1U);
        if (iLo > iHi) {
            return false;
        }
        srcOffset = static_cast<uint64_t>(lda) * j + (k + iLo - j);
        return true;
    }
    iLo = (j > c0) ? j : c0;
    iHi = j + k;
    if (iHi > (n - 1U)) {
        iHi = n - 1U;
    }
    if (iHi > (c1 - 1U)) {
        iHi = c1 - 1U;
    }
    if (iLo > iHi) {
        return false;
    }
    srcOffset = static_cast<uint64_t>(lda) * j + (iLo - j);
    return true;
}

/*!
 * \brief 把一列搬进本轮 seg 的第 slot 个坑位，并把它的落位参数写进 out。
 *
 * 列与本行块不相交时返回 false。段的头部 padding 由 DataCopyPad 的 leftPadding 吸收，
 * 使搬运后 seg[slot] 起算的向量运算基址保持 32B 对齐。
 */
__aicore__ inline bool SsbmvKernel::StageScatterColumn(
    uint32_t column, uint32_t c0, uint32_t c1, const LocalTensor<float>& seg, const LocalTensor<float>& xWin,
    uint32_t slot, uint32_t slotFloats, SsbmvScatterSlot& out)
{
    uint32_t iLo = 0;
    uint32_t iHi = 0;
    uint64_t srcOffset = 0;
    if (!ResolveScatterColumn(column, c0, c1, iLo, iHi, srcOffset)) {
        return false;
    }
    const uint32_t offsetInRow = iLo - c0;
    const uint8_t headPad = static_cast<uint8_t>(offsetInRow % SSBMV_ALIGN_ELEMS);
    const uint32_t count = iHi - iLo + 1U;
    /* The span is exactly the head plus the payload: the multiply below has to skip the tail
     * padding, because padding * xj is a NaN when xj is not finite and the reference, which never
     * forms that product, would not have one. Only the start address has to be aligned, the length
     * does not, so nothing is lost by not rounding up. */
    const uint32_t span = headPad + count;
    const uint8_t tailPad = static_cast<uint8_t>(AlignUp(span) - span);

    out.offset = offsetInRow - headPad;
    out.span = span;
    out.headPad = headPad;
    /* 列形式没有 x 窗口，乘数直接从 GM 按物理下标取（bandForm 下 incx 任意）。 */
    out.mult = (bandForm != 0U) ? xGM.GetValue(XPhysicalPos(column)) :
                                  xWin.GetValue(static_cast<uint32_t>(static_cast<int64_t>(column) - winLo));

    DataCopyExtParams copyParams(1, static_cast<uint32_t>(count * BYTES_PER_FLOAT32), 0, 0, 0);
    DataCopyPadExtParams<float> padParams(true, headPad, tailPad, 0.0f);
    DataCopyPad(seg[slot * slotFloats], aGM[srcOffset], copyParams, padParams);
    return true;
}

/*!
 * \brief 对本轮已暂存的各列做头部修正与累加。
 *
 * 头部 padding 在搬运时是 0，乘以 xj 后仍为 0 —— 除非 xj 非有限（0 * Inf = NaN），
 * 那几 lane 会落到本列并不触及的行上，必须显式清零。
 */
__aicore__ inline void SsbmvKernel::ApplyScatterBatch(
    uint32_t staged, const LocalTensor<float>& seg, uint32_t slotFloats, const SsbmvScatterSlot* slots)
{
    LocalTensor<float> yUb = yBuf.Get<float>();
    for (uint32_t i = 0; i < staged; i++) {
        if (slots[i].headPad != 0) {
            const float headNotFinite = (slots[i].mult - slots[i].mult);
            if (headNotFinite != SSBMV_FINITE_PROBE) {
                Duplicate(seg[i * slotFloats], 0.0f, slots[i].headPad);
            }
        }
        Axpy(yUb[slots[i].offset], seg[i * slotFloats], slots[i].mult, slots[i].span);
    }
}

__aicore__ inline void SsbmvKernel::ScatterChunk(uint32_t c0, uint32_t c1)
{
    LocalTensor<float> xWin = xBuf.Get<float>();
    const uint32_t jLo = (uplo == ACLBLAS_UPPER) ? c0 : ((c0 > k) ? (c0 - k) : 0U);
    const uint32_t jHi = (uplo == ACLBLAS_UPPER) ? ((c1 + k < n) ? (c1 + k) : n) : c1;
    const uint32_t slotFloats = segSlot;
    SsbmvScatterSlot slots[SSBMV_SEG_BATCH_MAX];

    /* A queue round costs four flag operations and that, not the bytes moved, is what a column
     * costs; so the columns are staged segBatch at a time and the flags are amortised over them. */
    uint32_t j = jLo;
    while (j < jHi) {
        uint32_t staged = 0;
        LocalTensor<float> seg = segQueue.AllocTensor<float>();
        while (j < jHi && staged < segBatch) {
            if (StageScatterColumn(j++, c0, c1, seg, xWin, staged, slotFloats, slots[staged])) {
                staged++;
            }
        }
        if (staged == 0U) {
            segQueue.FreeTensor(seg);
            break;
        }
        segQueue.EnQue(seg);
        seg = segQueue.DeQue<float>();
        ApplyScatterBatch(staged, seg, slotFloats, slots);
        segQueue.FreeTensor(seg);
    }
}

/*!
 * \brief scatter 的 UB 版：从 band span 里 gather 每一列，而不是每列发一次 GM 搬运。
 *
 * spanExt == 1 时调用。本 chunk 的 span 已把 scatter 要用的列（UPPER 右侧 / LOWER 左侧各 k 列）
 * 一起装进 UB，于是每列只剩「一次 Gather + 一次 Axpy + 若干标量地址运算」，不再有等待 GM 往返的
 * 小搬运 —— 这是小 k 形状的主要成本来源。
 *
 * 列 j 的元素在 span 中的下标是 (j - spanCol0) * lda + t，其中 t = i - j + k（UPPER）或 i - j
 * （LOWER）。Gather 的 srcBaseOffset 是**字节**偏移且允许非 32B 对齐（见 DECISIONS D-012(2)），
 * 因此 headPad 可以直接吸收进基址：把该列段搬到 seg[0..span)，再 Axpy 到 yUb[offsetInRow - headPad]，
 * 头部 padding 那几 lane 显式清零即可（它们的 x 乘数非有限时会是 0*Inf）。
 */
__aicore__ inline void SsbmvKernel::ScatterChunkFromSpan(uint32_t c0, uint32_t c1)
{
    /* UPPER 的行块起点一定落在有效区间内（t <= k 时 j = i + k - t >= i >= c0 >= 0），LOWER 的
     * 第一个 chunk（c0 < k）则存在 t > c0 的对角线，其目标起点是 yUb[t - c0]，不是 8 的倍数。
     * 那条路径回退到逐列版本：它只影响 c0 < k 的极少数核。 */
    if (uplo == ACLBLAS_LOWER && c0 < k) {
        ScatterChunkFromSpanPerColumn(c0, c1);
        return;
    }
    const uint32_t rows = c1 - c0;
    LocalTensor<float> span = spanBuf.Get<float>();
    LocalTensor<float> xWin = xBuf.Get<float>();
    LocalTensor<float> yUb = yBuf.Get<float>();
    LocalTensor<float> aVec = aVecBuf.Get<float>();
    LocalTensor<float> xv = xVecBuf.Get<float>();
    LocalTensor<uint32_t> offA = offBuf.Get<uint32_t>();
    LocalTensor<uint32_t> offX = offXBuf.Get<uint32_t>();

    /* 每条对角线上，A 侧沿行号步长 lda（offA 表），x 侧是窗口里的一段连续切片（offX 表）。
     * 目标一律是 yUb 的第 0 lane 起 —— 于是每次迭代的成本与列数无关，只随 k 增长。 */
    for (uint32_t t = 0; t <= k; t++) {
        int64_t iLast = 0;
        uint32_t aCol = 0;
        uint32_t xBase = 0;
        if (uplo == ACLBLAS_UPPER) {
            /* j = i + k - t 必须 <= n-1 */
            iLast = static_cast<int64_t>(n) - 1 - static_cast<int64_t>(k) + static_cast<int64_t>(t);
            aCol = c0 + k - t;
            xBase = 2U * k - t;
        } else {
            /* j = i - t 必须 >= 0 */
            iLast = static_cast<int64_t>(n) - 1;
            aCol = c0 - t;
            xBase = k - t;
        }
        const int64_t iFirst = static_cast<int64_t>(c0);
        uint32_t cnt = rows;
        if (iLast < iFirst + static_cast<int64_t>(rows) - 1) {
            if (iLast < iFirst) {
                continue;
            }
            cnt = static_cast<uint32_t>(iLast - iFirst + 1);
        }
        Gather(aVec, span, offA, static_cast<uint32_t>(spanOff + (aCol - spanCol0) * lda + t) * BYTES_PER_FLOAT32, cnt);
        Gather(xv, xWin, offX, xBase * BYTES_PER_FLOAT32, cnt);
        MulAddDst(yUb, aVec, xv, cnt);
    }
}

/*!
 * \brief 逐列版 scatter（spanExt == 1 但行块起点不满足对齐条件时使用）。
 */
__aicore__ inline void SsbmvKernel::ScatterChunkFromSpanPerColumn(uint32_t c0, uint32_t c1)
{
    LocalTensor<float> xWin = xBuf.Get<float>();
    LocalTensor<float> span = spanBuf.Get<float>();
    LocalTensor<float> yUb = yBuf.Get<float>();
    LocalTensor<uint32_t> offX = offXBuf.Get<uint32_t>();
    LocalTensor<float> seg = segQueue.AllocTensor<float>();

    const uint32_t jLo = (uplo == ACLBLAS_UPPER) ? c0 : ((c0 > k) ? (c0 - k) : 0U);
    const uint32_t jHi = (uplo == ACLBLAS_UPPER) ? ((c1 + k < n) ? (c1 + k) : n) : c1;
    for (uint32_t j = jLo; j < jHi; j++) {
        uint32_t iLo = 0;
        uint32_t iHi = 0;
        uint64_t srcOffset = 0;
        if (!ResolveScatterColumn(j, c0, c1, iLo, iHi, srcOffset)) {
            continue;
        }
        const uint32_t offsetInRow = iLo - c0;
        const uint8_t headPad = static_cast<uint8_t>(offsetInRow % SSBMV_ALIGN_ELEMS);
        const uint32_t count = iHi - iLo + 1U;
        const uint32_t segSpan = headPad + count;
        const uint32_t tLo = (uplo == ACLBLAS_UPPER) ? (iLo + k - j) : (iLo - j);
        const uint64_t elem = spanOff + (static_cast<uint64_t>(j - spanCol0) * lda) + tLo - headPad;
        const float mult = xWin.GetValue(static_cast<uint32_t>(static_cast<int64_t>(j) - winLo));
        Gather(seg, span, offX, static_cast<uint32_t>(elem * BYTES_PER_FLOAT32), segSpan);
        if (headPad != 0U) {
            Duplicate(seg, 0.0f, headPad);
        }
        Axpy(yUb[offsetInRow - headPad], seg, mult, segSpan);
    }
    segQueue.FreeTensor(seg);
}

/*!
 * \brief Loads every band entry the mirror of this chunk reads, in one contiguous run.
 *
 * The mirror of the chunk touches A(t, c0 + m) for t = 0..k and m = 0..rows-1 (LOWER: t = 1..k),
 * which is the GM range [lda * c0, lda * (c0 + rows - 1) + k]. Reading it as a whole costs the
 * same as the entries that are actually used, and it lets each t be extracted by Gather below.
 * The range is read in pieces so that one load never exceeds the copy interface's block length;
 * every piece but the last is a multiple of the vector alignment, so the destination offsets stay
 * 32B aligned.
 */
__aicore__ inline void SsbmvKernel::LoadSpan(const LocalTensor<float>& span, uint32_t sc, uint32_t cols)
{
    const uint32_t spanLen = lda * (cols - 1U) + k + 1U;
    uint32_t done = 0;
    while (done < spanLen) {
        uint32_t piece = spanLen - done;
        if (piece > SSBMV_SPAN_CHUNK) {
            piece = SSBMV_SPAN_CHUNK;
        }
        DataCopyExtParams copyParams(1, piece * BYTES_PER_FLOAT32, 0, 0, 0);
        DataCopyPad(
            span[spanOff + done], aGM[static_cast<uint64_t>(lda) * sc + done], copyParams,
            DataCopyPadExtParams<float>());
        done += piece;
    }
}

/*!
 * \brief 本 chunk 的 band span 覆盖的列范围 [sc, se)。
 *
 * spanExt == 0 时只覆盖行块自身的列（mirror 需要的那一段）；spanExt == 1 时向 UPPER 的右侧 /
 * LOWER 的左侧多覆盖 k 列，把 scatter 要读的那些列一起装进来，scatter 便能从 UB 里 gather 而
 * 不必逐列向 GM 发一次搬运。
 */
__aicore__ inline void SsbmvKernel::LoadChunkBand(uint32_t c0, uint32_t c1)
{
    uint32_t sc = c0;
    uint32_t se = c1;
    if (spanExt != 0U) {
        if (uplo == ACLBLAS_UPPER) {
            se = ((c1 + k) < n) ? (c1 + k) : n;
        } else {
            sc = (c0 > k) ? (c0 - k) : 0U;
        }
    }
    spanCol0 = sc;
    LoadSpan(spanBuf.Get<float>(), sc, se - sc);
}

/*!
 * \brief 把 band 存储里不属于任何矩阵元素的空槽排除掉，返回本步该 gather 的元素个数。
 *
 * 列被补齐到 lda，于是 UPPER 有 t < k-j 的空槽（矩阵元素落在矩阵上方），LOWER 有 t > n-1-j 的
 * 空槽（元素落在下方）。只有 mirror 会跨过它们，其乘数（越界的 x）读成 0 —— 但 0 * Inf 是 NaN，
 * 而参考实现根本不读这些槽，所以必须显式清零而不是靠乘 0 屏蔽。
 *
 * UPPER 的空槽是本行向量的**前缀**，用 Duplicate 从对齐的 0 号位清零；LOWER 的是**后缀**，
 * 由调用方按 t 递减遍历、这里只收紧 gather 长度，让缓冲区尾部保持初始化时的 0。
 * 两个判据在 chunk 内部自然为空操作，因此不需要额外的边界分支。
 */
__aicore__ inline uint32_t SsbmvKernel::MaskBandHoles(
    const LocalTensor<float>& aVec, uint32_t c0, uint32_t t, uint32_t rowsAligned)
{
    if (uplo == ACLBLAS_LOWER) {
        const int64_t lastValid = static_cast<int64_t>(n) - static_cast<int64_t>(c0) - static_cast<int64_t>(t);
        const uint32_t valid = (lastValid > 0) ? static_cast<uint32_t>(lastValid) : 0U;
        return (valid < rowsAligned) ? valid : rowsAligned;
    }
    return rowsAligned;
}

/*!
 * \brief 把 UPPER 的空槽（矩阵元素落在矩阵上方的那些列内槽位）从 Gather 结果里清掉。
 *
 * 这些槽位在 Gather 之后才可清：Gather 会写满 [0, gatherCount)，先清会被覆盖。它们的乘数是
 * 越界读到的 0，但 0 * Inf 是 NaN，而参考实现根本不读这些槽，所以必须显式清零而非靠乘 0。
 */
__aicore__ inline void SsbmvKernel::ClearBandHolePrefix(
    const LocalTensor<float>& aVec, uint32_t c0, uint32_t t, uint32_t rowsAligned)
{
    /* 只在 c0 < k 时才可能有空槽。必须先判 c0 < k：k - c0 在 uint32_t 下对 c0 > k 会下溢成
     * 巨大值，那样 hole 会大于 rowsAligned，整个向量被清零、mirror 全丢 —— 行块起点超过 k 的核
     * （即除最前面几个核以外的所有核）都会因此算错。 */
    if (c0 >= k) {
        return;
    }
    const uint32_t hole = ((k - c0) > t) ? (k - c0 - t) : 0U;
    if (hole > 0U) {
        Duplicate(aVec, 0.0f, (hole < rowsAligned) ? hole : rowsAligned);
    }
}

/*!
 * \brief 把本 chunk 的 gather 偏移表按实际行数建好。
 *
 * 偏移依赖 chunk 的行数，所以每个 chunk 重建；重建前必须等上一个 chunk 的 Gather 用完旧表。
 */
__aicore__ inline void SsbmvKernel::BuildSpanOffsets(uint32_t rows, uint32_t rowsAligned)
{
    SyncVectorToScalar();
    LocalTensor<uint32_t> offA = offBuf.Get<uint32_t>();
    for (uint32_t m = 0; m < rowsAligned; m++) {
        const uint32_t row = (m < rows) ? m : (rows - 1U);
        offA.SetValue(m, static_cast<uint32_t>(static_cast<uint64_t>(row) * lda * BYTES_PER_FLOAT32));
    }
    SyncScalarToVector();
}

/*!
 * \brief 宽带下的 mirror：逐列做「列点积」，而不是逐对角线做 FMA。
 *
 * 对角形式每 chunk 要 3(k+1) 次调用，每次只覆盖 chunkRows 个 lane；k 远大于每块行数时
 * (k+1)/rows 可达上百倍，调用数随之爆掉，mirror 成为绝对大头。
 *
 * 列形式把「每条对角线一次调用」换成「每列 4 次调用覆盖 k+1 个元素」：
 *   A 段（列 j 的 [tLo, tHi]，GM 上连续）+ x 段（x[j-k+tLo ..]，同样连续）
 *   → Mul → ReduceSum。
 * 两段都直接从 GM 搬，tLo 的错位由 DataCopyPad 的 leftPadding 吸收（见 DECISIONS D-010），
 * 因此不需要整块 band span、也不需要 tLo 相关的偏移表。x 的下标在本形式下必然落在 [0, n)：
 * UPPER 的 x 下标是 j-k+t >= 0（t >= k-j），LOWER 是 j+t <= n-1（t <= n-1-j）。
 * 段长超过 SSBMV_MIRROR_TILE 时按 tile 分块归约、块间累加，于是带宽本身大于 UB 时也能算
 * （每列实际长度受 n 限制，但 n 可以达到 UB 装不下的大小）。
 *
 * 归约结果按 8 lane 一组写进 redBuf（c*8 起，32B 对齐），每 8 列用收集表 off8 收成连续 8 lane，
 * 再整体 Add 进 yUb —— 目标起点 (j-c0) 因此始终是 8 的倍数，不需要逐列标量累加。
 */
/*!
 * \brief 把列 j 的 band 段 [tLo, tHi] 与对应 x 段分块做点积，部分和累加进 redBuf 的第
 *   redLane 个 8-lane 组。
 *
 * 段长超过 SSBMV_MIRROR_TILE 时切块：每块搬 A/x 各一段、乘完归约，块间把部分和累加进位。
 * prod/xsl/tmp 三个缓冲因此只按 tile 开，不随 k 或 n 增长 —— 这是带宽大于 UB 也能算的关键。
 */
__aicore__ inline void SsbmvKernel::ReduceBandColumn(uint32_t j, uint32_t tLo, uint32_t tHi, uint32_t redLane)
{
    const uint32_t segLen = tHi - tLo + 1U;
    const uint32_t tileFloats = (lda < SSBMV_MIRROR_TILE) ? lda : SSBMV_MIRROR_TILE;
    const int64_t xIdx = (uplo == ACLBLAS_UPPER) ? (static_cast<int64_t>(j) - static_cast<int64_t>(k) + tLo) :
                                                   (static_cast<int64_t>(j) + tLo);
    LocalTensor<float> prod = prodBuf.Get<float>();
    LocalTensor<float> xsl = xSlBuf.Get<float>();
    LocalTensor<float> red = redBuf.Get<float>();
    LocalTensor<float> redAcc = redAccBuf.Get<float>();
    LocalTensor<float> tmp = redTmpBuf.Get<float>();

    uint32_t done = 0;
    bool firstTile = true;
    while (done < segLen) {
        const uint32_t left = segLen - done;
        const uint32_t piece = (left < tileFloats) ? left : tileFloats;
        const uint32_t t = tLo + done;
        const uint8_t headPad = static_cast<uint8_t>(t % SSBMV_ALIGN_ELEMS);
        const uint32_t used = headPad + piece;
        const uint8_t tailPad = static_cast<uint8_t>(AlignUp(used) - used);
        DataCopyExtParams copyParams(1, static_cast<uint32_t>(piece * BYTES_PER_FLOAT32), 0, 0, 0);
        DataCopyPadExtParams<float> padParams(true, headPad, tailPad, 0.0f);
        DataCopyPad(prod, aGM[(static_cast<uint64_t>(lda) * j) + t], copyParams, padParams);
        if (incx == 1) {
            DataCopyPad(xsl, xGM[xIdx + done], copyParams, padParams);
        } else {
            /* 任意步长：x 段按物理下标逐 lane 填入，头尾 padding 显式清零后再参与乘法。 */
            Duplicate(xsl, 0.0f, AlignUp(used));
            SyncVectorToScalar();
            for (uint32_t i = 0; i < piece; i++) {
                xsl.SetValue(
                    headPad + i,
                    xGM.GetValue(XPhysicalPos(static_cast<uint32_t>(xIdx + done + static_cast<int64_t>(i)))));
            }
            SyncScalarToVector();
        }
        SyncMte2ToVector();
        Mul(prod, prod, xsl, used);
        if (firstTile) {
            ReduceSum(red[redLane], prod, tmp, used);
            firstTile = false;
        } else {
            ReduceSum(redAcc, prod, tmp, used);
            Add(red[redLane], red[redLane], redAcc, 1);
        }
        SyncVectorToMte2();
        done += piece;
    }
}

__aicore__ inline void SsbmvKernel::MirrorChunkByColumn(uint32_t c0, uint32_t c1)
{
    if (k == 0U) {
        return;
    }
    const uint32_t rows = c1 - c0;
    LocalTensor<float> red = redBuf.Get<float>();
    LocalTensor<float> yUb = yBuf.Get<float>();
    LocalTensor<float> compact = aVecBuf.Get<float>();
    LocalTensor<uint32_t> off8 = off8Buf.Get<uint32_t>();

    for (uint32_t base = 0; base < rows; base += SSBMV_ALIGN_ELEMS) {
        const uint32_t left = rows - base;
        const uint32_t cnt = (left < SSBMV_ALIGN_ELEMS) ? left : SSBMV_ALIGN_ELEMS;
        for (uint32_t c = 0; c < cnt; c++) {
            const uint32_t j = c0 + base + c;
            uint32_t tLo = 0;
            uint32_t tHi = 0;
            if (uplo == ACLBLAS_UPPER) {
                tLo = (j > k) ? 0U : (k - j);
                tHi = k - 1U;
            } else {
                tLo = 1U;
                tHi = ((n - 1U - j) < k) ? (n - 1U - j) : k;
            }
            if (tLo > tHi) {
                /* 该列没有 mirror 项：UPPER 的 j == 0（对角以外的 t 全落在矩阵上方）、LOWER 的
                 * j == n-1（t 上界为 n-1-j == 0）。这里必须把归约槽显式清零 —— 本批的 Add 是
                 * 无条件执行的，留上一列的残值就会把它加到 y 上，产生错误元素。 */
                Duplicate(red[c * SSBMV_ALIGN_ELEMS], 0.0f, SSBMV_ALIGN_ELEMS);
                continue;
            }
            ReduceBandColumn(j, tLo, tHi, c * SSBMV_ALIGN_ELEMS);
        }
        Gather(compact, red, off8, 0U, cnt);
        Add(yUb[base], yUb[base], compact, cnt);
    }
}

__aicore__ inline void SsbmvKernel::MirrorChunk(uint32_t c0, uint32_t c1)
{
    if (k == 0U) {
        return;
    }
    if (bandForm != 0U) {
        MirrorChunkByColumn(c0, c1);
        return;
    }
    const uint32_t rows = c1 - c0;
    const uint32_t rowsAligned = AlignUp(rows);
    /* span 的起点列可能与行块起点不同（spanExt），每行的 band 起点因此整体平移 */
    const uint32_t spanShift = (c0 - spanCol0) * lda;
    LocalTensor<float> yUb = yBuf.Get<float>();
    LocalTensor<float> span = spanBuf.Get<float>();
    LocalTensor<float> xWin = xBuf.Get<float>();
    LocalTensor<float> aVec = aVecBuf.Get<float>();
    LocalTensor<float> xv = xVecBuf.Get<float>();
    LocalTensor<uint32_t> offA = offBuf.Get<uint32_t>();
    LocalTensor<uint32_t> offX = offXBuf.Get<uint32_t>();

    /* LOWER 走 t 递减：每次 gather 的长度单调增长、写入都落在上一次之内，
     * 尾部因此始终是这里留下的 0（见 MaskBandHoles）。 */
    if (uplo == ACLBLAS_LOWER) {
        Duplicate(aVec, 0.0f, rowsAligned);
    }

    const uint32_t tBegin = (uplo == ACLBLAS_UPPER) ? 0U : 1U;
    const uint32_t tLast = (uplo == ACLBLAS_UPPER) ? (k - 1U) : k;
    for (uint32_t step = 0; step <= (tLast - tBegin); step++) {
        const uint32_t t = (uplo == ACLBLAS_UPPER) ? (tBegin + step) : (tLast - step);
        const uint32_t gatherCount = MaskBandHoles(aVec, c0, t, rowsAligned);
        if (gatherCount > 0U) {
            Gather(aVec, span, offA, static_cast<uint32_t>(spanOff + spanShift + t) * BYTES_PER_FLOAT32, gatherCount);
        }
        /* UPPER 的空槽是本行向量的前缀，必须在 Gather 之后清零（Gather 会覆盖 [0, gatherCount)）；
         * LOWER 的空槽是后缀，靠 LOWER 分支收紧 gatherCount、由缓冲区初值保持 0。 */
        if (uplo == ACLBLAS_UPPER) {
            ClearBandHolePrefix(aVec, c0, t, rowsAligned);
        }
        /* Lane m of the slice is x[c0 + m - k + t] on UPPER and x[c0 + m + t] on LOWER, which in
         * window coordinates (the window starts at c0 - k) is m + t resp. m + t + k. */
        const uint32_t xBaseBytes = (uplo == ACLBLAS_UPPER) ? (t * BYTES_PER_FLOAT32) : ((k + t) * BYTES_PER_FLOAT32);
        Gather(xv, xWin, offX, xBaseBytes, rowsAligned);
        MulAddDst(yUb, aVec, xv, rowsAligned);
    }
}

__aicore__ inline LocalTensor<float> SsbmvKernel::YAcc() { return (diagForm != 0U) ? yT : yBuf.Get<float>(); }

__aicore__ inline LocalTensor<float> SsbmvKernel::YOld() { return (diagForm != 0U) ? yOldT : yOldBuf.Get<float>(); }

__aicore__ inline void SsbmvKernel::FinalizeChunk(uint32_t c0, uint32_t c1)
{
    LocalTensor<float> yUb = YAcc();
    const uint32_t rowCount = c1 - c0;
    const uint32_t rowCountAligned = AlignUp(rowCount);

    if (incy == 1) {
        Muls(yUb, yUb, alpha, rowCountAligned);
        if (beta != 0.0f) {
            LocalTensor<float> yOld = YOld();
            DataCopyExtParams copyParams(1, static_cast<uint32_t>(rowCount * BYTES_PER_FLOAT32), 0, 0, 0);
            DataCopyPad(
                yOld, yGM[c0], copyParams,
                DataCopyPadExtParams<float>(true, 0, static_cast<uint8_t>(rowCountAligned - rowCount), 0.0f));
            SyncMte2ToVector();
            Muls(yOld, yOld, beta, rowCountAligned);
            Add(yUb, yUb, yOld, rowCountAligned);
        }
        SyncVectorToMte3();
        DataCopyExtParams outParams(1, static_cast<uint32_t>(rowCount * BYTES_PER_FLOAT32), 0, 0, 0);
        DataCopyPad(yGM[c0], yUb, outParams);
    } else {
        StoreStridedChunk(c0, c1, true);
    }
}

/*!
 * \brief Stores a chunk's rows into a y whose stride is not 1.
 *
 * With incy other than 1 the rows of a chunk are |incy| floats apart, so the chunk's slice of y is
 * a contiguous run of (rows - 1) * |incy| + 1 floats of which only every |incy|-th entry belongs to
 * this chunk. The run is read and written as a whole: the entries in between have to survive, and
 * the run is moved by DMA rather than by scalar stores.
 *
 * The scalar unit is deliberately not used to store into y here. It writes through a per-core
 * cache, so when two cores write different entries of one cache line only one of the two updates
 * survives - whichever core flushes last reinstates its stale copy of the whole line. That would
 * let whole rows of a core silently revert to their input values, in a pattern following the
 * 64B cache lines rather than the row blocks.
 *
 * \param useAcc  true: y = alpha * accumulator + beta * y, false: y = beta * y (the alpha == 0 path)
 */
__aicore__ inline void SsbmvKernel::StoreStridedChunk(uint32_t c0, uint32_t c1, bool useAcc)
{
    LocalTensor<float> yUb = YAcc();
    LocalTensor<float> yStage = yStageBuf.Get<float>();
    if (useAcc) {
        /* The accumulator comes out of the vector pipe; the scalar pipe reads it lane by lane. */
        SyncVectorToScalar();
    }
    const uint32_t rowCount = c1 - c0;
    for (uint32_t base = 0; base < rowCount; base += yStageRows) {
        const uint32_t rows = ((base + yStageRows) < rowCount) ? yStageRows : (rowCount - base);
        const uint32_t pFirst = YPhysicalPos(c0 + base);
        const uint32_t pLast = YPhysicalPos(c0 + base + rows - 1U);
        const uint32_t pLo = (pFirst < pLast) ? pFirst : pLast;
        const uint32_t spanLen = ((rows - 1U) * absIncy) + 1U;

        SyncMte3ToMte2();
        DataCopyExtParams copyParams(1, static_cast<uint32_t>(spanLen * BYTES_PER_FLOAT32), 0, 0, 0);
        DataCopyPad(yStage, yGM[pLo], copyParams, DataCopyPadExtParams<float>());
        SyncMte2ToScalar();

        for (uint32_t i = 0; i < rows; i++) {
            /* Lane of the staged run that row c0 + base + i occupies: the run is ascending in
             * memory, the rows are not (a negative incy walks it backwards). */
            const uint32_t lane = (incy > 0) ? (i * absIncy) : ((rows - 1U - i) * absIncy);
            float value = yStage.GetValue(lane);
            if (useAcc) {
                value = alpha * yUb.GetValue(base + i);
                /* beta == 0 means y is not read at all (BLAS: "y need not be set on input"); adding
                 * 0 * oldY would turn an Inf/NaN input into NaN instead of leaving alpha * (A * x). */
                if (beta != 0.0f) {
                    value += beta * yStage.GetValue(lane);
                }
            } else if (beta == 0.0f) {
                value = 0.0f;
            } else {
                value = beta * value;
            }
            yStage.SetValue(lane, value);
        }

        SyncScalarToMte3();
        DataCopyExtParams outParams(1, static_cast<uint32_t>(spanLen * BYTES_PER_FLOAT32), 0, 0, 0);
        DataCopyPad(yGM[pLo], yStage, outParams);
    }
}

/*!
 * \brief alpha == 0 fast path: y = beta * y with A and x never referenced.
 *
 * Kept separate from the general path so the result is a single float multiply of the incoming y,
 * which the reference implementation reproduces bit exactly.
 */
__aicore__ inline void SsbmvKernel::HandleAlphaZero()
{
    LocalTensor<float> yUb = YAcc();
    for (uint32_t c0 = r0; c0 < r1; c0 += chunkRows) {
        const uint32_t c1 = ((c0 + chunkRows) < r1) ? (c0 + chunkRows) : r1;
        const uint32_t rowCount = c1 - c0;
        const uint32_t rowCountAligned = AlignUp(rowCount);
        SyncPipesBetweenChunks();
        if (incy == 1) {
            if (beta == 0.0f) {
                Duplicate(yUb, 0.0f, rowCountAligned);
            } else {
                DataCopyExtParams copyParams(1, static_cast<uint32_t>(rowCount * BYTES_PER_FLOAT32), 0, 0, 0);
                DataCopyPad(
                    yUb, yGM[c0], copyParams,
                    DataCopyPadExtParams<float>(true, 0, static_cast<uint8_t>(rowCountAligned - rowCount), 0.0f));
                SyncMte2ToVector();
                Muls(yUb, yUb, beta, rowCountAligned);
            }
            SyncVectorToMte3();
            DataCopyExtParams outParams(1, static_cast<uint32_t>(rowCount * BYTES_PER_FLOAT32), 0, 0, 0);
            DataCopyPad(yGM[c0], yUb, outParams);
        } else {
            StoreStridedChunk(c0, c1, false);
        }
    }
}

/*!
 * \brief 行-对角形式需要的 band span 元素个数（与 host 的预算公式必须逐字一致）。
 *
 * span 覆盖本 chunk 的行块加上 k 列延伸：UPPER 向右（列 [c0, c1+k)），LOWER 向左（[c0-k, c1)）。
 * 最宽的一段是 lda * (chunkRows + k) + k + 1（最后一行再往前 k 个元素的尾巴）。
 */
/*! \brief 行-对角形式所需的 span 元素个数（含前置零哨兵）。两个 kernel 共用，故做成自由函数。
 *
 * 全程用 32 位算术：AI Core 的标量单元没有 64 位乘法指令，`uint64_t` 乘法要调软件例程，
 * 而这里是在每次下发的 Init 路径上（小 shape 的固定开销里这一项可观）。
 * issue #841 修正安全依据：host 的 ValidateSsbmvArgs 并不存在 n/lda <= 8192 的
 * 规格约束（n=3e8、k=7、lda=8 的形状可合法进入 diag-only 入口）；32 位不回绕的
 * 真实依据是准入链自身的乘积上界——本函数返回的 span 必须完整装入 UB
 * （与 host 的 SsbmvDiagSpanFloats 逐字一致、按 UB 容量装箱准入），即
 * lda * (chunkRows + k) + k + 1 连同 guard/对齐/余量以 UB 量级（约 50K floats）
 * 为上界，远小于 2^32；后续放宽准入或改 UB 容量时须同步复核此处的 32 位域。 */
__aicore__ inline uint32_t DiagSpanFloats(uint32_t chunkRows, uint32_t k, uint32_t lda)
{
    const uint32_t spanMax = lda * (chunkRows + k) + k + 1U;
    const uint32_t guard = SSBMV_ALIGN_ELEMS * lda;
    return guard + CeilAlignU32(spanMax) + SSBMV_UB_SLACK;
}

__aicore__ inline uint32_t SsbmvKernel::DiagSpanFloats() const { return ::DiagSpanFloats(chunkRows, k, lda); }

/*!
 * \brief 行-对角形式的 x 窗口：只搬 payload，不做前置清零、不等待。
 *
 * 与通用窗口的区别在两点，都是行-对角形式独有的性质换来的：
 *   - 不需要把窗口整体清零：本形式的 x 下标在裁剪后恒落在 [0, n)，窗口两端永远不会被读到，
 *     于是省掉一次 Duplicate 与一次 V->MTE2 等待；
 *   - 不在这里等搬运完成：调用方把它与 band span 的搬运合并成一次等待。
 * 越界 lane 由 DiagPass 显式清零，所以窗口里读到什么都不会影响结果。
 */
__aicore__ inline void SsbmvKernel::LoadXWindowDiag(uint32_t c0, uint32_t c1)
{
    const int64_t vLo = (winLo > 0) ? winLo : 0;
    const int64_t vHi = (static_cast<int64_t>(n) < static_cast<int64_t>(c1) + k) ? static_cast<int64_t>(n) :
                                                                                   (static_cast<int64_t>(c1) + k);
    if (vHi <= vLo) {
        return;
    }
    LocalTensor<float> xWin = xT;
    CopyXWindowAligned(xWin, xGM, vLo, static_cast<uint32_t>(vLo - winLo), static_cast<uint32_t>(vHi - vLo));
}

/*!
 * \brief 行-对角形式的 band span：UPPER 取列 [c0, c1+k)，LOWER 取 [c0-k, c1)。
 */
__aicore__ inline void SsbmvKernel::LoadDiagSpan(uint32_t c0, uint32_t c1)
{
    uint32_t sc = c0;
    uint32_t se = c1;
    if (uplo == ACLBLAS_UPPER) {
        se = ((c1 + k) < n) ? (c1 + k) : n;
    } else {
        sc = (c0 > k) ? (c0 - k) : 0U;
    }
    spanCol0 = sc;
    LoadSpan(spanT, sc, se - sc);
}

/*!
 * \brief 一条错位对角线：对本 chunk 的所有行 r 做 y[r] += A(r, r+e) * x[r+e]。
 *
 * 这条对角线的 A 元素在带存储里是等步长 lda 的一串（index = r * lda + C_e），一次 Gather 就能取出；
 * x 侧是窗口里的一段连续切片（用恒为 0,1,2,... 的偏移表 + 标量基址）。两段的行偏移都吸收进
 * Gather 的 srcBase，于是目标恒为张量 0 号位，向量指令的基址始终 32B 对齐。
 *
 * 系数 C_e（元素下标减去 r * lda）由存储约定推出：
 *   UPPER: A(t, j) = M(j - k + t, j)
 *     e >= 0：配对 (r, r+e) 直接存在列 r+e 的 t = k-e 处 -> index = (r+e)*lda + (k-e) = r*lda + e*(lda-1) + k
 *     e <  0：用对称性取 M(r+e, r)，存在列 r 的 t = k+e 处   -> index = r*lda + (k+e)
 *   LOWER: A(t, j) = M(j + t, j)
 *     e >= 0：存在列 r 的 t = e 处                            -> index = r*lda + e
 *     e <  0：存在列 r+e 的 t = |e| 处                        -> index = (r+e)*lda + |e| = r*lda + e*(lda-1)
 *
 * 边界：e >= 0 时行 r 要求 r+e <= n-1（列不存在则整体不存在）；e < 0 时要求 r >= |e|（否则元素落在
 * 带外的空槽里，其值是未定义字节，乘上 0 仍可能是 NaN）。两侧都按行裁剪，并且只对 [lo, hi) 做累加，
 * 裁剪掉的 lane 既不参与 Gather 也不参与 MulAddDst —— 因此不需要任何洞区清零。
 */
__aicore__ inline void SsbmvKernel::DiagPass(uint32_t c0, uint32_t c1, int32_t e, bool accumulate)
{
    const int64_t c0i = static_cast<int64_t>(c0);
    const int64_t eAbs = (e >= 0) ? static_cast<int64_t>(e) : (-static_cast<int64_t>(e));
    const int64_t loRaw = (e >= 0) ? c0i : ((eAbs > c0i) ? eAbs : c0i);
    const int64_t hiRaw =
        (e >= 0) ? (((static_cast<int64_t>(n) - e) < static_cast<int64_t>(c1)) ? (static_cast<int64_t>(n) - e) :
                                                                                 static_cast<int64_t>(c1)) :
                   static_cast<int64_t>(c1);
    if (hiRaw <= loRaw) {
        return;
    }
    /* 累加器的目标基址必须 32B 对齐，而 yUb 的下标是「相对本 chunk 起点」的，所以把 (行 - c0)
     * 向下取整到 8 的倍数：多出来的那几 lane（至多 7 个）不属于本 pass，必须清零 ——
     * 它们的 x 乘数为 0，若不清零则 0 * NaN(带外槽位) 会污染结果。 */
    const uint32_t localRaw = static_cast<uint32_t>(loRaw - c0i);
    const uint32_t loL = localRaw & ~(SSBMV_ALIGN_ELEMS - 1U);
    const uint32_t row0 = c0 + loL;
    const uint32_t cnt = static_cast<uint32_t>(hiRaw) - row0;

    int64_t cE = 0;
    if (uplo == ACLBLAS_UPPER) {
        cE = (e >= 0) ? (static_cast<int64_t>(e) * (static_cast<int64_t>(lda) - 1) + static_cast<int64_t>(k)) :
                        (static_cast<int64_t>(k) + e);
    } else {
        cE = (e >= 0) ? static_cast<int64_t>(e) : (static_cast<int64_t>(e) * (static_cast<int64_t>(lda) - 1));
    }
    /* 行-对角形式用的是单块 UB 手工切分出来的张量（见 InitBuffers） */
    LocalTensor<float> span = spanT;
    LocalTensor<float> xWin = xT;
    LocalTensor<float> yUb = yT;
    LocalTensor<float> aVec = aVecT;
    LocalTensor<float> xv = xvT;
    LocalTensor<uint32_t> offA = offAT;
    LocalTensor<uint32_t> offX = offXT;

    /* 被裁掉的那几 lane 落在有效区间之外：它们的 A 元素可能属于带外空槽、x 下标可能为负，
     * 所以偏移只要求「读在 UB 内」——负的一律夹到 0（读到的是本缓冲区的真实数据，随后被清零）。 */
    const int64_t aRel = cE + static_cast<int64_t>(row0) * lda - static_cast<int64_t>(spanCol0) * lda;
    const int64_t xRel = static_cast<int64_t>(row0) + e - winLo;
    /* aRel 可能为负（对齐下取整后，行块起点比本 pass 的首个有效行还靠前，至多 7 行 =>
     * 至多 -7*lda 个元素）：span 的前置哨兵正好吸收它，偏移因此不必夹断、每个 lane 都读到
     * 正确的元素。 */
    const int64_t aPhys = aRel + static_cast<int64_t>(spanOff);
    const uint32_t aBase = (aPhys > 0) ? static_cast<uint32_t>(aPhys) : 0U;
    const uint32_t xBase = (xRel > 0) ? static_cast<uint32_t>(xRel) : 0U;
    const uint32_t inval = static_cast<uint32_t>(loRaw - static_cast<int64_t>(row0));
    Gather(aVec, span, offA, aBase * BYTES_PER_FLOAT32, cnt);
    Gather(xv, xWin, offX, xBase * BYTES_PER_FLOAT32, cnt);
    if (inval > 0U) {
        Duplicate(aVec, 0.0f, inval);
        Duplicate(xv, 0.0f, inval);
    }
    if (accumulate) {
        MulAddDst(yUb[loL], aVec, xv, cnt);
    } else {
        /* 首条对角线（e == 0）覆盖本 chunk 的每一行，直接写入即可 —— 省掉一次 yUb 清零。
         * 未被它写到的尾部 lane（对齐补齐区）不会被存储（FinalizeChunk 只写 rowCount 行）。 */
        Mul(yUb[loL], aVec, xv, cnt);
    }
}

/*!
 * \brief 行-对角形式的主循环：每个 chunk 只搬一次带 + 一次 x 窗口，然后做 2k+1 条对角线。
 *
 * pass 循环里没有任何跨流水同步（三条指令都在向量流水线上），所以一轮 chunk 只有两次等待：
 * 搬入（MTE2）与写回前（MTE3）。
 */
__aicore__ inline void SsbmvKernel::ProcessDiag()
{
    bool firstChunk = true;
    for (uint32_t c0 = r0; c0 < r1; c0 += chunkRows) {
        const uint32_t c1 = ((c0 + chunkRows) < r1) ? (c0 + chunkRows) : r1;
        winLo = static_cast<int64_t>(c0) - static_cast<int64_t>(k);
        if (!firstChunk) {
            /* 这一组旗标只在多 chunk 复用同一批缓冲时才需要，否则是纯开销。 */
            SyncPipesBetweenChunks();
        }
        firstChunk = false;
        LoadXWindowDiag(c0, c1);
        LoadDiagSpan(c0, c1);
        SyncMte2ToVector();
        /* 先做 e = 0：它覆盖本 chunk 的每一行，于是不需要预先清零累加器。之后按 +e / -e 成对推进。 */
        DiagPass(c0, c1, 0, false);
        for (int32_t e = 1; e <= static_cast<int32_t>(k); e++) {
            DiagPass(c0, c1, e, true);
            DiagPass(c0, c1, -e, true);
        }
        FinalizeChunk(c0, c1);
    }
}

__aicore__ inline void SsbmvKernel::Process()
{
    if (!active) {
        return;
    }

    if (alpha == 0.0f) {
        HandleAlphaZero();
        return;
    }
    if (diagForm == 3U) {
        /* 实验档：只走 Init（缓冲/事件/偏移表），不进任何计算或搬运 —— 量入口成本。 */
        return;
    }
    if (diagForm != 0U) {
        ProcessDiag();
        return;
    }

    LocalTensor<float> yUb = yBuf.Get<float>();
    for (uint32_t c0 = r0; c0 < r1; c0 += chunkRows) {
        const uint32_t c1 = ((c0 + chunkRows) < r1) ? (c0 + chunkRows) : r1;
        const uint32_t rowCount = c1 - c0;
        winLo = static_cast<int64_t>(c0) - static_cast<int64_t>(k);
        SyncPipesBetweenChunks();
        Duplicate(yUb, 0.0f, AlignUp(rowCount));
        /* 列形式两半都从 GM 按段读 x / A（mirror 逐列、scatter 逐列），不需要 x 窗口与 band span。 */
        if (bandForm == 0U) {
            LoadXWindow(c0, c1);
            LoadChunkBand(c0, c1);
        }
        /* x 窗口与 band span 是两条互不依赖的搬运，一起发出去、只在这里等一次。 */
        SyncMte2ToScalar();
        SyncMte2ToVector();
        if (bandForm == 0U) {
            BuildSpanOffsets(rowCount, AlignUp(rowCount));
        }
        if (spanExt != 0U) {
            ScatterChunkFromSpan(c0, c1);
        } else {
            ScatterChunk(c0, c1);
        }
        MirrorChunk(c0, c1);
        FinalizeChunk(c0, c1);
    }
}

/* 拆成两个入口 —— 每个 __global__ 的设备侧代码只含它自己走到的路径：
 * 设备侧取指是固定开销的一项，代码体越大开销越高。 */
extern "C" __global__ __aicore__ void ssbmv_kernel(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, SsbmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workSpace;
    SsbmvKernel op;
    op.Init(a, x, y, tiling);
    op.Process();
}

/* ==========================================================================================
 * 小 n / 窄带专用入口的 kernel 对象：**只含行-对角形式走得到的成员与代码**。
 *
 * 动机：设备侧取指是固定开销的一项，代码体越大开销越高。通用 kernel 把宽带 column 形式、
 * 逐列 scatter、任意步长的 strided 存取等全部回退路径装进同一个函数，入口虽小但代码体大。
 * 这里把行-对角形式单独拆成第三个入口，让 n<=8192 窄带（即绝大多数形状）的设备侧代码体最小。
 * UB 用一个 TBuf 手工切分（InitBuffer 次数是每核固定开销的一部分），事件只取 2 个。
 * ========================================================================================== */
template <bool kPhase>
class SsbmvDiagOnly {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR x, GM_ADDR y, const SsbmvTilingData& tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitUbTensors();
    __aicore__ inline void InitOffsetTables();
    __aicore__ inline void LoadXWindow(uint32_t c0, uint32_t c1);
    __aicore__ inline void LoadPhases(uint32_t c0, uint32_t c1);
    __aicore__ inline void LoadSpan(uint32_t c0, uint32_t c1);
    __aicore__ inline void Pass(uint32_t c0, uint32_t c1, int32_t e, bool accumulate);
    __aicore__ inline void ApplyPassPhase(uint32_t loL, uint32_t cnt, int32_t xRel, uint32_t inval, bool accumulate);
    __aicore__ inline void ApplyPassGather(uint32_t loL, uint32_t cnt, uint32_t xBase, uint32_t inval, bool accumulate);
    __aicore__ inline void Finalize(uint32_t c0, uint32_t c1);

    LocalTensor<float> yT;
    LocalTensor<float> yOldT;
    LocalTensor<float> aVecT;
    LocalTensor<float> xvT;
    LocalTensor<float> xT;
    LocalTensor<float> spanT;
    /*!< phaseForm 时：8 份错位 x 窗口副本拼在一块缓冲里，第 p 份的下标 0 对应窗口下标 p。 */
    LocalTensor<float> phaseT;
    LocalTensor<uint32_t> offAT;
    LocalTensor<uint32_t> offXT;
    GlobalTensor<float> aGM;
    GlobalTensor<float> xGM;
    GlobalTensor<float> yGM;
    event_t evMte2V;
    event_t evVToMte3;
    event_t evMte3V;
    event_t evVtoMte2;

    uint32_t n = 0;
    uint32_t k = 0;
    uint32_t lda = 0;
    uint32_t uplo = ACLBLAS_UPPER;
    uint32_t chunkRows = 0;
    uint32_t blockRows = 0;
    uint32_t spanCol0 = 0;
    uint32_t spanOff = 0;
    uint32_t r0 = 0;
    uint32_t r1 = 0;
    int32_t winLo = 0;
    uint32_t phaseForm = 0;
    uint32_t phaseLen = 0;
    bool active = false;
    float alpha = 1.0f;
    float beta = 0.0f;
};

template <bool kPhase>
__aicore__ inline void SsbmvDiagOnly<kPhase>::Init(GM_ADDR a, GM_ADDR x, GM_ADDR y, const SsbmvTilingData& tiling)
{
    n = tiling.n;
    k = tiling.k;
    lda = tiling.lda;
    uplo = tiling.uplo;
    chunkRows = tiling.chunkRows;
    blockRows = tiling.blockRows;
    alpha = tiling.alpha;
    beta = tiling.beta;
    phaseForm = kPhase ? 1U : 0U;
    InitUbTensors();

    r0 = GetBlockIdx() * blockRows;
    active = (r0 < n);
    if (!active) {
        return;
    }
    r1 = ((r0 + blockRows) < n) ? (r0 + blockRows) : n;
    aGM.SetGlobalBuffer((__gm__ float*)a, static_cast<uint64_t>(n) * lda);
    xGM.SetGlobalBuffer((__gm__ float*)x, n);
    yGM.SetGlobalBuffer((__gm__ float*)y, n);

    /* 首个 chunk 的两次搬运**在这里就发出去**：搬运走 MTE2、与后面建偏移表/清零哨兵的
     * 标量+向量工作互不依赖，早点发出去就能把 DMA 的固定延迟（这两笔数据只有几百字节，
     * 耗时全是启动开销）藏在 Init 后面。等 MTE2 的位置在 Process 里，那里才是真正要用数据的时候。 */
    spanOff = SSBMV_ALIGN_ELEMS * lda; /* LoadSpan 往哨兵之后落数据，先用上它 */
    /* 哨兵（前置零区）只在 LOWER 下会被读到：LOWER 的 cE = e*(lda-1) 可为负，最坏 -7*lda 由哨兵吸收；
     * UPPER 的 cE 恒 >= 0（e>=0 时 e*(lda-1)+k，e<0 时 k+e），且 row0 >= spanCol0 = c0，故 aRel >= 0，
     * 哨兵永远不会被读 —— 省掉这次清零（1004 上是 520 个 lane）。 */
    if (uplo == ACLBLAS_LOWER) {
        Duplicate(spanT, 0.0f, spanOff);
    }
    winLo = static_cast<int32_t>(r0) - static_cast<int32_t>(k);
    const uint32_t firstC1 = ((r0 + chunkRows) < r1) ? (r0 + chunkRows) : r1;
    if constexpr (kPhase) {
        LoadPhases(r0, firstC1);
    } else {
        LoadXWindow(r0, firstC1);
    }
    LoadSpan(r0, firstC1);

    /* 事件号直接取常量：FetchEventID 要走 TPipe 的登记表，而这条路径没有任何需要登记的东西。 */
    evMte2V = EVENT_ID0;
    evVToMte3 = EVENT_ID1;
    evMte3V = EVENT_ID2;
    evVtoMte2 = EVENT_ID3;
    InitOffsetTables();
    if (uplo == ACLBLAS_LOWER) {
        Duplicate(spanT, 0.0f, spanOff);
    }
}

/*!
 * \brief UB 缓冲的手工切分（Init 的前半段）。
 *
 * UB 手工切分：**不用 TPipe/InitBuffer**。TBuf 的构造与 InitBuffer 的登记是每核固定开销的
 * 一部分，而这一段切分对 TPipe 没有任何依赖 —— 每个缓冲的起点都是 32B 的倍数，直接用 UB
 * 字节地址构造张量即可。
 * 偏移与 host 的预算公式（SsbmvDiagUbFloats）逐项对应，改动这里必须同步改那里。 */
template <bool kPhase>
__aicore__ inline void SsbmvDiagOnly<kPhase>::InitUbTensors()
{
    const uint32_t accRows = AlignUp(chunkRows) + SSBMV_UB_SLACK;
    const uint32_t xCnt = AlignUp(chunkRows + 2U * k) + SSBMV_UB_SLACK;
    const uint32_t spanCnt = DiagSpanFloats(chunkRows, k, lda);
    const uint32_t accBytes = accRows * BYTES_PER_FLOAT32;
    const uint32_t xBytes = xCnt * BYTES_PER_FLOAT32;
    yT = LocalTensor<float>(TPosition::VECCALC, 0U, accRows);
    aVecT = LocalTensor<float>(TPosition::VECCALC, 2U * accBytes, accRows);
    xvT = LocalTensor<float>(TPosition::VECCALC, 3U * accBytes, accRows);
    /* 偏移表直接按 uint32 构造（Gather 的偏移就是 uint32 字节数），省掉一次 float→uint32 的重解释。 */
    offAT = LocalTensor<uint32_t>(TPosition::VECCALC, 4U * accBytes, accRows);
    offXT = LocalTensor<uint32_t>(TPosition::VECCALC, 5U * accBytes, accRows);
    xT = LocalTensor<float>(TPosition::VECCALC, 6U * accBytes, xCnt);
    spanT = LocalTensor<float>(TPosition::VECCALC, 6U * accBytes + xBytes, spanCnt);
    if constexpr (kPhase) {
        phaseLen = AlignUp(xCnt);
        phaseT = LocalTensor<float>(
            TPosition::VECCALC, 6U * accBytes + xBytes + spanCnt * BYTES_PER_FLOAT32, SSBMV_PHASE_COUNT * phaseLen);
    }
    /* beta == 0 时 Finalize 不会碰 beta*y 那块缓冲，构造它纯属每次调用的固定开销。 */
    if (beta != 0.0f) {
        yOldT = LocalTensor<float>(TPosition::VECCALC, accBytes, accRows);
    }
}

/*!
 * \brief Gather 偏移表构建（Init 的后半段）。
 *
 * 偏移表只需覆盖一次 chunk 的行数 + 1（Gather 的 lane 数不会超过它），末尾补到 32B 对齐。
 * A 侧那张表直接由 X 侧乘 lda 得到（两侧都是字节偏移），省掉一整次「64 lane 标量播种」——
 * 标量写在这条路径上是可观成本，小 shape 上尤其明显。 */
template <bool kPhase>
__aicore__ inline void SsbmvDiagOnly<kPhase>::InitOffsetTables()
{
    const uint32_t tableRows = AlignUp(chunkRows + 1U);
    if constexpr (kPhase) {
        /* 相位副本路径只有 A 侧需要 Gather（x 侧是对齐直读），所以只建 offAT，
         * 省掉 offXT 的播种与一次 Muls。 */
        BuildOffsetTable(offAT, lda * BYTES_PER_FLOAT32, tableRows);
    } else {
        BuildOffsetTable(offXT, BYTES_PER_FLOAT32, tableRows);
        /* Muls 只支持 half/float/int16/int32，偏移表按 int32 视图参与计算（字节数都 < 2^31）。 */
        Muls(
            offAT.ReinterpretCast<int32_t>(), offXT.ReinterpretCast<int32_t>(), static_cast<int32_t>(lda),
            static_cast<int32_t>(tableRows));
    }
}

template <bool kPhase>
__aicore__ inline void SsbmvDiagOnly<kPhase>::LoadXWindow(uint32_t c0, uint32_t c1)
{
    /* 32 位算术，理由同 Pass()：winLo ∈ [-k, n)、n <= 8192，全在 int32 内。 */
    const uint32_t vLo = (winLo > 0) ? static_cast<uint32_t>(winLo) : 0U;
    const uint32_t hi = c1 + k;
    const uint32_t vHi = (n < hi) ? n : hi;
    if (vHi <= vLo) {
        return;
    }
    const uint32_t lead = vLo - static_cast<uint32_t>(winLo);
    const uint32_t payload = vHi - vLo;
    CopyXWindowAligned(xT, xGM, vLo, lead, payload);
}

/*!
 * \brief phaseForm 的 x 窗口装载：把窗口按 0..7 的错位各存一份。
 *
 * 第 p 份副本的「下标 0」对应窗口下标 p（即 x 下标 winLo + p），于是任何一条对角线要用的
 * 连续切片 x[row0+e .. ) 都能在 δ = (row0+e-winLo) & 7 那份副本里以**下标 (xRel - δ)** 读到，
 * 而 xRel - δ 必为 SSBMV_ALIGN_ELEMS 的倍数 ⇒ 32B 对齐 ⇒ 向量指令可以直接用它当操作数，
 * 不再需要每 pass 一次 Gather（Gather 的开销显著高于普通向量指令）。
 *
 * 头部（x 下标为负的部分）与整块先清零：无效 lane 对应的 x 下标必为负（见 Pass 的注释），
 * 因此它们读到 0；与 A 侧的显式清零相乘仍是 0，不会把 Inf/NaN 带进结果。
 */
template <bool kPhase>
__aicore__ inline void SsbmvDiagOnly<kPhase>::LoadPhases(uint32_t c0, uint32_t c1)
{
    const uint32_t hi = c1 + k;
    const uint32_t vHi = (n < hi) ? n : hi;
    for (uint32_t p = 0U; p < SSBMV_PHASE_COUNT; p++) {
        /* 副本 p 的「下标 0」对应 x 下标 Gp = winLo + p（可为负）。x[0] 因此落在副本下标 -Gp，
         * 它不一定 32B 对齐，于是用 DataCopyPad 的 leftPadding 把有效数据放到「≤ -Gp 的最大 8 倍数」
         * 之后：目的张量的基址对齐、leftPadding 是元素个数（0..7）不要求对齐。
         * 副本下标 0..-Gp 这一段（Gp<0 时）对应负的 x 下标，保持预清零的 0 —— 无效 lane 正好读到这里。 */
        const int32_t gp = winLo + static_cast<int32_t>(p);
        if (gp >= static_cast<int32_t>(vHi)) {
            continue;
        }
        const uint32_t gs = (gp >= 0) ? static_cast<uint32_t>(gp) : 0U; /* GM 读起点 */
        const uint32_t payloadStart = (gp >= 0) ? 0U : static_cast<uint32_t>(-gp);
        if (gs >= vHi || payloadStart >= phaseLen) {
            continue;
        }
        uint32_t payload = vHi - gs;
        if ((payloadStart + payload) > phaseLen) {
            payload = phaseLen - payloadStart; /* 理论上不会发生，保险起见截断 */
        }
        const uint32_t base8 = payloadStart & ~(SSBMV_ALIGN_ELEMS - 1U);
        const uint8_t leftPad = static_cast<uint8_t>(payloadStart - base8);
        const uint32_t used = payloadStart + payload;
        const uint8_t rightPad = static_cast<uint8_t>(AlignUp(used) - used);
        DataCopyExtParams copyParams(1, payload * BYTES_PER_FLOAT32, 0, 0, 0);
        DataCopyPad(
            phaseT[p * phaseLen + base8], xGM[gs], copyParams,
            DataCopyPadExtParams<float>(true, leftPad, rightPad, 0.0f));
        /* 副本下标 < payloadStart 的那一段对应负的 x 下标，必须显式清零（否则无效 lane 会读到上一次的
         * 残留）。只需清这一段（<= 8 个 lane）：pass 的读下标从 (xRel - delta) 起、lane 数 cnt，
         * 最高只到 x 下标 (hiRaw-1+e) <= n-1，永远不会越过 payload 末尾，尾部不需要清。 */
        if (payloadStart > 0U) {
            Duplicate(phaseT[p * phaseLen], 0.0f, payloadStart);
        }
    }
}

template <bool kPhase>
__aicore__ inline void SsbmvDiagOnly<kPhase>::LoadSpan(uint32_t c0, uint32_t c1)
{
    uint32_t sc = c0;
    uint32_t se = c1;
    if (uplo == ACLBLAS_UPPER) {
        se = ((c1 + k) < n) ? (c1 + k) : n;
    } else {
        sc = (c0 > k) ? (c0 - k) : 0U;
    }
    spanCol0 = sc;
    const uint32_t spanLen = lda * (se - sc - 1U) + k + 1U;
    uint32_t done = 0;
    while (done < spanLen) {
        uint32_t piece = spanLen - done;
        if (piece > SSBMV_SPAN_CHUNK) {
            piece = SSBMV_SPAN_CHUNK;
        }
        DataCopyExtParams copyParams(1, piece * BYTES_PER_FLOAT32, 0, 0, 0);
        DataCopyPad(
            spanT[spanOff + done], aGM[static_cast<uint64_t>(lda) * sc + done], copyParams,
            DataCopyPadExtParams<float>());
        done += piece;
    }
}

template <bool kPhase>
__aicore__ inline void SsbmvDiagOnly<kPhase>::Pass(uint32_t c0, uint32_t c1, int32_t e, bool accumulate)
{
    /* 这里全部用 32 位算术：AI Core 标量单元没有 64 位乘法指令，int64_t 的乘要调软件例程，
     * 而这段代码每条对角线都要跑一遍（小 shape 上 pass 数占比不低）。
     * issue #841 修正安全依据：host 并未校验 n <= 8192（n 可达 3e8 量级）。本函数
     * 的乘法项均以 span 窗口内的局部量为界——cE 的 e*(ldaI-1) 与 aRel 的
     * (row0 - spanCol0)*lda 都以准入 span（lda*(chunkRows+k) 级、必须完整装入
     * UB，见 DiagSpanFloats 注释）为上界；row0 本身虽为全局行号（int32 域内），
     * 但只作差参与局部下标；nI 仅参与与 c1i 的 min 截断。GM 侧全局地址换算在
     * 拷贝点用 uint64（见 aGM[static_cast<uint64_t>(lda) * sc]）。后续放宽准入
     * 或改 UB 容量时须同步复核此处。 */
    const int32_t c0i = static_cast<int32_t>(c0);
    const int32_t eAbs = (e >= 0) ? e : -e;
    const int32_t loRaw = (e >= 0) ? c0i : ((eAbs > c0i) ? eAbs : c0i);
    const int32_t nI = static_cast<int32_t>(n);
    const int32_t c1i = static_cast<int32_t>(c1);
    const int32_t hiRaw = (e >= 0) ? (((nI - e) < c1i) ? (nI - e) : c1i) : c1i;
    if (hiRaw <= loRaw) {
        return;
    }
    const uint32_t localRaw = static_cast<uint32_t>(loRaw - c0i);
    const uint32_t loL = localRaw & ~(SSBMV_ALIGN_ELEMS - 1U);
    const uint32_t row0 = c0 + loL;
    const uint32_t cnt = static_cast<uint32_t>(hiRaw) - row0;
    const int32_t ldaI = static_cast<int32_t>(lda);
    const int32_t kI = static_cast<int32_t>(k);
    int32_t cE = 0;
    if (uplo == ACLBLAS_UPPER) {
        cE = (e >= 0) ? (e * (ldaI - 1) + kI) : (kI + e);
    } else {
        cE = (e >= 0) ? e : (e * (ldaI - 1));
    }
    const int32_t row0I = static_cast<int32_t>(row0);
    const int32_t aRel = cE + (row0I - static_cast<int32_t>(spanCol0)) * ldaI;
    const int32_t xRel = row0I + e - static_cast<int32_t>(winLo);
    const int32_t aPhys = aRel + static_cast<int32_t>(spanOff);
    const uint32_t aBase = (aPhys > 0) ? static_cast<uint32_t>(aPhys) : 0U;
    const uint32_t xBase = (xRel > 0) ? static_cast<uint32_t>(xRel) : 0U;
    const uint32_t inval = static_cast<uint32_t>(loRaw - row0I);
    Gather(aVecT, spanT, offAT, aBase * BYTES_PER_FLOAT32, cnt);
    if constexpr (kPhase) {
        ApplyPassPhase(loL, cnt, xRel, inval, accumulate);
    } else {
        ApplyPassGather(loL, cnt, xBase, inval, accumulate);
    }
}

/*!
 * \brief 相位副本路径的取值与累加（Pass 的尾部，kPhase 专用）。
 *
 * x 操作数在 δ = xRel & 7 那份副本里，下标 (xRel - δ) 恒为 32B 对齐。
 * 无效 lane（行 < loRaw）的 x 下标必为负（e<0 时 loRaw = max(|e|,c0)，行 < loRaw ⇒ r+e < 0；
 * e>=0 时 loRaw = c0、inval 恒为 0），副本头部是 0，因此只需清 A 侧即可保证乘积为 0。 */
template <bool kPhase>
__aicore__ inline void SsbmvDiagOnly<kPhase>::ApplyPassPhase(
    uint32_t loL, uint32_t cnt, int32_t xRel, uint32_t inval, bool accumulate)
{
    const uint32_t delta = static_cast<uint32_t>(xRel) & (SSBMV_ALIGN_ELEMS - 1U);
    if (inval > 0U) {
        Duplicate(aVecT, 0.0f, inval);
    }
    /* 副本 δ 的下标 0 对应 x 下标 Gδ = floor8(winLo + δ)，故本 pass 要读的下标是
     * (row0 + e) - Gδ；由 Gδ ≡ winLo + δ (mod 8) 与 (row0+e) ≡ winLo + δ (mod 8) 可知它是 8 的倍数。*/
    const uint32_t xIdx = delta * phaseLen + (static_cast<uint32_t>(xRel) - delta);
    if (accumulate) {
        MulAddDst(yT[loL], aVecT, phaseT[xIdx], cnt);
    } else {
        Mul(yT[loL], aVecT, phaseT[xIdx], cnt);
    }
}

/*! \brief 逐 pass Gather 路径的取值与累加（Pass 的尾部，非相位副本路径）。 */
template <bool kPhase>
__aicore__ inline void SsbmvDiagOnly<kPhase>::ApplyPassGather(
    uint32_t loL, uint32_t cnt, uint32_t xBase, uint32_t inval, bool accumulate)
{
    Gather(xvT, xT, offXT, xBase * BYTES_PER_FLOAT32, cnt);
    if (inval > 0U) {
        Duplicate(aVecT, 0.0f, inval);
        Duplicate(xvT, 0.0f, inval);
    }
    if (accumulate) {
        MulAddDst(yT[loL], aVecT, xvT, cnt);
    } else {
        Mul(yT[loL], aVecT, xvT, cnt);
    }
}

template <bool kPhase>
__aicore__ inline void SsbmvDiagOnly<kPhase>::Finalize(uint32_t c0, uint32_t c1)
{
    const uint32_t rowCount = c1 - c0;
    const uint32_t rowCountAligned = AlignUp(rowCount);
    Muls(yT, yT, alpha, rowCountAligned);
    if (beta != 0.0f) {
        DataCopyExtParams copyParams(1, static_cast<uint32_t>(rowCount * BYTES_PER_FLOAT32), 0, 0, 0);
        DataCopyPad(
            yOldT, yGM[c0], copyParams,
            DataCopyPadExtParams<float>(true, 0, static_cast<uint8_t>(rowCountAligned - rowCount), 0.0f));
        SetFlag<HardEvent::MTE2_V>(evMte2V);
        WaitFlag<HardEvent::MTE2_V>(evMte2V);
        Muls(yOldT, yOldT, beta, rowCountAligned);
        Add(yT, yT, yOldT, rowCountAligned);
    }
    SetFlag<HardEvent::V_MTE3>(evVToMte3);
    WaitFlag<HardEvent::V_MTE3>(evVToMte3);
    DataCopyExtParams outParams(1, static_cast<uint32_t>(rowCount * BYTES_PER_FLOAT32), 0, 0, 0);
    DataCopyPad(yGM[c0], yT, outParams);
}

template <bool kPhase>
__aicore__ inline void SsbmvDiagOnly<kPhase>::Process()
{
    if (!active) {
        return;
    }
    bool firstChunk = true;
    for (uint32_t c0 = r0; c0 < r1; c0 += chunkRows) {
        const uint32_t c1 = ((c0 + chunkRows) < r1) ? (c0 + chunkRows) : r1;
        if (!firstChunk) {
            /* 缓冲要复用：先等上一轮的计算与写回读完 xT/spanT，再覆盖它们。 */
            SetFlag<HardEvent::MTE3_V>(evMte3V);
            WaitFlag<HardEvent::MTE3_V>(evMte3V);
            SetFlag<HardEvent::V_MTE2>(evVtoMte2);
            WaitFlag<HardEvent::V_MTE2>(evVtoMte2);
            winLo = static_cast<int32_t>(c0) - static_cast<int32_t>(k);
            if constexpr (kPhase) {
                LoadPhases(c0, c1);
            } else {
                LoadXWindow(c0, c1);
            }
            LoadSpan(c0, c1);
        }
        firstChunk = false;
        /* 首 chunk 的搬运已在 Init 里发出，这里只做等待。 */
        SetFlag<HardEvent::MTE2_V>(evMte2V);
        WaitFlag<HardEvent::MTE2_V>(evMte2V);
        Pass(c0, c1, 0, false);
        for (int32_t e = 1; e <= static_cast<int32_t>(k); e++) {
            Pass(c0, c1, e, true);
            Pass(c0, c1, -e, true);
        }
        Finalize(c0, c1);
    }
}

extern "C" __global__ __aicore__ void ssbmv_kernel_diag_only(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, SsbmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workSpace;
    SsbmvDiagOnly<false> op; /* 非相位：设备侧代码里没有相位副本那一段 */
    op.Init(a, x, y, tiling);
    op.Process();
}

extern "C" __global__ __aicore__ void ssbmv_kernel_diag_only(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, SsbmvTilingData tiling);

/* 相位副本单独一个入口：设备侧代码体积是固定开销的一项。 */
extern "C" __global__ __aicore__ void ssbmv_kernel_diag_phase(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, SsbmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workSpace;
    SsbmvDiagOnly<true> op;
    op.Init(a, x, y, tiling);
    op.Process();
}

extern "C" __global__ __aicore__ void ssbmv_kernel_diag_phase(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, SsbmvTilingData tiling);

void ssbmv_kernel_do(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, uint32_t numBlocks, const SsbmvTilingData& tiling,
    void* stream)
{
    /* 行-对角形式走专用入口：它的设备侧代码只含这条路径（通用入口里还装着宽带 column 形式、
     * 逐列 scatter、任意步长的 strided 存取等全部回退路径）。设备侧取指是固定开销的一项，
     * 代码体越小越省。其余形状仍走通用入口。 */
    if (tiling.diagForm != 0U && tiling.alpha != 0.0f && tiling.incy == 1) {
        /* 只接「alpha != 0 且 incy == 1」：alpha == 0 的位精确路径与任意步长的 strided 存回
         * 都留在通用入口里，这里不重复实现。相位副本与非相位各走一个入口，代码体积最小。 */
        if (tiling.phaseForm != 0U) {
            ssbmv_kernel_diag_phase<<<numBlocks, nullptr, stream>>>(a, x, y, workSpace, tiling);
        } else {
            ssbmv_kernel_diag_only<<<numBlocks, nullptr, stream>>>(a, x, y, workSpace, tiling);
        }
        return;
    }
    ssbmv_kernel<<<numBlocks, nullptr, stream>>>(a, x, y, workSpace, tiling);
}
