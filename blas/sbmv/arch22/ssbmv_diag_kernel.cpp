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
 * \file ssbmv_diag_kernel.cpp
 * \brief Diagonal and contiguous diagonal kernels for aclblasSsbmv on arch22.
 */

#include "cann_ops_blas_common.h"
#include "ssbmv_kernel_internal.h"

using namespace AscendC;
using namespace ssbmv;

/* ==========================================================================================
 * 小 n / 窄带专用入口的 kernel 对象：**只含行-对角形式走得到的成员与代码**。
 *
 * 动机：设备侧取指是固定开销的一项，代码体越大开销越高。通用 kernel 把宽带 column 形式、
 * 逐列 scatter、任意步长的 strided 存取等全部回退路径装进同一个函数，入口虽小但代码体大。
 * 这里把行-对角形式单独拆成第三个入口，让 n<=8192 窄带（即绝大多数形状）的设备侧代码体最小。
 * UB 用一个 TBuf 手工切分（InitBuffer 次数是每核固定开销的一部分），事件只取 2 个。
 * ========================================================================================== */
template <bool kPhase, SsbmvRowDotMode mode = SsbmvRowDotMode::Disabled, bool kClipGroups = false>
class SsbmvDiagKernel {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR x, GM_ADDR y, const SsbmvTilingData& tiling);
    __aicore__ inline void Process();

private:
    __aicore__ inline void InitUbTensors();
    __aicore__ inline bool UseRowDots() const
    {
        if constexpr (mode == SsbmvRowDotMode::Edges) {
            return r0 < k || k > n - r1;
        }
        return mode == SsbmvRowDotMode::AllRows;
    }
    template <bool kUpper>
    __aicore__ inline void ComputeInterior(uint32_t c0, uint32_t count);
    template <bool kUpper>
    __aicore__ inline void ComputeGroupedInterior(uint32_t c0, uint32_t count);
    template <bool kUpper>
    __aicore__ inline void ComputeGroupedEdge(uint32_t c0, bool leftEdge);
    template <bool kUpper>
    __aicore__ inline void ProcessGroupedTile();
    __aicore__ inline void ComputeRowDots();
    __aicore__ inline void SumRowTerms(
        const LocalTensor<float>& dst, const LocalTensor<float>& src, const LocalTensor<float>& work, uint32_t count);
    __aicore__ inline bool UseGroupedInterior(uint32_t c0, uint32_t c1) const;
    __aicore__ inline void LoadInputs(uint32_t c0, uint32_t c1);
    __aicore__ inline void SelectRowRange(const SsbmvTilingData& tiling);
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
    LocalTensor<float> groupedScratch;
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
    bool groupedDiag = false;
    bool groupedXReuse = false;
    bool active = false;
    float alpha = 1.0f;
    float beta = 0.0f;
};

template <bool kPhase, SsbmvRowDotMode mode, bool kClipGroups>
__aicore__ inline void SsbmvDiagKernel<kPhase, mode, kClipGroups>::Init(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, const SsbmvTilingData& tiling)
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
    groupedDiag = tiling.groupedDiag != 0U;
    groupedXReuse = tiling.groupedXReuse != 0U;
    if constexpr (mode != SsbmvRowDotMode::Edges) {
        InitUbTensors();
    }

    SelectRowRange(tiling);
    active = (r0 < r1);
    if (!active) {
        return;
    }
    if constexpr (mode == SsbmvRowDotMode::Edges) {
        if (UseRowDots()) {
            chunkRows = SSBMV_ROW_DOT_MAX_ROWS;
        }
        InitUbTensors();
    }
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
    winLo = static_cast<int32_t>(r0) - static_cast<int32_t>(k);
    const uint32_t firstC1 = ((r0 + chunkRows) < r1) ? (r0 + chunkRows) : r1;
    LoadInputs(r0, firstC1);

    /* 事件号直接取常量：FetchEventID 要走 TPipe 的登记表，而这条路径没有任何需要登记的东西。 */
    evMte2V = EVENT_ID0;
    evVToMte3 = EVENT_ID1;
    evMte3V = EVENT_ID2;
    evVtoMte2 = EVENT_ID3;
    if (UseRowDots()) {
        return;
    }
    InitOffsetTables();
    if constexpr (!kClipGroups) {
        if (uplo == ACLBLAS_LOWER) {
            Duplicate(spanT, 0.0f, spanOff);
        }
    }
}

template <bool kPhase, SsbmvRowDotMode mode, bool kClipGroups>
__aicore__ inline void SsbmvDiagKernel<kPhase, mode, kClipGroups>::SelectRowRange(const SsbmvTilingData& tiling)
{
    const uint32_t block = GetBlockIdx();
    const uint32_t left = tiling.leftEdgeRows;
    const uint32_t right = tiling.rightEdgeStart;
    if constexpr (mode == SsbmvRowDotMode::Edges) {
        constexpr uint32_t edgeRows = SSBMV_ROW_DOT_MAX_ROWS;
        const uint32_t leftBlocks = (left + edgeRows - 1U) / edgeRows;
        const uint32_t bodyBlocks = (right - left + blockRows - 1U) / blockRows;
        uint32_t limit = n;
        uint32_t rows = edgeRows;
        if (block < leftBlocks) {
            r0 = block * edgeRows;
            limit = left;
        } else if (block - leftBlocks < bodyBlocks) {
            r0 = left + (block - leftBlocks) * blockRows;
            limit = right;
            rows = blockRows;
        } else {
            r0 = right + (block - leftBlocks - bodyBlocks) * edgeRows;
        }
        r1 = r0 + rows < limit ? r0 + rows : limit;
        return;
    }
    if (left == 0U) {
        r0 = block * blockRows;
        r1 = r0 + blockRows < n ? r0 + blockRows : n;
    } else if (block == 0U) {
        r0 = 0U;
        r1 = left;
    } else {
        const uint32_t first = left + (block - 1U) * blockRows;
        r0 = first < right ? first : right;
        r1 = first >= right ? n : (first + blockRows < right ? first + blockRows : right);
    }
}

/*!
 * \brief UB 缓冲的手工切分（Init 的前半段）。
 *
 * UB 手工切分：**不用 TPipe/InitBuffer**。TBuf 的构造与 InitBuffer 的登记是每核固定开销的
 * 一部分，而这一段切分对 TPipe 没有任何依赖 —— 每个缓冲的起点都是 32B 的倍数，直接用 UB
 * 字节地址构造张量即可。
 * 偏移与 host 的预算公式（SsbmvDiagUbFloats）逐项对应，改动这里必须同步改那里。 */
template <bool kPhase, SsbmvRowDotMode mode, bool kClipGroups>
__aicore__ inline void SsbmvDiagKernel<kPhase, mode, kClipGroups>::InitUbTensors()
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
    const uint32_t scratchStart = 6U * accBytes + xBytes + spanCnt * BYTES_PER_FLOAT32;
    // LoadInputs builds phase replicas only for tiles that do not use the
    // grouped layout. Process waits for Vector before refilling the next tile.
    if constexpr (kPhase && !kClipGroups) {
        if (!UseRowDots()) {
            phaseLen = AlignUp(xCnt);
            phaseT = LocalTensor<float>(TPosition::VECCALC, scratchStart, SSBMV_PHASE_COUNT * phaseLen);
        }
    }
    if (UseRowDots()) {
        const uint32_t dotLanes = AlignUp(k + 1U) + SSBMV_UB_SLACK;
        const uint32_t scratchFloats = 5U * dotLanes + (chunkRows + 1U) * SSBMV_ALIGN_ELEMS;
        groupedScratch = LocalTensor<float>(TPosition::VECCALC, scratchStart, scratchFloats);
    } else if (groupedDiag) {
        constexpr uint32_t width = SSBMV_DIAG_GROUP_WIDTH;
        const uint32_t haloFloats = groupedXReuse ? 2U * (2U * k + 1U - width) * width : 0U;
        const uint32_t scratchFloats = AlignUp(chunkRows) * width * SSBMV_DIAG_GROUP_BUFFERS + haloFloats;
        groupedScratch = LocalTensor<float>(TPosition::VECCALC, scratchStart, scratchFloats);
    }
    /* beta == 0 时 Finalize 不会碰 beta*y 那块缓冲，构造它纯属每次调用的固定开销。 */
    yOldT = LocalTensor<float>(TPosition::VECCALC, accBytes, accRows);
}

/*!
 * \brief Gather 偏移表构建（Init 的后半段）。
 *
 * 偏移表只需覆盖一次 chunk 的行数 + 1（Gather 的 lane 数不会超过它），末尾补到 32B 对齐。
 * A 侧那张表直接由 X 侧乘 lda 得到（两侧都是字节偏移），省掉一次标量播种和向量递推——
 * 标量写在这条路径上是可观成本，小 shape 上尤其明显。 */
template <bool kPhase, SsbmvRowDotMode mode, bool kClipGroups>
__aicore__ inline void SsbmvDiagKernel<kPhase, mode, kClipGroups>::InitOffsetTables()
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

template <bool kPhase, SsbmvRowDotMode mode, bool kClipGroups>
__aicore__ inline bool SsbmvDiagKernel<kPhase, mode, kClipGroups>::UseGroupedInterior(uint32_t c0, uint32_t c1) const
{
    return groupedDiag && (c1 - c0) % SSBMV_DIAG_GROUP_WIDTH == 0U && c0 >= k && k <= n - c1;
}

template <bool kPhase, SsbmvRowDotMode mode, bool kClipGroups>
__aicore__ inline void SsbmvDiagKernel<kPhase, mode, kClipGroups>::LoadInputs(uint32_t c0, uint32_t c1)
{
    if constexpr (kClipGroups) {
        LoadXWindow(c0, c1);
    } else if constexpr (kPhase) {
        if (UseRowDots() || UseGroupedInterior(c0, c1)) {
            LoadXWindow(c0, c1);
        } else {
            LoadPhases(c0, c1);
        }
    } else {
        LoadXWindow(c0, c1);
    }
    LoadSpan(c0, c1);
}

template <bool kPhase, SsbmvRowDotMode mode, bool kClipGroups>
__aicore__ inline void SsbmvDiagKernel<kPhase, mode, kClipGroups>::LoadXWindow(uint32_t c0, uint32_t c1)
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
template <bool kPhase, SsbmvRowDotMode mode, bool kClipGroups>
__aicore__ inline void SsbmvDiagKernel<kPhase, mode, kClipGroups>::LoadPhases(uint32_t c0, uint32_t c1)
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
        /* DataCopyPad 已清零从 base8 到 payloadStart 的左侧填充。Vector 只清零之前的完整块，
         * 使两条流水的写入地址按 32B 分开。 */
        if (base8 > 0U) {
            Duplicate(phaseT[p * phaseLen], 0.0f, base8);
        }
    }
}

template <bool kPhase, SsbmvRowDotMode mode, bool kClipGroups>
__aicore__ inline void SsbmvDiagKernel<kPhase, mode, kClipGroups>::LoadSpan(uint32_t c0, uint32_t c1)
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

template <bool kPhase, SsbmvRowDotMode mode, bool kClipGroups>
__aicore__ inline void SsbmvDiagKernel<kPhase, mode, kClipGroups>::Pass(
    uint32_t c0, uint32_t c1, int32_t e, bool accumulate)
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
template <bool kPhase, SsbmvRowDotMode mode, bool kClipGroups>
__aicore__ inline void SsbmvDiagKernel<kPhase, mode, kClipGroups>::ApplyPassPhase(
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
template <bool kPhase, SsbmvRowDotMode mode, bool kClipGroups>
__aicore__ inline void SsbmvDiagKernel<kPhase, mode, kClipGroups>::ApplyPassGather(
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

template <bool kPhase, SsbmvRowDotMode mode, bool kClipGroups>
__aicore__ inline void SsbmvDiagKernel<kPhase, mode, kClipGroups>::Finalize(uint32_t c0, uint32_t c1)
{
    const uint32_t rowCount = c1 - c0;
    const uint32_t rowCountAligned = AlignUp(rowCount);
    if (alpha != 1.0f) {
        Muls(yT, yT, alpha, rowCountAligned);
    }
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

// A full interior tile has no clipped lanes. The two sides of the
// symmetric band can advance by fixed steps and use independent accumulators.
template <bool kPhase, SsbmvRowDotMode mode, bool kClipGroups>
template <bool kUpper>
__aicore__ inline void SsbmvDiagKernel<kPhase, mode, kClipGroups>::ComputeInterior(uint32_t c0, uint32_t count)
{
    const uint32_t main = spanOff + (c0 - spanCol0) * lda + (kUpper ? k : 0U);
    const uint32_t rowX = c0 - static_cast<uint32_t>(winLo);
    const uint32_t forwardStep = kUpper ? lda - 1U : 1U;
    const uint32_t backwardStep = kUpper ? 1U : lda - 1U;

    Gather(aVecT, spanT, offAT, main * BYTES_PER_FLOAT32, count);
    if constexpr (kPhase) {
        ApplyPassPhase(0U, count, static_cast<int32_t>(rowX), 0U, false);
    } else {
        ApplyPassGather(0U, count, rowX, 0U, false);
    }
    Duplicate(yOldT, 0.0f, count);
    uint32_t forward = main;
    uint32_t backward = main;
    for (uint32_t e = 1U; e <= k; ++e) {
        forward += forwardStep;
        backward -= backwardStep;
        const uint32_t xRight = rowX + e;
        const uint32_t xLeft = rowX - e;
        Gather(aVecT, spanT, offAT, forward * BYTES_PER_FLOAT32, count);
        if constexpr (kPhase) {
            const uint32_t phase = xRight % SSBMV_ALIGN_ELEMS;
            MulAddDst(yT, aVecT, phaseT[phase * phaseLen + xRight - phase], count);
        } else {
            Gather(xvT, xT, offXT, xRight * BYTES_PER_FLOAT32, count);
            MulAddDst(yT, aVecT, xvT, count);
        }
        Gather(aVecT, spanT, offAT, backward * BYTES_PER_FLOAT32, count);
        if constexpr (kPhase) {
            const uint32_t phase = xLeft % SSBMV_ALIGN_ELEMS;
            MulAddDst(yOldT, aVecT, phaseT[phase * phaseLen + xLeft - phase], count);
        } else {
            Gather(xvT, xT, offXT, xLeft * BYTES_PER_FLOAT32, count);
            MulAddDst(yOldT, aVecT, xvT, count);
        }
    }
    Add(yT, yT, yOldT, count);
    // Finalize may DMA the original y into the second accumulator.
    if (beta != 0.0f) {
        SetFlag<HardEvent::V_MTE2>(evVtoMte2);
        WaitFlag<HardEvent::V_MTE2>(evVtoMte2);
    }
}

template <bool kPhase, SsbmvRowDotMode mode, bool kClipGroups>
template <bool kUpper>
__aicore__ inline void SsbmvDiagKernel<kPhase, mode, kClipGroups>::ComputeGroupedInterior(uint32_t c0, uint32_t count)
{
    constexpr uint32_t width = SSBMV_DIAG_GROUP_WIDTH;
    const uint32_t lanes = count * width;
    const uint32_t expandedRows = count + 2U * k + 1U - width;
    const bool reuseX = groupedXReuse && expandedRows < count * (2U * (k / width));
    const uint32_t xLanes = reuseX ? expandedRows * width : lanes;
    const auto forwardOffsets = groupedScratch.ReinterpretCast<uint32_t>();
    const auto backwardOffsets = groupedScratch[lanes].ReinterpretCast<uint32_t>();
    const auto xOffsets = groupedScratch[2U * lanes].ReinterpretCast<uint32_t>();
    const auto aTerms = groupedScratch[2U * lanes + xLanes];
    const auto xTerms = groupedScratch[3U * lanes + xLanes];
    const auto sums = groupedScratch[3U * lanes + 2U * xLanes];
    const uint32_t aStep = lda * BYTES_PER_FLOAT32;
    const uint32_t forwardStep = kUpper ? lda - 1U : 1U;
    const uint32_t backwardStep = kUpper ? 1U : lda - 1U;
    const uint32_t main = spanOff + (c0 - spanCol0) * lda + (kUpper ? k : 0U);
    const uint32_t rowX = c0 - static_cast<uint32_t>(winLo);
    const bool directMainX = rowX % SSBMV_ALIGN_ELEMS == 0U;
    const uint32_t grouped = k / width * width;
    const uint32_t remainder = k - grouped;
    // Small leftovers use fewer active lanes with the per-diagonal path.
    const bool maskedRemainder = remainder >= width / 2U;
    [[maybe_unused]] const bool needXTable = !directMainX || (remainder != 0U && !maskedRemainder);
    SeedOffsetTable<width>(forwardOffsets, aStep, lanes, forwardStep * BYTES_PER_FLOAT32);
    SeedOffsetTable<width>(backwardOffsets, aStep, lanes, backwardStep * BYTES_PER_FLOAT32);
    SeedOffsetTable<width>(xOffsets, BYTES_PER_FLOAT32, xLanes, BYTES_PER_FLOAT32);
    if constexpr (kPhase) {
        if (needXTable) {
            SeedOffsetTable(offXT, BYTES_PER_FLOAT32, count, 0U);
        }
    }
    SetFlag<HardEvent::S_V>(EVENT_ID4);
    WaitFlag<HardEvent::S_V>(EVENT_ID4);
    ExtendOffsetTable<width>(forwardOffsets, aStep, lanes);
    ExtendOffsetTable<width>(backwardOffsets, aStep, lanes);
    ExtendOffsetTable<width>(xOffsets, BYTES_PER_FLOAT32, xLanes);
    if constexpr (kPhase) {
        if (needXTable) {
            ExtendOffsetTable(offXT, BYTES_PER_FLOAT32, count);
        }
    }
    if (reuseX) {
        // Each aligned UB block holds x[i..i+7] for one window position i.
        // A group starting at i reads count such blocks sequentially.
        Gather(xTerms, xT, xOffsets, 0U, xLanes);
    }
    Duplicate(sums, 0.0f, lanes);
    for (uint32_t first = 0; first < grouped; first += width) {
        // For each row, gather offsets e = first + [1, 8].
        const uint32_t positive = first + 1U;
        Gather(aTerms, spanT, forwardOffsets, (main + positive * forwardStep) * BYTES_PER_FLOAT32, lanes);
        const uint32_t right = rowX + positive;
        if (!reuseX) {
            Gather(xTerms, xT, xOffsets, right * BYTES_PER_FLOAT32, lanes);
        }
        MulAddDst(sums, aTerms, xTerms[reuseX ? right * width : 0U], lanes);

        // Negative groups advance towards the diagonal, so all offsets in
        // the UB table are nonnegative relative to the group's first term.
        const uint32_t distance = grouped - first;
        Gather(aTerms, spanT, backwardOffsets, (main - distance * backwardStep) * BYTES_PER_FLOAT32, lanes);
        const uint32_t left = rowX - distance;
        if (!reuseX) {
            Gather(xTerms, xT, xOffsets, left * BYTES_PER_FLOAT32, lanes);
        }
        MulAddDst(sums, aTerms, xTerms[reuseX ? left * width : 0U], lanes);
    }
    if (maskedRemainder) {
        // Both windows stay in band. Mask out the diagonals already
        // accumulated by the full groups, including any non-finite terms.
        constexpr uint64_t repeatedByte = 0x0101010101010101ULL;
        constexpr uint32_t fullByte = (1U << width) - 1U;
        const uint32_t lowBits = (1U << remainder) - 1U;
        const uint64_t positiveMask[2] = {(fullByte ^ (fullByte >> remainder)) * repeatedByte, 0U};
        const uint64_t negativeMask[2] = {lowBits * repeatedByte, 0U};
        const BinaryRepeatParams strides(1U, 1U, 1U, width, width, width);
        const uint8_t repeats = static_cast<uint8_t>(count / width);
        const uint32_t positive = k - width + 1U;
        const uint32_t right = rowX + positive;
        Gather(aTerms, spanT, forwardOffsets, (main + positive * forwardStep) * BYTES_PER_FLOAT32, lanes);
        if (!reuseX) {
            Gather(xTerms, xT, xOffsets, right * BYTES_PER_FLOAT32, lanes);
        }
        MulAddDst(sums, aTerms, xTerms[reuseX ? right * width : 0U], positiveMask, repeats, strides);

        const uint32_t left = rowX - k;
        Gather(aTerms, spanT, backwardOffsets, (main - k * backwardStep) * BYTES_PER_FLOAT32, lanes);
        if (!reuseX) {
            Gather(xTerms, xT, xOffsets, left * BYTES_PER_FLOAT32, lanes);
        }
        MulAddDst(sums, aTerms, xTerms[reuseX ? left * width : 0U], negativeMask, repeats, strides);
    }
    BlockReduceSum(yT, sums, count / width, 64, 1, 1, 8);

    Gather(aVecT, spanT, offAT, main * BYTES_PER_FLOAT32, count);
    if (directMainX) {
        MulAddDst(yT, aVecT, xT[rowX], count);
    } else {
        ApplyPassGather(0U, count, rowX, 0U, true);
    }
    if (!maskedRemainder) {
        for (uint32_t e = grouped + 1U; e <= k; ++e) {
            Gather(aVecT, spanT, offAT, (main + e * forwardStep) * BYTES_PER_FLOAT32, count);
            ApplyPassGather(0U, count, rowX + e, 0U, true);
            Gather(aVecT, spanT, offAT, (main - e * backwardStep) * BYTES_PER_FLOAT32, count);
            ApplyPassGather(0U, count, rowX - e, 0U, true);
        }
    }
}

// Each clipped group is a rectangular run and one 8 x 8 triangle.
// Aligned edge ownership makes the triangle mask and its x values
// independent of the diagonal group's position.
template <bool kPhase, SsbmvRowDotMode mode, bool kClipGroups>
template <bool kUpper>
__aicore__ inline void SsbmvDiagKernel<kPhase, mode, kClipGroups>::ComputeGroupedEdge(uint32_t c0, bool leftEdge)
{
    constexpr uint32_t width = SSBMV_DIAG_GROUP_WIDTH;
    constexpr uint32_t triangleSize = width * width;
    const uint32_t lanes = k * width;
    const uint32_t xLanes = (3U * k + 1U - width) * width;
    const uint32_t validXLanes = (2U * k + 1U - width) * width;
    const auto forwardOffsets = groupedScratch.ReinterpretCast<uint32_t>();
    const auto backwardOffsets = groupedScratch[lanes].ReinterpretCast<uint32_t>();
    const auto xOffsets = groupedScratch[2U * lanes].ReinterpretCast<uint32_t>();
    const auto aTerms = groupedScratch[2U * lanes + xLanes];
    const auto xTerms = groupedScratch[3U * lanes + xLanes];
    const auto sums = groupedScratch[3U * lanes + 2U * xLanes];
    const auto triangleX = xTerms[leftEdge ? 0U : validXLanes];

    const uint32_t xFirst = leftEdge ? k : 0U;
    const uint32_t triangleXFirst = leftEdge ? k - width : 2U * k - width + 1U;
    const uint64_t triangleMask[2] = {leftEdge ? 0xFEFCF8F0E0C08000ULL : 0x000103070F1F3F7FULL, 0U};
    const uint32_t forwardStep = kUpper ? lda - 1U : 1U;
    const uint32_t backwardStep = kUpper ? 1U : lda - 1U;
    const uint32_t aStep = lda * BYTES_PER_FLOAT32;
    const uint32_t main = spanOff + (c0 - spanCol0) * lda + (kUpper ? k : 0U);
    SeedOffsetTable<width>(forwardOffsets, aStep, lanes, forwardStep * BYTES_PER_FLOAT32);
    SeedOffsetTable<width>(backwardOffsets, aStep, lanes, backwardStep * BYTES_PER_FLOAT32);
    SeedOffsetTable<width>(xOffsets, BYTES_PER_FLOAT32, validXLanes, BYTES_PER_FLOAT32);
    SetFlag<HardEvent::S_V>(EVENT_ID4);
    WaitFlag<HardEvent::S_V>(EVENT_ID4);
    ExtendOffsetTable<width>(forwardOffsets, aStep, lanes);
    ExtendOffsetTable<width>(backwardOffsets, aStep, lanes);
    ExtendOffsetTable<width>(xOffsets, BYTES_PER_FLOAT32, validXLanes);
    Gather(xTerms[xFirst * width], xT, xOffsets, xFirst * BYTES_PER_FLOAT32, validXLanes);
    Gather(triangleX, xT, xOffsets, triangleXFirst * BYTES_PER_FLOAT32, triangleMask, 1U, width);
    Duplicate(sums, 0.0f, lanes);

    const BinaryRepeatParams triangleStrides(1U, 1U, 1U, width, width, width);
    for (uint32_t first = 0U; first < k; first += width) {
        const uint32_t positive = first + 1U;
        const uint32_t distance = k - first;
        const uint32_t forwardBase = main + positive * forwardStep;
        if (leftEdge) {
            Gather(aTerms, spanT, forwardOffsets, forwardBase * BYTES_PER_FLOAT32, lanes);
            MulAddDst(sums, aTerms, xTerms[(k + positive) * width], lanes);

            const uint32_t start = distance - width;
            const uint32_t triangleBase = main + start * lda - distance * backwardStep;
            Gather(aTerms, spanT, backwardOffsets, triangleBase * BYTES_PER_FLOAT32, triangleMask, 1U, width);
            MulAddDst(sums[start * width], aTerms, triangleX, triangleMask, 1U, triangleStrides);
            if (distance != k) {
                const uint32_t fullLanes = (k - distance) * width;
                const uint32_t fullBase = triangleBase + width * lda;
                Gather(aTerms, spanT, backwardOffsets, fullBase * BYTES_PER_FLOAT32, fullLanes);
                MulAddDst(sums[distance * width], aTerms, xTerms[k * width], fullLanes);
            }
        } else {
            const uint32_t fullRows = k - first - width;
            if (fullRows != 0U) {
                Gather(aTerms, spanT, forwardOffsets, forwardBase * BYTES_PER_FLOAT32, fullRows * width);
                MulAddDst(sums, aTerms, xTerms[(k + positive) * width], fullRows * width);
            }
            const uint32_t triangleBase = forwardBase + fullRows * lda;
            Gather(aTerms, spanT, forwardOffsets, triangleBase * BYTES_PER_FLOAT32, triangleMask, 1U, width);
            MulAddDst(sums[fullRows * width], aTerms, triangleX, triangleMask, 1U, triangleStrides);

            const uint32_t backwardBase = main - distance * backwardStep;
            Gather(aTerms, spanT, backwardOffsets, backwardBase * BYTES_PER_FLOAT32, lanes);
            MulAddDst(sums, aTerms, xTerms[(k - distance) * width], lanes);
        }
    }
    BlockReduceSum(yT, sums, lanes / triangleSize, triangleSize, 1, 1, width);
    Gather(aVecT, spanT, offAT, main * BYTES_PER_FLOAT32, k);
    MulAddDst(yT, aVecT, xT[k], k);
}

template <bool kPhase, SsbmvRowDotMode mode, bool kClipGroups>
template <bool kUpper>
__aicore__ inline void SsbmvDiagKernel<kPhase, mode, kClipGroups>::ProcessGroupedTile()
{
    if (r0 == 0U || r1 == n) {
        ComputeGroupedEdge<kUpper>(r0, r0 == 0U);
    } else {
        ComputeGroupedInterior<kUpper>(r0, r1 - r0);
    }
}

template <bool kPhase, SsbmvRowDotMode mode, bool kClipGroups>
__aicore__ inline void SsbmvDiagKernel<kPhase, mode, kClipGroups>::SumRowTerms(
    const LocalTensor<float>& dst, const LocalTensor<float>& src, const LocalTensor<float>& work, uint32_t count)
{
    constexpr uint32_t vectorLanes = 64U;
    if (count <= vectorLanes) {
        WholeReduceSum(dst, src, count, 1, 1, 1, 8);
        return;
    }
    const uint32_t repeats = count / vectorLanes;
    const uint32_t tail = count % vectorLanes;
    BlockReduceSum(work, src, repeats, vectorLanes, 1, 1, 8);
    if (tail != 0U) {
        BlockReduceSum(work[repeats * SSBMV_ALIGN_ELEMS], src[repeats * vectorLanes], 1, tail, 1, 1, 8);
    }
    // The host caps each side at 512 terms, so the partials fit in one repeat.
    WholeReduceSum(dst, work, AlignUp(count) / SSBMV_ALIGN_ELEMS, 1, 1, 1, 8);
}

template <bool kPhase, SsbmvRowDotMode mode, bool kClipGroups>
__aicore__ inline void SsbmvDiagKernel<kPhase, mode, kClipGroups>::ComputeRowDots()
{
    const uint32_t dotLanes = AlignUp(k + 1U) + SSBMV_UB_SLACK;
    const auto aTerms = groupedScratch;
    const auto xTerms = groupedScratch[dotLanes];
    const auto work = groupedScratch[2U * dotLanes];
    const auto linear = groupedScratch[3U * dotLanes].ReinterpretCast<uint32_t>();
    const auto strided = groupedScratch[4U * dotLanes].ReinterpretCast<uint32_t>();
    const auto sums = groupedScratch[5U * dotLanes];
    const auto rightSum = sums[chunkRows * SSBMV_ALIGN_ELEMS];
    BuildOffsetTable(linear, BYTES_PER_FLOAT32, k + 1U);
    Muls(strided.ReinterpretCast<int32_t>(), linear.ReinterpretCast<int32_t>(), static_cast<int32_t>(lda - 1U), k + 1U);
    const bool upper = uplo == ACLBLAS_UPPER;
    const auto leftOffsets = upper ? linear : strided;
    const auto rightOffsets = upper ? strided : linear;
    const uint32_t leftStep = upper ? 1U : lda - 1U;
    const uint32_t rightStep = upper ? lda - 1U : 1U;
    for (uint32_t row = r0; row < r1; ++row) {
        const uint32_t left = row < k ? row : k;
        const uint32_t available = n - 1U - row;
        const uint32_t right = available < k ? available : k;
        const uint32_t main = spanOff + (row - spanCol0) * lda + (upper ? k : 0U);
        const uint32_t rowX = row - static_cast<uint32_t>(winLo);
        const auto sum = sums[(row - r0) * SSBMV_ALIGN_ELEMS];
        Gather(aTerms, spanT, leftOffsets, (main - left * leftStep) * BYTES_PER_FLOAT32, left + 1U);
        Gather(xTerms, xT, linear, (rowX - left) * BYTES_PER_FLOAT32, left + 1U);
        Mul(aTerms, aTerms, xTerms, left + 1U);
        SumRowTerms(sum, aTerms, work, left + 1U);
        if (right != 0U) {
            Gather(aTerms, spanT, rightOffsets, (main + rightStep) * BYTES_PER_FLOAT32, right);
            Gather(xTerms, xT, linear, (rowX + 1U) * BYTES_PER_FLOAT32, right);
            Mul(aTerms, aTerms, xTerms, right);
            SumRowTerms(rightSum, aTerms, work, right);
            Add(sum, sum, rightSum, 1);
        }
    }
    const uint32_t rows = r1 - r0;
    BuildOffsetTable(offXT, SSBMV_ALIGN_ELEMS * BYTES_PER_FLOAT32, rows);
    Gather(yT, sums, offXT, 0U, rows);
}

template <bool kPhase, SsbmvRowDotMode mode, bool kClipGroups>
__aicore__ inline void SsbmvDiagKernel<kPhase, mode, kClipGroups>::Process()
{
    if (!active) {
        return;
    }
    if constexpr (kClipGroups) {
        SetFlag<HardEvent::MTE2_V>(evMte2V);
        WaitFlag<HardEvent::MTE2_V>(evMte2V);
        if (uplo == ACLBLAS_UPPER) {
            ProcessGroupedTile<true>();
        } else {
            ProcessGroupedTile<false>();
        }
        Finalize(r0, r1);
        return;
    }
    if (UseRowDots()) {
        SetFlag<HardEvent::MTE2_V>(evMte2V);
        WaitFlag<HardEvent::MTE2_V>(evMte2V);
        ComputeRowDots();
        Finalize(r0, r1);
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
            if (UseGroupedInterior(c0, c1)) {
                SetFlag<HardEvent::V_S>(EVENT_ID5);
                WaitFlag<HardEvent::V_S>(EVENT_ID5);
            }
            winLo = static_cast<int32_t>(c0) - static_cast<int32_t>(k);
            LoadInputs(c0, c1);
        }
        firstChunk = false;
        /* 首 chunk 的搬运已在 Init 里发出，这里只做等待。 */
        SetFlag<HardEvent::MTE2_V>(evMte2V);
        WaitFlag<HardEvent::MTE2_V>(evMte2V);
        if (UseGroupedInterior(c0, c1)) {
            if (uplo == ACLBLAS_UPPER) {
                ComputeGroupedInterior<true>(c0, c1 - c0);
            } else {
                ComputeGroupedInterior<false>(c0, c1 - c0);
            }
        } else if (k >= 8U && c0 >= k && k <= n - c1) {
            if (uplo == ACLBLAS_UPPER) {
                ComputeInterior<true>(c0, c1 - c0);
            } else {
                ComputeInterior<false>(c0, c1 - c0);
            }
        } else {
            Pass(c0, c1, 0, false);
            for (int32_t e = 1; e <= static_cast<int32_t>(k); e++) {
                Pass(c0, c1, e, true);
                Pass(c0, c1, -e, true);
            }
        }
        Finalize(c0, c1);
    }
}

extern "C" __global__ __aicore__ void ssbmv_kernel_diag_only(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, SsbmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workSpace;
    SsbmvDiagKernel<false> op; /* 非相位：设备侧代码里没有相位副本那一段 */
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
    SsbmvDiagKernel<true> op;
    op.Init(a, x, y, tiling);
    op.Process();
}

extern "C" __global__ __aicore__ void ssbmv_kernel_diag_phase(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, SsbmvTilingData tiling);

extern "C" __global__ __aicore__ void ssbmv_kernel_diag_grouped_edges(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, SsbmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workSpace;
    SsbmvDiagKernel<true, SsbmvRowDotMode::Disabled, true> op;
    op.Init(a, x, y, tiling);
    op.Process();
}

extern "C" __global__ __aicore__ void ssbmv_kernel_diag_grouped_edges(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, SsbmvTilingData tiling);

extern "C" __global__ __aicore__ void ssbmv_kernel_diag_row_dot(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, SsbmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workSpace;
    SsbmvDiagKernel<false, SsbmvRowDotMode::AllRows> op;
    op.Init(a, x, y, tiling);
    op.Process();
}

extern "C" __global__ __aicore__ void ssbmv_kernel_diag_row_dot(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, SsbmvTilingData tiling);

extern "C" __global__ __aicore__ void ssbmv_kernel_diag_edge_dot(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, SsbmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workSpace;
    SsbmvDiagKernel<false, SsbmvRowDotMode::Edges> op;
    op.Init(a, x, y, tiling);
    op.Process();
}

extern "C" __global__ __aicore__ void ssbmv_kernel_diag_phase_edge_dot(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, SsbmvTilingData tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    (void)workSpace;
    SsbmvDiagKernel<true, SsbmvRowDotMode::Edges> op;
    op.Init(a, x, y, tiling);
    op.Process();
}

extern "C" __global__ __aicore__ void ssbmv_kernel_diag_edge_dot(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, SsbmvTilingData tiling);
extern "C" __global__ __aicore__ void ssbmv_kernel_diag_phase_edge_dot(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, GM_ADDR workSpace, SsbmvTilingData tiling);

extern "C" __global__ __aicore__ void ssbmv_kernel_diagonal(
    GM_ADDR a, GM_ADDR x, GM_ADDR y, uint32_t n, uint32_t lda, uint32_t diagonal, float alpha, float beta,
    uint32_t blockRows)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    constexpr uint32_t TILE = 256U;
    constexpr uint32_t TILE_BYTES = TILE * BYTES_PER_FLOAT32;
    LocalTensor<float> aUb(TPosition::VECCALC, 0, TILE);
    LocalTensor<float> xUb(TPosition::VECCALC, TILE_BYTES, TILE);
    LocalTensor<float> yUb(TPosition::VECCALC, 2U * TILE_BYTES, TILE);
    GlobalTensor<float> aGm;
    GlobalTensor<float> xGm;
    GlobalTensor<float> yGm;
    aGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a), static_cast<uint64_t>(n) * lda);
    xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), n);
    yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(y), n);
    const uint32_t begin = GetBlockIdx() * blockRows;
    const uint32_t end = begin + blockRows < n ? begin + blockRows : n;
    for (uint32_t i = begin; i < end; i += TILE) {
        if (i != begin) {
            SetFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
            WaitFlag<HardEvent::MTE3_MTE2>(EVENT_ID0);
        }
        const uint32_t count = end - i < TILE ? end - i : TILE;
        const DataCopyExtParams copy(1, count * BYTES_PER_FLOAT32, 0, 0, 0);
        const DataCopyPadExtParams<float> padding;
        DataCopyPad(aUb, aGm[static_cast<uint64_t>(i) * lda + diagonal], copy, padding);
        DataCopyPad(xUb, xGm[i], copy, padding);
        if (beta != 0.0f) {
            DataCopyPad(yUb, yGm[i], copy, padding);
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Mul(aUb, aUb, xUb, count);
        if (alpha != 1.0f) {
            Muls(aUb, aUb, alpha, count);
        }
        if (beta != 0.0f) {
            Muls(yUb, yUb, beta, count);
            Add(aUb, aUb, yUb, count);
        }
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopyPad(yGm[i], aUb, copy);
    }
}

void ssbmv_diagonal_do(
    uint8_t* a, uint8_t* x, uint8_t* y, uint32_t n, uint32_t lda, uint32_t diagonal, float alpha, float beta,
    uint32_t blockRows, uint32_t numBlocks, void* stream)
{
    ssbmv_kernel_diagonal<<<numBlocks, nullptr, stream>>>(a, x, y, n, lda, diagonal, alpha, beta, blockRows);
}

aclError ssbmv_diag_kernel_do(
    uint8_t* a, uint8_t* x, uint8_t* y, uint8_t* workSpace, uint32_t numBlocks, const SsbmvTilingData& tiling,
    void* stream)
{
    const void* symbol = nullptr;
    if (tiling.rowDotMode == SsbmvRowDotMode::AllRows) {
        symbol = reinterpret_cast<const void*>(ssbmv_kernel_diag_row_dot);
    } else if (tiling.rowDotMode == SsbmvRowDotMode::Edges) {
        symbol = tiling.phaseForm != 0U ? reinterpret_cast<const void*>(ssbmv_kernel_diag_phase_edge_dot) :
                                          reinterpret_cast<const void*>(ssbmv_kernel_diag_edge_dot);
    } else if (
        tiling.groupedDiag != 0U && tiling.groupedXReuse != 0U && tiling.k <= 64U &&
        tiling.k % SSBMV_DIAG_GROUP_WIDTH == 0U && tiling.n % SSBMV_DIAG_GROUP_WIDTH == 0U &&
        tiling.leftEdgeRows == tiling.k && tiling.rightEdgeStart == tiling.n - tiling.k &&
        tiling.chunkRows == tiling.blockRows) {
        symbol = reinterpret_cast<const void*>(ssbmv_kernel_diag_grouped_edges);
    } else {
        symbol = tiling.phaseForm != 0U ? reinterpret_cast<const void*>(ssbmv_kernel_diag_phase) :
                                          reinterpret_cast<const void*>(ssbmv_kernel_diag_only);
    }
    return LaunchKernel(symbol, a, x, y, workSpace, numBlocks, tiling, stream);
}
