/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lt_matmul_tiling.h"
#if defined(__ASCENDC_ENABLE_SUPER_KERNEL__)
#error "This private endpoint experiment relies on normal TPipe::Destroy PIPE_ALL"
#endif
using namespace AscendC;
namespace {
template <HardEvent E>
__aicore__ inline void Sync()
{
    SetFlag<E>(EVENT_ID0);
    WaitFlag<E>(EVENT_ID0);
}
__aicore__ inline uint32_t Min(uint32_t a, uint32_t b) { return a < b ? a : b; }
__aicore__ inline uint32_t Pad16(uint32_t a) { return (a + 15U) / 16U * 16U; }

__aicore__ inline void CopyND(
    const LocalTensor<float>& dst, const GlobalTensor<float>& src, uint64_t offset, uint32_t rows, uint32_t cols,
    uint32_t ld, uint32_t physicalRows)
{
    Nd2NzParams p;
    p.ndNum = 1;
    p.nValue = static_cast<uint16_t>(rows);
    p.dValue = static_cast<uint16_t>(cols);
    p.srcNdMatrixStride = 0;
    p.srcDValue = static_cast<uint16_t>(ld);
    // This field is the padded row count, not the count of 16-row blocks.
    p.dstNzC0Stride = static_cast<uint16_t>(physicalRows);
    p.dstNzNStride = 1;
    p.dstNzMatrixStride = 0;
    DataCopy(dst, src[offset], p);
}
__aicore__ inline void ClearL1(const LocalTensor<float>& dst, uint32_t bytes)
{
    InitConstValueParams<float> p;
    p.repeatTimes = 1;
    p.blockNum = static_cast<uint16_t>(bytes / 32U);
    p.dstGap = 0;
    p.initValue = 0.0F;
    InitConstValue(dst, p);
}

template <
    uint32_t TM, uint32_t TN, uint32_t KPANEL, bool WHOLE_N, bool TRIM, bool UNIT, uint32_t ORDER, bool BYPASS,
    bool ELIDE_LAST = false, uint32_t PANEL = 0U, bool ONLY_TAIL = false>
class H9Producer {
    static constexpr uint32_t KPAD = 32U;
    const LtMatmul22Tiling& t;
    TBuf<TPosition::A1> a1Buffer;
    TBuf<TPosition::B1> b1Buffer;
    TBuf<TPosition::A2> a2Buffer;
    TBuf<TPosition::B2> b2Buffer;
    TBuf<TPosition::CO1> cBuffer;
    LocalTensor<float> a1, b1, a2, b2, c;
    GlobalTensor<float> ag, bg, dg;
    uint32_t row = 0, col = 0, rows = 0, cols = 0, mp = 0, np = 0;

    __aicore__ inline void CopyInputs()
    {
        if constexpr (!TRIM) {
            ClearL1(a1, TM * KPAD * 4U);
            ClearL1(b1, KPAD * TN * 4U);
            PipeBarrier<PIPE_MTE2>();
        }
        if constexpr (ORDER == 0U) {
            CopyND(a1, ag, uint64_t(row) * t.lda, rows, t.k, t.lda, mp);
            CopyND(b1, bg, col, t.k, cols, t.ldb, KPAD);
        } else if (ORDER == 1U || (GetBlockIdx() & 1U)) {
            CopyND(b1, bg, col, t.k, cols, t.ldb, KPAD);
            CopyND(a1, ag, uint64_t(row) * t.lda, rows, t.k, t.lda, mp);
        } else {
            CopyND(a1, ag, uint64_t(row) * t.lda, rows, t.k, t.lda, mp);
            CopyND(b1, bg, col, t.k, cols, t.ldb, KPAD);
        }
    }

    __aicore__ inline void Reduce()
    {
        const uint8_t noPad[4] = {0, 0, 0, 0};
        SetFmatrix(1, static_cast<uint16_t>(mp), noPad, FmatrixMode::FMATRIX_LEFT);
        SetFmatrix(1, KPAD, noPad, FmatrixMode::FMATRIX_RIGHT);
        for (uint32_t k = 0; k < t.k; k += KPANEL) {
            if (k)
                Sync<HardEvent::M_MTE1>();
            // ARCH2201 FP32 uses LOAD3DV2: A2 is Zz and B2 is Zn.
            // Match the selected SDK LoadToL0A/B path; offsets use elements.
            const uint32_t loadK = TRIM ? Pad16(Min(KPANEL, t.k - k)) : KPANEL;
            LoadData3DParamsV2Pro pa;
            pa.channelSize = KPAD;
            pa.extConfig = (uint64_t(k) << 32U) | (uint64_t(mp) << 16U) | loadK;
            LoadData(a2, a1, pa);
            LoadData3DParamsV2Pro pb;
            pb.channelSize = static_cast<uint16_t>(np);
            pb.extConfig = (uint64_t(k) << 48U) | (uint64_t(loadK) << 16U) | np;
            pb.fMatrixCtrl = true;
            LoadData(b2, b1, pb);
            Sync<HardEvent::MTE1_M>();
            MmadParams mad;
            // m=1 enables GEMV's different L0A layout. Compute an extra zero
            // row in that tail and copy only the caller's logical rows.
            mad.m = static_cast<uint16_t>(rows == 1U ? 2U : rows);
            mad.n = static_cast<uint16_t>(cols);
            mad.k = static_cast<uint16_t>(TRIM ? Min(KPANEL, t.k - k) : KPANEL);
            mad.cmatrixInitVal = k == 0U;
            mad.cmatrixSource = false;
            mad.unitFlag = UNIT && rows > 1U ? 3U : 0U;
            mad.kDirectionAlign = TRIM;
            Mmad(c, a2, b2, mad);
        }
    }

    __aicore__ inline void Store()
    {
        auto* dst = (__gm__ float*)dg[uint64_t(row) * t.ldd + col].GetPhyAddr();
        auto* cc = (__cc__ float*)c.GetPhyAddr();
        const uint8_t uf = UNIT && rows > 1U ? 3U : 0U;
        if constexpr (!PANEL) {
            SetFixpipeNz2ndFlag(1U, 0U, 0U);
            copy_matrix_cc_to_gm(
                dst, cc, 0, static_cast<uint16_t>(cols), static_cast<uint16_t>(rows), t.ldd, static_cast<uint16_t>(mp),
                uf, QuantMode_t::NoQuant, uint8_t(0), false, true);
        } else if (ONLY_TAIL && cols % 16U == 0U) {
            SetFixpipeNz2ndFlag(1U, 0U, 0U);
            copy_matrix_cc_to_gm(
                dst, cc, 0, static_cast<uint16_t>(cols), static_cast<uint16_t>(rows), t.ldd, static_cast<uint16_t>(mp),
                uf, QuantMode_t::NoQuant, uint8_t(0), false, true);
        } else {
            const uint32_t count = cols / PANEL, head = count * PANEL;
            if (count) {
                SetFixpipeNz2ndFlag(
                    static_cast<uint16_t>(count), static_cast<uint16_t>(PANEL / 16U * (mp / 16U)), PANEL);
                copy_matrix_cc_to_gm(
                    dst, cc, 0, PANEL, static_cast<uint16_t>(rows), t.ldd, static_cast<uint16_t>(mp), uf,
                    QuantMode_t::NoQuant, uint8_t(0), false, true);
            }
            if (head != cols) {
                if (count)
                    PipeBarrier<PIPE_FIX>();
                SetFixpipeNz2ndFlag(1U, 0U, 0U);
                copy_matrix_cc_to_gm(
                    dst + head, cc + uint64_t(head) * mp, 0, static_cast<uint16_t>(cols - head),
                    static_cast<uint16_t>(rows), t.ldd, static_cast<uint16_t>(mp), uf, QuantMode_t::NoQuant, uint8_t(0),
                    false, true);
            }
        }
    }

public:
    __aicore__ inline H9Producer(GM_ADDR a, GM_ADDR b, GM_ADDR output, const LtMatmul22Tiling& tiling, TPipe& pipe)
        : t(tiling)
    {
        static_assert(TM * TN * 4U <= 131072U, "CO1 must fit actual card1 L0C");
        static_assert(
            TM * KPANEL * 4U <= 65536U && KPANEL * TN * 4U <= 65536U, "L0 operands must fit actual card1 L0A/B");
        static_assert((TM + TN) * KPAD * 4U <= 524032U, "L1 operands must fit actual card1 L1");
        pipe.InitBuffer(a1Buffer, TM * KPAD * 4U);
        pipe.InitBuffer(b1Buffer, KPAD * TN * 4U);
        pipe.InitBuffer(a2Buffer, TM * KPANEL * 4U);
        pipe.InitBuffer(b2Buffer, KPANEL * TN * 4U);
        pipe.InitBuffer(cBuffer, TM * TN * 4U);
        a1 = a1Buffer.Get<float>();
        b1 = b1Buffer.Get<float>();
        a2 = a2Buffer.Get<float>();
        b2 = b2Buffer.Get<float>();
        c = cBuffer.Get<float>();
        ag.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(a));
        bg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(b));
        dg.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(output));
        if constexpr (BYPASS)
            dg.SetL2CacheHint(CacheMode::CACHE_MODE_DISABLE);
        SetHF32Mode(false);
        SetHF32TransMode(false);
        AscendCUtils::SetOverflow(1);
        SetAtomicNone();
        SetFixpipeNz2ndFlag(1, 0, 0);
        if constexpr (UNIT)
            SetMMRowMajor();
        SetLoadDataRepeat({0, 1, 0});
        SetLoadDataBoundary(0);
        SetLoadDataPaddingValue(0.0F);
    }

    __aicore__ inline void Run()
    {
        const uint32_t columns = WHOLE_N ? 1U : (t.n + TN - 1U) / TN;
        const uint32_t rowTiles = WHOLE_N ? GetBlockNum() : (t.m + TM - 1U) / TM;
        const uint64_t chunks = (static_cast<uint64_t>(t.m) + 7U) / 8U;
        for (uint64_t job = GetBlockIdx(); job < uint64_t(rowTiles) * columns; job += GetBlockNum()) {
            row = WHOLE_N ? chunks * GetBlockIdx() / GetBlockNum() * 8U : static_cast<uint32_t>(job / columns) * TM;
            col = WHOLE_N ? 0U : static_cast<uint32_t>(job % columns) * TN;
            const uint32_t next = WHOLE_N ? chunks * (GetBlockIdx() + 1U) / GetBlockNum() * 8U : row + TM;
            const uint32_t end = Min(t.m, next);
            if (row >= end)
                continue;
            rows = end - row;
            cols = Min(TN, t.n - col);
            mp = Pad16(rows);
            np = Pad16(cols);
            CopyInputs();
            Sync<HardEvent::MTE2_MTE1>();
            Reduce();
            if constexpr (!UNIT) {
                Sync<HardEvent::M_FIX>();
            } else if (rows == 1U) {
                Sync<HardEvent::M_FIX>();
            }

            Store();
            // Normal TPipe destruction drains PIPE_ALL; keep a FIX fence only
            // when this core must reuse C/L0 operands for another job.
            if constexpr (ELIDE_LAST) {
                if (job + GetBlockNum() < uint64_t(rowTiles) * columns)
                    PipeBarrier<PIPE_FIX>();
            } else {
                PipeBarrier<PIPE_FIX>();
            }
        }
    }
};
} // namespace

extern "C" __global__ __aicore__ __cube__ void lt_matmul22_h9_nd_columns32(
    GM_ADDR a, GM_ADDR b, GM_ADDR output, LtMatmul22Tiling t)
{
    TPipe pipe;
    H9Producer<80U, 288U, 32U, false, true, true, 2U, false, true, 32U, false> producer(a, b, output, t, pipe);
    producer.Run();
}
void LtMatmul22ShortKColumns(uint32_t blocks, void* stream, uint8_t* a, uint8_t* b, uint8_t* output, LtMatmul22Tiling t)
{
    lt_matmul22_h9_nd_columns32<<<blocks, nullptr, stream>>>(a, b, output, t);
}
