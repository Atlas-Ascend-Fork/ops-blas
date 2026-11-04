/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "common/helper/devkit_version_compat.h"

#if ASC_DEVKIT_GE_9_1

#include "kernel_operator.h"
#include "cgemm_ex_kernel.h"

namespace cgemm_ex_kern {
constexpr int32_t TILE = 2048;
constexpr int32_t BUF_NUM = 2;

#include "cgemm_ex_combine_impl.h"
#include "cgemm_ex_combine_impl_tail.h"
using namespace AscendC;

class DeinterleaveKernel {
public:
    __aicore__ inline void Init(
        __gm__ uint8_t* src, __gm__ uint8_t* reOut, __gm__ uint8_t* imOut, int32_t physRows, int32_t physCols,
        int32_t lda, int32_t logicalRows, int32_t transFlag, int32_t conjFlag, __gm__ uint8_t* sumOut, TPipe* pipe)
    {
        Init(
            src, reOut, imOut, physRows, physCols, lda, logicalRows, transFlag, conjFlag, sumOut, pipe, GetBlockNum(),
            GetBlockIdx());
    }

    __aicore__ inline void Init(
        __gm__ uint8_t* src, __gm__ uint8_t* reOut, __gm__ uint8_t* imOut, int32_t physRows, int32_t physCols,
        int32_t lda, int32_t logicalRows, int32_t transFlag, int32_t conjFlag, __gm__ uint8_t* sumOut, TPipe* pipe,
        uint32_t blockNum, uint32_t blockIdx)
    {
        pipe_ = pipe;
        physRows_ = physRows;
        physCols_ = physCols;
        lda_ = lda;
        logicalRows_ = logicalRows;
        transFlag_ = transFlag;
        conjFlag_ = conjFlag;
        writeSum_ = (sumOut != nullptr) ? 1 : 0;
        uint64_t srcCount = static_cast<uint64_t>(lda) * physCols * 2;
        uint64_t outCols = transFlag ? physRows : physCols;
        uint64_t outCount = static_cast<uint64_t>(logicalRows) * outCols;
        srcGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src), srcCount);
        reGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(reOut), outCount);
        imGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(imOut), outCount);
        if (writeSum_) {
            sumGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(sumOut), outCount);
        }
        contig_ = (transFlag_ == 0 && lda_ == physRows_ && logicalRows_ == physRows_) ? 1 : 0;
        if (contig_) {
            AssignCoreRange(physRows_ * physCols_, blockNum, blockIdx, startCol_, endCol_);
        } else {
            AssignCoreRange(physCols_, blockNum, blockIdx, startCol_, endCol_);
            colsPerCore_ = endCol_ - startCol_;
        }
    }

    __aicore__ inline void Process()
    {
        if (startCol_ >= endCol_) {
            return;
        }
        constexpr int32_t DEINT_TILE = 4096;
        if (contig_) {
            int32_t span = endCol_ - startCol_;
            int32_t tileCap = span > DEINT_TILE ? DEINT_TILE : span;
            tileCap = (tileCap + 15) & ~15;
            if (tileCap < 16) {
                tileCap = 16;
            }
            int32_t bufNum = span > DEINT_TILE ? BUF_NUM : 1;
            pipe_->InitBuffer(inQue_, bufNum, tileCap * 2 * sizeof(float));
            pipe_->InitBuffer(reQue_, bufNum, tileCap * sizeof(float));
            pipe_->InitBuffer(imQue_, bufNum, tileCap * sizeof(float));
            if (writeSum_) {
                pipe_->InitBuffer(sumQue_, bufNum, tileCap * sizeof(float));
            }
            int32_t offset = startCol_;
            while (offset < endCol_) {
                int32_t count = endCol_ - offset;
                if (count > tileCap) {
                    count = tileCap;
                }
                ProcessContigTile(offset, count);
                offset += count;
            }
            return;
        }
        pipe_->InitBuffer(inQue_, BUF_NUM, DEINT_TILE * 2 * sizeof(float));
        pipe_->InitBuffer(reQue_, BUF_NUM, DEINT_TILE * sizeof(float));
        pipe_->InitBuffer(imQue_, BUF_NUM, DEINT_TILE * sizeof(float));
        if (writeSum_) {
            pipe_->InitBuffer(sumQue_, BUF_NUM, DEINT_TILE * sizeof(float));
        }
        for (int32_t col = startCol_; col < endCol_; col++) {
            int32_t rowOffset = 0;
            while (rowOffset < physRows_) {
                int32_t count = physRows_ - rowOffset;
                if (count > DEINT_TILE) {
                    count = DEINT_TILE;
                }
                ProcessTile(col, rowOffset, count);
                rowOffset += count;
            }
        }
    }

private:
    __aicore__ inline void LoadSplitCplx(
        uint64_t srcOff, int32_t count, LocalTensor<float>& inLocal, LocalTensor<float>& reLocal,
        LocalTensor<float>& imLocal)
    {
        uint32_t cplxBytes = static_cast<uint32_t>(count) * 2 * sizeof(float);
        DataCopyExtParams ext{1, cplxBytes, 0, 0, 0};
        DataCopyPadExtParams<float> padParams{false, 0, 0, 0};
        LocalTensor<float> inTile = inQue_.AllocTensor<float>();
        DataCopyPad(inTile, srcGlobal_[srcOff], ext, padParams);
        inQue_.EnQue(inTile);
        inLocal = inQue_.DeQue<float>();
        LocalTensor<float> reTile = reQue_.AllocTensor<float>();
        LocalTensor<float> imTile = imQue_.AllocTensor<float>();
        uint64_t rsvdCnt = 0;
        uint16_t repeatTimes = static_cast<uint16_t>((count * 2 + 63) / 64);
        if (repeatTimes < 1) {
            repeatTimes = 1;
        }
        GatherMask(reTile, inLocal, 1, false, 0, {1, repeatTimes, 8, 8}, rsvdCnt);
        GatherMask(imTile, inLocal, 2, false, 0, {1, repeatTimes, 8, 8}, rsvdCnt);
        if (conjFlag_) {
            Muls(imTile, imTile, -1.0f, count);
        }
        reQue_.EnQue(reTile);
        imQue_.EnQue(imTile);
        reLocal = reQue_.DeQue<float>();
        imLocal = imQue_.DeQue<float>();
    }

    __aicore__ inline void ProcessContigTile(int32_t offset, int32_t count)
    {
        uint64_t srcOff = static_cast<uint64_t>(offset) * 2;
        LocalTensor<float> inLocal;
        LocalTensor<float> reLocal;
        LocalTensor<float> imLocal;
        LoadSplitCplx(srcOff, count, inLocal, reLocal, imLocal);
        uint32_t nbytes = static_cast<uint32_t>(count) * sizeof(float);
        DataCopyExtParams oext{1, nbytes, 0, 0, 0};
        uint64_t dstOff = static_cast<uint64_t>(offset);
        DataCopyPad(reGlobal_[dstOff], reLocal, oext);
        DataCopyPad(imGlobal_[dstOff], imLocal, oext);
        if (writeSum_) {
            LocalTensor<float> sumTile = sumQue_.AllocTensor<float>();
            Add(sumTile, reLocal, imLocal, count);
            sumQue_.EnQue(sumTile);
            LocalTensor<float> sumLocal = sumQue_.DeQue<float>();
            DataCopyPad(sumGlobal_[dstOff], sumLocal, oext);
            sumQue_.FreeTensor(sumLocal);
        }
        reQue_.FreeTensor(reLocal);
        imQue_.FreeTensor(imLocal);
        inQue_.FreeTensor(inLocal);
    }

    __aicore__ inline void ProcessTile(int32_t col, int32_t rowOffset, int32_t count)
    {
        uint64_t srcOff = static_cast<uint64_t>(col) * lda_ * 2 + rowOffset * 2;
        LocalTensor<float> inLocal;
        LocalTensor<float> reLocal;
        LocalTensor<float> imLocal;
        LoadSplitCplx(srcOff, count, inLocal, reLocal, imLocal);
        if (transFlag_ == 0) {
            WriteNoTrans(reLocal, imLocal, col, rowOffset, count);
        } else {
            WriteTrans(reLocal, imLocal, col, rowOffset, count);
        }
        reQue_.FreeTensor(reLocal);
        imQue_.FreeTensor(imLocal);
        inQue_.FreeTensor(inLocal);
    }

    __aicore__ inline void WriteNoTrans(
        LocalTensor<float>& reLocal, LocalTensor<float>& imLocal, int32_t col, int32_t rowOffset, int32_t count)
    {
        uint32_t nbytes = static_cast<uint32_t>(count) * sizeof(float);
        DataCopyExtParams oext{1, nbytes, 0, 0, 0};
        uint64_t dstOff = static_cast<uint64_t>(col) * logicalRows_ + rowOffset;
        DataCopyPad(reGlobal_[dstOff], reLocal, oext);
        DataCopyPad(imGlobal_[dstOff], imLocal, oext);
        if (writeSum_ == 0) {
            return;
        }
        LocalTensor<float> sumTile = sumQue_.AllocTensor<float>();
        Add(sumTile, reLocal, imLocal, count);
        sumQue_.EnQue(sumTile);
        LocalTensor<float> sumLocal = sumQue_.DeQue<float>();
        DataCopyPad(sumGlobal_[dstOff], sumLocal, oext);
        sumQue_.FreeTensor(sumLocal);
    }

    __aicore__ inline void WriteTrans(
        LocalTensor<float>& reLocal, LocalTensor<float>& imLocal, int32_t col, int32_t rowOffset, int32_t count)
    {
        for (int32_t i = 0; i < count; i++) {
            int32_t lr = col;
            int32_t lc = rowOffset + i;
            uint64_t dstOff = static_cast<uint64_t>(lc) * logicalRows_ + lr;
            reGlobal_.SetValue(dstOff, reLocal.GetValue(i));
            imGlobal_.SetValue(dstOff, imLocal.GetValue(i));
        }
    }

    TPipe* pipe_ = nullptr;
    TQue<QuePosition::VECIN, BUF_NUM> inQue_;
    TQue<QuePosition::VECOUT, BUF_NUM> reQue_;
    TQue<QuePosition::VECOUT, BUF_NUM> imQue_;
    TQue<QuePosition::VECOUT, BUF_NUM> sumQue_;
    GlobalTensor<float> srcGlobal_;
    GlobalTensor<float> reGlobal_;
    GlobalTensor<float> imGlobal_;
    GlobalTensor<float> sumGlobal_;
    int32_t physRows_ = 0;
    int32_t physCols_ = 0;
    int32_t lda_ = 0;
    int32_t logicalRows_ = 0;
    int32_t transFlag_ = 0;
    int32_t conjFlag_ = 0;
    int32_t writeSum_ = 0;
    int32_t contig_ = 0;
    int32_t startCol_ = 0;
    int32_t endCol_ = 0;
    int32_t colsPerCore_ = 0;
};

class C32MatBase {
public:
    __aicore__ inline void InitMats(
        __gm__ uint8_t* A, int32_t lda, int32_t transA, int32_t conjA, __gm__ uint8_t* B, int32_t ldb, int32_t transB,
        int32_t conjB, __gm__ uint8_t* C, int32_t ldc, int32_t cIsComplex, int32_t m, int32_t n, int32_t k, float ar,
        float ai, float br, float bi, TPipe* pipe)
    {
        pipe_ = pipe;
        m_ = m;
        n_ = n;
        k_ = k;
        lda_ = lda;
        ldb_ = ldb;
        ldc_ = ldc;
        transA_ = transA;
        transB_ = transB;
        conjA_ = conjA;
        conjB_ = conjB;
        cIsComplex_ = cIsComplex;
        ar_ = ar;
        ai_ = ai;
        br_ = br;
        bi_ = bi;
        aGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(A), static_cast<uint64_t>(lda) * (transA ? m : k) * 2);
        bGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(B), static_cast<uint64_t>(ldb) * (transB ? k : n) * 2);
        uint64_t cCount = static_cast<uint64_t>(ldc) * n * (cIsComplex ? 2 : 1);
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(C), cCount);
        AssignCoreRange(n_, GetBlockNum(), GetBlockIdx(), startCol_, endCol_);
    }

protected:
    TPipe* pipe_ = nullptr;
    GlobalTensor<float> aGlobal_;
    GlobalTensor<float> bGlobal_;
    GlobalTensor<float> cGlobal_;
    int32_t m_ = 0;
    int32_t n_ = 0;
    int32_t k_ = 0;
    int32_t lda_ = 0;
    int32_t ldb_ = 0;
    int32_t ldc_ = 0;
    int32_t transA_ = 0;
    int32_t transB_ = 0;
    int32_t conjA_ = 0;
    int32_t conjB_ = 0;
    int32_t cIsComplex_ = 1;
    float ar_ = 1.0f;
    float ai_ = 0.0f;
    float br_ = 0.0f;
    float bi_ = 0.0f;
    int32_t startCol_ = 0;
    int32_t endCol_ = 0;
};

class SmallC32Kernel : public C32MatBase {
public:
    __aicore__ inline void Init(
        __gm__ uint8_t* A, int32_t lda, int32_t transA, int32_t conjA, __gm__ uint8_t* B, int32_t ldb, int32_t transB,
        int32_t conjB, __gm__ uint8_t* C, int32_t ldc, int32_t cIsComplex, int32_t m, int32_t n, int32_t k, float ar,
        float ai, float br, float bi, TPipe* pipe)
    {
        InitMats(A, lda, transA, conjA, B, ldb, transB, conjB, C, ldc, cIsComplex, m, n, k, ar, ai, br, bi, pipe);
        int32_t outCount = cIsComplex ? (m * 2) : m;
        if (outCount < 8) {
            outCount = 8;
        }
        pipe_->InitBuffer(outQue_, 1, static_cast<uint32_t>(outCount) * sizeof(float));
    }

    __aicore__ inline void WriteColOut(LocalTensor<float>& out, int32_t j, int32_t count)
    {
        uint32_t nbytes = static_cast<uint32_t>(count) * sizeof(float);
        DataCopyExtParams cext{1, nbytes, 0, 0, 0};
        uint64_t cOff = cIsComplex_ ? (static_cast<uint64_t>(j) * ldc_ * 2) : (static_cast<uint64_t>(j) * ldc_);
        outQue_.EnQue(out);
        LocalTensor<float> outLocal = outQue_.DeQue<float>();
        DataCopyPad(cGlobal_[cOff], outLocal, cext);
        outQue_.FreeTensor(outLocal);
    }

    __aicore__ inline void AccDot(int32_t i, int32_t j, float& sr, float& si)
    {
        sr = 0.0f;
        si = 0.0f;
        for (int32_t kk = 0; kk < k_; ++kk) {
            float ar;
            float ai;
            float br;
            float bi;
            LoadA(i, kk, ar, ai);
            LoadB(kk, j, br, bi);
            sr += ar * br - ai * bi;
            si += ar * bi + ai * br;
        }
    }

    __aicore__ inline void StoreElem(LocalTensor<float>& out, int32_t i, int32_t j, float cr, float ci)
    {
        if (cIsComplex_) {
            if (br_ != 0.0f || bi_ != 0.0f) {
                uint64_t off = (static_cast<uint64_t>(j) * ldc_ + i) * 2;
                float orr = cGlobal_.GetValue(off);
                float ori = cGlobal_.GetValue(off + 1);
                cr += br_ * orr - bi_ * ori;
                ci += br_ * ori + bi_ * orr;
            }
            out.SetValue(i * 2, cr);
            out.SetValue(i * 2 + 1, ci);
            return;
        }
        if (br_ != 0.0f) {
            uint64_t off = static_cast<uint64_t>(j) * ldc_ + i;
            cr += br_ * cGlobal_.GetValue(off);
        }
        out.SetValue(i, cr);
    }

    __aicore__ inline void Process()
    {
        for (int32_t j = startCol_; j < endCol_; ++j) {
            LocalTensor<float> out = outQue_.AllocTensor<float>();
            for (int32_t i = 0; i < m_; ++i) {
                float sr;
                float si;
                AccDot(i, j, sr, si);
                float cr = ar_ * sr - ai_ * si;
                float ci = ar_ * si + ai_ * sr;
                StoreElem(out, i, j, cr, ci);
            }
            int32_t count = cIsComplex_ ? (m_ * 2) : m_;
            WriteColOut(out, j, count);
        }
    }

private:
    __aicore__ inline void LoadA(int32_t i, int32_t kk, float& ar, float& ai)
    {
        uint64_t off =
            transA_ ? (static_cast<uint64_t>(i) * lda_ + kk) * 2 : (static_cast<uint64_t>(kk) * lda_ + i) * 2;
        ar = aGlobal_.GetValue(off);
        ai = aGlobal_.GetValue(off + 1);
        if (conjA_) {
            ai = -ai;
        }
    }

    __aicore__ inline void LoadB(int32_t kk, int32_t j, float& br, float& bi)
    {
        uint64_t off =
            transB_ ? (static_cast<uint64_t>(kk) * ldb_ + j) * 2 : (static_cast<uint64_t>(j) * ldb_ + kk) * 2;
        br = bGlobal_.GetValue(off);
        bi = bGlobal_.GetValue(off + 1);
        if (conjB_) {
            bi = -bi;
        }
    }

    TQue<QuePosition::VECOUT, 1> outQue_;
};

} // namespace cgemm_ex_kern

extern "C" __global__ __aicore__ void cgemm_ex_deinterleave_kernel(
    __gm__ uint8_t* src, __gm__ uint8_t* reOut, __gm__ uint8_t* imOut, int32_t physRows, int32_t physCols, int32_t lda,
    int32_t logicalRows, int32_t transFlag, int32_t conjFlag, __gm__ uint8_t* sumOut)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    cgemm_ex_kern::DeinterleaveKernel op;
    op.Init(src, reOut, imOut, physRows, physCols, lda, logicalRows, transFlag, conjFlag, sumOut, &pipe);
    op.Process();
}

extern "C" __global__ __aicore__ void cgemm_ex_combine3_kernel(
    __gm__ uint8_t* t1, __gm__ uint8_t* t2, __gm__ uint8_t* t3, int32_t tempLdc, __gm__ uint8_t* cInOut, int32_t m,
    int32_t n, int32_t ldc, float ar, float ai, float br, float bi, int32_t cIsComplex, int32_t kParts)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    cgemm_ex_kern::Combine3Kernel op;
    op.Init(t1, t2, t3, tempLdc, cInOut, m, n, ldc, ar, ai, br, bi, cIsComplex, kParts, &pipe);
    op.Process();
}

void cgemm_ex_deinterleave_do(
    uint32_t numBlocks, void* stream, GM_ADDR src, GM_ADDR reOut, GM_ADDR imOut, int32_t physRows, int32_t physCols,
    int32_t lda, int32_t logicalRows, int32_t transFlag, int32_t conjFlag, GM_ADDR sumOut)
{
    cgemm_ex_deinterleave_kernel<<<numBlocks, nullptr, stream>>>(
        src, reOut, imOut, physRows, physCols, lda, logicalRows, transFlag, conjFlag, sumOut);
}

extern "C" __global__ __aicore__ void cgemm_ex_deint2_kernel(
    __gm__ uint8_t* srcA, __gm__ uint8_t* reA, __gm__ uint8_t* imA, int32_t physRowsA, int32_t physColsA, int32_t ldaA,
    int32_t logicalRowsA, int32_t conjA, __gm__ uint8_t* sumA, __gm__ uint8_t* srcB, __gm__ uint8_t* reB,
    __gm__ uint8_t* imB, int32_t physRowsB, int32_t physColsB, int32_t ldaB, int32_t logicalRowsB, int32_t conjB,
    __gm__ uint8_t* sumB)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    uint32_t n = static_cast<uint32_t>(AscendC::GetBlockNum());
    uint32_t blk = static_cast<uint32_t>(AscendC::GetBlockIdx());
    if (n < 2) {
        {
            AscendC::TPipe pipe;
            cgemm_ex_kern::DeinterleaveKernel op;
            op.Init(srcA, reA, imA, physRowsA, physColsA, ldaA, logicalRowsA, 0, conjA, sumA, &pipe);
            op.Process();
        }
        {
            AscendC::TPipe pipe;
            cgemm_ex_kern::DeinterleaveKernel op;
            op.Init(srcB, reB, imB, physRowsB, physColsB, ldaB, logicalRowsB, 0, conjB, sumB, &pipe);
            op.Process();
        }
        return;
    }
    uint64_t sizeA = static_cast<uint64_t>(physRowsA) * static_cast<uint64_t>(physColsA);
    uint64_t sizeB = static_cast<uint64_t>(physRowsB) * static_cast<uint64_t>(physColsB);
    uint64_t tot = sizeA + sizeB;
    uint32_t nA = n / 2;
    if (tot > 0) {
        nA = static_cast<uint32_t>((static_cast<uint64_t>(n) * sizeA + tot / 2) / tot);
    }
    if (nA < 1) {
        nA = 1;
    }
    if (nA >= n) {
        nA = n - 1;
    }
    AscendC::TPipe pipe;
    cgemm_ex_kern::DeinterleaveKernel op;
    if (blk < nA) {
        op.Init(srcA, reA, imA, physRowsA, physColsA, ldaA, logicalRowsA, 0, conjA, sumA, &pipe, nA, blk);
    } else {
        op.Init(srcB, reB, imB, physRowsB, physColsB, ldaB, logicalRowsB, 0, conjB, sumB, &pipe, n - nA, blk - nA);
    }
    op.Process();
}

void cgemm_ex_deint2_do(
    uint32_t numBlocks, void* stream, GM_ADDR srcA, GM_ADDR reA, GM_ADDR imA, int32_t physRowsA, int32_t physColsA,
    int32_t ldaA, int32_t logicalRowsA, int32_t conjA, GM_ADDR sumA, GM_ADDR srcB, GM_ADDR reB, GM_ADDR imB,
    int32_t physRowsB, int32_t physColsB, int32_t ldaB, int32_t logicalRowsB, int32_t conjB, GM_ADDR sumB)
{
    cgemm_ex_deint2_kernel<<<numBlocks, nullptr, stream>>>(
        srcA, reA, imA, physRowsA, physColsA, ldaA, logicalRowsA, conjA, sumA, srcB, reB, imB, physRowsB, physColsB,
        ldaB, logicalRowsB, conjB, sumB);
}

extern "C" __global__ __aicore__ void cgemm_ex_small_c32_kernel(
    __gm__ uint8_t* A, int32_t lda, int32_t transA, int32_t conjA, __gm__ uint8_t* B, int32_t ldb, int32_t transB,
    int32_t conjB, __gm__ uint8_t* C, int32_t ldc, int32_t cIsComplex, int32_t m, int32_t n, int32_t k, float ar,
    float ai, float br, float bi)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    // issue #766: 本 kernel 仅由 host UseSmallC32 门控（m<=16 && n<=16 && k<=16）下发，
    // 小 C32 维度契约由 host 保证；原 AivC32Kernel 兜底分支不可达且自带 48 截断死雷，已删除
    cgemm_ex_kern::SmallC32Kernel op;
    op.Init(A, lda, transA, conjA, B, ldb, transB, conjB, C, ldc, cIsComplex, m, n, k, ar, ai, br, bi, &pipe);
    op.Process();
}

void cgemm_ex_small_c32_do(
    uint32_t numBlocks, void* stream, GM_ADDR A, int32_t lda, int32_t transA, int32_t conjA, GM_ADDR B, int32_t ldb,
    int32_t transB, int32_t conjB, GM_ADDR C, int32_t ldc, int32_t cIsComplex, int32_t m, int32_t n, int32_t k,
    float ar, float ai, float br, float bi)
{
    cgemm_ex_small_c32_kernel<<<numBlocks, nullptr, stream>>>(
        A, lda, transA, conjA, B, ldb, transB, conjB, C, ldc, cIsComplex, m, n, k, ar, ai, br, bi);
}

void cgemm_ex_combine3_do(
    uint32_t numBlocks, void* stream, GM_ADDR t1, GM_ADDR t2, GM_ADDR t3, int32_t tempLdc, GM_ADDR cInOut, int32_t m,
    int32_t n, int32_t ldc, float ar, float ai, float br, float bi, int32_t cIsComplex, int32_t kParts)
{
    cgemm_ex_combine3_kernel<<<numBlocks, nullptr, stream>>>(
        t1, t2, t3, tempLdc, cInOut, m, n, ldc, ar, ai, br, bi, cIsComplex, kParts);
}

extern "C" __global__ __aicore__ void cgemm_ex_neg_kernel(__gm__ uint8_t* src, __gm__ uint8_t* dst, int32_t count)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    using namespace AscendC;
    using namespace cgemm_ex_kern;
    uint32_t blockNum = GetBlockNum();
    uint32_t blockIdx = GetBlockIdx();
    int32_t start = 0;
    int32_t end = 0;
    AssignCoreRange(count, blockNum, blockIdx, start, end);
    if (start >= end) {
        return;
    }
    AscendC::TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQue;
    TQue<QuePosition::VECOUT, 1> outQue;
    pipe.InitBuffer(inQue, 1, TILE * sizeof(float));
    pipe.InitBuffer(outQue, 1, TILE * sizeof(float));
    GlobalTensor<float> srcG;
    GlobalTensor<float> dstG;
    srcG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(src), count);
    dstG.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(dst), count);
    int32_t offset = start;
    while (offset < end) {
        int32_t n = end - offset;
        if (n > TILE) {
            n = TILE;
        }
        uint32_t nbytes = static_cast<uint32_t>(n) * sizeof(float);
        DataCopyExtParams ext{1, nbytes, 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        LocalTensor<float> in = inQue.AllocTensor<float>();
        DataCopyPad(in, srcG[offset], ext, pad);
        inQue.EnQue(in);
        LocalTensor<float> inL = inQue.DeQue<float>();
        LocalTensor<float> out = outQue.AllocTensor<float>();
        Muls(out, inL, -1.0f, n);
        outQue.EnQue(out);
        LocalTensor<float> outL = outQue.DeQue<float>();
        DataCopyPad(dstG[offset], outL, ext);
        outQue.FreeTensor(outL);
        inQue.FreeTensor(inL);
        offset += n;
    }
}

void cgemm_ex_neg_do(uint32_t numBlocks, void* stream, GM_ADDR src, GM_ADDR dst, int32_t count)
{
    if (count <= 0) {
        return;
    }
    cgemm_ex_neg_kernel<<<numBlocks, nullptr, stream>>>(src, dst, count);
}

#endif
