/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

__aicore__ inline void LoadBlockPanel(
        uint64_t off, const AscendC::DataCopyExtParams& text, const AscendC::DataCopyPadExtParams<float>& pad, AscendC::LocalTensor<float>& t1,
        AscendC::LocalTensor<float>& t2, AscendC::LocalTensor<float>& t3)
    {
        AscendC::LocalTensor<float> a = t1Que_.AllocTensor<float>();
        AscendC::DataCopyPad(a, t1Global_[off], text, pad);
        t1Que_.EnQue(a);
        t1 = t1Que_.DeQue<float>();
        AscendC::LocalTensor<float> b = t2Que_.AllocTensor<float>();
        AscendC::DataCopyPad(b, t2Global_[off], text, pad);
        t2Que_.EnQue(b);
        t2 = t2Que_.DeQue<float>();
        AscendC::LocalTensor<float> c = t1Que_.AllocTensor<float>();
        AscendC::DataCopyPad(c, t3Global_[off], text, pad);
        t1Que_.EnQue(c);
        t3 = t1Que_.DeQue<float>();
    }

    __aicore__ inline void FreeBlockPanel(AscendC::LocalTensor<float>& t1, AscendC::LocalTensor<float>& t2, AscendC::LocalTensor<float>& t3)
    {
        t1Que_.FreeTensor(t1);
        t2Que_.FreeTensor(t2);
        t1Que_.FreeTensor(t3);
    }

    __aicore__ inline void AccBlockPanels(
        uint64_t tempOff, int32_t count, const AscendC::DataCopyExtParams& text, const AscendC::DataCopyPadExtParams<float>& pad,
        AscendC::LocalTensor<float>& pack, AscendC::LocalTensor<float>& abR, AscendC::LocalTensor<float>& abI)
    {
        AscendC::LocalTensor<float> t1;
        AscendC::LocalTensor<float> t2;
        AscendC::LocalTensor<float> t3;
        LoadBlockPanel(tempOff, text, pad, t1, t2, t3);
        AscendC::Sub(abR, t1, t2, count);
        AscendC::Sub(abI, t3, t1, count);
        AscendC::Sub(abI, abI, t2, count);
        FreeBlockPanel(t1, t2, t3);
        for (int32_t p = 1; p < kParts_; p++) {
            uint64_t off = tempOff + static_cast<uint64_t>(p) * panelElems_;
            AscendC::LocalTensor<float> a;
            AscendC::LocalTensor<float> b;
            AscendC::LocalTensor<float> c;
            LoadBlockPanel(off, text, pad, a, b, c);
            AscendC::LocalTensor<float> tmpR = pack[TILE * 2];
            AscendC::LocalTensor<float> tmpI = pack[TILE * 3];
            AscendC::Sub(tmpR, a, b, count);
            AscendC::Sub(tmpI, c, a, count);
            AscendC::Sub(tmpI, tmpI, b, count);
            AscendC::Add(abR, abR, tmpR, count);
            AscendC::Add(abI, abI, tmpI, count);
            FreeBlockPanel(a, b, c);
        }
    }

    __aicore__ inline void AccBlockSingle(
        uint64_t tempOff, int32_t count, const AscendC::DataCopyExtParams& text, const AscendC::DataCopyPadExtParams<float>& pad,
        AscendC::LocalTensor<float>& abR, AscendC::LocalTensor<float>& abI)
    {
        AscendC::LocalTensor<float> t1;
        AscendC::LocalTensor<float> t2;
        AscendC::LocalTensor<float> t3;
        LoadBlockPanel(tempOff, text, pad, t1, t2, t3);
        AscendC::Sub(abR, t1, t2, count);
        AscendC::Sub(abI, t3, t1, count);
        AscendC::Sub(abI, abI, t2, count);
        FreeBlockPanel(t1, t2, t3);
    }

    __aicore__ inline void LoadBlockC(
        int32_t row0, int32_t loadRows, int32_t storeRows, int32_t col0, int32_t nCols, int32_t cMul, bool c2d,
        uint32_t cRowBytes, uint32_t cGap, uint64_t cOff, const AscendC::DataCopyPadExtParams<float>& pad,
        AscendC::LocalTensor<float>& cLocal)
    {
        AscendC::LocalTensor<float> cTile = cQue_.AllocTensor<float>();
        if (c2d) {
            AscendC::DataCopyExtParams cin{static_cast<uint16_t>(nCols), cRowBytes, cGap, 0, 0};
            AscendC::DataCopyPad(cTile, cGlobal_[cOff], cin, pad);
        } else {
            AscendC::DataCopyExtParams cin1{1, cRowBytes, 0, 0, 0};
            for (int32_t c = 0; c < nCols; ++c) {
                uint64_t src = static_cast<uint64_t>(col0 + c) * ldc_ * cMul + static_cast<uint64_t>(row0) * cMul;
                AscendC::DataCopyPad(cTile[c * loadRows * cMul], cGlobal_[src], cin1, pad);
            }
        }
        (void)storeRows;
        cQue_.EnQue(cTile);
        cLocal = cQue_.DeQue<float>();
    }

    __aicore__ inline void CombineBlockOut(
        AscendC::LocalTensor<float>& outTile, AscendC::LocalTensor<float>& pack, AscendC::LocalTensor<float>& abR, AscendC::LocalTensor<float>& abI,
        AscendC::LocalTensor<float>& cLocal, bool needC, int32_t count)
    {
        if (cIsComplex_ == 0 && ai_ == 0.0f && bi_ == 0.0f) {
            CombineReal(outTile, abR, abI, cLocal, needC, count);
        } else if (cIsComplex_) {
            CombineComplex(outTile, pack, abR, abI, cLocal, needC, count);
        } else {
            CombineFallback(outTile, abR, abI, cLocal, needC, count);
        }
    }

    __aicore__ inline void ProcessBlock(int32_t row0, int32_t loadRows, int32_t storeRows, int32_t col0, int32_t nCols)
    {
        int32_t count = loadRows * nCols;
        uint32_t rowBytes = static_cast<uint32_t>(loadRows) * sizeof(float);
        uint32_t srcGap = static_cast<uint32_t>(tempLdc_ - loadRows) * sizeof(float);
        AscendC::DataCopyExtParams text{static_cast<uint16_t>(nCols), rowBytes, srcGap, 0, 0};
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        uint64_t tempOff = static_cast<uint64_t>(col0) * tempLdc_ + row0;
        AscendC::LocalTensor<float> pack = calcBuf_.Get<float>();
        AscendC::LocalTensor<float> abR = pack;
        AscendC::LocalTensor<float> abI = pack[TILE];
        if (kParts_ > 1) {
            AccBlockPanels(tempOff, count, text, pad, pack, abR, abI);
        } else {
            AccBlockSingle(tempOff, count, text, pad, abR, abI);
        }
        AscendC::PipeBarrier<PIPE_V>();
        int32_t cMul = cIsComplex_ ? 2 : 1;
        uint32_t cRowBytes = static_cast<uint32_t>(storeRows * cMul) * sizeof(float);
        uint32_t cGap = static_cast<uint32_t>((ldc_ - storeRows) * cMul) * sizeof(float);
        bool c2d = (loadRows == storeRows) && ((cGap & 31U) == 0U);
        AscendC::DataCopyExtParams cext{static_cast<uint16_t>(nCols), cRowBytes, 0, cGap, 0};
        uint64_t cOff = static_cast<uint64_t>(col0) * ldc_ * cMul + row0 * cMul;
        AscendC::LocalTensor<float> outTile = outQue_.AllocTensor<float>();
        bool needC = (br_ != 0.0f || bi_ != 0.0f);
        AscendC::LocalTensor<float> cLocal = pack;
        if (needC) {
            LoadBlockC(row0, loadRows, storeRows, col0, nCols, cMul, c2d, cRowBytes, cGap, cOff, pad, cLocal);
        }
        CombineBlockOut(outTile, pack, abR, abI, cLocal, needC, count);
        if (c2d) {
            StoreOut(outTile, cOff, cext);
        } else {
            StorePackedCols(outTile, row0, storeRows, loadRows, col0, nCols);
        }
        if (needC) {
            cQue_.FreeTensor(cLocal);
        }
    }

    __aicore__ inline void Process()
    {
        if (startCol_ >= endCol_) {
            return;
        }
        SetupBuf();
        if (contig_) {
            ProcessContig();
            return;
        }
        ProcessStrided();
    }

private:
    __aicore__ inline bool CanBlock(int32_t rows)
    {
        if (rows < 8 || (rows & 7) != 0) {
            return false;
        }
        uint32_t srcGap = static_cast<uint32_t>(tempLdc_ - rows) * sizeof(float);
        return (srcGap & 31U) == 0U;
    }

    __aicore__ inline void CombineReal(
        AscendC::LocalTensor<float>& outTile, AscendC::LocalTensor<float>& abR, AscendC::LocalTensor<float>& abI, AscendC::LocalTensor<float>& cLocal,
        bool needC, int32_t count)
    {
        AscendC::Muls(outTile, abR, ar_, count);
        if (needC && br_ != 0.0f) {
            AscendC::Muls(abI, cLocal, br_, count);
            AscendC::Add(outTile, outTile, abI, count);
        }
    }

    __aicore__ inline void AddScaledC(
        AscendC::LocalTensor<float>& pack, AscendC::LocalTensor<float>& oR, AscendC::LocalTensor<float>& oI, AscendC::LocalTensor<float>& abR,
        AscendC::LocalTensor<float>& abI, AscendC::LocalTensor<float>& cLocal, int32_t count)
    {
        AscendC::LocalTensor<float> cR = pack[TILE * 4];
        AscendC::LocalTensor<float> cI = pack[TILE * 5];
        uint64_t rsvdCnt = 0;
        uint16_t rpt = static_cast<uint16_t>((count * 2 + 63) / 64);
        if (rpt < 1) {
            rpt = 1;
        }
        AscendC::GatherMask(cR, cLocal, 1, false, 0, {1, rpt, 8, 8}, rsvdCnt);
        AscendC::GatherMask(cI, cLocal, 2, false, 0, {1, rpt, 8, 8}, rsvdCnt);
        AscendC::Muls(abR, cR, br_, count);
        AscendC::Add(oR, oR, abR, count);
        AscendC::Muls(abI, cI, bi_, count);
        AscendC::Sub(oR, oR, abI, count);
        AscendC::Muls(abR, cI, br_, count);
        AscendC::Add(oI, oI, abR, count);
        AscendC::Muls(abI, cR, bi_, count);
        AscendC::Add(oI, oI, abI, count);
    }

    __aicore__ inline void CombineComplex(
        AscendC::LocalTensor<float>& outTile, AscendC::LocalTensor<float>& pack, AscendC::LocalTensor<float>& abR, AscendC::LocalTensor<float>& abI,
        AscendC::LocalTensor<float>& cLocal, bool needC, int32_t count)
    {
        AscendC::LocalTensor<float> oR = pack[TILE * 2];
        AscendC::LocalTensor<float> oI = pack[TILE * 3];
        if (!needC && ar_ == 1.0f && ai_ == 0.0f && count >= 256 && (count & 255) == 0) {
            AscendC::Interleave(outTile, outTile[count], abR, abI, count);
            return;
        }
        if (ar_ == 1.0f && ai_ == 0.0f) {
            AscendC::Adds(oR, abR, 0.0f, count);
            AscendC::Adds(oI, abI, 0.0f, count);
        } else {
            AscendC::Muls(oR, abR, ar_, count);
            AscendC::Muls(oI, abI, ai_, count);
            AscendC::Sub(oR, oR, oI, count);
            AscendC::Muls(oI, abI, ar_, count);
            AscendC::Muls(abR, abR, ai_, count);
            AscendC::Add(oI, oI, abR, count);
        }
        if (needC) {
            AddScaledC(pack, oR, oI, abR, abI, cLocal, count);
        }
        AscendC::PipeBarrier<PIPE_V>();
        EnsureGatherIdx(outTile);
        AscendC::LocalTensor<uint32_t> idx = idxBuf_.Get<uint32_t>();
        AscendC::Gather(outTile, oR, idx, 0, static_cast<uint32_t>(count * 2));
    }

    __aicore__ inline void EnsureGatherIdx(AscendC::LocalTensor<float>& scratchTile)
    {
        if (idxReady_) {
            return;
        }
        AscendC::LocalTensor<int32_t> scratch = scratchTile.template ReinterpretCast<int32_t>();
        AscendC::LocalTensor<int32_t> reOff = scratch;
        AscendC::LocalTensor<int32_t> imOff = scratch[TILE];
        AscendC::CreateVecIndex(reOff, static_cast<int32_t>(0), static_cast<uint32_t>(TILE));
        AscendC::Muls(reOff, reOff, static_cast<int32_t>(sizeof(float)), TILE);
        AscendC::Adds(imOff, reOff, static_cast<int32_t>(TILE * sizeof(float)), TILE);
        AscendC::PipeBarrier<PIPE_V>();
        AscendC::LocalTensor<int32_t> idx = idxBuf_.Get<int32_t>();
        AscendC::Interleave(idx, idx[TILE], reOff, imOff, TILE);
        AscendC::PipeBarrier<PIPE_V>();
        idxReady_ = 1;
    }

    __aicore__ inline void CombineFallback(
        AscendC::LocalTensor<float>& outTile, AscendC::LocalTensor<float>& abR, AscendC::LocalTensor<float>& abI, AscendC::LocalTensor<float>& cLocal,
        bool needC, int32_t count)
    {
        for (int32_t i = 0; i < count; i++) {
            float r = abR.GetValue(i);
            float im = abI.GetValue(i);
            float cR = needC ? cLocal.GetValue(i) : 0.0f;
            outTile.SetValue(i, ar_ * r - ai_ * im + br_ * cR);
        }
    }

    __aicore__ inline void LoadTemps(
        uint64_t tempOff, int32_t count, AscendC::LocalTensor<float>& t1, AscendC::LocalTensor<float>& t2, AscendC::LocalTensor<float>& t3)
    {
        uint32_t nbytes = static_cast<uint32_t>(count) * sizeof(float);
        AscendC::DataCopyExtParams ext{1, nbytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        AscendC::LocalTensor<float> a = t1Que_.AllocTensor<float>();
        AscendC::DataCopyPad(a, t1Global_[tempOff], ext, pad);
        t1Que_.EnQue(a);
        t1 = t1Que_.DeQue<float>();
        AscendC::LocalTensor<float> b = t2Que_.AllocTensor<float>();
        AscendC::DataCopyPad(b, t2Global_[tempOff], ext, pad);
        t2Que_.EnQue(b);
        t2 = t2Que_.DeQue<float>();
        AscendC::LocalTensor<float> c = t1Que_.AllocTensor<float>();
        AscendC::DataCopyPad(c, t3Global_[tempOff], ext, pad);
        t1Que_.EnQue(c);
        t3 = t1Que_.DeQue<float>();
    }

    __aicore__ inline void StoreOut(AscendC::LocalTensor<float>& outTile, uint64_t cOff, const AscendC::DataCopyExtParams& cext)
    {
        outQue_.EnQue(outTile);
        AscendC::LocalTensor<float> outLocal = outQue_.DeQue<float>();
        AscendC::DataCopyPad(cGlobal_[cOff], outLocal, cext);
        outQue_.FreeTensor(outLocal);
    }

    __aicore__ inline void StorePackedCols(
        AscendC::LocalTensor<float>& outTile, int32_t row0, int32_t storeRows, int32_t packRows, int32_t col0, int32_t nCols)
    {
        int32_t cMul = cIsComplex_ ? 2 : 1;
        uint32_t nbytes = static_cast<uint32_t>(storeRows * cMul) * sizeof(float);
        AscendC::DataCopyExtParams cext{1, nbytes, 0, 0, 0};
        outQue_.EnQue(outTile);
        AscendC::LocalTensor<float> outLocal = outQue_.DeQue<float>();
        for (int32_t c = 0; c < nCols; ++c) {
            uint64_t dst = static_cast<uint64_t>(col0 + c) * ldc_ * cMul + static_cast<uint64_t>(row0) * cMul;
            AscendC::DataCopyPad(cGlobal_[dst], outLocal[c * packRows * cMul], cext);
        }
        outQue_.FreeTensor(outLocal);
    }

    __aicore__ inline void AccumulatePanels(
        uint64_t tempOff, int32_t count, AscendC::LocalTensor<float> abR, AscendC::LocalTensor<float> abI, AscendC::LocalTensor<float> tmpR,
        AscendC::LocalTensor<float> tmpI)
    {
        for (int32_t p = 0; p < kParts_; p++) {
            AscendC::LocalTensor<float> t1;
            AscendC::LocalTensor<float> t2;
            AscendC::LocalTensor<float> t3;
            uint64_t off = tempOff + static_cast<uint64_t>(p) * panelElems_;
            LoadTemps(off, count, t1, t2, t3);
            if (packMode_) {
                if (p == 0) {
                    AscendC::Adds(abR, t1, 0.0f, count);
                    AscendC::Adds(abI, t2, 0.0f, count);
                } else {
                    AscendC::Add(abR, abR, t1, count);
                    AscendC::Add(abI, abI, t2, count);
                }
            } else if (p == 0) {
                AscendC::Sub(abR, t1, t2, count);
                AscendC::Sub(abI, t3, t1, count);
                AscendC::Sub(abI, abI, t2, count);
            } else {
                AscendC::Sub(tmpR, t1, t2, count);
                AscendC::Sub(tmpI, t3, t1, count);
                AscendC::Sub(tmpI, tmpI, t2, count);
                AscendC::Add(abR, abR, tmpR, count);
                AscendC::Add(abI, abI, tmpI, count);
            }
            t1Que_.FreeTensor(t1);
            t2Que_.FreeTensor(t2);
            t1Que_.FreeTensor(t3);
        }
    }

    __aicore__ inline void CombineAndStore(
        uint64_t cOff, int32_t count, AscendC::LocalTensor<float> pack, AscendC::LocalTensor<float> abR, AscendC::LocalTensor<float> abI)
    {
        AscendC::PipeBarrier<PIPE_V>();
        int32_t cMul = cIsComplex_ ? 2 : 1;
        uint32_t cBytes = static_cast<uint32_t>(count * cMul) * sizeof(float);
        AscendC::DataCopyExtParams cext{1, cBytes, 0, 0, 0};
        AscendC::DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        AscendC::LocalTensor<float> outTile = outQue_.AllocTensor<float>();
        bool needC = (br_ != 0.0f || bi_ != 0.0f);
        AscendC::LocalTensor<float> cLocal = pack;
        if (needC) {
            AscendC::LocalTensor<float> cTile = cQue_.AllocTensor<float>();
            AscendC::DataCopyPad(cTile, cGlobal_[cOff], cext, pad);
            cQue_.EnQue(cTile);
            cLocal = cQue_.DeQue<float>();
        }
        if (cIsComplex_ == 0 && ai_ == 0.0f && bi_ == 0.0f) {
            CombineReal(outTile, abR, abI, cLocal, needC, count);
        } else if (cIsComplex_) {
            CombineComplex(outTile, pack, abR, abI, cLocal, needC, count);
        } else {
            CombineFallback(outTile, abR, abI, cLocal, needC, count);
        }
        StoreOut(outTile, cOff, cext);
        if (needC) {
            cQue_.FreeTensor(cLocal);
        }
    }

    __aicore__ inline void ProcessTile(uint64_t tempOff, uint64_t cOff, int32_t count)
    {
        AscendC::LocalTensor<float> pack = calcBuf_.Get<float>();
        AscendC::LocalTensor<float> abR = pack;
        AscendC::LocalTensor<float> abI = pack[TILE];
        AscendC::LocalTensor<float> tmpR = pack[TILE * 2];
        AscendC::LocalTensor<float> tmpI = pack[TILE * 3];
        AccumulatePanels(tempOff, count, abR, abI, tmpR, tmpI);
        CombineAndStore(cOff, count, pack, abR, abI);
    }

    AscendC::TPipe* pipe_ = nullptr;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUF_NUM> t1Que_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUF_NUM> t2Que_;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUF_NUM> cQue_;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUF_NUM> outQue_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuf_;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> idxBuf_;
    AscendC::GlobalTensor<float> t1Global_;
    AscendC::GlobalTensor<float> t2Global_;
    AscendC::GlobalTensor<float> t3Global_;
    AscendC::GlobalTensor<float> cGlobal_;
    int32_t m_ = 0;
    int32_t n_ = 0;
    int32_t ldc_ = 0;
    int32_t tempLdc_ = 0;
    float ar_ = 1.0f;
    float ai_ = 0.0f;
    float br_ = 0.0f;
    float bi_ = 0.0f;
    int32_t cIsComplex_ = 1;
    int32_t contig_ = 0;
    int32_t startCol_ = 0;
    int32_t endCol_ = 0;
    int32_t colsPerCore_ = 0;
    int32_t kParts_ = 1;
    int32_t packMode_ = 0;
    int32_t idxReady_ = 0;
    uint64_t panelElems_ = 0;
};
