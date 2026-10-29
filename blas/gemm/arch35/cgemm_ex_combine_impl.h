/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

__aicore__ inline void AssignCoreRange(
    int32_t total, uint32_t blockNum, uint32_t blockIdx, int32_t& startCol, int32_t& endCol)
{
    if (blockNum == 0 || total <= 0) {
        startCol = 0;
        endCol = 0;
        return;
    }
    int32_t per = (total + static_cast<int32_t>(blockNum) - 1) / static_cast<int32_t>(blockNum);
    startCol = static_cast<int32_t>(blockIdx) * per;
    endCol = startCol + per;
    if (endCol > total) {
        endCol = total;
    }
    if (startCol >= total) {
        startCol = 0;
        endCol = 0;
    }
}

class Combine3Kernel {
public:
    __aicore__ inline void Init(
        __gm__ uint8_t* t1, __gm__ uint8_t* t2, __gm__ uint8_t* t3, int32_t tempLdc, __gm__ uint8_t* cInOut, int32_t m,
        int32_t n, int32_t ldc, float ar, float ai, float br, float bi, int32_t cIsComplex, int32_t kParts,
        AscendC::TPipe* pipe)
    {
        Init(
            t1, t2, t3, tempLdc, cInOut, m, n, ldc, ar, ai, br, bi, cIsComplex, kParts, pipe, AscendC::GetBlockNum(),
            AscendC::GetBlockIdx());
    }

    __aicore__ inline void Init(
        __gm__ uint8_t* t1, __gm__ uint8_t* t2, __gm__ uint8_t* t3, int32_t tempLdc, __gm__ uint8_t* cInOut, int32_t m,
        int32_t n, int32_t ldc, float ar, float ai, float br, float bi, int32_t cIsComplex, int32_t kParts,
        AscendC::TPipe* pipe,
        uint32_t blockNum, uint32_t blockIdx)
    {
        pipe_ = pipe;
        m_ = m;
        n_ = n;
        ldc_ = ldc;
        tempLdc_ = tempLdc;
        ar_ = ar;
        ai_ = ai;
        br_ = br;
        bi_ = bi;
        cIsComplex_ = cIsComplex;
        kParts_ = kParts;
        packMode_ = 0;
        if (kParts_ < 0) {
            packMode_ = 1;
            kParts_ = -kParts_;
        }
        if (kParts_ < 1) {
            kParts_ = 1;
        }
        panelElems_ = static_cast<uint64_t>(tempLdc) * static_cast<uint64_t>(n);
        uint64_t tempCount = panelElems_ * static_cast<uint64_t>(kParts_);
        t1Global_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t1), tempCount);
        t2Global_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t2), tempCount);
        t3Global_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(t3), tempCount);
        uint64_t cCount = static_cast<uint64_t>(ldc) * n * (cIsComplex ? 2 : 1);
        cGlobal_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cInOut), cCount);
        contig_ = (tempLdc_ == m_ && ldc_ == m_) ? 1 : 0;
        if (contig_) {
            AssignCoreRange(m_ * n_, blockNum, blockIdx, startCol_, endCol_);
        } else {
            AssignCoreRange(n_, blockNum, blockIdx, startCol_, endCol_);
            colsPerCore_ = endCol_ - startCol_;
        }
    }

    __aicore__ inline void ProcessContig()
    {
        int32_t offset = startCol_;
        int32_t cMul = cIsComplex_ ? 2 : 1;
        while (offset < endCol_) {
            int32_t count = endCol_ - offset;
            if (count > TILE) {
                count = TILE;
            }
            ProcessTile(static_cast<uint64_t>(offset), static_cast<uint64_t>(offset) * cMul, count);
            offset += count;
        }
    }

    __aicore__ inline void ProcessStrided()
    {
        int32_t cMul = cIsComplex_ ? 2 : 1;
        if ((endCol_ - startCol_) > 1) {
            ProcessRect(0, m_, startCol_, endCol_);
            return;
        }
        for (int32_t col = startCol_; col < endCol_; col++) {
            int32_t rowOffset = 0;
            while (rowOffset < m_) {
                int32_t count = m_ - rowOffset;
                if (count > TILE) {
                    count = TILE;
                }
                uint64_t tempOff = static_cast<uint64_t>(col) * tempLdc_ + rowOffset;
                uint64_t cOff = static_cast<uint64_t>(col) * ldc_ * cMul + rowOffset * cMul;
                ProcessTile(tempOff, cOff, count);
                rowOffset += count;
            }
        }
    }

    __aicore__ inline void SetupBuf()
    {
        pipe_->InitBuffer(t1Que_, BUF_NUM, TILE * sizeof(float));
        pipe_->InitBuffer(t2Que_, BUF_NUM, TILE * sizeof(float));
        pipe_->InitBuffer(cQue_, BUF_NUM, TILE * 2 * sizeof(float));
        pipe_->InitBuffer(outQue_, BUF_NUM, TILE * 2 * sizeof(float));
        pipe_->InitBuffer(calcBuf_, TILE * 6 * sizeof(float));
        pipe_->InitBuffer(idxBuf_, TILE * 2 * sizeof(uint32_t));
        idxReady_ = 0;
    }

    __aicore__ inline void ProcessRect(int32_t row0, int32_t row1, int32_t col0, int32_t col1)
    {
        if (row0 < 0) {
            row0 = 0;
        }
        if (col0 < 0) {
            col0 = 0;
        }
        if (row1 > m_) {
            row1 = m_;
        }
        if (col1 > n_) {
            col1 = n_;
        }
        if (row0 >= row1 || col0 >= col1) {
            return;
        }
        ProcessRectBody(row0, row1, col0, col1);
    }

    __aicore__ inline void ProcessRectCol(int32_t row0, int32_t row1, int32_t col, int32_t cMul)
    {
        int32_t rowOffset = row0;
        while (rowOffset < row1) {
            int32_t count = row1 - rowOffset;
            if (count > TILE) {
                count = TILE;
            }
            uint64_t tempOff = static_cast<uint64_t>(col) * tempLdc_ + rowOffset;
            uint64_t cOff = static_cast<uint64_t>(col) * ldc_ * cMul + rowOffset * cMul;
            ProcessTile(tempOff, cOff, count);
            rowOffset += count;
        }
    }

    __aicore__ inline void ProcessRectBody(int32_t row0, int32_t row1, int32_t col0, int32_t col1)
    {
        int32_t rows = row1 - row0;
        int32_t loadRows = (rows + 7) & ~7;
        if (loadRows < 8) {
            loadRows = rows;
        }
        if (row0 + loadRows > tempLdc_) {
            loadRows = rows;
        }
        int32_t cMul = cIsComplex_ ? 2 : 1;
        if (rows < 1 || col0 >= col1) {
            return;
        }
        int32_t maxCols = (loadRows > 0) ? (TILE / loadRows) : 1;
        if (maxCols < 1) {
            maxCols = 1;
        }
        bool useBlk = CanBlock(loadRows) && (row0 + loadRows <= tempLdc_);
        for (int32_t col = col0; col < col1;) {
            int32_t nc = col1 - col;
            if (nc > maxCols) {
                nc = maxCols;
            }
            if (nc > 1 && loadRows * nc <= TILE && useBlk) {
                ProcessBlock(row0, loadRows, rows, col, nc);
                col += nc;
                continue;
            }
            ProcessRectCol(row0, row1, col, cMul);
            col += 1;
        }
    }
