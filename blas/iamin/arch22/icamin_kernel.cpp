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
#include "icamin_kernel.h"

using namespace AscendC;

namespace {
constexpr uint32_t VECTOR_FLOATS = 64;
constexpr uint32_t BLOCK_FLOATS = 8;
constexpr uint32_t REDUCE_SCRATCH_BYTES = 2048;

template <IcaminLayout LAYOUT, bool SHORT_SLICE = false>
class IcaminKernel {
public:
    __aicore__ inline void Init(TPipe& pipe, GM_ADDR x, GM_ADDR result, GM_ADDR workspace, const IcaminArch22Tiling& t)
    {
        pipe_ = &pipe;
        tiling_ = &t;
        block_ = GetBlockIdx();
        start_ = block_ * t.perCore;
        count_ = start_ < t.n ? min(t.perCore, t.n - start_) : 0U;
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x));
        result_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(result), 1);
        if (t.cores > 1) {
            workspace_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
        }
        pipe_->InitBuffer(inputQueue_, t.buffers, t.inputFloats * sizeof(float));
        if constexpr (SHORT_SLICE) {
            InitShortStorage();
        } else {
            InitSummaryStorage<true>(t.tile);
        }
        if constexpr (LAYOUT != IcaminLayout::CONTIGUOUS) {
            pipe_->InitBuffer(imagBuf_, t.tile * sizeof(float));
            pipe_->InitBuffer(inputOffsetsBuf_, t.tile * sizeof(uint32_t));
            auto offsets = inputOffsetsBuf_.Get<int32_t>();
            uint32_t pitch = LAYOUT == IcaminLayout::DENSE_STRIDED ? t.incx * 8 : 32;
            CreateVecIndex(offsets, int32_t{0}, t.tile);
            PipeBarrier<PIPE_V>();
            Muls(offsets, offsets, static_cast<int32_t>(pitch), t.tile);
        }
    }

    template <IcaminReduction REDUCTION>
    __aicore__ inline void Process()
    {
        if constexpr (SHORT_SLICE) {
            ReduceShortSlice();
        } else {
            ReduceLocal();
        }
        if constexpr (REDUCTION == IcaminReduction::SINGLE_CORE) {
            WriteResult();
        } else {
            StorePartial<REDUCTION>();
            if constexpr (REDUCTION == IcaminReduction::FUSED) {
                SyncAll();
                if (block_ == 0) {
                    LoadSummary<false>();
                }
            }
        }
    }

    __aicore__ inline void Finish(TPipe& pipe, GM_ADDR result, GM_ADDR workspace, const IcaminArch22Tiling& t)
    {
        pipe_ = &pipe;
        tiling_ = &t;
        result_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(result), 1);
        workspace_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(workspace));
        InitSummaryStorage<false>(t.slots);
        LoadSummary<true>();
    }

private:
    __aicore__ inline void InitArena(uint32_t normCount, uint32_t slots)
    {
        // A2/A3 Compare packs one result per bit; each 32-byte block holds 256 results.
        uint32_t maskBytes = (normCount + 255) / 256 * 32;
        uint32_t bytes = normCount * sizeof(float) + maskBytes + slots * (ICAMIN_SLOT_FLOATS + 1) * sizeof(float);
        bytes += REDUCE_SCRATCH_BYTES + ICAMIN_METADATA_FLOATS * sizeof(float);
        pipe_->InitBuffer(arena_, bytes);
        auto data = arena_.Get<float>();
        norm_ = data;
        uint32_t offset = normCount;
        mask_ = data[offset].template ReinterpretCast<uint8_t>();
        offset += maskBytes / sizeof(float);
        slots_ = data[offset];
        offset += slots * ICAMIN_SLOT_FLOATS;
        slotOffsets_ = data[offset].template ReinterpretCast<uint32_t>();
        offset += slots;
        scratch_ = data[offset];
        offset += REDUCE_SCRATCH_BYTES / sizeof(float);
        meta_ = data[offset];
        offset += 3 * VECTOR_FLOATS + 6 * BLOCK_FLOATS;
        best_ = data[offset];
        offset += BLOCK_FLOATS;
        out_ = data[offset].template ReinterpretCast<int32_t>();
        // Compare writes packed bits, while Select reads aligned mask blocks.
        Duplicate(mask_.template ReinterpretCast<uint32_t>(), uint32_t{0}, maskBytes / sizeof(uint32_t));
        // Gather fetches a whole offset vector, including its masked-off lanes.
        Duplicate(meta_, 0.0f, ICAMIN_METADATA_FLOATS);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void InitShortStorage()
    {
        InitArena(tiling_->tile, BLOCK_FLOATS);
        Duplicate(slots_, 0.0f, BLOCK_FLOATS);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void ReduceShortSlice()
    {
        for (uint32_t offset = 0; offset < count_; offset += tiling_->tile) {
            uint32_t valid = min(tiling_->tile, count_ - offset);
            CopyIn(offset, valid);
            ComputeTile(valid, start_ + offset, 0);
            DataCopy(Candidate(), slots_, BLOCK_FLOATS);
            PipeBarrier<PIPE_V>();
            Accumulate(offset == 0);
        }
    }

    template <bool INITIALIZE_SLOTS>
    __aicore__ inline void InitSummaryStorage(uint32_t normCount)
    {
        InitArena(normCount, tiling_->slots);
        auto slotOffsets = slotOffsets_.template ReinterpretCast<int32_t>();
        CreateVecIndex(slotOffsets, int32_t{0}, tiling_->slots);
        CreateVecIndex(LaneOffsets(), int32_t{0}, BLOCK_FLOATS);
        Duplicate(IndexOffsets(), uint32_t{4}, BLOCK_FLOATS);
        Duplicate(One().template ReinterpretCast<int32_t>(), int32_t{1}, BLOCK_FLOATS);
        if constexpr (INITIALIZE_SLOTS) {
            Duplicate(slots_, 0.0f, tiling_->slots * ICAMIN_SLOT_FLOATS);
        }
        PipeBarrier<PIPE_V>();
        ShiftLeft(
            slotOffsets.template ReinterpretCast<uint32_t>(), slotOffsets.template ReinterpretCast<uint32_t>(),
            uint32_t{5}, tiling_->slots);
        ShiftLeft(
            LaneOffsets().template ReinterpretCast<uint32_t>(), LaneOffsets().template ReinterpretCast<uint32_t>(),
            uint32_t{2}, BLOCK_FLOATS);
    }

    __aicore__ inline void ReduceLocal()
    {
        uint32_t tiles = (count_ + tiling_->tile - 1) / tiling_->tile;
        if (tiling_->buffers == 2) {
            CopyIn(0, min(count_, tiling_->tile));
        }
        uint32_t groupCount = 0;
        bool firstGroup = true;
        for (uint32_t tile = 0; tile < tiles; ++tile) {
            uint32_t offset = tile * tiling_->tile;
            uint32_t valid = min(tiling_->tile, count_ - offset);
            if (tiling_->buffers == 1) {
                CopyIn(offset, valid);
            } else if (tile + 1 < tiles) {
                CopyIn(offset + tiling_->tile, min(tiling_->tile, count_ - offset - tiling_->tile));
            }
            ComputeTile(valid, start_ + offset, groupCount);
            ++groupCount;
            if (groupCount == ICAMIN_GROUP_TILES || tile + 1 == tiles) {
                SelectSlot(groupCount);
                Accumulate(firstGroup);
                firstGroup = false;
                groupCount = 0;
            }
        }
    }

    template <IcaminReduction REDUCTION>
    __aicore__ inline void StorePartial()
    {
        // Cache-line-separated slots avoid partial writes from different AIVs.
        auto ready = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(ready);
        WaitFlag<HardEvent::V_MTE3>(ready);
        DataCopy(workspace_[static_cast<uint64_t>(block_) * ICAMIN_SLOT_FLOATS], best_, BLOCK_FLOATS);
        if constexpr (REDUCTION == IcaminReduction::STAGED) {
            if (block_ == 0) {
                DataCopy(
                    workspace_[static_cast<uint64_t>(tiling_->cores) * ICAMIN_SLOT_FLOATS], FirstNorm(),
                    ICAMIN_FIRST_NORM_FLOATS);
            }
        }
    }

    template <bool LOAD_FIRST_NORM>
    __aicore__ inline void LoadSummary()
    {
        DataCopy(slots_, workspace_, tiling_->cores * ICAMIN_SLOT_FLOATS);
        if constexpr (LOAD_FIRST_NORM) {
            DataCopy(
                FirstNorm(), workspace_[static_cast<uint64_t>(tiling_->cores) * ICAMIN_SLOT_FLOATS],
                ICAMIN_FIRST_NORM_FLOATS);
        }
        auto loaded = static_cast<event_t>(pipe_->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(loaded);
        WaitFlag<HardEvent::MTE2_V>(loaded);
        SelectSlot(tiling_->cores);
        DataCopy(best_, Candidate(), BLOCK_FLOATS);
        PipeBarrier<PIPE_V>();
        WriteResult();
    }

    __aicore__ inline LocalTensor<float> FirstNorm() { return meta_; }
    __aicore__ inline LocalTensor<float> NewBroadcast() { return meta_[VECTOR_FLOATS]; }
    __aicore__ inline LocalTensor<float> OldBroadcast() { return meta_[2 * VECTOR_FLOATS]; }
    __aicore__ inline LocalTensor<float> Summary() { return meta_[3 * VECTOR_FLOATS]; }
    __aicore__ inline LocalTensor<float> Candidate() { return meta_[3 * VECTOR_FLOATS + 8]; }
    __aicore__ inline LocalTensor<uint32_t> IndexWork()
    {
        return meta_.template ReinterpretCast<uint32_t>()[3 * VECTOR_FLOATS + 16];
    }
    __aicore__ inline LocalTensor<int32_t> LaneOffsets()
    {
        return meta_.template ReinterpretCast<int32_t>()[3 * VECTOR_FLOATS + 24];
    }
    __aicore__ inline LocalTensor<uint32_t> IndexOffsets()
    {
        return meta_.template ReinterpretCast<uint32_t>()[3 * VECTOR_FLOATS + 32];
    }
    __aicore__ inline LocalTensor<float> One() { return meta_[3 * VECTOR_FLOATS + 40]; }

    __aicore__ inline void CopyIn(uint32_t offset, uint32_t valid)
    {
        auto local = inputQueue_.AllocTensor<float>();
        // Logical complex offsets are validated on host; float addresses still require 64 bits.
        uint64_t gmOffset = (static_cast<uint64_t>(start_) + offset) * tiling_->incx * 2;
        DataCopyExtParams copy{1, valid * 8U, 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0};
        if constexpr (LAYOUT == IcaminLayout::DENSE_STRIDED) {
            copy.blockLen = ((valid - 1) * tiling_->incx + 1) * 8U;
        } else if constexpr (LAYOUT == IcaminLayout::SPARSE_STRIDED) {
            copy.blockLen = 2 * sizeof(float);
            if (tiling_->incx <= ICAMIN_MAX_DMA_INCX) {
                copy.blockCount = static_cast<uint16_t>(valid);
                copy.srcStride = static_cast<uint32_t>(static_cast<uint64_t>(tiling_->incx - 1) * 8);
            } else {
                // The int32 span bound limits this case to at most four elements.
                // Keep their loads on MTE2 even when the byte gap exceeds the DMA field.
                for (uint32_t i = 0; i < valid; ++i) {
                    DataCopyPad(
                        local[i * BLOCK_FLOATS], input_[gmOffset + static_cast<uint64_t>(i) * tiling_->incx * 2], copy,
                        pad);
                }
                inputQueue_.EnQue(local);
                return;
            }
        }
        DataCopyPad(local, input_[gmOffset], copy, pad);
        inputQueue_.EnQue(local);
    }

    __aicore__ inline void ComputeMagnitudes(const LocalTensor<float>& input, uint32_t valid)
    {
        auto norm = norm_;
        if constexpr (LAYOUT == IcaminLayout::CONTIGUOUS) {
            Abs(input, input, valid * 2);
            PipeBarrier<PIPE_V>();
            // Repeat count is uint8_t. Split the larger tile into legal pair-reduction groups.
            for (uint32_t offset = 0; offset < valid; offset += ICAMIN_CONTIGUOUS_TILE) {
                uint32_t count = min(ICAMIN_CONTIGUOUS_TILE, valid - offset);
                uint32_t repeats = count / 32;
                if (repeats > 0) {
                    ReducePairElem<ReduceType::SUM>(norm[offset], input[offset * 2], VECTOR_FLOATS, repeats, 1, 1, 8);
                }
                if (count % 32 != 0) {
                    ReducePairElem<ReduceType::SUM>(
                        norm[offset + repeats * 32], input[offset * 2 + repeats * VECTOR_FLOATS],
                        static_cast<int32_t>((count % 32) * 2), 1, 1, 1, 8);
                }
            }
        } else {
            auto imag = imagBuf_.Get<float>();
            auto offsets = inputOffsetsBuf_.Get<uint32_t>();
            Gather(norm, input, offsets, 0, valid);
            Gather(imag, input, offsets, sizeof(float), valid);
            PipeBarrier<PIPE_V>();
            Abs(norm, norm, valid);
            Abs(imag, imag, valid);
            PipeBarrier<PIPE_V>();
            Add(norm, norm, imag, valid);
        }
    }

    __aicore__ inline void ComputeTile(uint32_t valid, uint32_t logicalOffset, uint32_t slot)
    {
        auto input = inputQueue_.DeQue<float>();
        auto norm = norm_;
        uint32_t aligned = (valid + VECTOR_FLOATS - 1) / VECTOR_FLOATS * VECTOR_FLOATS;
        if (aligned != valid) {
            Duplicate(norm, AscendC::NumericLimits<float>::Infinity(), aligned);
            PipeBarrier<PIPE_V>();
        }
        ComputeMagnitudes(input, valid);
        PipeBarrier<PIPE_V>();
        if (logicalOffset == 0) {
            DataCopy(FirstNorm(), norm, VECTOR_FLOATS);
            PipeBarrier<PIPE_V>();
        }
        // Later NaNs are ignored. A first-element NaN is handled at the final write.
        auto mask = mask_;
        Compare(mask, norm, norm, CMPMODE::EQ, aligned);
        PipeBarrier<PIPE_V>();
        Select(norm, mask, norm, AscendC::NumericLimits<float>::Infinity(), SELMODE::VSEL_TENSOR_SCALAR_MODE, aligned);
        PipeBarrier<PIPE_V>();
        auto tileResult = slots_[slot * ICAMIN_SLOT_FLOATS];
        // A2/A3 ReduceMin handles a partial repeat; exclude padding from the index domain.
        ReduceMin(tileResult, norm, scratch_, valid, true);
        PipeBarrier<PIPE_V>();
        // Only lane 1 holds index bits. Never convert a global index through float32.
        uint64_t indexMask[2] = {2, 0};
        auto index = tileResult.template ReinterpretCast<int32_t>();
        Adds(index, index, static_cast<int32_t>(logicalOffset), indexMask, 1, {1, 1, 8, 8});
        PipeBarrier<PIPE_V>();
        inputQueue_.FreeTensor(input);
    }

    __aicore__ inline void SelectSlot(uint32_t count)
    {
        auto slots = slots_;
        auto norm = norm_;
        Gather(norm, slots, slotOffsets_, 0, count);
        PipeBarrier<PIPE_V>();
        ReduceMin(Summary(), norm, scratch_, count, true);
        PipeBarrier<PIPE_V>();
        // Expand the winning slot number into eight byte offsets entirely on the V pipe.
        Gather(IndexWork(), Summary().template ReinterpretCast<uint32_t>(), IndexOffsets(), 0, BLOCK_FLOATS);
        PipeBarrier<PIPE_V>();
        ShiftLeft(IndexWork(), IndexWork(), uint32_t{5}, BLOCK_FLOATS);
        PipeBarrier<PIPE_V>();
        Add(IndexWork().template ReinterpretCast<int32_t>(), IndexWork().template ReinterpretCast<int32_t>(),
            LaneOffsets(), BLOCK_FLOATS);
        PipeBarrier<PIPE_V>();
        Gather(Candidate(), slots, IndexWork(), 0, BLOCK_FLOATS);
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void Accumulate(bool first)
    {
        auto best = best_;
        if (first) {
            DataCopy(best, Candidate(), BLOCK_FLOATS);
        } else {
            // Brcb expands eight floats into eight 32-byte blocks (64 floats).
            // Compare requires 64 float lanes on A2/A3; Select uses the first eight value bits.
            Brcb(NewBroadcast(), Candidate(), 1, {1, 8});
            Brcb(OldBroadcast(), best, 1, {1, 8});
            PipeBarrier<PIPE_V>();
            Compare(mask_, NewBroadcast(), OldBroadcast(), CMPMODE::LT, VECTOR_FLOATS);
            PipeBarrier<PIPE_V>();
            Select(best, mask_, Candidate(), best, SELMODE::VSEL_TENSOR_TENSOR_MODE, BLOCK_FLOATS);
        }
        PipeBarrier<PIPE_V>();
    }

    __aicore__ inline void WriteResult()
    {
        auto out = out_;
        Gather(
            out.template ReinterpretCast<uint32_t>(), best_.template ReinterpretCast<uint32_t>(), IndexOffsets(), 0,
            BLOCK_FLOATS);
        PipeBarrier<PIPE_V>();
        Adds(out, out, int32_t{1}, BLOCK_FLOATS);
        Compare(mask_, FirstNorm(), FirstNorm(), CMPMODE::EQ, VECTOR_FLOATS);
        PipeBarrier<PIPE_V>();
        Select(
            out.template ReinterpretCast<float>(), mask_, out.template ReinterpretCast<float>(), One(),
            SELMODE::VSEL_TENSOR_TENSOR_MODE, BLOCK_FLOATS);
        auto ready = static_cast<event_t>(pipe_->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(ready);
        WaitFlag<HardEvent::V_MTE3>(ready);
        DataCopyExtParams copy{1, sizeof(int32_t), 0, 0, 0};
        DataCopyPad(result_, out, copy);
    }

    TPipe* pipe_ = nullptr;
    TBuf<QuePosition::VECCALC> arena_;
    LocalTensor<float> norm_;
    LocalTensor<uint8_t> mask_;
    LocalTensor<float> slots_;
    LocalTensor<uint32_t> slotOffsets_;
    LocalTensor<float> scratch_;
    LocalTensor<float> meta_;
    LocalTensor<float> best_;
    LocalTensor<int32_t> out_;
    TQue<QuePosition::VECIN, 2> inputQueue_;
    TBuf<QuePosition::VECCALC> imagBuf_;
    TBuf<QuePosition::VECCALC> inputOffsetsBuf_;
    GlobalTensor<float> input_;
    GlobalTensor<float> workspace_;
    GlobalTensor<int32_t> result_;
    const IcaminArch22Tiling* tiling_ = nullptr;
    uint32_t block_ = 0;
    uint32_t start_ = 0;
    uint32_t count_ = 0;
};
} // namespace

template <IcaminLayout LAYOUT, IcaminReduction REDUCTION, bool SHORT_SLICE = false>
__global__ __aicore__ void icamin_arch22_vector_kernel(
    GM_ADDR x, GM_ADDR result, GM_ADDR workspace, IcaminArch22Tiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    IcaminKernel<LAYOUT, SHORT_SLICE> op;
    op.Init(pipe, x, result, workspace, tiling);
    op.template Process<REDUCTION>();
    pipe.Destroy();
    ResetMask();
}

template <IcaminLayout LAYOUT>
__schedmode__(1) __global__ __mix__(0, 1) __aicore__
    void icamin_arch22_kernel(GM_ADDR x, GM_ADDR result, GM_ADDR workspace, IcaminArch22Tiling tiling)
{
    InitSocState();
    TPipe pipe;
    IcaminKernel<LAYOUT> op;
    op.Init(pipe, x, result, workspace, tiling);
    op.template Process<IcaminReduction::FUSED>();
    pipe.Destroy();
    ResetMask();
}

__global__ __aicore__ void icamin_arch22_finish_kernel(GM_ADDR result, GM_ADDR workspace, IcaminArch22Tiling tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    IcaminKernel<IcaminLayout::CONTIGUOUS> op;
    op.Finish(pipe, result, workspace, tiling);
    pipe.Destroy();
    ResetMask();
}

__global__ __aicore__ void icamin_arch22_one_kernel(GM_ADDR result)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    TBuf<QuePosition::VECOUT> buffer;
    pipe.InitBuffer(buffer, 32);
    auto out = buffer.Get<int32_t>();
    Duplicate(out, int32_t{1}, BLOCK_FLOATS);
    auto ready = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(ready);
    WaitFlag<HardEvent::V_MTE3>(ready);
    GlobalTensor<int32_t> dst;
    dst.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(result), 1);
    DataCopyExtParams copy{1, sizeof(int32_t), 0, 0, 0};
    DataCopyPad(dst, out, copy);
    pipe.Destroy();
    ResetMask();
}

template <IcaminLayout LAYOUT>
void LaunchIcamin(uint8_t* x, uint8_t* result, uint8_t* workspace, const IcaminArch22Tiling& t, void* stream)
{
    if (t.reduction == IcaminReduction::SINGLE_CORE) {
        icamin_arch22_vector_kernel<LAYOUT, IcaminReduction::SINGLE_CORE>
            <<<1, nullptr, stream>>>(x, result, workspace, t);
    } else if (t.reduction == IcaminReduction::STAGED) {
        if (t.buffers == 1) {
            icamin_arch22_vector_kernel<LAYOUT, IcaminReduction::STAGED, true>
                <<<t.cores, nullptr, stream>>>(x, result, workspace, t);
        } else {
            icamin_arch22_vector_kernel<LAYOUT, IcaminReduction::STAGED>
                <<<t.cores, nullptr, stream>>>(x, result, workspace, t);
        }
        // Kernel completion on the bound stream makes every partial visible to this AIV.
        icamin_arch22_finish_kernel<<<1, nullptr, stream>>>(result, workspace, t);
    } else {
        icamin_arch22_kernel<LAYOUT><<<t.cores, nullptr, stream>>>(x, result, workspace, t);
    }
}

void icamin_arch22_do(uint8_t* x, uint8_t* result, uint8_t* workspace, const IcaminArch22Tiling& tiling, void* stream)
{
    if (tiling.n == 1) {
        icamin_arch22_one_kernel<<<1, nullptr, stream>>>(result);
    } else if (tiling.layout == IcaminLayout::CONTIGUOUS) {
        LaunchIcamin<IcaminLayout::CONTIGUOUS>(x, result, workspace, tiling, stream);
    } else if (tiling.layout == IcaminLayout::DENSE_STRIDED) {
        LaunchIcamin<IcaminLayout::DENSE_STRIDED>(x, result, workspace, tiling, stream);
    } else {
        LaunchIcamin<IcaminLayout::SPARSE_STRIDED>(x, result, workspace, tiling, stream);
    }
}
