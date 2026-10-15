/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
// Target: Ascend 950, DAV-3510. Not compatible with 910B/A2/A3 SIMT.
#include <cstdint>
#include "acl/acl.h"
#include "kernel_operator.h"
#include "simt_api/asc_simt.h"
#include "cscal_config.h"
#include "cscal_tiling_data.h"
using namespace AscendC;

__SIMT_DEVICE_FUNCTIONS_DECL__ inline float CscalCopySign(float magnitude, float sign)
{
    return __uint_as_float((__float_as_uint(magnitude) & 0x7fffffffu) | (__float_as_uint(sign) & 0x80000000u));
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bool CscalIsInf(float value)
{
    const float inf = __builtin_inff();
    return value == inf || value == -inf;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline float CscalClearNaN(float value)
{
    return value != value ? CscalCopySign(0.0f, value) : value;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bool CscalNormalizeInfPair(float& a, float& b, float& c, float& d)
{
    if (!CscalIsInf(a) && !CscalIsInf(b))
        return false;
    a = CscalCopySign(CscalIsInf(a) ? 1.0f : 0.0f, a);
    b = CscalCopySign(CscalIsInf(b) ? 1.0f : 0.0f, b);
    c = CscalClearNaN(c);
    d = CscalClearNaN(d);
    return true;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline void CscalMultiply(float a, float b, float c, float d, float& rr, float& ri)
{
    const float p0 = a * c, p1 = b * d;
    const float p2 = a * d, p3 = b * c;
    rr = p0 - p1;
    ri = p2 + p3;
    if (!(rr != rr && ri != ri))
        return;

    // Recovery observes the first normalization before handling the input.
    bool repair = CscalNormalizeInfPair(a, b, c, d);
    const bool inputInf = CscalNormalizeInfPair(c, d, a, b);
    repair = repair || inputInf;
    if (!repair && (CscalIsInf(p0) || CscalIsInf(p1) || CscalIsInf(p2) || CscalIsInf(p3))) {
        a = CscalClearNaN(a);
        b = CscalClearNaN(b);
        c = CscalClearNaN(c);
        d = CscalClearNaN(d);
        repair = true;
    }
    if (repair) {
        const float inf = __builtin_inff();
        rr = inf * (a * c - b * d);
        ri = inf * (a * d + b * c);
    }
}

__simt_vf__ __aicore__ LAUNCH_BOUND(256) inline void CscalSimtBody(
    uint64_t begin, uint64_t end, uint64_t incx, float ar, float ai, uint32_t writeZero, __gm__ float* x)
{
    for (uint64_t i = begin + threadIdx.x; i < end; i += blockDim.x) {
        const uint64_t k = uint64_t(2) * i * incx; // avoid int32 address overflow
        if (writeZero) {
            x[k] = 0.0f;
            x[k + 1] = 0.0f;
        } else {
            const float r = x[k], im = x[k + 1]; // read BOTH before either store
            float rr, ri;
            CscalMultiply(ar, ai, r, im, rr, ri);
            x[k] = rr;
            x[k + 1] = ri;
        }
    }
}

__global__ __aicore__ void cscal_simt_kernel(GM_ADDR x, CscalTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    const uint64_t core = GetBlockIdx();
    const uint64_t begin = t.n * core / t.coreCount;
    const uint64_t end = t.n * (core + 1) / t.coreCount;
    asc_vf_call<CscalSimtBody>(
        dim3{t.simtThreads, 1, 1}, begin, end, t.incx, t.alphaReal, t.alphaImag, t.writeZero,
        reinterpret_cast<__gm__ float*>(x));
}

#if CSCAL_ENABLE_VECTOR
namespace CR = AscendC::Reg;

// Classify by bits so infinities, signed zeros and NaNs follow the SIMT path
// independently of the device's floating comparison behavior.
template <CMPMODE mode>
__simd_callee__ inline void CscalRegClassify(
    CR::RegTensor<float>& value, CR::MaskReg& result, CR::RegTensor<uint32_t>& absMask,
    CR::RegTensor<uint32_t>& infBits, CR::MaskReg& valid)
{
    CR::RegTensor<uint32_t> magnitude;
    CR::And(magnitude, (CR::RegTensor<uint32_t>&)value, absMask, valid);
    CR::Compare<uint32_t, mode>(result, magnitude, infBits, valid);
}

__simd_callee__ inline void CscalRegSignedUnit(
    CR::RegTensor<float>& dst, CR::RegTensor<float>& value, CR::MaskReg& isInf, CR::RegTensor<float>& one,
    CR::RegTensor<float>& zero, CR::RegTensor<uint32_t>& signMask, CR::MaskReg& valid)
{
    CR::RegTensor<uint32_t> sign;
    CR::And(sign, (CR::RegTensor<uint32_t>&)value, signMask, valid);
    CR::Select(dst, one, zero, isInf);
    CR::Or((CR::RegTensor<uint32_t>&)dst, (CR::RegTensor<uint32_t>&)dst, sign, valid);
}

__simd_callee__ inline void CscalRegClearNaN(
    CR::RegTensor<float>& value, CR::MaskReg& condition, CR::RegTensor<uint32_t>& absMask,
    CR::RegTensor<uint32_t>& infBits, CR::RegTensor<uint32_t>& signMask, CR::MaskReg& valid)
{
    CR::RegTensor<uint32_t> signedZero;
    CR::MaskReg isNaN, clear;
    CscalRegClassify<CMPMODE::GT>(value, isNaN, absMask, infBits, valid);
    CR::And(clear, isNaN, condition, valid);
    CR::And(signedZero, (CR::RegTensor<uint32_t>&)value, signMask, valid);
    CR::Select(value, (CR::RegTensor<float>&)signedZero, value, clear);
}

__simd_callee__ inline void CscalRegProducts(
    CR::RegTensor<float>& c, CR::RegTensor<float>& d, float ar, float ai, CR::RegTensor<float>& p0,
    CR::RegTensor<float>& p1, CR::RegTensor<float>& p2, CR::RegTensor<float>& p3, CR::MaskReg& valid)
{
    // Keep all four rounded products separate, as in the accepted SIMT.
    CR::Muls(p0, c, ar, valid);
    CR::Muls(p1, d, ai, valid);
    CR::Muls(p2, d, ar, valid);
    CR::Muls(p3, c, ai, valid);
}

__simd_callee__ inline void CscalRegCombineProducts(
    CR::RegTensor<float>& rr, CR::RegTensor<float>& ri, CR::RegTensor<float>& p0, CR::RegTensor<float>& p1,
    CR::RegTensor<float>& p2, CR::RegTensor<float>& p3, CR::MaskReg& valid)
{
    CR::Sub(rr, p0, p1, valid);
    CR::Add(ri, p2, p3, valid);
}

__simd_callee__ inline void CscalRegBothNaN(
    CR::RegTensor<float>& rr, CR::RegTensor<float>& ri, CR::MaskReg& bothNaN, CR::RegTensor<uint32_t>& absMask,
    CR::RegTensor<uint32_t>& infBits, CR::MaskReg& valid)
{
    CR::MaskReg rrNaN, riNaN;
    CscalRegClassify<CMPMODE::GT>(rr, rrNaN, absMask, infBits, valid);
    CscalRegClassify<CMPMODE::GT>(ri, riNaN, absMask, infBits, valid);
    CR::And(bothNaN, rrNaN, riNaN, valid);
}

__simd_callee__ inline void CscalRegNormalizeInfPair(
    CR::RegTensor<float>& a, CR::RegTensor<float>& b, CR::MaskReg& pairInf, CR::RegTensor<uint32_t>& absMask,
    CR::RegTensor<uint32_t>& infBits, CR::RegTensor<float>& one, CR::RegTensor<float>& zero,
    CR::RegTensor<uint32_t>& signMask, CR::MaskReg& valid)
{
    CR::MaskReg aInf, bInf;
    CR::RegTensor<float> normalized;
    CscalRegClassify<CMPMODE::EQ>(a, aInf, absMask, infBits, valid);
    CscalRegClassify<CMPMODE::EQ>(b, bInf, absMask, infBits, valid);
    CR::Or(pairInf, aInf, bInf, valid);
    CscalRegSignedUnit(normalized, a, aInf, one, zero, signMask, valid);
    CR::Select(a, normalized, a, pairInf);
    CscalRegSignedUnit(normalized, b, bInf, one, zero, signMask, valid);
    CR::Select(b, normalized, b, pairInf);
}

__simd_callee__ inline void CscalRegProductInf(
    CR::RegTensor<float>& p0, CR::RegTensor<float>& p1, CR::RegTensor<float>& p2, CR::RegTensor<float>& p3,
    CR::MaskReg& productInf, CR::RegTensor<uint32_t>& absMask, CR::RegTensor<uint32_t>& infBits, CR::MaskReg& valid)
{
    CR::MaskReg partInf;
    CscalRegClassify<CMPMODE::EQ>(p0, productInf, absMask, infBits, valid);
    CscalRegClassify<CMPMODE::EQ>(p1, partInf, absMask, infBits, valid);
    CR::Or(productInf, productInf, partInf, valid);
    CscalRegClassify<CMPMODE::EQ>(p2, partInf, absMask, infBits, valid);
    CR::Or(productInf, productInf, partInf, valid);
    CscalRegClassify<CMPMODE::EQ>(p3, partInf, absMask, infBits, valid);
    CR::Or(productInf, productInf, partInf, valid);
}

__simd_callee__ inline void CscalRegSelectRecovery(
    CR::RegTensor<float>& rr, CR::RegTensor<float>& ri, CR::RegTensor<float>& p0, CR::RegTensor<float>& p1,
    CR::RegTensor<float>& p2, CR::RegTensor<float>& p3, CR::MaskReg& repair, CR::MaskReg& valid)
{
    CR::Sub(p0, p0, p1, valid);
    CR::Add(p2, p2, p3, valid);
    CR::Muls(p0, p0, __builtin_inff(), valid);
    CR::Muls(p2, p2, __builtin_inff(), valid);
    CR::Select(rr, p0, rr, repair);
    CR::Select(ri, p2, ri, repair);
}

__simd_vf__ inline void CscalRegBody(
    __ubuf__ float* src, __ubuf__ float* dst, float ar, float ai, uint32_t count, uint16_t loops)
{
    // Each paired load reads 64 complex values, including the masked UB tail.
    // Queue allocations are rounded to that width; GM transfers remain exact.
    CR::RegTensor<uint32_t> absMask, signMask, infBits;
    CR::RegTensor<float> one, zero;
    CR::Duplicate(absMask, uint32_t(0x7fffffffu));
    CR::Duplicate(signMask, uint32_t(0x80000000u));
    CR::Duplicate(infBits, uint32_t(0x7f800000u));
    CR::Duplicate(one, 1.0f);
    CR::Duplicate(zero, 0.0f);

    for (uint16_t i = 0; i < loops; ++i) {
        CR::MaskReg valid = CR::UpdateMask<float>(count);
        CR::RegTensor<float> c, d, p0, p1, p2, p3, rr, ri;
        CR::LoadAlign<float, CR::LoadDist::DIST_DINTLV_B32>(c, d, src + uint32_t(i) * 128);
        CscalRegProducts(c, d, ar, ai, p0, p1, p2, p3, valid);
        CscalRegCombineProducts(rr, ri, p0, p1, p2, p3, valid);

        CR::MaskReg bothNaN;
        CscalRegBothNaN(rr, ri, bothNaN, absMask, infBits, valid);

        CR::RegTensor<float> a, b;
        CR::Duplicate(a, ar);
        CR::Duplicate(b, ai);
        CR::MaskReg repair, inputInf;

        // First SIMT recovery branch: normalize an infinite alpha, and replace
        // NaN input components with zeros carrying their original sign bits.
        CscalRegNormalizeInfPair(a, b, repair, absMask, infBits, one, zero, signMask, valid);
        CscalRegClearNaN(c, repair, absMask, infBits, signMask, valid);
        CscalRegClearNaN(d, repair, absMask, infBits, signMask, valid);

        // Second branch observes the values after the first branch, matching
        // the accepted SIMT recovery order exactly.
        CscalRegNormalizeInfPair(c, d, inputInf, absMask, infBits, one, zero, signMask, valid);
        CscalRegClearNaN(a, inputInf, absMask, infBits, signMask, valid);
        CscalRegClearNaN(b, inputInf, absMask, infBits, signMask, valid);
        CR::Or(repair, repair, inputInf, valid);

        // Third branch is enabled only if neither prior branch applied and an
        // original product overflowed. Preserve the original p0..p3 until here.
        {
            CR::MaskReg noRepair, productInf, overflow;
            CR::Not(noRepair, repair, valid);
            CscalRegProductInf(p0, p1, p2, p3, productInf, absMask, infBits, valid);
            CR::And(overflow, productInf, noRepair, valid);
            CscalRegClearNaN(a, overflow, absMask, infBits, signMask, valid);
            CscalRegClearNaN(b, overflow, absMask, infBits, signMask, valid);
            CscalRegClearNaN(c, overflow, absMask, infBits, signMask, valid);
            CscalRegClearNaN(d, overflow, absMask, infBits, signMask, valid);
            CR::Or(repair, repair, overflow, valid);
        }
        CR::And(repair, repair, bothNaN, valid);

        // Only selected NaN pairs receive the recomputed result. Other lanes
        // retain the ordinary four-product result, including any signed zeros.
        CR::Mul(p0, a, c, valid);
        CR::Mul(p1, b, d, valid);
        CR::Mul(p2, a, d, valid);
        CR::Mul(p3, b, c, valid);
        CscalRegSelectRecovery(rr, ri, p0, p1, p2, p3, repair, valid);
        CR::StoreAlign<float, CR::StoreDist::DIST_INTLV_B32>(dst + uint32_t(i) * 128, rr, ri, valid);
    }
}

// Host classification is uniform for the whole launch. With finite alpha the
// first recovery branch is impossible, and alpha never needs NaN cleanup.
__simd_vf__ inline void CscalRegFiniteAlphaBody(
    __ubuf__ float* src, __ubuf__ float* dst, float ar, float ai, uint32_t count, uint16_t loops)
{
    CR::RegTensor<uint32_t> absMask, infBits;
    CR::RegTensor<float> zero;
    CR::Duplicate(absMask, uint32_t(0x7fffffffu));
    CR::Duplicate(infBits, uint32_t(0x7f800000u));
    CR::Duplicate(zero, 0.0f);

    for (uint16_t i = 0; i < loops; ++i) {
        CR::MaskReg valid = CR::UpdateMask<float>(count);
        CR::RegTensor<float> c, d, p0, p1, p2, p3, rr, ri;
        CR::LoadAlign<float, CR::LoadDist::DIST_DINTLV_B32>(c, d, src + uint32_t(i) * 128);
        CscalRegProducts(c, d, ar, ai, p0, p1, p2, p3, valid);
        CscalRegCombineProducts(rr, ri, p0, p1, p2, p3, valid);

        CR::MaskReg bothNaN, repair;
        CscalRegBothNaN(rr, ri, bothNaN, absMask, infBits, valid);
        // Preserve overflow recovery for finite alpha and arbitrary input,
        // e.g. a large finite component paired with a NaN component.
        CscalRegProductInf(p0, p1, p2, p3, repair, absMask, infBits, valid);
        CR::And(repair, repair, bothNaN, valid);
        {
            // For finite alpha, recovery can reuse the original rounded
            // products after replacing NaNs with +0. Signed-zero differences
            // inside recovery become NaN under the final Inf multiplication;
            // non-recovery lanes retain rr/ri, including their signed zeros.
            CR::MaskReg isNaN;
            CscalRegClassify<CMPMODE::GT>(p0, isNaN, absMask, infBits, valid);
            CR::Select(p0, zero, p0, isNaN);
            CscalRegClassify<CMPMODE::GT>(p1, isNaN, absMask, infBits, valid);
            CR::Select(p1, zero, p1, isNaN);
            CscalRegClassify<CMPMODE::GT>(p2, isNaN, absMask, infBits, valid);
            CR::Select(p2, zero, p2, isNaN);
            CscalRegClassify<CMPMODE::GT>(p3, isNaN, absMask, infBits, valid);
            CR::Select(p3, zero, p3, isNaN);
        }

        // Select only when the ordinary pair is NaN and an original product
        // is infinite. No input normalization or recovery products are needed.
        CscalRegSelectRecovery(rr, ri, p0, p1, p2, p3, repair, valid);
        CR::StoreAlign<float, CR::StoreDist::DIST_INTLV_B32>(dst + uint32_t(i) * 128, rr, ri, valid);
    }
}

// Queue/DMA scheduling is unchanged; each tile's compute stays in registers.
class CscalVector {
public:
    __aicore__ inline CscalVector() {}
    __aicore__ inline void Init(GM_ADDR x, const CscalTilingData& t, TPipe* pipe)
    {
        t_ = t;
        pipe_ = pipe;
        const uint64_t c = GetBlockIdx(), groups = t.n / 8;
        const uint64_t q = groups / t.coreCount, rem = groups % t.coreCount;
        start_ = (c * q + (c < rem ? c : rem)) * 8;
        count_ = (q + (c < rem)) * 8;
        if (c + 1 == t.coreCount)
            count_ += t.n % 8;
        x_.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(x), 2 * t.n);
        // Queue buffers are 32-byte aligned; RegBase uses 64-complex UB vectors.
        const uint32_t regTileComplex = (t.tileComplex + 63) / 64 * 64;
        pipe_->InitBuffer(in_, CSCAL_QUEUE_DEPTH, 8 * regTileComplex);
        pipe_->InitBuffer(out_, CSCAL_QUEUE_DEPTH, 8 * regTileComplex);
    }
    __aicore__ inline void Process()
    {
        if (!count_)
            return;
        const uint64_t tiles = (count_ + t_.tileComplex - 1) / t_.tileComplex;
        if (t_.writeZero) {
            for (uint64_t j = 0; j < tiles; ++j) {
                auto dst = out_.AllocTensor<float>();
                Duplicate(dst, 0.0f, int32_t(2 * PaddedCount(j)));
                out_.EnQue(dst);
                CopyOut(j);
            }
            return;
        }
#if CSCAL_QUEUE_DEPTH == 2
        CopyIn(0);
        for (uint64_t j = 0; j < tiles; ++j) {
            if (j + 1 < tiles)
                CopyIn(j + 1); // real one-tile look-ahead, not just depth=2
            Compute(j);
            CopyOut(j);
        }
#else
        for (uint64_t j = 0; j < tiles; ++j) {
            CopyIn(j);
            Compute(j);
            CopyOut(j);
        }
#endif
    }

private:
    __aicore__ inline uint32_t Count(uint64_t j)
    {
        uint64_t remain = count_ - j * t_.tileComplex;
        return uint32_t(remain < t_.tileComplex ? remain : t_.tileComplex);
    }
    __aicore__ inline uint32_t PaddedCount(uint64_t j) { return (Count(j) + 7) / 8 * 8; }
    __aicore__ inline void CopyIn(uint64_t j)
    {
        const uint32_t n = Count(j), m = PaddedCount(j);
        LocalTensor<float> src = in_.AllocTensor<float>();
        if (n != m) {
            // Only the final tile needs zero padding. Initialize the entire local span
            // before MTE2 overwrites its valid prefix; never read past the GM allocation.
            Duplicate(src, 0.0f, int32_t(2 * m));
            const auto event = pipe_->FetchEventID<HardEvent::V_MTE2>();
            SetFlag<HardEvent::V_MTE2>(event);
            WaitFlag<HardEvent::V_MTE2>(event);
        }
        DataCopyExtParams p{1, uint32_t(n * 8), 0, 0, 0};
        DataCopyPadExtParams<float> pad{false, 0, 0, 0.0f};
        DataCopyPad(src, x_[2 * (start_ + j * t_.tileComplex)], p, pad);
        in_.EnQue(src);
    }
    __aicore__ inline void Compute(uint64_t j)
    {
        const uint32_t n = Count(j);
        LocalTensor<float> src = in_.DeQue<float>();
        LocalTensor<float> dst = out_.AllocTensor<float>();
        if (t_.finiteAlpha) {
            asc_vf_call<CscalRegFiniteAlphaBody>(
                reinterpret_cast<__ubuf__ float*>(src.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(dst.GetPhyAddr()), t_.alphaReal, t_.alphaImag, n,
                uint16_t((n + 63) / 64));
        } else {
            asc_vf_call<CscalRegBody>(
                reinterpret_cast<__ubuf__ float*>(src.GetPhyAddr()),
                reinterpret_cast<__ubuf__ float*>(dst.GetPhyAddr()), t_.alphaReal, t_.alphaImag, n,
                uint16_t((n + 63) / 64));
        }
        in_.FreeTensor(src);
        out_.EnQue(dst);
    }
    __aicore__ inline void CopyOut(uint64_t j)
    {
        auto dst = out_.DeQue<float>();
        DataCopyExtParams p{1, uint32_t(Count(j) * 8), 0, 0, 0};
        // Only valid bytes are written: no rounded GM store over neighbours/guards.
        DataCopyPad(x_[2 * (start_ + j * t_.tileComplex)], dst, p);
        out_.FreeTensor(dst);
    }
    CscalTilingData t_{};
    TPipe* pipe_;
    uint64_t start_, count_;
    GlobalTensor<float> x_;
    TQue<QuePosition::VECIN, CSCAL_QUEUE_DEPTH> in_;
    TQue<QuePosition::VECOUT, CSCAL_QUEUE_DEPTH> out_;
};
__global__ __aicore__ void cscal_vector_kernel(GM_ADDR x, CscalTilingData t)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CscalVector op;
    op.Init(x, t, &pipe);
    op.Process();
}
#endif

void cscal_kernel_do(uint8_t* x, const CscalTilingData& t, void* stream)
{
    auto s = static_cast<aclrtStream>(stream);
#if CSCAL_ENABLE_VECTOR
    if (t.useVector) {
        cscal_vector_kernel<<<t.coreCount, nullptr, s>>>(x, t);
        return;
    }
#endif
    cscal_simt_kernel<<<t.coreCount, nullptr, s>>>(x, t);
}
