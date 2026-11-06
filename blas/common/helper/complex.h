/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <math.h>
#include "cann_ops_blas_common.h"

#ifdef __cplusplus

/* ── aclblasComplex ── */

inline aclblasComplex operator+(aclblasComplex a, aclblasComplex b) { return {a.real + b.real, a.imag + b.imag}; }
inline aclblasComplex operator-(aclblasComplex a, aclblasComplex b) { return {a.real - b.real, a.imag - b.imag}; }
inline aclblasComplex operator*(aclblasComplex a, aclblasComplex b)
{
    return {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
}
inline aclblasComplex operator/(aclblasComplex a, aclblasComplex b)
{
    // issue #828: Smith 单边缩放除法。原实现 (ac+bd)/(c^2+d^2) 在除数模长较大
    // （|c| 或 |d| 约 > 1.8e19，float 域）时 c^2+d^2 上溢为 inf，分子/inf 把本可
    // 精确表示的有限结果错误归零。以模长较大的分量缩放分子分母后不存在平方项：
    // |r|<=1，denom 有界，仅 c、d 同为零时 r=0/0=NaN，结果自然为 NaN（与原
    // denom==0 特判 {NAN,NAN} 口径一致）。operator/= 与 double 版本同步。
    if (std::fabs(b.real) >= std::fabs(b.imag)) {
        const float r = b.imag / b.real;
        const float denom = b.real + b.imag * r;
        return {(a.real + a.imag * r) / denom, (a.imag - a.real * r) / denom};
    }
    const float r = b.real / b.imag;
    const float denom = b.imag + b.real * r;
    return {(a.real * r + a.imag) / denom, (a.imag * r - a.real) / denom};
}
inline bool operator==(aclblasComplex a, aclblasComplex b) { return a.real == b.real && a.imag == b.imag; }
inline bool operator!=(aclblasComplex a, aclblasComplex b) { return !(a == b); }

inline aclblasComplex& operator+=(aclblasComplex& a, aclblasComplex b)
{
    a.real += b.real;
    a.imag += b.imag;
    return a;
}
inline aclblasComplex& operator-=(aclblasComplex& a, aclblasComplex b)
{
    a.real -= b.real;
    a.imag -= b.imag;
    return a;
}
inline aclblasComplex& operator*=(aclblasComplex& a, aclblasComplex b)
{
    a = a * b;
    return a;
}
inline aclblasComplex& operator/=(aclblasComplex& a, aclblasComplex b)
{
    // issue #828: 复用 Smith 缩放版 operator/，与单目除法保持同一实现
    a = a / b;
    return a;
}

inline aclblasComplex operator*(aclblasComplex a, float s) { return {a.real * s, a.imag * s}; }
inline aclblasComplex operator*(float s, aclblasComplex a) { return {s * a.real, s * a.imag}; }

inline float blasComplexAbs(aclblasComplex z) { return hypotf(z.real, z.imag); }

/* ── aclblasDoubleComplex ── */

inline aclblasDoubleComplex operator+(aclblasDoubleComplex a, aclblasDoubleComplex b)
{
    return {a.real + b.real, a.imag + b.imag};
}
inline aclblasDoubleComplex operator-(aclblasDoubleComplex a, aclblasDoubleComplex b)
{
    return {a.real - b.real, a.imag - b.imag};
}
inline aclblasDoubleComplex operator*(aclblasDoubleComplex a, aclblasDoubleComplex b)
{
    return {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
}
inline aclblasDoubleComplex operator/(aclblasDoubleComplex a, aclblasDoubleComplex b)
{
    // issue #828: Smith 单边缩放除法（double 版，推导见 float 版注释）
    if (std::fabs(b.real) >= std::fabs(b.imag)) {
        const double r = b.imag / b.real;
        const double denom = b.real + b.imag * r;
        return {(a.real + a.imag * r) / denom, (a.imag - a.real * r) / denom};
    }
    const double r = b.real / b.imag;
    const double denom = b.imag + b.real * r;
    return {(a.real * r + a.imag) / denom, (a.imag * r - a.real) / denom};
}
inline bool operator==(aclblasDoubleComplex a, aclblasDoubleComplex b) { return a.real == b.real && a.imag == b.imag; }
inline bool operator!=(aclblasDoubleComplex a, aclblasDoubleComplex b) { return !(a == b); }

inline aclblasDoubleComplex& operator+=(aclblasDoubleComplex& a, aclblasDoubleComplex b)
{
    a.real += b.real;
    a.imag += b.imag;
    return a;
}
inline aclblasDoubleComplex& operator-=(aclblasDoubleComplex& a, aclblasDoubleComplex b)
{
    a.real -= b.real;
    a.imag -= b.imag;
    return a;
}
inline aclblasDoubleComplex& operator*=(aclblasDoubleComplex& a, aclblasDoubleComplex b)
{
    a = a * b;
    return a;
}
inline aclblasDoubleComplex& operator/=(aclblasDoubleComplex& a, aclblasDoubleComplex b)
{
    // issue #828: 复用 Smith 缩放版 operator/，与单目除法保持同一实现
    a = a / b;
    return a;
}

inline aclblasDoubleComplex operator*(aclblasDoubleComplex a, double s) { return {a.real * s, a.imag * s}; }
inline aclblasDoubleComplex operator*(double s, aclblasDoubleComplex a) { return {s * a.real, s * a.imag}; }

inline double blasComplexAbs(aclblasDoubleComplex z) { return hypot(z.real, z.imag); }

#endif
