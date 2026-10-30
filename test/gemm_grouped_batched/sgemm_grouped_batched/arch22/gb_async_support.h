/*
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include "gb_test_support.h"
namespace gb_test {
inline Row AsyncCase(bool large = false)
{
    Row r{
        {"case_name", "ASYNC_SNAPSHOT"},
        {"description", "async_metadata_lifetime"},
        {"dtype", "0"},
        {"groupCount", "2"},
        {"transaArray", "N;T"},
        {"transbArray", "T;N"},
        {"mArray", "3;5"},
        {"nArray", "5;3"},
        {"kArray", "7;9"},
        {"ldaArray", "5;11"},
        {"ldbArray", "7;11"},
        {"ldcArray", "7;9"},
        {"alphaArray", "1;-1"},
        {"betaArray", "0;1"},
        {"groupSizeArray", "33;2"},
        {"A_fill", "RANDOM_NORM_5_5"},
        {"B_fill", "RANDOM_NORM_5_5"},
        {"C_fill", "RANDOM_NORM_5_5"},
        {"random_seed", "20260923"},
        {"expect_result", "ACLBLAS_STATUS_SUCCESS"},
        {"offset_floats", "1"}};
    if (large) {
        r["mArray"] = "256;256";
        r["nArray"] = "384;384";
        r["kArray"] = "256;256";
        r["ldaArray"] = "256;256";
        r["ldbArray"] = "384;256";
        r["ldcArray"] = "256;256";
        r["alphaArray"] = "1;1";
        r["betaArray"] = "0;0";
        r["groupSizeArray"] = "2;2";
        r["offset_floats"] = "0";
    }
    return r;
}
inline void AssertVerified(const Fixture& f, const Options& opt)
{
    Result r;
    f.Verify(r, opt);
    if (!r.metrics.Passed() || !r.message.empty())
        throw std::runtime_error("async output mismatch: " + JSON(r));
}
inline void AsyncSnapshot(Context& ctx, bool large = false)
{
    Options opt;
    opt.maxFlops = 0;
    Fixture f(AsyncCase(large), opt);
    {
        Parameters temporary = f.p;
        CheckBLAS(temporary.Call(ctx.handle), "temporary parameter enqueue");
#ifdef GB_CPU_MOCK
        if (gb_mock::Pending(ctx.stream) == 0)
            throw std::runtime_error("mock launch unexpectedly synchronized");
#endif
        // Destroy host argument arrays immediately after return, before stream synchronization.
        temporary.ta.clear();
        temporary.tb.clear();
        temporary.m.clear();
        temporary.n.clear();
        temporary.k.clear();
        temporary.a.clear();
        temporary.b.clear();
        temporary.c.clear();
        temporary.alpha.clear();
        temporary.beta.clear();
    }
    CheckACL(aclrtSynchronizeStream(ctx.stream), "snapshot sync");
    AssertVerified(f, opt);
}
inline void AsyncSerialC(Context& ctx)
{
    Options opt;
    auto row = AsyncCase();
    row["groupSizeArray"] = "2;1";
    row["betaArray"] = "1;1";
    Fixture f(row, opt);
    for (int i = 0; i < 2; ++i)
        CheckBLAS(f.p.Call(ctx.handle), "serial same-C enqueue");
    CheckACL(aclrtSynchronizeStream(ctx.stream), "serial same-C sync");
    // beta=1 twice => C_final=2*alpha*AB+C_initial.
    for (float& alpha : f.p.alpha)
        alpha *= 2;
    AssertVerified(f, opt);
}
inline void AsyncTwoStreams(Context& ctx, bool large = false)
{
    Options opt;
    auto row = AsyncCase(large);
    Fixture first(row, opt);
    row["random_seed"] = "20260924";
    Fixture second(row, opt);
    aclrtStream stream = nullptr;
    aclblasHandle_t handle = nullptr;
    try {
        CheckACL(aclrtCreateStream(&stream), "second stream create");
        CheckBLAS(aclblasCreate(&handle), "second handle create");
        CheckBLAS(aclblasSetStream(handle, stream), "second handle bind");
        CheckBLAS(first.p.Call(ctx.handle), "first stream enqueue");
        CheckBLAS(second.p.Call(handle), "second stream enqueue");
        CheckACL(aclrtSynchronizeStream(stream), "second stream sync");
        AssertVerified(second, opt);
        CheckACL(aclrtSynchronizeStream(ctx.stream), "first stream sync");
        AssertVerified(first, opt);
    } catch (...) {
        if (stream)
            aclrtSynchronizeStream(stream);
        aclrtSynchronizeStream(ctx.stream);
        if (handle)
            aclblasDestroy(handle);
        if (stream)
            aclrtDestroyStream(stream);
        throw;
    }
    CheckBLAS(aclblasDestroy(handle), "second handle destroy");
    CheckACL(aclrtDestroyStream(stream), "second stream destroy");
}
} // namespace gb_test
