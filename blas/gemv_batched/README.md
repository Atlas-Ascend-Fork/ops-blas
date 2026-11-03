# GemvBatched算子

## 算子概述

GemvBatched（批量矩阵-向量乘法）实现了对一批矩阵分别进行矩阵-向量乘法的运算。

数学表达式：

```
y[i] = alpha * op(A[i]) * x[i] + beta * y[i]
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSgemvBatched | 单精度批量矩阵-向量乘法 |
| aclblasHSHgemvBatched | FP16 入/出批量矩阵-向量乘法 |
| aclblasHSSgemvBatched | FP16 入/FP32 出批量矩阵-向量乘法 |
| aclblasTSTgemvBatched | BF16 入/出批量矩阵-向量乘法 |
| aclblasTSSgemvBatched | BF16 入/FP32 出批量矩阵-向量乘法 |
| aclblasCgemvBatched | 复数批量矩阵-向量乘法 |

## 算子执行接口

### aclblasSgemvBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSgemvBatched(aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float *alpha, const float *const Aarray[], int lda, const float *const xarray[], int incx, const float *beta, float *const yarray[], int incy, int batchCount)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，Host 内存 |
| trans | 输入 | aclblasOperation_t | 矩阵操作类型：ACLBLAS_OP_N / ACLBLAS_OP_T，Host 内存 |
| m | 输入 | int | 矩阵 A 的行数，Host 内存 |
| n | 输入 | int | 矩阵 A 的列数，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量乘数，Host 内存 |
| Aarray | 输入 | const float *const[] | Device 侧指针数组，每个元素 `Aarray[i]` 指向第 i 个输入矩阵（FP32，列主序，m×n），Device 内存 |
| lda | 输入 | int | A 矩阵的 leading dimension，Host 内存 |
| xarray | 输入 | const float *const[] | Device 侧指针数组，每个元素 `xarray[i]` 指向第 i 个输入向量（FP32），Device 内存 |
| incx | 输入 | int | x 向量元素步长，Host 内存 |
| beta | 输入 | const float*（FP32） | 标量乘数，Host 内存 |
| yarray | 输入/输出 | float *const[] | Device 侧指针数组，每个元素 `yarray[i]` 指向第 i 个输出向量（FP32），Device 内存 |
| incy | 输入 | int | y 向量元素步长，Host 内存 |
| batchCount | 输入 | int | 批量大小，Host 内存 |

#### 约束说明

- m >= 0, n >= 0
- lda >= max(1, m)
- incx != 0, incy != 0
- batchCount >= 0
- trans 必须为 ACLBLAS_OP_N 或 ACLBLAS_OP_T

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cstdio>
#include <functional>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

class AclContext {
public:
    explicit AclContext(int32_t deviceId) : deviceId_(deviceId) {}

    ~AclContext()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
            deviceSet_ = false;
        }
        if (aclInited_) {
            aclFinalize();
            aclInited_ = false;
        }
    }

    int Init()
    {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        aclInited_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(&stream_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        return ACL_SUCCESS;
    }

    aclrtStream Stream() const { return stream_; }

private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

struct BlasHandleDeleter {
    void operator()(aclblasHandle_t h) const { aclblasDestroy(h); }
};

int aclblasSgemvBatchedTest(AclContext& ctx)
{
    aclError aclRet;
    aclrtStream stream = ctx.Stream();

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasHandle_t>::type, BlasHandleDeleter> handlePtr(rawHandle);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    constexpr int m = 2;
    constexpr int n = 2;
    constexpr int lda = m;
    constexpr int incx = 1;
    constexpr int incy = 1;
    constexpr int batchCount = 2;
    constexpr size_t matBytes = (size_t)lda * n * sizeof(float);
    constexpr size_t xBytes = (size_t)n * sizeof(float);
    constexpr size_t yBytes = (size_t)m * sizeof(float);

    float alpha = 1.0f, beta = 0.0f;
    std::vector<float> hA = {1.0f, 3.0f, 2.0f, 4.0f, 5.0f, 7.0f, 6.0f, 8.0f};
    std::vector<float> hX = {1.0f, 1.0f, 1.0f, 1.0f};
    std::vector<float> hY(batchCount * m, 0.0f);

    std::vector<void*> dABufs(batchCount, nullptr), dXBufs(batchCount, nullptr), dYBufs(batchCount, nullptr);
    auto cleanupBufs = [&]() {
        for (int i = 0; i < batchCount; i++) {
            if (dABufs[i]) aclrtFree(dABufs[i]);
            if (dXBufs[i]) aclrtFree(dXBufs[i]);
            if (dYBufs[i]) aclrtFree(dYBufs[i]);
        }
    };
    struct BufGuard {
        std::function<void()> cleanup;
        ~BufGuard() { cleanup(); }
    };
    BufGuard guard{cleanupBufs};

    std::vector<float*> hAPtrs(batchCount), hXPtrs(batchCount);
    std::vector<float*> hYPtrs(batchCount);

    for (int b = 0; b < batchCount; b++) {
        aclRet = aclrtMalloc(&dABufs[b], matBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
        aclRet = aclrtMemcpy(dABufs[b], matBytes, hA.data() + b * lda * n, matBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
        aclRet = aclrtMalloc(&dXBufs[b], xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for X[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
        aclRet = aclrtMemcpy(dXBufs[b], xBytes, hX.data() + b * n, xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for X[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
        aclRet = aclrtMalloc(&dYBufs[b], yBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for Y[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
        aclRet = aclrtMemcpy(dYBufs[b], yBytes, hY.data() + b * m, yBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for Y[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
        hAPtrs[b] = (float*)dABufs[b];
        hXPtrs[b] = (float*)dXBufs[b];
        hYPtrs[b] = (float*)dYBufs[b];
    }

    const size_t ptrArrBytesA = batchCount * sizeof(float*);
    const size_t ptrArrBytesY = batchCount * sizeof(float*);
    void *rawDAPtrArr = nullptr, *rawDXPtrArr = nullptr, *rawDYPtrArr = nullptr;
    aclRet = aclrtMalloc(&rawDAPtrArr, ptrArrBytesA, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for dAPtrArr failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtrArrPtr(rawDAPtrArr, aclrtFree);
    aclRet = aclrtMalloc(&rawDXPtrArr, ptrArrBytesA, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for dXPtrArr failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dXPtrArrPtr(rawDXPtrArr, aclrtFree);
    aclRet = aclrtMalloc(&rawDYPtrArr, ptrArrBytesY, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for dYPtrArr failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dYPtrArrPtr(rawDYPtrArr, aclrtFree);
    aclRet = aclrtMemcpy(dAPtrArrPtr.get(), ptrArrBytesA, hAPtrs.data(), ptrArrBytesA, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for dAPtrArr failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dXPtrArrPtr.get(), ptrArrBytesA, hXPtrs.data(), ptrArrBytesA, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for dXPtrArr failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dYPtrArrPtr.get(), ptrArrBytesY, hYPtrs.data(), ptrArrBytesY, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for dYPtrArr failed. ERROR: %d\n", aclRet); return aclRet);

    blasRet = aclblasSgemvBatched(
        static_cast<aclblasHandle_t>(handlePtr.get()), ACLBLAS_OP_N, m, n, &alpha,
        (const float* const*)dAPtrArrPtr.get(), lda,
        (const float* const*)dXPtrArrPtr.get(), incx,
        &beta,
        (float* const*)dYPtrArrPtr.get(), incy,
        batchCount);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSgemvBatched failed. ERROR: %d\n", blasRet);
              return blasRet);

    aclError aclRet2 = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet2 == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet2); return aclRet2);

    for (int b = 0; b < batchCount; b++) {
        aclRet = aclrtMemcpy(hY.data() + b * m, yBytes, dYBufs[b], yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for Y[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
    }

    for (int b = 0; b < batchCount; b++)
        for (int i = 0; i < m; i++)
            LOG_PRINT("batch %d, hY[%d] = %f\n", b, i, hY[b * m + i]);

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSgemvBatchedTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSgemvBatchedTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

### aclblasHSHgemvBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasHSHgemvBatched(aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float *alpha, const uint16_t *const Aarray[], int lda, const uint16_t *const xarray[], int incx, const float *beta, uint16_t *const yarray[], int incy, int batchCount)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，Host 内存 |
| trans | 输入 | aclblasOperation_t | 矩阵操作类型：ACLBLAS_OP_N / ACLBLAS_OP_T，Host 内存 |
| m | 输入 | int | 矩阵 A 的行数，Host 内存 |
| n | 输入 | int | 矩阵 A 的列数，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量乘数，Host 内存 |
| Aarray | 输入 | const uint16_t *const[] | Device 侧指针数组，每个元素 `Aarray[i]` 指向第 i 个输入矩阵（FP16），Device 内存 |
| lda | 输入 | int | A 矩阵的 leading dimension，Host 内存 |
| xarray | 输入 | const uint16_t *const[] | Device 侧指针数组，每个元素 `xarray[i]` 指向第 i 个输入向量（FP16），Device 内存 |
| incx | 输入 | int | x 向量元素步长，Host 内存 |
| beta | 输入 | const float*（FP32） | 标量乘数，Host 内存 |
| yarray | 输入/输出 | uint16_t *const[] | Device 侧指针数组，每个元素 `yarray[i]` 指向第 i 个输出向量（FP16），Device 内存 |
| incy | 输入 | int | y 向量元素步长，Host 内存 |
| batchCount | 输入 | int | 批量大小，Host 内存 |

#### 约束说明

- m >= 0, n >= 0
- lda >= max(1, m)
- incx != 0, incy != 0
- batchCount >= 0
- trans 必须为 ACLBLAS_OP_N 或 ACLBLAS_OP_T

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

class AclContext {
public:
    explicit AclContext(int32_t deviceId) : deviceId_(deviceId) {}

    ~AclContext()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
            deviceSet_ = false;
        }
        if (aclInited_) {
            aclFinalize();
            aclInited_ = false;
        }
    }

    int Init()
    {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        aclInited_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(&stream_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        return ACL_SUCCESS;
    }

    aclrtStream Stream() const { return stream_; }

private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

static float fp16_to_float(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        bits = sign;
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}


struct BlasHandleDeleter {
    void operator()(aclblasHandle_t h) const { aclblasDestroy(h); }
};

int aclblasHSHgemvBatchedTest(AclContext& ctx)
{
    aclError aclRet;
    aclrtStream stream = ctx.Stream();

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasHandle_t>::type, BlasHandleDeleter> handlePtr(rawHandle);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    constexpr int m = 2;
    constexpr int n = 2;
    constexpr int lda = m;
    constexpr int incx = 1;
    constexpr int incy = 1;
    constexpr int batchCount = 2;
    constexpr size_t matBytes = (size_t)lda * n * sizeof(uint16_t);
    constexpr size_t xBytes = (size_t)n * sizeof(uint16_t);
    constexpr size_t yBytes = (size_t)m * sizeof(uint16_t);

    float alpha = 1.0f, beta = 0.0f;
    std::vector<uint16_t> hA = {0x3C00, 0x4000, 0x3E00, 0x4200,
                                0x4500, 0x4700, 0x4600, 0x4800};
    std::vector<uint16_t> hX = {0x3C00, 0x3C00, 0x3C00, 0x3C00};
    std::vector<uint16_t> hY(batchCount * m, 0);

    std::vector<void*> dABufs(batchCount, nullptr), dXBufs(batchCount, nullptr), dYBufs(batchCount, nullptr);
    auto cleanupBufs = [&]() {
        for (int i = 0; i < batchCount; i++) {
            if (dABufs[i]) aclrtFree(dABufs[i]);
            if (dXBufs[i]) aclrtFree(dXBufs[i]);
            if (dYBufs[i]) aclrtFree(dYBufs[i]);
        }
    };
    struct BufGuard {
        std::function<void()> cleanup;
        ~BufGuard() { cleanup(); }
    };
    BufGuard guard{cleanupBufs};

    std::vector<uint16_t*> hAPtrs(batchCount), hXPtrs(batchCount);
    std::vector<uint16_t*> hYPtrs(batchCount);

    for (int b = 0; b < batchCount; b++) {
        aclRet = aclrtMalloc(&dABufs[b], matBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
        aclRet = aclrtMemcpy(dABufs[b], matBytes, hA.data() + b * lda * n, matBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
        aclRet = aclrtMalloc(&dXBufs[b], xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for X[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
        aclRet = aclrtMemcpy(dXBufs[b], xBytes, hX.data() + b * n, xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for X[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
        aclRet = aclrtMalloc(&dYBufs[b], yBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for Y[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
        aclRet = aclrtMemcpy(dYBufs[b], yBytes, hY.data() + b * m, yBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for Y[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
        hAPtrs[b] = (uint16_t*)dABufs[b];
        hXPtrs[b] = (uint16_t*)dXBufs[b];
        hYPtrs[b] = (uint16_t*)dYBufs[b];
    }

    const size_t ptrArrBytesA = batchCount * sizeof(uint16_t*);
    const size_t ptrArrBytesY = batchCount * sizeof(uint16_t*);
    void *rawDAPtrArr = nullptr, *rawDXPtrArr = nullptr, *rawDYPtrArr = nullptr;
    aclRet = aclrtMalloc(&rawDAPtrArr, ptrArrBytesA, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for dAPtrArr failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtrArrPtr(rawDAPtrArr, aclrtFree);
    aclRet = aclrtMalloc(&rawDXPtrArr, ptrArrBytesA, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for dXPtrArr failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dXPtrArrPtr(rawDXPtrArr, aclrtFree);
    aclRet = aclrtMalloc(&rawDYPtrArr, ptrArrBytesY, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for dYPtrArr failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dYPtrArrPtr(rawDYPtrArr, aclrtFree);
    aclRet = aclrtMemcpy(dAPtrArrPtr.get(), ptrArrBytesA, hAPtrs.data(), ptrArrBytesA, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for dAPtrArr failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dXPtrArrPtr.get(), ptrArrBytesA, hXPtrs.data(), ptrArrBytesA, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for dXPtrArr failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dYPtrArrPtr.get(), ptrArrBytesY, hYPtrs.data(), ptrArrBytesY, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for dYPtrArr failed. ERROR: %d\n", aclRet); return aclRet);

    blasRet = aclblasHSHgemvBatched(
        static_cast<aclblasHandle_t>(handlePtr.get()), ACLBLAS_OP_N, m, n, &alpha,
        (const uint16_t* const*)dAPtrArrPtr.get(), lda,
        (const uint16_t* const*)dXPtrArrPtr.get(), incx,
        &beta,
        (uint16_t* const*)dYPtrArrPtr.get(), incy,
        batchCount);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasHSHgemvBatched failed. ERROR: %d\n", blasRet);
              return blasRet);

    aclError aclRet2 = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet2 == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet2); return aclRet2);

    for (int b = 0; b < batchCount; b++) {
        aclRet = aclrtMemcpy(hY.data() + b * m, yBytes, dYBufs[b], yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for Y[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
    }

    for (int b = 0; b < batchCount; b++)
        for (int i = 0; i < m; i++)
            LOG_PRINT("batch %d, hY[%d] = %f\n", b, i, fp16_to_float(hY[b * m + i]));

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasHSHgemvBatchedTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasHSHgemvBatchedTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

### aclblasHSSgemvBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasHSSgemvBatched(aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float *alpha, const uint16_t *const Aarray[], int lda, const uint16_t *const xarray[], int incx, const float *beta, float *const yarray[], int incy, int batchCount)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，Host 内存 |
| trans | 输入 | aclblasOperation_t | 矩阵操作类型：ACLBLAS_OP_N / ACLBLAS_OP_T，Host 内存 |
| m | 输入 | int | 矩阵 A 的行数，Host 内存 |
| n | 输入 | int | 矩阵 A 的列数，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量乘数，Host 内存 |
| Aarray | 输入 | const uint16_t *const[] | Device 侧指针数组，每个元素 `Aarray[i]` 指向第 i 个输入矩阵（FP16），Device 内存 |
| lda | 输入 | int | A 矩阵的 leading dimension，Host 内存 |
| xarray | 输入 | const uint16_t *const[] | Device 侧指针数组，每个元素 `xarray[i]` 指向第 i 个输入向量（FP16），Device 内存 |
| incx | 输入 | int | x 向量元素步长，Host 内存 |
| beta | 输入 | const float*（FP32） | 标量乘数，Host 内存 |
| yarray | 输入/输出 | float *const[] | Device 侧指针数组，每个元素 `yarray[i]` 指向第 i 个输出向量（FP32），Device 内存 |
| incy | 输入 | int | y 向量元素步长，Host 内存 |
| batchCount | 输入 | int | 批量大小，Host 内存 |

#### 约束说明

- m >= 0, n >= 0
- lda >= max(1, m)
- incx != 0, incy != 0
- batchCount >= 0
- trans 必须为 ACLBLAS_OP_N 或 ACLBLAS_OP_T

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cstdio>
#include <functional>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

class AclContext {
public:
    explicit AclContext(int32_t deviceId) : deviceId_(deviceId) {}

    ~AclContext()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
            deviceSet_ = false;
        }
        if (aclInited_) {
            aclFinalize();
            aclInited_ = false;
        }
    }

    int Init()
    {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        aclInited_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(&stream_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        return ACL_SUCCESS;
    }

    aclrtStream Stream() const { return stream_; }

private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

struct BlasHandleDeleter {
    void operator()(aclblasHandle_t h) const { aclblasDestroy(h); }
};

int aclblasHSSgemvBatchedTest(AclContext& ctx)
{
    aclError aclRet;
    aclrtStream stream = ctx.Stream();

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasHandle_t>::type, BlasHandleDeleter> handlePtr(rawHandle);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    constexpr int m = 2;
    constexpr int n = 2;
    constexpr int lda = m;
    constexpr int incx = 1;
    constexpr int incy = 1;
    constexpr int batchCount = 2;
    constexpr size_t matBytes = (size_t)lda * n * sizeof(uint16_t);
    constexpr size_t xBytes = (size_t)n * sizeof(uint16_t);
    constexpr size_t yBytes = (size_t)m * sizeof(float);

    float alpha = 1.0f, beta = 0.0f;
    std::vector<uint16_t> hA = {0x3C00, 0x4000, 0x3E00, 0x4200,
                                0x4500, 0x4700, 0x4600, 0x4800};
    std::vector<uint16_t> hX = {0x3C00, 0x3C00, 0x3C00, 0x3C00};
    std::vector<float> hY(batchCount * m, 0.0f);

    std::vector<void*> dABufs(batchCount, nullptr), dXBufs(batchCount, nullptr), dYBufs(batchCount, nullptr);
    auto cleanupBufs = [&]() {
        for (int i = 0; i < batchCount; i++) {
            if (dABufs[i]) aclrtFree(dABufs[i]);
            if (dXBufs[i]) aclrtFree(dXBufs[i]);
            if (dYBufs[i]) aclrtFree(dYBufs[i]);
        }
    };
    struct BufGuard {
        std::function<void()> cleanup;
        ~BufGuard() { cleanup(); }
    };
    BufGuard guard{cleanupBufs};

    std::vector<uint16_t*> hAPtrs(batchCount), hXPtrs(batchCount);
    std::vector<float*> hYPtrs(batchCount);

    for (int b = 0; b < batchCount; b++) {
        aclRet = aclrtMalloc(&dABufs[b], matBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
        aclRet = aclrtMemcpy(dABufs[b], matBytes, hA.data() + b * lda * n, matBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
        aclRet = aclrtMalloc(&dXBufs[b], xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for X[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
        aclRet = aclrtMemcpy(dXBufs[b], xBytes, hX.data() + b * n, xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for X[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
        aclRet = aclrtMalloc(&dYBufs[b], yBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for Y[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
        aclRet = aclrtMemcpy(dYBufs[b], yBytes, hY.data() + b * m, yBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for Y[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
        hAPtrs[b] = (uint16_t*)dABufs[b];
        hXPtrs[b] = (uint16_t*)dXBufs[b];
        hYPtrs[b] = (float*)dYBufs[b];
    }

    const size_t ptrArrBytesA = batchCount * sizeof(uint16_t*);
    const size_t ptrArrBytesY = batchCount * sizeof(float*);
    void *rawDAPtrArr = nullptr, *rawDXPtrArr = nullptr, *rawDYPtrArr = nullptr;
    aclRet = aclrtMalloc(&rawDAPtrArr, ptrArrBytesA, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for dAPtrArr failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtrArrPtr(rawDAPtrArr, aclrtFree);
    aclRet = aclrtMalloc(&rawDXPtrArr, ptrArrBytesA, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for dXPtrArr failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dXPtrArrPtr(rawDXPtrArr, aclrtFree);
    aclRet = aclrtMalloc(&rawDYPtrArr, ptrArrBytesY, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for dYPtrArr failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dYPtrArrPtr(rawDYPtrArr, aclrtFree);
    aclRet = aclrtMemcpy(dAPtrArrPtr.get(), ptrArrBytesA, hAPtrs.data(), ptrArrBytesA, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for dAPtrArr failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dXPtrArrPtr.get(), ptrArrBytesA, hXPtrs.data(), ptrArrBytesA, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for dXPtrArr failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dYPtrArrPtr.get(), ptrArrBytesY, hYPtrs.data(), ptrArrBytesY, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for dYPtrArr failed. ERROR: %d\n", aclRet); return aclRet);

    blasRet = aclblasHSSgemvBatched(
        static_cast<aclblasHandle_t>(handlePtr.get()), ACLBLAS_OP_N, m, n, &alpha,
        (const uint16_t* const*)dAPtrArrPtr.get(), lda,
        (const uint16_t* const*)dXPtrArrPtr.get(), incx,
        &beta,
        (float* const*)dYPtrArrPtr.get(), incy,
        batchCount);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasHSSgemvBatched failed. ERROR: %d\n", blasRet);
              return blasRet);

    aclError aclRet2 = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet2 == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet2); return aclRet2);

    for (int b = 0; b < batchCount; b++) {
        aclRet = aclrtMemcpy(hY.data() + b * m, yBytes, dYBufs[b], yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for Y[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
    }

    for (int b = 0; b < batchCount; b++)
        for (int i = 0; i < m; i++)
            LOG_PRINT("batch %d, hY[%d] = %f\n", b, i, hY[b * m + i]);

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasHSSgemvBatchedTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasHSSgemvBatchedTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

### aclblasCgemvBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：不支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasCgemvBatched(aclblasHandle_t handle, aclblasOperation trans, const int64_t m, const int64_t n, const aclblasComplex alpha, aclblasComplex* A, const int64_t lda, aclblasComplex* x, const int64_t incx, const aclblasComplex beta, aclblasComplex* y, const int64_t incy, const int64_t batchCount)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，Host 内存 |
| trans | 输入 | aclblasOperation | 矩阵操作类型：N=不转置，T=转置，C=共轭转置，Host 内存 |
| m | 输入 | int64_t | 矩阵 A 的行数，Host 内存 |
| n | 输入 | int64_t | 矩阵 A 的列数，Host 内存 |
| alpha | 输入 | const aclblasComplex | 复数标量 alpha，Host 内存 |
| A | 输入 | aclblasComplex* | 批量复数矩阵，batchCount 个 m x n 矩阵，Device 内存 |
| lda | 输入 | int64_t | 矩阵 A 的主维长度，Host 内存 |
| x | 输入 | aclblasComplex* | 批量复数向量，Device 内存 |
| incx | 输入 | int64_t | x 中连续元素之间的步长，Host 内存 |
| beta | 输入 | const aclblasComplex | 复数标量 beta，Host 内存 |
| y | 输入/输出 | aclblasComplex* | 批量复数向量，Device 内存 |
| incy | 输入 | int64_t | y 中连续元素之间的步长，Host 内存 |
| batchCount | 输入 | int64_t | 批次数，Host 内存 |

#### 约束说明

- batchCount >= 0, m >= 0, n >= 0
- trans 必须为 ACLBLAS_OP_N、ACLBLAS_OP_T 或 ACLBLAS_OP_C

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cstdio>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

class AclContext {
public:
    explicit AclContext(int32_t deviceId) : deviceId_(deviceId) {}

    ~AclContext()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
            deviceSet_ = false;
        }
        if (aclInited_) {
            aclFinalize();
            aclInited_ = false;
        }
    }

    int Init()
    {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        aclInited_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(&stream_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        return ACL_SUCCESS;
    }

    aclrtStream Stream() const { return stream_; }

private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

struct BlasHandleDeleter {
    void operator()(aclblasHandle_t h) const { aclblasDestroy(h); }
};

int aclblasCgemvBatchedTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasHandle_t>::type, BlasHandleDeleter> handlePtr(rawHandle);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    constexpr int64_t m = 2;
    constexpr int64_t n = 2;
    constexpr int64_t lda = m;
    constexpr int64_t incx = 1;
    constexpr int64_t incy = 1;
    constexpr int64_t batchCount = 2;
    constexpr size_t aElem = static_cast<size_t>(batchCount) * lda * n;
    constexpr size_t xElem = static_cast<size_t>(batchCount) * n;
    constexpr size_t yElem = static_cast<size_t>(batchCount) * m;
    constexpr size_t aSize = aElem * sizeof(aclblasComplex);
    constexpr size_t xSize = xElem * sizeof(aclblasComplex);
    constexpr size_t ySize = yElem * sizeof(aclblasComplex);

    aclblasComplex alpha = {1.0f, 0.0f};
    aclblasComplex beta = {0.0f, 0.0f};
    std::vector<aclblasComplex> hA = {
        {1.0f, 0.0f}, {0.0f, 1.0f}, {2.0f, 0.0f}, {0.0f, 2.0f},
        {3.0f, 0.0f}, {0.0f, 3.0f}, {4.0f, 0.0f}, {0.0f, 4.0f}};
    std::vector<aclblasComplex> hX = {
        {1.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 0.0f}, {1.0f, 0.0f}};
    std::vector<aclblasComplex> hY(yElem);

    void *rawA = nullptr, *rawX = nullptr, *rawY = nullptr;
    auto aclRet = aclrtMalloc(&rawA, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtr(rawA, aclrtFree);

    aclRet = aclrtMalloc(&rawX, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for X failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dXPtr(rawX, aclrtFree);

    aclRet = aclrtMalloc(&rawY, ySize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for Y failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dYPtr(rawY, aclrtFree);

    aclRet = aclrtMemcpy(dAPtr.get(), aSize, hA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dXPtr.get(), xSize, hX.data(), xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for X failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dYPtr.get(), ySize, hY.data(), ySize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for Y failed. ERROR: %d\n", aclRet); return aclRet);

    blasRet = aclblasCgemvBatched(
        static_cast<aclblasHandle_t>(handlePtr.get()), ACLBLAS_OP_N, m, n, alpha,
        static_cast<aclblasComplex*>(dAPtr.get()), lda, static_cast<aclblasComplex*>(dXPtr.get()), incx, beta,
        static_cast<aclblasComplex*>(dYPtr.get()), incy, batchCount);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCgemvBatched failed. ERROR: %d\n", blasRet);
              return blasRet);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(hY.data(), ySize, dYPtr.get(), ySize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for Y failed. ERROR: %d\n", aclRet); return aclRet);

    for (int b = 0; b < batchCount; b++)
        for (int i = 0; i < m; i++)
            LOG_PRINT("batch %d, hY[%d] = (%f, %f)\n", b, i, hY[b * m + i].real, hY[b * m + i].imag);

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasCgemvBatchedTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasCgemvBatchedTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```
### aclblasTSTgemvBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasTSTgemvBatched(aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float *alpha, const uint16_t *const Aarray[], int lda, const uint16_t *const xarray[], int incx, const float *beta, uint16_t *const yarray[], int incy, int batchCount)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，Host 内存 |
| trans | 输入 | aclblasOperation_t | 矩阵操作类型：ACLBLAS_OP_N / ACLBLAS_OP_T，Host 内存 |
| m | 输入 | int | 矩阵 A 的行数，Host 内存 |
| n | 输入 | int | 矩阵 A 的列数，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量乘数，Host 内存 |
| Aarray | 输入 | const uint16_t *const[] | Device 侧指针数组，每个元素 `Aarray[i]` 指向第 i 个输入矩阵（BF16），Device 内存 |
| lda | 输入 | int | A 矩阵的 leading dimension，Host 内存 |
| xarray | 输入 | const uint16_t *const[] | Device 侧指针数组，每个元素 `xarray[i]` 指向第 i 个输入向量（BF16），Device 内存 |
| incx | 输入 | int | x 向量元素步长，Host 内存 |
| beta | 输入 | const float*（FP32） | 标量乘数，Host 内存 |
| yarray | 输入/输出 | uint16_t *const[] | Device 侧指针数组，每个元素 `yarray[i]` 指向第 i 个输出向量（BF16），Device 内存 |
| incy | 输入 | int | y 向量元素步长，Host 内存 |
| batchCount | 输入 | int | 批量大小，Host 内存 |

#### 约束说明

- m >= 0, n >= 0
- lda >= max(1, m)
- incx != 0, incy != 0
- batchCount >= 0
- trans 必须为 ACLBLAS_OP_N 或 ACLBLAS_OP_T

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

class AclContext {
public:
    explicit AclContext(int32_t deviceId) : deviceId_(deviceId) {}

    ~AclContext()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
            deviceSet_ = false;
        }
        if (aclInited_) {
            aclFinalize();
            aclInited_ = false;
        }
    }

    int Init()
    {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        aclInited_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(&stream_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        return ACL_SUCCESS;
    }

    aclrtStream Stream() const { return stream_; }

private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

static float bf16_to_float(uint16_t bf)
{
    uint32_t bits = (uint32_t)bf << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}


struct BlasHandleDeleter {
    void operator()(aclblasHandle_t h) const { aclblasDestroy(h); }
};

int aclblasTSTgemvBatchedTest(AclContext& ctx)
{
    aclError aclRet;
    aclrtStream stream = ctx.Stream();

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasHandle_t>::type, BlasHandleDeleter> handlePtr(rawHandle);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    constexpr int m = 2;
    constexpr int n = 2;
    constexpr int lda = m;
    constexpr int incx = 1;
    constexpr int incy = 1;
    constexpr int batchCount = 2;
    constexpr size_t matBytes = (size_t)lda * n * sizeof(uint16_t);
    constexpr size_t xBytes = (size_t)n * sizeof(uint16_t);
    constexpr size_t yBytes = (size_t)m * sizeof(uint16_t);

    float alpha = 1.0f, beta = 0.0f;
    std::vector<uint16_t> hA = {0x3F80, 0x4000, 0x3FC0, 0x4080,
                                0x40A0, 0x40E0, 0x40C0, 0x4100};
    std::vector<uint16_t> hX = {0x3F80, 0x3F80, 0x3F80, 0x3F80};
    std::vector<uint16_t> hY(batchCount * m, 0);

    std::vector<void*> dABufs(batchCount, nullptr), dXBufs(batchCount, nullptr), dYBufs(batchCount, nullptr);
    auto cleanupBufs = [&]() {
        for (int i = 0; i < batchCount; i++) {
            if (dABufs[i]) aclrtFree(dABufs[i]);
            if (dXBufs[i]) aclrtFree(dXBufs[i]);
            if (dYBufs[i]) aclrtFree(dYBufs[i]);
        }
    };
    struct BufGuard {
        std::function<void()> cleanup;
        ~BufGuard() { cleanup(); }
    };
    BufGuard guard{cleanupBufs};

    std::vector<uint16_t*> hAPtrs(batchCount), hXPtrs(batchCount);
    std::vector<uint16_t*> hYPtrs(batchCount);

    for (int b = 0; b < batchCount; b++) {
        aclRet = aclrtMalloc(&dABufs[b], matBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
        aclRet = aclrtMemcpy(dABufs[b], matBytes, hA.data() + b * lda * n, matBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
        aclRet = aclrtMalloc(&dXBufs[b], xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for X[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
        aclRet = aclrtMemcpy(dXBufs[b], xBytes, hX.data() + b * n, xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for X[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
        aclRet = aclrtMalloc(&dYBufs[b], yBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for Y[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
        aclRet = aclrtMemcpy(dYBufs[b], yBytes, hY.data() + b * m, yBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for Y[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
        hAPtrs[b] = (uint16_t*)dABufs[b];
        hXPtrs[b] = (uint16_t*)dXBufs[b];
        hYPtrs[b] = (uint16_t*)dYBufs[b];
    }

    const size_t ptrArrBytesA = batchCount * sizeof(uint16_t*);
    const size_t ptrArrBytesY = batchCount * sizeof(uint16_t*);
    void *rawDAPtrArr = nullptr, *rawDXPtrArr = nullptr, *rawDYPtrArr = nullptr;
    aclRet = aclrtMalloc(&rawDAPtrArr, ptrArrBytesA, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for dAPtrArr failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtrArrPtr(rawDAPtrArr, aclrtFree);
    aclRet = aclrtMalloc(&rawDXPtrArr, ptrArrBytesA, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for dXPtrArr failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dXPtrArrPtr(rawDXPtrArr, aclrtFree);
    aclRet = aclrtMalloc(&rawDYPtrArr, ptrArrBytesY, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for dYPtrArr failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dYPtrArrPtr(rawDYPtrArr, aclrtFree);
    aclRet = aclrtMemcpy(dAPtrArrPtr.get(), ptrArrBytesA, hAPtrs.data(), ptrArrBytesA, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for dAPtrArr failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dXPtrArrPtr.get(), ptrArrBytesA, hXPtrs.data(), ptrArrBytesA, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for dXPtrArr failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dYPtrArrPtr.get(), ptrArrBytesY, hYPtrs.data(), ptrArrBytesY, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for dYPtrArr failed. ERROR: %d\n", aclRet); return aclRet);

    blasRet = aclblasTSTgemvBatched(
        static_cast<aclblasHandle_t>(handlePtr.get()), ACLBLAS_OP_N, m, n, &alpha,
        (const uint16_t* const*)dAPtrArrPtr.get(), lda,
        (const uint16_t* const*)dXPtrArrPtr.get(), incx,
        &beta,
        (uint16_t* const*)dYPtrArrPtr.get(), incy,
        batchCount);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasTSTgemvBatched failed. ERROR: %d\n", blasRet);
              return blasRet);

    aclError aclRet2 = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet2 == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet2); return aclRet2);

    for (int b = 0; b < batchCount; b++) {
        aclRet = aclrtMemcpy(hY.data() + b * m, yBytes, dYBufs[b], yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for Y[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
    }

    for (int b = 0; b < batchCount; b++)
        for (int i = 0; i < m; i++)
            LOG_PRINT("batch %d, hY[%d] = %f\n", b, i, bf16_to_float(hY[b * m + i]));

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasTSTgemvBatchedTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasTSTgemvBatchedTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

### aclblasTSSgemvBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasTSSgemvBatched(aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float *alpha, const uint16_t *const Aarray[], int lda, const uint16_t *const xarray[], int incx, const float *beta, float *const yarray[], int incy, int batchCount)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，Host 内存 |
| trans | 输入 | aclblasOperation_t | 矩阵操作类型：ACLBLAS_OP_N / ACLBLAS_OP_T，Host 内存 |
| m | 输入 | int | 矩阵 A 的行数，Host 内存 |
| n | 输入 | int | 矩阵 A 的列数，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量乘数，Host 内存 |
| Aarray | 输入 | const uint16_t *const[] | Device 侧指针数组，每个元素 `Aarray[i]` 指向第 i 个输入矩阵（BF16），Device 内存 |
| lda | 输入 | int | A 矩阵的 leading dimension，Host 内存 |
| xarray | 输入 | const uint16_t *const[] | Device 侧指针数组，每个元素 `xarray[i]` 指向第 i 个输入向量（BF16），Device 内存 |
| incx | 输入 | int | x 向量元素步长，Host 内存 |
| beta | 输入 | const float*（FP32） | 标量乘数，Host 内存 |
| yarray | 输入/输出 | float *const[] | Device 侧指针数组，每个元素 `yarray[i]` 指向第 i 个输出向量（FP32），Device 内存 |
| incy | 输入 | int | y 向量元素步长，Host 内存 |
| batchCount | 输入 | int | 批量大小，Host 内存 |

#### 约束说明

- m >= 0, n >= 0
- lda >= max(1, m)
- incx != 0, incy != 0
- batchCount >= 0
- trans 必须为 ACLBLAS_OP_N 或 ACLBLAS_OP_T

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cstdio>
#include <functional>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

class AclContext {
public:
    explicit AclContext(int32_t deviceId) : deviceId_(deviceId) {}

    ~AclContext()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
            deviceSet_ = false;
        }
        if (aclInited_) {
            aclFinalize();
            aclInited_ = false;
        }
    }

    int Init()
    {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        aclInited_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(&stream_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        return ACL_SUCCESS;
    }

    aclrtStream Stream() const { return stream_; }

private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

struct BlasHandleDeleter {
    void operator()(aclblasHandle_t h) const { aclblasDestroy(h); }
};

int aclblasTSSgemvBatchedTest(AclContext& ctx)
{
    aclError aclRet;
    aclrtStream stream = ctx.Stream();

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasHandle_t>::type, BlasHandleDeleter> handlePtr(rawHandle);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    constexpr int m = 2;
    constexpr int n = 2;
    constexpr int lda = m;
    constexpr int incx = 1;
    constexpr int incy = 1;
    constexpr int batchCount = 2;
    constexpr size_t matBytes = (size_t)lda * n * sizeof(uint16_t);
    constexpr size_t xBytes = (size_t)n * sizeof(uint16_t);
    constexpr size_t yBytes = (size_t)m * sizeof(float);

    float alpha = 1.0f, beta = 0.0f;
    std::vector<uint16_t> hA = {0x3F80, 0x4000, 0x3FC0, 0x4080,
                                0x40A0, 0x40E0, 0x40C0, 0x4100};
    std::vector<uint16_t> hX = {0x3F80, 0x3F80, 0x3F80, 0x3F80};
    std::vector<float> hY(batchCount * m, 0.0f);

    std::vector<void*> dABufs(batchCount, nullptr), dXBufs(batchCount, nullptr), dYBufs(batchCount, nullptr);
    auto cleanupBufs = [&]() {
        for (int i = 0; i < batchCount; i++) {
            if (dABufs[i]) aclrtFree(dABufs[i]);
            if (dXBufs[i]) aclrtFree(dXBufs[i]);
            if (dYBufs[i]) aclrtFree(dYBufs[i]);
        }
    };
    struct BufGuard {
        std::function<void()> cleanup;
        ~BufGuard() { cleanup(); }
    };
    BufGuard guard{cleanupBufs};

    std::vector<uint16_t*> hAPtrs(batchCount), hXPtrs(batchCount);
    std::vector<float*> hYPtrs(batchCount);

    for (int b = 0; b < batchCount; b++) {
        aclRet = aclrtMalloc(&dABufs[b], matBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
        aclRet = aclrtMemcpy(dABufs[b], matBytes, hA.data() + b * lda * n, matBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
        aclRet = aclrtMalloc(&dXBufs[b], xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for X[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
        aclRet = aclrtMemcpy(dXBufs[b], xBytes, hX.data() + b * n, xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for X[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
        aclRet = aclrtMalloc(&dYBufs[b], yBytes, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for Y[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
        aclRet = aclrtMemcpy(dYBufs[b], yBytes, hY.data() + b * m, yBytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for Y[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
        hAPtrs[b] = (uint16_t*)dABufs[b];
        hXPtrs[b] = (uint16_t*)dXBufs[b];
        hYPtrs[b] = (float*)dYBufs[b];
    }

    const size_t ptrArrBytesA = batchCount * sizeof(uint16_t*);
    const size_t ptrArrBytesY = batchCount * sizeof(float*);
    void *rawDAPtrArr = nullptr, *rawDXPtrArr = nullptr, *rawDYPtrArr = nullptr;
    aclRet = aclrtMalloc(&rawDAPtrArr, ptrArrBytesA, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for dAPtrArr failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtrArrPtr(rawDAPtrArr, aclrtFree);
    aclRet = aclrtMalloc(&rawDXPtrArr, ptrArrBytesA, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for dXPtrArr failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dXPtrArrPtr(rawDXPtrArr, aclrtFree);
    aclRet = aclrtMalloc(&rawDYPtrArr, ptrArrBytesY, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for dYPtrArr failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dYPtrArrPtr(rawDYPtrArr, aclrtFree);
    aclRet = aclrtMemcpy(dAPtrArrPtr.get(), ptrArrBytesA, hAPtrs.data(), ptrArrBytesA, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for dAPtrArr failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dXPtrArrPtr.get(), ptrArrBytesA, hXPtrs.data(), ptrArrBytesA, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for dXPtrArr failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dYPtrArrPtr.get(), ptrArrBytesY, hYPtrs.data(), ptrArrBytesY, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for dYPtrArr failed. ERROR: %d\n", aclRet); return aclRet);

    blasRet = aclblasTSSgemvBatched(
        static_cast<aclblasHandle_t>(handlePtr.get()), ACLBLAS_OP_N, m, n, &alpha,
        (const uint16_t* const*)dAPtrArrPtr.get(), lda,
        (const uint16_t* const*)dXPtrArrPtr.get(), incx,
        &beta,
        (float* const*)dYPtrArrPtr.get(), incy,
        batchCount);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasTSSgemvBatched failed. ERROR: %d\n", blasRet);
              return blasRet);

    aclError aclRet2 = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet2 == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet2); return aclRet2);

    for (int b = 0; b < batchCount; b++) {
        aclRet = aclrtMemcpy(hY.data() + b * m, yBytes, dYBufs[b], yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for Y[%d] failed. ERROR: %d\n", b, aclRet); return aclRet);
    }

    for (int b = 0; b < batchCount; b++)
        for (int i = 0; i < m; i++)
            LOG_PRINT("batch %d, hY[%d] = %f\n", b, i, hY[b * m + i]);

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasTSSgemvBatchedTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasTSSgemvBatchedTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```
