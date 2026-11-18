# Gemv算子

## 算子概述

Gemv（General Matrix-Vector multiplication）算子实现了通用矩阵与向量的乘法运算。

数学表达式：

```
y = alpha * op(A) * x + beta * y
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSgemv | 单精度浮点矩阵-向量乘法 |
| aclblasCgemv | 复数矩阵-向量乘法 |

## 算子执行接口

### aclblasSgemv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSgemv(aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，Host 内存 |
| trans | 输入 | aclblasOperation_t | 矩阵操作类型：ACLBLAS_OP_N（不转置）、ACLBLAS_OP_T（转置）、ACLBLAS_OP_C（共轭转置，实数等价于转置），Host 内存 |
| m | 输入 | int | 矩阵 A 的行数，m >= 0，Host 内存 |
| n | 输入 | int | 矩阵 A 的列数，n >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量 alpha，不可为 nullptr，Host 内存 |
| A | 输入 | const float*（FP32） | 列主序 m x n 矩阵，维度为 lda x n，Device 内存 |
| lda | 输入 | int | 矩阵 A 的主维，lda >= max(1, m)，Host 内存 |
| x | 输入 | const float*（FP32） | 输入向量，trans=N 时逻辑长度 n，trans=T/C 时逻辑长度 m，Device 内存 |
| incx | 输入 | int | 向量 x 的元素步长，incx != 0，支持正负值，Host 内存 |
| beta | 输入 | const float*（FP32） | 标量 beta，不可为 nullptr。若 beta == 0，则 y 的输入值不被使用，Host 内存 |
| y | 输入/输出 | float*（FP32） | 输入/输出向量，trans=N 时逻辑长度 m，trans=T/C 时逻辑长度 n，Device 内存 |
| incy | 输入 | int | 向量 y 的元素步长，incy != 0，支持正负值，Host 内存 |

#### 约束说明

- m >= 0, n >= 0
- lda >= max(1, m)
- incx、incy 非零且不为 `INT32_MIN`；向量跨度 `(len - 1) * |inc|` 不超过 `INT32_MAX`
- alpha、beta 不可为 nullptr
- arch22 在 alpha == 0 时计算 y = beta * y，跳过 A/x 数据读取；beta == 0 时直接覆写 y，beta == 1 时保持 y 不变

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cstdio>
#include <memory>

#include "acl/acl.h"
#include "cann_ops_blas.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

class AclContext {
public:
    explicit AclContext(int deviceId) : deviceId_(deviceId) {}

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
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        aclInited_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(&stream_);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        return ACL_SUCCESS;
    }

    aclrtStream Stream() const { return stream_; }

private:
    int deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

struct AclrtMemDeleter {
    void operator()(void* ptr) const
    {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
};

struct AclblasHandleDeleter {
    void operator()(aclblasHandle_t handle) const
    {
        if (handle != nullptr) {
            aclblasDestroy(handle);
        }
    }
};

int aclblasSgemvTest(AclContext& ctx)
{
    constexpr int m = 2;
    constexpr int n = 2;
    constexpr int lda = 2;
    constexpr int incx = 1;
    constexpr int incy = 1;
    constexpr size_t aSize = lda * n * sizeof(float);
    constexpr size_t xSize = n * sizeof(float);
    constexpr size_t ySize = m * sizeof(float);
    float alpha = 1.0f;
    float beta = 0.0f;

    // A 按列主序存储：[[1,3],[2,4]]
    float hA[lda * n] = {1.0f, 2.0f, 3.0f, 4.0f};
    float hX[n] = {1.0f, 1.0f};
    float hY[m] = {0.0f, 0.0f};

    void *rawA = nullptr;
    auto aclRet = aclrtMalloc(&rawA, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dA(rawA);

    void *rawX = nullptr;
    aclRet = aclrtMalloc(&rawX, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dX(rawX);

    void *rawY = nullptr;
    aclRet = aclrtMalloc(&rawY, ySize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dY(rawY);

    aclRet = aclrtMemcpy(dA.get(), aSize, hA, aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dX.get(), xSize, hX, xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);
    std::unique_ptr<void, AclblasHandleDeleter> handle(rawHandle);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handle.get()), ctx.Stream());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    blasRet = aclblasSgemv(
        static_cast<aclblasHandle_t>(handle.get()), ACLBLAS_OP_N, m, n, &alpha,
        static_cast<const float*>(dA.get()), lda,
        static_cast<const float*>(dX.get()), incx, &beta,
        static_cast<float*>(dY.get()), incy);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    aclRet = aclrtSynchronizeStream(ctx.Stream());
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclRet = aclrtMemcpy(hY, ySize, dY.get(), ySize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 预期结果：y = A*x = {4, 6}
    for (int i = 0; i < m; i++) {
        printf("y[%d] = %f\n", i, hY[i]);
    }

    return 0;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSgemvTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}
```

### aclblasCgemv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasCgemv(aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const aclblasComplex *alpha, const aclblasComplex *A, int lda, const aclblasComplex *x, int incx, const aclblasComplex *beta, aclblasComplex *y, int incy)
```

`m`、`n`、`lda`、`incx`、`incy` 沿用现有公共接口的 `int` 类型，与 [cann_ops_blas.h](../../include/cann_ops_blas.h) 中的声明一致。调用时，这些参数的值须在 `int` 的取值范围内。

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，Host 内存 |
| trans | 输入 | aclblasOperation_t | 矩阵操作类型：ACLBLAS_OP_N（不转置）、ACLBLAS_OP_T（转置）、ACLBLAS_OP_C（共轭转置，先转置再逐元素取共轭），Host 内存 |
| m | 输入 | int | 矩阵 A 的行数，m >= 0，Host 内存 |
| n | 输入 | int | 矩阵 A 的列数，n >= 0，Host 内存 |
| alpha | 输入 | const aclblasComplex*（COMPLEX64） | 复数标量 alpha，不可为 nullptr，Host 内存 |
| A | 输入 | const aclblasComplex*（COMPLEX64） | 列主序复数矩阵，数组维度 lda x n，前 m x n 部分为有效系数矩阵，Device 内存 |
| lda | 输入 | int | 矩阵 A 的主维，lda >= max(1, m)，Host 内存 |
| x | 输入 | const aclblasComplex*（COMPLEX64） | 输入向量，trans=N 时逻辑长度 n，trans=T/C 时逻辑长度 m，Device 内存 |
| incx | 输入 | int | 向量 x 的元素步长，incx != 0，Host 内存 |
| beta | 输入 | const aclblasComplex*（COMPLEX64） | 复数标量 beta，不可为 nullptr。若 beta == (0,0)，则 y 的输入值不被使用，Host 内存 |
| y | 输入/输出 | aclblasComplex*（COMPLEX64） | 输入/输出向量，原地覆写，trans=N 时逻辑长度 m，trans=T/C 时逻辑长度 n，Device 内存 |
| incy | 输入 | int | 向量 y 的元素步长，incy != 0，Host 内存 |

#### 约束说明

- m >= 0, n >= 0
- lda >= max(1, m)
- incx != 0, incy != 0
- alpha、beta 不可为 nullptr
- Ascend 950PR 支持正负步长，负步长按 Netlib 语义反向遍历
- Ascend 950PR 上，trans 取值不在 {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C} 时返回 ACLBLAS_STATUS_INVALID_ENUM
- Ascend 950PR 上，m == 0 或 n == 0 为合法 quick return；alpha == (0,0) 且 beta == (1,0) 时不写 y；alpha == (0,0) 且 beta != (1,0) 时只计算 y = beta * y，不读 A、x；beta == (0,0) 时不读取 y 的输入值

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cstdio>
#include <memory>

#include "acl/acl.h"
#include "cann_ops_blas.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

class AclContext {
public:
    explicit AclContext(int deviceId) : deviceId_(deviceId) {}

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
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        aclInited_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(&stream_);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        return ACL_SUCCESS;
    }

    aclrtStream Stream() const { return stream_; }

private:
    int deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

struct AclrtMemDeleter {
    void operator()(void* ptr) const
    {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
};

struct AclblasHandleDeleter {
    void operator()(aclblasHandle_t handle) const
    {
        if (handle != nullptr) {
            aclblasDestroy(handle);
        }
    }
};

int aclblasCgemvTest(AclContext& ctx)
{
    constexpr int m = 2;
    constexpr int n = 2;
    constexpr int lda = 2;
    constexpr int incx = 1;
    constexpr int incy = 1;
    constexpr size_t aSize = static_cast<size_t>(lda) * n * sizeof(aclblasComplex);
    constexpr size_t xSize = static_cast<size_t>(n) * sizeof(aclblasComplex);
    constexpr size_t ySize = static_cast<size_t>(m) * sizeof(aclblasComplex);
    aclblasComplex alpha = {1.0f, 0.0f};
    aclblasComplex beta = {0.0f, 0.0f};

    // A 按列主序存储复数矩阵：[[1+0i,3+0i],[2+0i,4+0i]]
    aclblasComplex hA[lda * n] = {{1.0f, 0.0f}, {2.0f, 0.0f}, {3.0f, 0.0f}, {4.0f, 0.0f}};
    aclblasComplex hX[n] = {{1.0f, 0.0f}, {1.0f, 0.0f}};
    aclblasComplex hY[m] = {{0.0f, 0.0f}, {0.0f, 0.0f}};

    void *rawA = nullptr;
    auto aclRet = aclrtMalloc(&rawA, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dA(rawA);

    void *rawX = nullptr;
    aclRet = aclrtMalloc(&rawX, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dX(rawX);

    void *rawY = nullptr;
    aclRet = aclrtMalloc(&rawY, ySize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dY(rawY);

    aclRet = aclrtMemcpy(dA.get(), aSize, hA, aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dX.get(), xSize, hX, xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);
    std::unique_ptr<void, AclblasHandleDeleter> handle(rawHandle);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handle.get()), ctx.Stream());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    blasRet = aclblasCgemv(
        static_cast<aclblasHandle_t>(handle.get()), ACLBLAS_OP_N, m, n, &alpha,
        static_cast<const aclblasComplex*>(dA.get()), lda,
        static_cast<const aclblasComplex*>(dX.get()), incx, &beta,
        static_cast<aclblasComplex*>(dY.get()), incy);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    aclRet = aclrtSynchronizeStream(ctx.Stream());
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclRet = aclrtMemcpy(hY, ySize, dY.get(), ySize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 预期结果：y = A*x = {(4,0), (6,0)}
    for (int64_t i = 0; i < m; i++) {
        printf("y[%lld] = (%f, %f)\n", static_cast<long long>(i), hY[i].real, hY[i].imag);
    }

    return 0;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasCgemvTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}
```
