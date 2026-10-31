# Trsv算子

## 算子概述

trsv (Triangular matrix Solve) 求解三角线性系统。该算子支持前向和后向求解，支持单位对角线和非单位对角线，支持转置和共轭转置。

数学表达式：

```
op(A) * x = b
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasStrsv | 单精度三角矩阵求解 |
| aclblasCtrsv | 单精度复数三角矩阵求解 |

## 算子执行接口

### aclblasStrsv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasStrsv(aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, const float *A, int lda, float *x, int incx)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | ACLBLAS_UPPER(121) — A 为上三角矩阵；ACLBLAS_LOWER(122) — A 为下三角矩阵，Host 内存 |
| trans | 输入 | aclblasOperation_t | ACLBLAS_OP_N(111) — op(A) = A；ACLBLAS_OP_T(112) — op(A) = A^T；ACLBLAS_OP_C(113) — op(A) = A^H（FP32 下与 T 等价），Host 内存 |
| diag | 输入 | aclblasDiagType_t | ACLBLAS_NON_UNIT(131) — 对角元从 A 读取；ACLBLAS_UNIT(132) — 对角元固定为 1，Host 内存 |
| n | 输入 | int | 矩阵阶数，n >= 0。n == 0 时为空操作直接返回成功，Host 内存 |
| A | 输入 | const float*（FP32） | n x lda 三角矩阵指针，仅相关三角部分被访问，Device 内存 |
| lda | 输入 | int | A 的 leading dimension，lda >= max(1, n)，Host 内存 |
| x | 输入/输出 | float*（FP32） | 输入时存储右端向量 b，输出时原地覆盖为解向量 x，Device 内存 |
| incx | 输入 | int | x 的存储增量，incx != 0（可正可负）。incx < 0 时 x 反向存储，Host 内存 |

#### 约束说明

- n >= 0，n == 0 时为空操作直接返回成功
- uplo 必须为 ACLBLAS_UPPER 或 ACLBLAS_LOWER
- trans 必须为 ACLBLAS_OP_N、ACLBLAS_OP_T 或 ACLBLAS_OP_C
- diag 必须为 ACLBLAS_NON_UNIT 或 ACLBLAS_UNIT
- lda >= max(1, n)
- incx != 0（可正可负）
- A、x 不可为 nullptr

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
int aclblasStrsvTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<void, aclblasStatus_t (*)(void*)> handlePtr(rawHandle, aclblasDestroy);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    constexpr int n = 4;
    constexpr int incx = 1;
    constexpr int lda = 4;
    constexpr size_t aSize = n * lda * sizeof(float);
    constexpr size_t xSize = n * sizeof(float);

    std::vector<float> hA = {
        1.0f, 2.0f, 4.0f, 7.0f,
        0.0f, 3.0f, 5.0f, 8.0f,
        0.0f, 0.0f, 6.0f, 9.0f,
        0.0f, 0.0f, 0.0f, 10.0f};
    std::vector<float> hX = {1.0f, 4.0f, 9.0f, 16.0f};

    void* rawA = nullptr;
    aclError aclRet;
    aclRet = aclrtMalloc(&rawA, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtr(rawA, aclrtFree);

    void* rawX = nullptr;
    aclRet = aclrtMalloc(&rawX, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for x failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dXPtr(rawX, aclrtFree);

    aclRet = aclrtMemcpy(dAPtr.get(), aSize, hA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(dXPtr.get(), xSize, hX.data(), xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for x failed. ERROR: %d\n", aclRet); return aclRet);

    blasRet = aclblasStrsv(
        static_cast<aclblasHandle_t>(handlePtr.get()), ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
        n, static_cast<const float*>(dAPtr.get()), lda, static_cast<float*>(dXPtr.get()), incx);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasStrsv failed. ERROR: %d\n", blasRet);
              return blasRet);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    std::vector<float> xResult(n, 0.0f);
    aclRet = aclrtMemcpy(xResult.data(), xSize, dXPtr.get(), xSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet); return aclRet);
    for (int i = 0; i < n; i++) {
        LOG_PRINT("x[%d] = %f\n", i, xResult[i]);
    }

    LOG_PRINT("aclblasStrsv test passed\n");
    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasStrsvTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasStrsvTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

### aclblasCtrsv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持（arch22）
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持（arch22）

#### 函数原型

```cpp
aclblasStatus_t aclblasCtrsv(aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, const aclblasComplex* A, int lda, aclblasComplex* x, int incx)
```

求解 `op(A) * x = b`，将输入向量 b 原地覆盖为解 x。A 按列主序存储，
`A[i + j * lda]` 表示第 i 行、第 j 列；复数实部、虚部均为 float32。

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
| --- | --- | --- | --- |
| handle | 输入 | aclblasHandle_t | 有效 BLAS 句柄，通过 aclblasSetStream 绑定执行流，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | ACLBLAS_UPPER / ACLBLAS_LOWER，仅引用对应三角，Host 内存 |
| trans | 输入 | aclblasOperation_t | ACLBLAS_OP_N / ACLBLAS_OP_T / ACLBLAS_OP_C，分别表示 A / 转置 / 共轭转置，Host 内存 |
| diag | 输入 | aclblasDiagType_t | ACLBLAS_NON_UNIT / ACLBLAS_UNIT；UNIT 不读取 A 对角元素，Host 内存 |
| n | 输入 | int | 矩阵阶数，n >= 0，Host 内存 |
| A | 输入 | const aclblasComplex*（COMPLEX64） | 只读列主序矩阵，物理容量至少 lda*n 个元素，Device 内存 |
| lda | 输入 | int | lda >= max(1,n)，支持列尾 padding，Host 内存 |
| x | 输入/输出 | aclblasComplex*（COMPLEX64） | 原地向量，n>0 时物理容量至少 1+(n-1)*abs(incx) 个元素，Device 内存 |
| incx | 输入 | int | 非零步长，可正可负；负步长时 x 仍指向分配区起始地址，Host 内存 |

#### 约束说明

- 标量参数满足表中条件；负步长的第 i 个逻辑元素位于 `(n-1-i)*abs(incx)`，步长间隙保持不变。
- n>0 时 A/x 必须指向独立且容量足够的设备缓冲区。容量计算须使用足够宽的整数类型。
- NON_UNIT 对角由调用方保证非零，算子不检测奇异性或近奇异性；仅引用指定三角。
- 空 handle 返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR；非法枚举、n、lda、incx 或 n>0 时空 A/x 返回 ACLBLAS_STATUS_INVALID_VALUE。
- n=0 且标量参数合法时返回成功，允许空 A/x，不访问设备数据或下发 kernel。
- 接口异步执行，读回 x 前须同步 handle 对应 stream。
- 950PR 实现运行时按尺寸、diag 和工作区选择路径。4096 是缓存分界，不是接口尺寸上限；工作区不足时回退到无需额外工作区的路径。

#### 调用示例

下面是可独立编译的示例，使用 AclContext 和 std::unique_ptr 管理资源。
编译步骤见[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)，ST 步骤见[Ctrsv 测试说明](../../test/trsv/ctrsv/README.md)。

```cpp
#include <cmath>
#include <cstdio>
#include <memory>
#include <type_traits>
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
int aclblasCtrsvTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasHandle_t>::type, decltype(&aclblasDestroy)> handlePtr(
        rawHandle, aclblasDestroy);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    constexpr int n = 2;
    constexpr int incx = 1;
    constexpr int lda = 2;
    constexpr size_t aSize = n * lda * sizeof(aclblasComplex);
    constexpr size_t xSize = n * sizeof(aclblasComplex);

    // Column-major lower triangle: [[2, 0], [1+i, 3]].
    std::vector<aclblasComplex> hA = {{2.0f, 0.0f}, {1.0f, 1.0f}, {0.0f, 0.0f}, {3.0f, 0.0f}};
    std::vector<aclblasComplex> hX = {{2.0f, 2.0f}, {6.0f, -1.0f}};

    void* rawA = nullptr;
    aclError aclRet;
    aclRet = aclrtMalloc(&rawA, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtr(rawA, aclrtFree);

    void* rawX = nullptr;
    aclRet = aclrtMalloc(&rawX, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for x failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dXPtr(rawX, aclrtFree);

    aclRet = aclrtMemcpy(dAPtr.get(), aSize, hA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(dXPtr.get(), xSize, hX.data(), xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for x failed. ERROR: %d\n", aclRet); return aclRet);

    blasRet = aclblasCtrsv(
        static_cast<aclblasHandle_t>(handlePtr.get()), ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
        n, static_cast<const aclblasComplex*>(dAPtr.get()), lda, static_cast<aclblasComplex*>(dXPtr.get()), incx);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCtrsv failed. ERROR: %d\n", blasRet);
              return blasRet);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    std::vector<aclblasComplex> xResult(n);
    aclRet = aclrtMemcpy(xResult.data(), xSize, dXPtr.get(), xSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet); return aclRet);
    for (int i = 0; i < n; i++) {
        LOG_PRINT("x[%d] = (%f, %f)\n", i, xResult[i].real, xResult[i].imag);
    }

    CHECK_RET(std::abs(xResult[0].real - 1.0f) < 1e-5f && std::abs(xResult[0].imag - 1.0f) < 1e-5f &&
              std::abs(xResult[1].real - 2.0f) < 1e-5f && std::abs(xResult[1].imag + 1.0f) < 1e-5f,
              LOG_PRINT("Unexpected solution\n"); return 1);
    LOG_PRINT("aclblasCtrsv test passed\n");
    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasCtrsvTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasCtrsvTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```
