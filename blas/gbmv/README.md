# Gbmv算子

## 算子概述

BLAS Gbmv（General Banded Matrix-Vector Multiplication）算子实现了带状矩阵与向量的乘法运算，针对带状矩阵的稀疏存储特性进行了优化，支持转置操作和多核并行归约。

数学表达式：

```
y = alpha * op(A) * x + beta * y
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSgbmv | 单精度浮点带状矩阵-向量乘法 |
| aclblasCgbmv | 单精度复数带状矩阵-向量乘法 |

## 算子执行接口

### aclblasSgbmv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSgbmv(aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int kl, int ku, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| trans | 输入 | aclblasOperation_t | 矩阵操作类型：ACLBLAS_OP_N（不转置）、ACLBLAS_OP_T（转置）、ACLBLAS_OP_C（共轭转置，实数下同 T），Host 内存 |
| m | 输入 | int | 矩阵 A 的行数，Host 内存 |
| n | 输入 | int | 矩阵 A 的列数，Host 内存 |
| kl | 输入 | int | A 的下带宽（主对角线以下的非零对角线数），Host 内存 |
| ku | 输入 | int | A 的上带宽（主对角线以上的非零对角线数），Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量乘数，Host/Device 内存 |
| A | 输入 | const float*（FP32） | 带状矩阵，维度为 lda x n，Device 内存 |
| lda | 输入 | int | 带状矩阵 A 存储的主维长度，lda >= kl + ku + 1，Host 内存 |
| x | 输入 | const float*（FP32） | 向量，trans='N' 时包含 n 个元素，否则包含 m 个元素，Device 内存 |
| incx | 输入 | int | x 中连续元素之间的步长，Host 内存 |
| beta | 输入 | const float*（FP32） | 标量乘数。如果 beta == 0，则 y 不必是有效输入，Host/Device 内存 |
| y | 输入/输出 | float*（FP32） | 向量，trans='N' 时包含 m 个元素，否则包含 n 个元素，Device 内存 |
| incy | 输入 | int | y 中连续元素之间的步长，Host 内存 |

#### 约束说明

- lda >= kl + ku + 1
- incx != 0, incy != 0

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
int aclblasSgbmvTest(AclContext& ctx)
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

    constexpr int m = 3;
    constexpr int n = 3;
    constexpr int kl = 1;
    constexpr int ku = 1;
    constexpr int lda = kl + ku + 1;
    constexpr int incx = 1;
    constexpr int incy = 1;
    constexpr size_t aSize = lda * n * sizeof(float);
    constexpr size_t xSize = n * sizeof(float);
    constexpr size_t ySize = m * sizeof(float);

    float alpha = 1.0f;
    float beta = 0.0f;
    std::vector<float> hA = {0.0f, 1.0f, 3.0f,
                             2.0f, 1.0f, 3.0f,
                             2.0f, 1.0f, 0.0f};
    std::vector<float> hX = {1.0f, 1.0f, 1.0f};
    std::vector<float> hY = {0.0f, 0.0f, 0.0f};

    void* rawA = nullptr;
    aclError aclRet;
    aclRet = aclrtMalloc(&rawA, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtr(rawA, aclrtFree);

    void* rawX = nullptr;
    aclRet = aclrtMalloc(&rawX, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for x failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dXPtr(rawX, aclrtFree);

    void* rawY = nullptr;
    aclRet = aclrtMalloc(&rawY, ySize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for y failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dYPtr(rawY, aclrtFree);

    aclRet = aclrtMemcpy(dAPtr.get(), aSize, hA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(dXPtr.get(), xSize, hX.data(), xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for x failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(dYPtr.get(), ySize, hY.data(), ySize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for y failed. ERROR: %d\n", aclRet); return aclRet);

    blasRet = aclblasSgbmv(
        static_cast<aclblasHandle_t>(handlePtr.get()), ACLBLAS_OP_N, m, n, kl, ku, &alpha,
        static_cast<const float*>(dAPtr.get()), lda, static_cast<const float*>(dXPtr.get()), incx,
        &beta, static_cast<float*>(dYPtr.get()), incy);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSgbmv failed. ERROR: %d\n", blasRet);
              return blasRet);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    std::vector<float> yResult(m, 0.0f);
    aclRet = aclrtMemcpy(yResult.data(), ySize, dYPtr.get(), ySize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet); return aclRet);
    for (int i = 0; i < m; i++) {
        LOG_PRINT("y[%d] = %f\n", i, yResult[i]);
    }

    LOG_PRINT("aclblasSgbmv test passed\n");
    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSgbmvTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSgbmvTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

### aclblasCgbmv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasCgbmv(aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int kl, int ku, const aclblasComplex *alpha, const aclblasComplex *A, int lda, const aclblasComplex *x, int incx, const aclblasComplex *beta, aclblasComplex *y, int incy)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| trans | 输入 | aclblasOperation_t | 矩阵操作类型：ACLBLAS_OP_N（op(A)=A）、ACLBLAS_OP_T（op(A)=A^T）、ACLBLAS_OP_C（op(A)=A^H，共轭转置），Host 内存 |
| m | 输入 | int | 矩阵 A 的行数，Host 内存 |
| n | 输入 | int | 矩阵 A 的列数，Host 内存 |
| kl | 输入 | int | A 的下带宽（主对角线以下的次对角线条数），Host 内存 |
| ku | 输入 | int | A 的上带宽（主对角线以上的次对角线条数），Host 内存 |
| alpha | 输入 | const aclblasComplex*（COMPLEX64） | 复数标量乘数，Host/Device 内存 |
| A | 输入 | const aclblasComplex*（COMPLEX64） | 列主序带状存储的矩阵，数组维度 lda x n，元素 A(i,j)（1-based）存于 A(ku+1+i-j, j)，Device 内存 |
| lda | 输入 | int | 带状矩阵 A 存储的主维长度，Host 内存 |
| x | 输入 | const aclblasComplex*（COMPLEX64） | 向量，trans=ACLBLAS_OP_N 时包含 n 个元素，否则包含 m 个元素，Device 内存 |
| incx | 输入 | int | x 中连续元素之间的步长，可为负，Host 内存 |
| beta | 输入 | const aclblasComplex*（COMPLEX64） | 复数标量乘数。beta 为 (0,0) 时不读取 y 的旧值，Host/Device 内存 |
| y | 输入/输出 | aclblasComplex*（COMPLEX64） | 向量，trans=ACLBLAS_OP_N 时包含 m 个元素，否则包含 n 个元素，原地覆写，Device 内存 |
| incy | 输入 | int | y 中连续元素之间的步长，可为负，Host 内存 |

#### 约束说明

- m >= 0，n >= 0；m == 0 或 n == 0 时为合法 no-op：trans、kl/ku 下界、lda、incx/incy、alpha/beta 非空等检查仍先执行，通过后返回 ACLBLAS_STATUS_SUCCESS，不访问 A、x、y。
- kl >= 0，ku >= 0，lda >= kl + ku + 1；m、n 均大于 0 时还要求 kl <= m - 1，ku <= n - 1。
- incx != 0，incy != 0；负步长按 BLAS 语义从物理存储的高地址端开始遍历，调用方传入物理存储的起始地址。
- alpha、beta 不可为 nullptr，可各自位于 Host 或 Device 内存，由算子内部通过 aclrtPointerGetAttributes 自动判断；位于 Device 时由 kernel 直接读取，不引入同步。
- m、n 均大于 0 时 A、x、y 不可为 nullptr（y 为输出，beta 为 (0,0) 时同样要求非空）。
- 非法参数返回 ACLBLAS_STATUS_INVALID_VALUE（含 alpha/beta 指针位置查询失败），trans 取值非法同样返回 ACLBLAS_STATUS_INVALID_VALUE（与 aclblasSgbmv 一致），handle 为 nullptr 返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR，获取 AI Vector 核数失败返回 ACLBLAS_STATUS_EXECUTION_FAILED。
- alpha == (0,0) 且 beta == (1,0) 时直接返回；alpha == (0,0) 时仅计算 y = beta * y，不读取 A、x。
- 只访问带内元素：带外三角区与 lda 填充行不被读取，y 的步长间隙保持不变。
- x、y 不得与 A 重叠。函数在 handle 绑定的 stream 上异步执行，读取 y 前需同步该 stream。

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
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

struct AclMemDeleter {
    void operator()(void* p) const { aclrtFree(p); }
};
struct BlasHandleDeleter {
    void operator()(aclblasHandle_t h) const { aclblasDestroy(h); }
};

int aclblasCgbmvTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasHandle_t>::type, BlasHandleDeleter> handlePtr(rawHandle);

    blasRet = aclblasSetStream(handlePtr.get(), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 3 x 3 tridiagonal matrix (kl = ku = 1) in band storage, column by column:
    // column 0: [unused, A(0,0), A(1,0)], column 1: [A(0,1), A(1,1), A(2,1)], column 2: [A(1,2), A(2,2), unused]
    constexpr int m = 3;
    constexpr int n = 3;
    constexpr int kl = 1;
    constexpr int ku = 1;
    constexpr int lda = kl + ku + 1;
    constexpr int incx = 1;
    constexpr int incy = 1;
    constexpr size_t aSize = lda * n * sizeof(aclblasComplex);
    constexpr size_t xSize = n * sizeof(aclblasComplex);
    constexpr size_t ySize = m * sizeof(aclblasComplex);

    aclblasComplex alpha = {1.0f, 0.0f};
    aclblasComplex beta = {0.0f, 0.0f};
    std::vector<aclblasComplex> hA = {{0.0f, 0.0f}, {1.0f, 1.0f}, {2.0f, 0.0f},
                                      {3.0f, 0.0f}, {4.0f, -1.0f}, {5.0f, 0.0f},
                                      {6.0f, 0.0f}, {7.0f, 2.0f}, {0.0f, 0.0f}};
    std::vector<aclblasComplex> hX = {{1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}};
    std::vector<aclblasComplex> hY(m, {0.0f, 0.0f});

    void* rawA = nullptr;
    aclError aclRet = aclrtMalloc(&rawA, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, AclMemDeleter> dAPtr(rawA);

    void* rawX = nullptr;
    aclRet = aclrtMalloc(&rawX, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for x failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, AclMemDeleter> dXPtr(rawX);

    void* rawY = nullptr;
    aclRet = aclrtMalloc(&rawY, ySize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for y failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, AclMemDeleter> dYPtr(rawY);

    aclRet = aclrtMemcpy(dAPtr.get(), aSize, hA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dXPtr.get(), xSize, hX.data(), xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for x failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dYPtr.get(), ySize, hY.data(), ySize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for y failed. ERROR: %d\n", aclRet); return aclRet);

    blasRet = aclblasCgbmv(
        handlePtr.get(), ACLBLAS_OP_N, m, n, kl, ku, &alpha,
        static_cast<const aclblasComplex*>(dAPtr.get()), lda, static_cast<const aclblasComplex*>(dXPtr.get()), incx,
        &beta, static_cast<aclblasComplex*>(dYPtr.get()), incy);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCgbmv failed. ERROR: %d\n", blasRet);
              return blasRet);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    std::vector<aclblasComplex> yResult(m);
    aclRet = aclrtMemcpy(yResult.data(), ySize, dYPtr.get(), ySize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet); return aclRet);
    // expected: y = (1, 4), (9, 10), (5, 14)
    for (int i = 0; i < m; i++) {
        LOG_PRINT("y[%d] = (%f, %f)\n", i, yResult[i].real, yResult[i].imag);
    }

    LOG_PRINT("aclblasCgbmv test passed\n");
    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasCgbmvTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasCgbmvTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```
