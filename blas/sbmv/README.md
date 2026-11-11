# Sbmv算子

## 算子概述

Sbmv（Symmetric Banded Matrix-Vector Multiplication）算子实现了对称带状矩阵与向量的乘法运算。

数学表达式：

```
y = alpha * A * x + beta * y
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSsbmv | 单精度浮点对称带状矩阵-向量乘法 |
| aclblasChbmv | 单精度复数 Hermitian 带状矩阵-向量乘法 |

## 算子执行接口

### aclblasSsbmv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSsbmv(aclblasHandle_t handle, aclblasFillMode_t uplo, int n, int k, const float *alpha, const float *A, int lda, const float *x, int incx, const float *beta, float *y, int incy)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | ACLBLAS_UPPER(121) 或 ACLBLAS_LOWER(122)，Host 内存 |
| n | 输入 | int | 方阵阶数，Host 内存 |
| k | 输入 | int | 次对角线/超对角线数量，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量 alpha 的指针，Host 内存 |
| A | 输入 | const float*（FP32） | 带状对称矩阵，列主序，维度 (k+1)×n，Device 内存 |
| lda | 输入 | int | A 的主维数，Host 内存 |
| x | 输入 | const float*（FP32） | 输入向量，n 个元素，Device 内存 |
| incx | 输入 | int | x 的步长（可正可负），Host 内存 |
| beta | 输入 | const float*（FP32） | 标量 beta 的指针，Host 内存 |
| y | 输入/输出 | float*（FP32） | 输入/输出向量，n 个元素，Device 内存 |
| incy | 输入 | int | y 的步长（可正可负），Host 内存 |

#### 约束说明

- n >= 0, k >= 0
- lda >= k + 1
- incx != 0, incy != 0

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
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

int aclblasSsbmvTest(AclContext& ctx)
{
    constexpr int n = 4;
    constexpr int k = 1;
    constexpr int lda = k + 1;
    constexpr int incx = 1;
    constexpr int incy = 1;
    constexpr size_t aSize = lda * n * sizeof(float);
    constexpr size_t xSize = n * sizeof(float);
    constexpr size_t ySize = n * sizeof(float);
    float alpha = 1.0f;
    float beta = 1.0f;

    float hA[lda * n] = {
        4.0f, 1.0f,
        5.0f, 2.0f,
        6.0f, 3.0f,
        7.0f, 0.0f
    };
    float hX[n] = {1.0f, 2.0f, 3.0f, 4.0f};
    float hY[n] = {10.0f, 20.0f, 30.0f, 40.0f};

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
    aclRet = aclrtMemcpy(dY.get(), ySize, hY, ySize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);
    std::unique_ptr<void, AclblasHandleDeleter> handle(rawHandle);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handle.get()), ctx.Stream());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    blasRet = aclblasSsbmv(
        static_cast<aclblasHandle_t>(handle.get()), ACLBLAS_LOWER, n, k, &alpha,
        static_cast<float*>(dA.get()), lda, static_cast<float*>(dX.get()), incx, &beta,
        static_cast<float*>(dY.get()), incy);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    aclRet = aclrtSynchronizeStream(ctx.Stream());
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclRet = aclrtMemcpy(hY, ySize, dY.get(), ySize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    return 0;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSsbmvTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}
```

### aclblasChbmv

#### 产品支持情况

- Ascend 950PR：支持
- Atlas A2 / A3：支持
- 其他产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasChbmv(aclblasHandle_t handle, aclblasFillMode_t uplo, int n, int k, const aclblasComplex* alpha, const aclblasComplex* A, int lda, const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y, int incy)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带执行 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | 指定厄米特带状矩阵 A 存储上三角带（`ACLBLAS_UPPER`）或下三角带（`ACLBLAS_LOWER`），Host 内存 |
| n | 输入 | int | 厄米特带状矩阵 A 的行数和列数，n >= 0，Host 内存 |
| k | 输入 | int | 矩阵 A 的超对角线（或次对角线）条数（半带宽），k >= 0，Host 内存 |
| alpha | 输入 | const aclblasComplex*（COMPLEX64） | 复数标量乘数，Host 内存 |
| A | 输入 | const aclblasComplex*（COMPLEX64） | 厄米特带状矩阵，列主序带状紧凑存储，物理维度为 lda x n，对角线元素虚部假定为 0（不参与读取），未存储侧按共轭对称推断，Device 内存 |
| lda | 输入 | int | 用于存储带状矩阵 A 的二维数组的主维，lda >= k + 1，Host 内存 |
| x | 输入 | const aclblasComplex*（COMPLEX64） | 输入复数向量，包含 n 个逻辑元素，Device 内存 |
| incx | 输入 | int | x 中连续元素之间的步长，支持正负非零值（incx != 0），Host 内存 |
| beta | 输入 | const aclblasComplex*（COMPLEX64） | 复数标量乘数；当 beta == (0, 0) 时不读取 y 的输入值，Host 内存 |
| y | 输入/输出 | aclblasComplex*（COMPLEX64） | 输入/输出复数向量，包含 n 个逻辑元素，原地写回计算结果，Device 内存 |
| incy | 输入 | int | y 中连续元素之间的步长，支持正负非零值（incy != 0），Host 内存 |

#### 约束说明

- n >= 0（n == 0 时在校验 handle 非空后直接返回 `ACLBLAS_STATUS_SUCCESS`）
- k >= 0
- lda >= k + 1
- incx != 0，incy != 0
- uplo 仅支持 `ACLBLAS_UPPER` 和 `ACLBLAS_LOWER`
- 对角线元素仅实部参与计算，虚部忽略（按 0 处理）；非对角对称位置自动取复共轭 `conj(A[j, i])`
