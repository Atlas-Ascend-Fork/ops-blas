# Trmv算子

## 算子概述

trmv (Triangular Matrix-Vector Multiplication) 实现三角矩阵与向量的乘法运算。本算子包含实数三角矩阵-向量乘法（Strmv）和复数三角矩阵-向量乘法（Ctrmv）。

数学表达式：

```
x = op(A) * x
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasStrmv | 实数三角矩阵-向量乘法 |
| aclblasCtrmv | 复数三角矩阵-向量乘法 |

## 算子执行接口

### aclblasStrmv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasStrmv(aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, const float *A, int lda, float *x, int incx)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | 矩阵填充类型：ACLBLAS_UPPER(上三角)、ACLBLAS_LOWER(下三角)，Host 内存 |
| trans | 输入 | aclblasOperation_t | 矩阵操作类型：ACLBLAS_OP_N(不转置)、ACLBLAS_OP_T(转置)、ACLBLAS_OP_C(共轭转置，实数下同 T)，Host 内存 |
| diag | 输入 | aclblasDiagType_t | 对角线类型：ACLBLAS_NON_UNIT(非单位对角线)、ACLBLAS_UNIT(单位对角线，对角元素视为 1)，Host 内存 |
| n | 输入 | int | 三角矩阵 A 的行数和列数，Host 内存 |
| A | 输入 | const float*（FP32） | 三角矩阵数组，维度为 lda x n，Device 内存 |
| lda | 输入 | int | 矩阵 A 存储的主维长度，lda >= n，Host 内存 |
| x | 输入/输出 | float*（FP32） | 向量，包含 n 个元素。输入为原始向量，输出为计算结果（原地覆盖），Device 内存 |
| incx | 输入 | int | x 中连续元素之间的步长，不可为 0，Host 内存 |

#### 约束说明

- n >= 0
- incx != 0
- lda >= n

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

int aclblasStrmvTest(AclContext& ctx)
{
    constexpr int n = 3;
    constexpr int lda = 3;
    constexpr int incx = 1;
    constexpr size_t aSize = lda * n * sizeof(float);
    constexpr size_t xSize = n * sizeof(float);

    // A 按列主序存储，此处存储下三角部分。
    float hA[lda * n] = {
        1.0f, 2.0f, 4.0f,
        0.0f, 3.0f, 5.0f,
        0.0f, 0.0f, 6.0f
    };
    float hX[n] = {1.0f, 2.0f, 3.0f};

    void *rawA = nullptr;
    auto aclRet = aclrtMalloc(&rawA, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dA(rawA);

    void *rawX = nullptr;
    aclRet = aclrtMalloc(&rawX, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dX(rawX);

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

    blasRet = aclblasStrmv(
        static_cast<aclblasHandle_t>(handle.get()), ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
        n, static_cast<float*>(dA.get()), lda, static_cast<float*>(dX.get()), incx);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    aclRet = aclrtSynchronizeStream(ctx.Stream());
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclRet = aclrtMemcpy(hX, xSize, dX.get(), xSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    return 0;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasStrmvTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}
```

### aclblasCtrmv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasCtrmv(aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, const aclblasComplex *A, int lda, aclblasComplex *x, int incx)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | 指定矩阵 A 的上三角或下三角部分。ACLBLAS_UPPER 或 ACLBLAS_LOWER，Host 内存 |
| trans | 输入 | aclblasOperation_t | 指定对矩阵 A 的操作类型。ACLBLAS_OP_N(不转置)、ACLBLAS_OP_T(转置)、ACLBLAS_OP_C(共轭转置)，Host 内存 |
| diag | 输入 | aclblasDiagType_t | 指定对角线元素是否为单位元。ACLBLAS_UNIT(单位对角线，对角视为 1 且不访问)或 ACLBLAS_NON_UNIT(非单位对角线)，Host 内存 |
| n | 输入 | int | 三角矩阵 A 的阶数（行/列数），即向量的逻辑长度，Host 内存 |
| A | 输入 | const aclblasComplex*（COMPLEX64） | 复数三角矩阵，按列主序以 lda x n 全存储，仅 uplo 指定三角被引用，Device 内存 |
| lda | 输入 | int | 矩阵 A 的主维度，lda >= max(1, n)，Host 内存 |
| x | 输入/输出 | aclblasComplex*（COMPLEX64） | 复数向量，逻辑长度 n。输入为原始向量，输出为计算结果（原地覆写），Device 内存 |
| incx | 输入 | int | x 中连续元素之间的步长，不可为 0（可为负，反向遍历），Host 内存 |

#### 约束说明

- n >= 0；n = 0 为合法 no-op，返回 ACLBLAS_STATUS_SUCCESS 且不执行计算
- incx != 0（负步长合法）
- lda >= max(1, n)
- 仅引用 uplo 指定三角，另一部分不读取；diag = ACLBLAS_UNIT 时不访问对角元素
- 接口声明与 cuBLAS `cublasCtrmv` 语义对齐（Netlib `ctrmv`）

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

int aclblasCtrmvTest(aclblasHandle_t handle, aclrtStream stream)
{
    constexpr int n = 3;
    constexpr int lda = 3;
    constexpr int incx = 1;
    constexpr size_t aSize = lda * n * sizeof(aclblasComplex);
    constexpr size_t xSize = n * sizeof(aclblasComplex);

    // A 按列主序存储，此处存储下三角部分（每列 3 个元素，列主序展平）。
    aclblasComplex hA[lda * n] = {
        {1.0f, 2.0f}, {2.0f, 0.0f}, {4.0f, 1.0f},
        {0.0f, 0.0f}, {3.0f, 3.0f}, {5.0f, 2.0f},
        {0.0f, 0.0f}, {0.0f, 0.0f}, {6.0f, 4.0f}
    };
    aclblasComplex hX[n] = {{1.0f, 0.0f}, {2.0f, 1.0f}, {3.0f, 2.0f}};

    void* rawA = nullptr;
    auto aclRet = aclrtMalloc(&rawA, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dA(rawA);

    void* rawX = nullptr;
    aclRet = aclrtMalloc(&rawX, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dX(rawX);

    aclRet = aclrtMemcpy(dA.get(), aSize, hA, aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dX.get(), xSize, hX, xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclblasStatus_t blasRet = aclblasCtrmv(
        handle, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, n,
        static_cast<const aclblasComplex*>(dA.get()), lda, static_cast<aclblasComplex*>(dX.get()), incx);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclRet = aclrtMemcpy(hX, xSize, dX.get(), xSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    return 0;
}
```

`AclrtMemDeleter` 定义见 `aclblasStrmv` 调用示例；`handle` 与 `stream` 的创建方式同上。
