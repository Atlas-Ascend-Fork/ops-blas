# Tbmv算子

## 算子概述

tbmv (Triangular Band Matrix-Vector Multiplication) 实现三角带状矩阵与向量的乘法运算。该算子支持上三角和下三角矩阵，支持转置和共轭转置操作，支持单位对角线和非单位对角线。

数学表达式：

```
x = op(A) * x    （标准接口，原地覆盖）
y = A * x        （arch22 legacy 接口，输入输出分离）
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasStbmv | 单精度三角带状矩阵-向量乘法（标准接口） |
| aclblasCtbmv | 单精度复数三角带状矩阵-向量乘法（标准接口） |
| aclblasStbmv_legacy | 单精度三角带状矩阵-向量乘法（早期接口） |

## 算子执行接口

### aclblasCtbmv

#### 产品支持情况

| 产品 | 支持情况 |
| --- | --- |
| Ascend 950PR / Ascend 950DT | 不支持 |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | 支持 |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | 支持 |

#### 函数原型

```cpp
aclblasStatus_t aclblasCtbmv(aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int k, const aclblasComplex* A, int lda, aclblasComplex* x, int incx)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
| --- | --- | --- | --- |
| handle | 输入 | aclblasHandle_t | 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | UPPER/LOWER，指定上/下三角带状存储，Host 内存 |
| trans | 输入 | aclblasOperation_t | N/T/C，指定不转置/转置/共轭转置，Host 内存 |
| diag | 输入 | aclblasDiagType_t | NON_UNIT/UNIT，指定实际/隐式单位对角，Host 内存 |
| n | 输入 | int | 方阵的阶数及逻辑向量长度，Host 内存 |
| k | 输入 | int | 半带宽，非负；大于等于 n-1 时覆盖整个三角，Host 内存 |
| A | 输入 | const aclblasComplex* | COMPLEX64 带状矩阵，lda × n，Device 内存 |
| lda | 输入 | int | 带状存储前导维度，至少 k+1，Host 内存 |
| x | 输入/输出 | aclblasComplex* | COMPLEX64 向量，结果原地覆写，Device 内存 |
| incx | 输入 | int | 向量逻辑元素步长，非零，支持负值，Host 内存 |

#### 约束说明

- `A` 为 COMPLEX64 三角带状矩阵，按列主序存放在 `lda x n` 数组中。
- `ACLBLAS_OP_C` 计算共轭转置；`ACLBLAS_OP_T` 仅转置，不取共轭。
- `ACLBLAS_UNIT` 不读取对角元素并将其视为 `(1, 0)`。
- `n >= 0`、`k >= 0`、`lda >= k + 1`、`incx != 0`；`n == 0` 合法且不启动 kernel。
- 支持正、负 `incx`，结果原地写回 `x`。
- 调用异步提交到 handle 绑定的 stream，读取结果前需要同步该 stream。

#### 参数与存储

参数顺序与 cuBLAS `cublasCtbmv` 一致，复数类型为公共头文件中的 `aclblasComplex`。
`handle`、枚举及维数参数位于 Host；`A`、`x` 位于 Device。
用 0-based 下标表示时，UPPER 的带内元素存于 `A[k + row - col + col * lda]`，
LOWER 存于 `A[row - col + col * lda]`；带外及未映射三角区域不参与计算。
`x` 需要至少 `1 + (n-1)*abs(incx)` 个复数的存储空间（n > 0），
负步长的首个逻辑元素位于 `(n-1)*abs(incx)`。没有 alpha/beta 参数。

空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；
非法枚举返回 `ACLBLAS_STATUS_INVALID_ENUM`；
非法维数、步长或 n > 0 时的空 A/x 返回 `ACLBLAS_STATUS_INVALID_VALUE`。
需要使用有效的 handle 工作空间；运行时资源不足或搬运失败返回
`ACLBLAS_STATUS_EXECUTION_FAILED`。

完整调用、精度验证和性能复现见
[Ctbmv 自测说明](../../test/tbmv/ctbmv/README.md)。

#### 调用示例

以下示例使用 AclContext 和 std::unique_ptr 管理资源，预期结果为
`[(3, 3), (7, 1)]`（每项依次为实部、虚部）。
编译方式参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include "acl/acl.h"
#include "cann_ops_blas.h"

void Check(bool success)
{
    if (!success) {
        throw std::runtime_error("Ctbmv example failed");
    }
}

class AclContext {
public:
    ~AclContext()
    {
        if (stream_ != nullptr) {
            aclrtSynchronizeStream(stream_);
            aclrtDestroyStream(stream_);
        }
        if (deviceSet_) {
            aclrtResetDevice(0);
        }
        if (initialized_) {
            aclFinalize();
        }
    }
    void Init()
    {
        Check(aclInit(nullptr) == ACL_SUCCESS);
        initialized_ = true;
        Check(aclrtSetDevice(0) == ACL_SUCCESS);
        deviceSet_ = true;
        Check(aclrtCreateStream(&stream_) == ACL_SUCCESS);
    }
    aclrtStream Stream() const { return stream_; }

private:
    aclrtStream stream_ = nullptr;
    bool initialized_ = false;
    bool deviceSet_ = false;
};

void RunCtbmv(AclContext& context)
{
    aclblasHandle_t rawHandle = nullptr;
    Check(aclblasCreate(&rawHandle) == ACLBLAS_STATUS_SUCCESS);
    std::unique_ptr<std::remove_pointer_t<aclblasHandle_t>, decltype(&aclblasDestroy)> handle(rawHandle, aclblasDestroy);
    Check(aclblasSetStream(handle.get(), context.Stream()) == ACLBLAS_STATUS_SUCCESS);

    // Column-major upper band: A = [[2, 1+i], [0, 3-i]], n=2, k=1, lda=2.
    aclblasComplex a[] = {{0, 0}, {2, 0}, {1, 1}, {3, -1}};
    aclblasComplex x[] = {{1, 0}, {2, 1}};
    auto allocate = [](size_t bytes) {
        void* pointer = nullptr;
        Check(aclrtMalloc(&pointer, bytes, ACL_MEM_MALLOC_HUGE_FIRST) == ACL_SUCCESS);
        return std::unique_ptr<void, decltype(&aclrtFree)>(pointer, aclrtFree);
    };
    auto deviceA = allocate(sizeof(a));
    auto deviceX = allocate(sizeof(x));
    Check(aclrtMemcpy(deviceA.get(), sizeof(a), a, sizeof(a), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS);
    Check(aclrtMemcpy(deviceX.get(), sizeof(x), x, sizeof(x), ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS);
    Check(aclblasCtbmv(handle.get(), ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 2, 1,
                      static_cast<const aclblasComplex*>(deviceA.get()), 2,
                      static_cast<aclblasComplex*>(deviceX.get()), 1) == ACLBLAS_STATUS_SUCCESS);
    Check(aclrtSynchronizeStream(context.Stream()) == ACL_SUCCESS);
    Check(aclrtMemcpy(x, sizeof(x), deviceX.get(), sizeof(x), ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS);
    Check(std::fabs(x[0].real - 3) < 1e-5F && std::fabs(x[0].imag - 3) < 1e-5F &&
          std::fabs(x[1].real - 7) < 1e-5F && std::fabs(x[1].imag - 1) < 1e-5F);
    std::printf("x = [(%.1f, %.1f), (%.1f, %.1f)] PASS\n", x[0].real, x[0].imag, x[1].real, x[1].imag);
}

int main()
{
    try {
        AclContext context;
        context.Init();
        RunCtbmv(context);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
```

### aclblasStbmv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasStbmv(aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int k, const float *A, int lda, float *x, int incx)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | 矩阵填充类型：ACLBLAS_UPPER(上三角)、ACLBLAS_LOWER(下三角)，Host 内存 |
| trans | 输入 | aclblasOperation_t | 矩阵操作类型：ACLBLAS_OP_N(不转置)、ACLBLAS_OP_T(转置)、ACLBLAS_OP_C(共轭转置，实数下同 T)，Host 内存 |
| diag | 输入 | aclblasDiagType_t | 对角线类型：ACLBLAS_NON_UNIT(非单位对角线)、ACLBLAS_UNIT(单位对角线，对角元素视为 1)，Host 内存 |
| n | 输入 | int | 三角带状矩阵 A 的行数和列数，Host 内存 |
| k | 输入 | int | 三角带状矩阵的半带宽，Host 内存 |
| A | 输入 | const float*（FP32） | 三角带状矩阵 float 数组，维度为 lda x n，Device 内存 |
| lda | 输入 | int | 矩阵 A 存储的主维长度，lda >= k + 1，Host 内存 |
| x | 输入/输出 | float*（FP32） | float 向量，包含 n 个元素。输入为原始向量，输出为计算结果（原地覆盖），Device 内存 |
| incx | 输入 | int | x 中连续元素之间的步长，不可为 0，Host 内存 |

#### 约束说明

- n >= 0
- k >= 0
- lda >= k + 1
- incx != 0

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

int aclblasStbmvTest(AclContext& ctx)
{
    constexpr int n = 4;
    constexpr int k = 1;
    constexpr int lda = k + 1;
    constexpr int incx = 1;
    constexpr size_t aSize = lda * n * sizeof(float);
    constexpr size_t xSize = n * sizeof(float);

    float hA[lda * n] = {
        4.0f, 1.0f,
        5.0f, 2.0f,
        6.0f, 3.0f,
        7.0f, 0.0f
    };
    float hX[n] = {1.0f, 2.0f, 3.0f, 4.0f};

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

    blasRet = aclblasStbmv(
        static_cast<aclblasHandle_t>(handle.get()), ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
        n, k, static_cast<float*>(dA.get()), lda, static_cast<float*>(dX.get()), incx);
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

    ret = aclblasStbmvTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}
```

### aclblasStbmv_legacy

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasStbmv_legacy(aclblasHandle_t handle, const float *a, const int64_t lda, const float *x, float *y, const int64_t n, const int64_t k, const int64_t incx)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| a | 输入 | const float*（FP32） | 下三角带状矩阵 float 数组，维度为 lda x n，Device 内存 |
| lda | 输入 | int64_t | 矩阵 a 存储的主维长度，lda >= k + 1，Host 内存 |
| x | 输入 | const float*（FP32） | float 输入向量，包含 n 个元素，Device 内存 |
| y | 输出 | float*（FP32） | float 输出向量，包含 n 个元素，Device 内存 |
| n | 输入 | int64_t | 矩阵 A 的行数和列数，Host 内存 |
| k | 输入 | int64_t | 下三角带状矩阵的半带宽，Host 内存 |
| incx | 输入 | int64_t | x 中连续元素之间的步长，Host 内存 |

#### 约束说明

- n >= 0
- k >= 0
- lda >= k + 1
- incx != 0
