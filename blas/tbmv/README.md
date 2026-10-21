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

### aclblasCtbmv

#### 产品支持情况

| 产品 | 支持情况 |
| --- | --- |
| Ascend 950PR / Ascend 950DT | 支持 |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | 支持 |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | 支持 |

#### 函数原型

```cpp
aclblasStatus_t aclblasCtbmv(aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int k, const aclblasComplex *A, int lda, aclblasComplex *x, int incx)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | 矩阵填充类型：ACLBLAS_UPPER(上三角)、ACLBLAS_LOWER(下三角)，Host 内存 |
| trans | 输入 | aclblasOperation_t | 矩阵操作类型：ACLBLAS_OP_N(不转置)、ACLBLAS_OP_T(转置)、ACLBLAS_OP_C(共轭转置，复数路径对元素取共轭)，Host 内存 |
| diag | 输入 | aclblasDiagType_t | 对角线类型：ACLBLAS_NON_UNIT(非单位对角线)、ACLBLAS_UNIT(单位对角线，对角元素不读、视为 1)，Host 内存 |
| n | 输入 | int | 三角带状矩阵 A 的行数和列数，Host 内存，n >= 0 |
| k | 输入 | int | 三角带状矩阵的半带宽，Host 内存，k >= 0 |
| A | 输入 | const aclblasComplex*（COMPLEX64） | 三角带状矩阵复数数组，维度为 lda x n，按带状格式列主序存储，Device 内存 |
| lda | 输入 | int | 矩阵 A 存储的主维长度，lda >= k + 1，Host 内存 |
| x | 输入/输出 | aclblasComplex*（COMPLEX64） | 复数向量，包含 n 个元素。输入为原始向量，输出为计算结果（原地覆盖），Device 内存 |
| incx | 输入 | int | x 中连续元素之间的步长，不可为 0，支持负步长（从末端反向遍历），Host 内存 |

#### 计算公式

```
x = op(A) * x    （原地覆写）
```

- `trans = ACLBLAS_OP_N` 时 `op(A) = A`
- `trans = ACLBLAS_OP_T` 时 `op(A) = Aᵀ`
- `trans = ACLBLAS_OP_C` 时 `op(A) = Aᴴ`（共轭转置）

#### 约束说明

- n >= 0
- k >= 0
- lda >= k + 1
- incx != 0
- 仅引用 uplo 指定三角的带内元素；diag = ACLBLAS_UNIT 时不读取对角元素
- n = 0 为合法 no-op，直接返回成功

#### 调用示例

```cpp
#include "acl/acl.h"
#include "cann_ops_blas.h"

// 假设已完成 aclInit / aclrtSetDevice / aclblasCreate / aclblasSetStream
// 下三角带状 A（lda=2, n=4, k=1, incx=1），A(0,0)=1+2i ... 按带状列主序存储
// aclblasComplex A[2*4] = {{1,2},{3,4}, {0,0},{5,6}, {0,0},{7,8}, {0,0},{9,10}};
// aclblasComplex x[4] = {{1,0},{1,0},{1,0},{1,0}};
// aclblasCtbmv(handle, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, 4, 1, A, 2, x, 1);
```

（完整可编译示例请参考仓库 docs/zh/develop/compile_and_run_example.md）

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

