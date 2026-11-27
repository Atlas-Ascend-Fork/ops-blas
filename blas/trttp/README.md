# Trttp算子

## 算子概述

Trttp（Triangular matrix, Regular storage To Triangular matrix, Packed format）算子将常规二维三角矩阵压缩为 packed format 存储。属于 LAPACK 格式转换算子。

数学表达式：

```
A(lda x n) -> AP(n*(n+1)/2, packed format)
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasStrttp | 单精度实数常规三角矩阵压缩为 packed 格式 |
| aclblasCtrttp | 单精度复数（complex64）常规三角矩阵压缩为 packed 格式 |

## 算子执行接口

### aclblasStrttp

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasStrttp(aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float *A, int lda, float *AP)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | 三角存储方式：ACLBLAS_UPPER(121)、ACLBLAS_LOWER(122)，Host 内存 |
| n | 输入 | int | 方阵维数，Host 内存 |
| A | 输入 | const float*（FP32） | 常规三角矩阵，维度 lda × n，Device 内存 |
| lda | 输入 | int | A 的 leading dimension，lda >= max(1, n)，Host 内存 |
| AP | 输出 | float*（FP32） | 输出压缩格式，长度 n*(n+1)/2，Device 内存 |

#### 约束说明

- n >= 0
- lda >= max(1, n)
- uplo 必须为 ACLBLAS_UPPER 或 ACLBLAS_LOWER

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

int aclblasStrttpTest(AclContext& ctx)
{
    constexpr int n = 3;
    constexpr int lda = 3;
    constexpr size_t aSize = lda * n * sizeof(float);
    constexpr size_t apSize = n * (n + 1) / 2 * sizeof(float);

    // A 按列主序存储上三角部分：[[1,2,3],[0,5,6],[0,0,9]]
    float hA[lda * n] = {
        1.0f, 0.0f, 0.0f,
        2.0f, 5.0f, 0.0f,
        3.0f, 6.0f, 9.0f
    };
    float hAP[n * (n + 1) / 2] = {0.0f};

    void *rawA = nullptr;
    auto aclRet = aclrtMalloc(&rawA, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dA(rawA);

    void *rawAP = nullptr;
    aclRet = aclrtMalloc(&rawAP, apSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dAP(rawAP);

    aclRet = aclrtMemcpy(dA.get(), aSize, hA, aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);
    std::unique_ptr<void, AclblasHandleDeleter> handle(rawHandle);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handle.get()), ctx.Stream());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    blasRet = aclblasStrttp(
        static_cast<aclblasHandle_t>(handle.get()), ACLBLAS_UPPER, n,
        static_cast<const float*>(dA.get()), lda, static_cast<float*>(dAP.get()));
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    aclRet = aclrtSynchronizeStream(ctx.Stream());
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclRet = aclrtMemcpy(hAP, apSize, dAP.get(), apSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 预期结果（按列打包上三角）：1 2 5 3 6 9
    for (int i = 0; i < n * (n + 1) / 2; i++) {
        printf("AP[%d] = %f\n", i, hAP[i]);
    }

    return 0;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasStrttpTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}
```

### aclblasCtrttp

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasCtrttp(aclblasHandle_t handle, aclblasFillMode_t uplo, const int n, const aclblasComplex *A, const int lda, aclblasComplex *AP)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | 三角存储方式：ACLBLAS_UPPER(121)、ACLBLAS_LOWER(122)，Host 内存 |
| n | 输入 | int | 方阵维数，Host 内存 |
| A | 输入 | const aclblasComplex*（COMPLEX64） | 常规三角矩阵，维度 lda × n，列主序，Device 内存，仅引用 uplo 指定三角 |
| lda | 输入 | int | A 的 leading dimension，lda >= max(1, n)，Host 内存 |
| AP | 输出 | aclblasComplex*（COMPLEX64） | 输出压缩格式，长度 n*(n+1)/2，Device 内存，不复用 A 的内存 |

#### 约束说明

- `n >= 0`，`n < 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`
- `lda >= max(1, n)`；`n = 0` 时同样要求 `lda >= 1`，即 `n = 0` 且 `lda = 0` 返回 `ACLBLAS_STATUS_INVALID_VALUE`
- `uplo` 必须为 `ACLBLAS_UPPER` 或 `ACLBLAS_LOWER`，非法枚举返回 `ACLBLAS_STATUS_INVALID_VALUE`（与 cuBLAS `cublasCtrttp` 一致，不返回 `ACLBLAS_STATUS_INVALID_ENUM`）
- `n = 0` 且 `lda >= 1` 为合法 no-op，不访问 A / AP；`n > 0` 时 A、AP 为 `nullptr` 返回 `ACLBLAS_STATUS_INVALID_VALUE`
- 仅读取 uplo 指定三角，A 的另一三角与 lda padding 行不读；AP 写满 n(n+1)/2 个元素，尾部不得越界写
- A 与 AP 不得重叠；本算子是纯数据搬运，输出与输入的实部/虚部逐位相等（含 Inf/NaN 载荷）
- packed 偏移：UPPER 为 `AP[j*(j+1)/2 + i] = A(i,j)`（i <= j），LOWER 为 `AP[j*n - j*(j-1)/2 + (i-j)] = A(i,j)`（j <= i）

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

int aclblasCtrttpTest(AclContext& ctx)
{
    constexpr int n = 3;
    constexpr int lda = 3;
    constexpr size_t aSize = static_cast<size_t>(lda) * n * sizeof(aclblasComplex);
    constexpr size_t apSize = static_cast<size_t>(n) * (n + 1) / 2 * sizeof(aclblasComplex);

    // A 按列主序存储：第 0 列 (1+1i, 0, 0)，第 1 列 (2+2i, 5+5i, 0)，第 2 列 (3+3i, 6+6i, 9+9i)
    aclblasComplex hA[lda * n] = {
        {1.0f, 1.0f}, {0.0f, 0.0f}, {0.0f, 0.0f},
        {2.0f, 2.0f}, {5.0f, 5.0f}, {0.0f, 0.0f},
        {3.0f, 3.0f}, {6.0f, 6.0f}, {9.0f, 9.0f}
    };
    aclblasComplex hAP[n * (n + 1) / 2] = {};

    void *rawA = nullptr;
    auto aclRet = aclrtMalloc(&rawA, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dA(rawA);

    void *rawAP = nullptr;
    aclRet = aclrtMalloc(&rawAP, apSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dAP(rawAP);

    aclRet = aclrtMemcpy(dA.get(), aSize, hA, aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);
    std::unique_ptr<void, AclblasHandleDeleter> handle(rawHandle);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handle.get()), ctx.Stream());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    blasRet = aclblasCtrttp(
        static_cast<aclblasHandle_t>(handle.get()), ACLBLAS_UPPER, n,
        static_cast<const aclblasComplex*>(dA.get()), lda, static_cast<aclblasComplex*>(dAP.get()));
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    aclRet = aclrtSynchronizeStream(ctx.Stream());
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclRet = aclrtMemcpy(hAP, apSize, dAP.get(), apSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 预期结果（按列打包上三角）：(1+1i) (2+2i) (5+5i) (3+3i) (6+6i) (9+9i)
    for (int i = 0; i < n * (n + 1) / 2; i++) {
        printf("AP[%d] = %f + %fi\n", i, hAP[i].real, hAP[i].imag);
    }

    return 0;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasCtrttpTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}
```