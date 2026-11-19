# Tpttr算子

## 算子概述

Tpttr（Triangular matrix, Packed format To Triangular matrix, Regular storage）算子将 LAPACK 压缩格式（packed format）中的三角矩阵展开为按列主序存储的常规二维矩阵。展开是 packed 与常规存储之间的纯搬运，不做转置或共轭运算（复数接口同样只搬运元素值）。仅写入 `uplo` 指定的三角区域，矩阵另一三角及未参与运算的元素保持原值不变。属于 LAPACK 格式转换算子。

数学表达式：

```
A[triangular] = unpack(AP)
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasStpttr | 单精度压缩三角矩阵展开为常规矩阵 |
| aclblasCtpttr | 单精度复数 packed 三角矩阵按列解包 |

## 算子执行接口

### aclblasStpttr

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasStpttr(aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* AP, float* A, int lda)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | 三角存储方式：ACLBLAS_UPPER(121)、ACLBLAS_LOWER(122)，Host 内存 |
| n | 输入 | int | 方阵维数，须 >= 0；为 0 时在校验 handle、uplo 和 lda 后立即返回成功，Host 内存 |
| AP | 输入 | const float*（FP32） | 压缩格式输入，float 数组，长度 n*(n+1)/2，Device 内存 |
| A | 输入/输出 | float*（FP32） | 常规输出矩阵，float 数组，维度 lda × n；非目标三角保持原值，Device 内存 |
| lda | 输入 | int | A 的主维长度，须满足 lda >= max(1, n)，Host 内存 |

#### 约束说明

- n >= 0
- lda >= max(1, n)（n = 0 时仍须满足，即 lda >= 1，(n = 0, lda = 0) 返回 ACLBLAS_STATUS_INVALID_VALUE）
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

int aclblasStpttrTest(AclContext& ctx)
{
    constexpr int n = 3;
    constexpr int lda = 3;
    constexpr size_t aSize = lda * n * sizeof(float);
    constexpr size_t apSize = n * (n + 1) / 2 * sizeof(float);

    // 输入压缩格式（按列打包上三角）：1 2 5 3 6 9
    float hAP[n * (n + 1) / 2] = {1.0f, 2.0f, 5.0f, 3.0f, 6.0f, 9.0f};
    float hA[lda * n] = {0.0f};

    void *rawAP = nullptr;
    auto aclRet = aclrtMalloc(&rawAP, apSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dAP(rawAP);

    void *rawA = nullptr;
    aclRet = aclrtMalloc(&rawA, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dA(rawA);

    aclRet = aclrtMemcpy(dAP.get(), apSize, hAP, apSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);
    std::unique_ptr<void, AclblasHandleDeleter> handle(rawHandle);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handle.get()), ctx.Stream());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    blasRet = aclblasStpttr(
        static_cast<aclblasHandle_t>(handle.get()), ACLBLAS_UPPER, n,
        static_cast<const float*>(dAP.get()), static_cast<float*>(dA.get()), lda);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    aclRet = aclrtSynchronizeStream(ctx.Stream());
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclRet = aclrtMemcpy(hA, aSize, dA.get(), aSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 预期结果（列主序上三角）：1 0 0 2 5 0 3 6 9
    for (int i = 0; i < lda * n; i++) {
        printf("A[%d] = %f\n", i, hA[i]);
    }

    return 0;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasStpttrTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}
```

### aclblasCtpttr

#### 产品支持情况

| 产品 | 支持情况 |
|---|---|
| Ascend 950PR | 支持 |
| Ascend 950DT | 支持 |

#### 函数原型

```cpp
aclblasStatus_t aclblasCtpttr(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    const int n,
    const aclblasComplex* AP,
    aclblasComplex* A,
    const int lda);
```

将单精度复数 packed 数组 AP 按列解包到前导维度为 lda 的列主序矩阵 A。
UPPER 每列写入第 0 行到对角线，LOWER 每列写入对角线到最后一行。另一三角和 padding 保留原值。

- uplo 取 ACLBLAS_UPPER 或 ACLBLAS_LOWER。
- n ≥ 0，lda ≥ max(1, n)，AP 至少包含 n × (n + 1) / 2 个复数，A 至少容纳 lda × n 个复数。
- AP 和 A 均在 Device 内存且须互不重叠。n = 0 时在校验 handle、n、uplo 和 lda 后直接返回。
- 通过 handle 绑定的 stream 异步执行。调用方需要等待 stream 完成后再读取 A。
