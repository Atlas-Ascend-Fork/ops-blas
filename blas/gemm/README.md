# Gemm算子

## 算子概述

aclblasSgemm / aclblasCgemm / aclblasCgemmEx 算子实现了通用矩阵乘法，核心运算为 C = alpha * op(A) * op(B) + beta * C。aclblasSgemm 支持单精度（FP32）数据类型，aclblasCgemm 支持复数单精度（COMPLEX64，每个复数由两个 FP32 分量组成）数据类型，aclblasCgemmEx 在此基础上允许 A/B/C 独立选择 FP32 / Complex32 / FP16 / Complex16 元素类型。均采用 BLAS 标准列主序存储。

数学表达式：

```
C = alpha * op(A) * op(B) + beta * C
```

其中：
- op(A) = A（transA = ACLBLAS_OP_N）、A^T（transA = ACLBLAS_OP_T）或 A^H（transA = ACLBLAS_OP_C，FP32 实数场景下等价于 A^T）
- op(B) = B（transB = ACLBLAS_OP_N）、B^T（transB = ACLBLAS_OP_T）或 B^H（transB = ACLBLAS_OP_C，FP32 实数场景下等价于 B^T）
- A 为 M×K 矩阵，B 为 K×N 矩阵，C 为 M×N 矩阵

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSgemm | 单精度（FP32）通用矩阵乘法，支持矩阵转置和 alpha/beta 缩放 |
| aclblasCgemm | 复数单精度（COMPLEX64）通用矩阵乘法，支持矩阵转置和 alpha/beta 缩放 |
| aclblasCgemmEx | 扩展复数通用矩阵乘法，A/B/C 可独立指定元素类型（Ascend 950） |

## 算子执行接口

### aclblasSgemm

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSgemm(aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, const float* alpha, const float* A, int lda, const float* B, int ldb, const float* beta, float* C, int ldc)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| transa | 输入 | aclblasOperation_t | 矩阵 A 的操作类型：ACLBLAS_OP_N（不转置）、ACLBLAS_OP_T（转置）、ACLBLAS_OP_C（共轭转置，FP32 实数等价于转置），Host 内存 |
| transb | 输入 | aclblasOperation_t | 矩阵 B 的操作类型（同 transa），Host 内存 |
| m | 输入 | int | op(A) 和 C 的行数，M >= 0，Host 内存 |
| n | 输入 | int | op(B) 和 C 的列数，N >= 0，Host 内存 |
| k | 输入 | int | op(A) 的列数和 op(B) 的行数，K >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量 alpha 指针，指向 FP32 标量，不可为 nullptr，Host 内存 |
| A | 输入 | const float*（FP32） | 矩阵 A 的设备内存指针，FP32，列主序；当 K > 0 时不可为 nullptr，Device 内存 |
| lda | 输入 | int | 矩阵 A 的主维度（列主序），transA=N 时 lda >= max(1, M)，transA=T/C 时 lda >= max(1, K)，Host 内存 |
| B | 输入 | const float*（FP32） | 矩阵 B 的设备内存指针，FP32，列主序；当 K > 0 时不可为 nullptr，Device 内存 |
| ldb | 输入 | int | 矩阵 B 的主维度（列主序），transB=N 时 ldb >= max(1, K)，transB=T/C 时 ldb >= max(1, N)，Host 内存 |
| beta | 输入 | const float*（FP32） | 标量 beta 指针，指向 FP32 标量，不可为 nullptr，Host 内存 |
| C | 输入/输出 | float*（FP32） | 矩阵 C 的设备内存指针，FP32，列主序；当 K > 0 或 beta != 0 时不可为 nullptr，Device 内存 |
| ldc | 输入 | int | 矩阵 C 的主维度（列主序），ldc >= max(1, M)，Host 内存 |

#### 约束说明

- m >= 0
- n >= 0
- k >= 0
- transa 必须为 ACLBLAS_OP_N、ACLBLAS_OP_T 或 ACLBLAS_OP_C
- transb 必须为 ACLBLAS_OP_N、ACLBLAS_OP_T 或 ACLBLAS_OP_C
- transA = N 时 lda >= max(1, m)；transA = T/C 时 lda >= max(1, k)
- transB = N 时 ldb >= max(1, k)；transB = T/C 时 ldb >= max(1, n)
- ldc >= max(1, m)
- alpha 不可为 nullptr
- beta 不可为 nullptr
- k > 0 时 A 不可为 nullptr
- k > 0 时 B 不可为 nullptr
- k > 0 或 beta != 0 时 C 不可为 nullptr

边界情况处理：
- m == 0 或 n == 0 时直接返回 ACLBLAS_STATUS_SUCCESS，不执行计算
- k == 0 或 alpha == 0 时跳过矩阵乘，执行 C = beta * C（beta == 0 时置零，beta == 1 时不变，其他值逐元素缩放）
- beta == 0.0f 时 C 无需在调用前初始化

依赖限制：
- arch35（Ascend 950）的 Cube kernel 使用 tensor_api，需 asc-devkit >= 9.1（ASC_DEVKIT_MAJOR >= 9 且 ASC_DEVKIT_MINOR > 0）

#### 精度验证

采用 `MIXED_TOLERANCE` 混合容差策略（对齐[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) §2.1）：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
|----------|------|------|----------------------|-------------------|
| FLOAT32 | 2^-10 (9.77e-4) | 2^-16 (1.53e-5) | 0.99 | 1e-2 或 32·ULP |

其中，ULP（Unit in the Last Place，末位单位）表示给定浮点数与相邻可表示值之间的间距。
`max_abs_error_limit` 按输出元素逐个计算：第`i`个元素的误差上限为
`max(1e-2, 32 * ULP(abs(golden[i])))`。精度验证需同时满足以下条件：

- 至少 99% 的输出元素满足 `abs(actual[i] - golden[i]) <= atol + rtol * abs(golden[i])`。
- 所有输出元素均满足 `abs(actual[i] - golden[i]) <= max_abs_error_limit[i]`。

因此，即使 `required_matched_ratio` 已达到 0.99 或 1.0，只要任一元素超过对应的
`max_abs_error_limit`，用例仍会校验失败。当 `golden[i]` 接近 0 时，固定上限`1e-2`
起主要作用；设计大倍率`alpha`或`beta`的测试用例时，需同时考虑绝对误差的放大效应。

alpha == 0（且 alpha 非空）时使用 `EXACT` 位精确校验（C = beta * C 结果应位精确）。

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

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

struct AclMemDeleter {
    void operator()(void* p) const { aclrtFree(p); }
};

int aclblasSgemmTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-blas 句柄
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<void, aclblasStatus_t (*)(void*)> handlePtr(rawHandle, aclblasDestroy);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 2. 准备 Host 数据
    // C = alpha * op(A) * op(B) + beta * C
    // M=2, N=2, K=3, transA=N, transB=N, alpha=1.0, beta=0.0
    int m = 2, n = 2, k = 3;
    int lda = m, ldb = k, ldc = m;
    float alpha = 1.0f;
    float beta = 0.0f;
    aclblasOperation_t transA = ACLBLAS_OP_N;
    aclblasOperation_t transB = ACLBLAS_OP_N;

    // A (M=2, K=3, column-major, lda=2): [[1,2,3],[4,5,6]]
    std::vector<float> hA = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
    // B (K=3, N=2, column-major, ldb=3): [[1,0],[0,1],[1,1]]
    std::vector<float> hB = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f};
    // C (M=2, N=2, column-major, ldc=2): expected result [[4,5],[10,11]]
    std::vector<float> hC(static_cast<size_t>(ldc) * n, 0.0f);

    size_t aBytes = hA.size() * sizeof(float);
    size_t bBytes = hB.size() * sizeof(float);
    size_t cBytes = hC.size() * sizeof(float);

    // 3. 申请 Device 内存并拷贝数据
    void* rawA = nullptr;
    auto aclRet = aclrtMalloc(&rawA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<float, AclMemDeleter> aDevicePtr(static_cast<float*>(rawA));

    void* rawB = nullptr;
    aclRet = aclrtMalloc(&rawB, bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for B failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<float, AclMemDeleter> bDevicePtr(static_cast<float*>(rawB));

    void* rawC = nullptr;
    aclRet = aclrtMalloc(&rawC, cBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for C failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<float, AclMemDeleter> cDevicePtr(static_cast<float*>(rawC));

    aclRet = aclrtMemcpy(aDevicePtr.get(), aBytes, hA.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(bDevicePtr.get(), bBytes, hB.data(), bBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for B failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(cDevicePtr.get(), cBytes, hC.data(), cBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for C failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasSgemm
    blasRet = aclblasSgemm(static_cast<aclblasHandle_t>(handlePtr.get()),
                           transA, transB, m, n, k, &alpha,
                           aDevicePtr.get(), lda, bDevicePtr.get(), ldb,
                           &beta, cDevicePtr.get(), ldc);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSgemm failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    aclRet = aclrtMemcpy(hC.data(), cBytes, cDevicePtr.get(), cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", aclRet); return aclRet);

    // 打印结果（列主序存储：hC[col * ldc + row] = C[row][col]）
    LOG_PRINT("result C (column-major):\n");
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < m; row++) {
            LOG_PRINT("  C[%d][%d] = %f\n", row, col, hC[static_cast<size_t>(col) * ldc + row]);
        }
    }

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSgemmTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSgemmTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

预期输出：

```
result C (column-major):
  C[0][0] = 4.000000
  C[1][0] = 10.000000
  C[0][1] = 5.000000
  C[1][1] = 11.000000
```

### aclblasCgemm

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### Ascend 950PR 实现说明

- 实现目录：`blas/gemm/arch35/`
- 调用方式：通过 `aclblasHandle_t` 绑定 stream，直接下发 Ascend C Kernel
- 数据类型：`aclblasComplex`（单精度复数，实部和虚部均为 FP32）
- 存储格式：BLAS 列主序，支持 `N`、`T`、`C` 全部转置组合
- 自测目录：`test/gemm/cgemm/arch35/`
- 精度测试：`./build/test/gemm/cgemm/cgemm_test`

#### 函数原型

```cpp
aclblasStatus_t aclblasCgemm(aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, const aclblasComplex* alpha, const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb, const aclblasComplex* beta, aclblasComplex* C, int ldc)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| transa | 输入 | aclblasOperation_t | 矩阵 A 的操作类型：ACLBLAS_OP_N（不转置）、ACLBLAS_OP_T（转置）、ACLBLAS_OP_C（共轭转置），Host 内存 |
| transb | 输入 | aclblasOperation_t | 矩阵 B 的操作类型（同 transa），Host 内存 |
| m | 输入 | int | op(A) 和 C 的行数，M >= 0，Host 内存 |
| n | 输入 | int | op(B) 和 C 的列数，N >= 0，Host 内存 |
| k | 输入 | int | op(A) 的列数和 op(B) 的行数，K >= 0，Host 内存 |
| alpha | 输入 | const aclblasComplex* | 标量 alpha 指针，不可为 nullptr，Host 内存 |
| A | 输入 | const aclblasComplex* | 矩阵 A 的设备内存指针，COMPLEX64，列主序；当 K > 0 且 alpha != 0 时不可为 nullptr，Device 内存 |
| lda | 输入 | int | 矩阵 A 的主维度（列主序），transA=N 时 lda >= max(1, M)，transA=T/C 时 lda >= max(1, K)，Host 内存 |
| B | 输入 | const aclblasComplex* | 矩阵 B 的设备内存指针，COMPLEX64，列主序；当 K > 0 且 alpha != 0 时不可为 nullptr，Device 内存 |
| ldb | 输入 | int | 矩阵 B 的主维度（列主序），transB=N 时 ldb >= max(1, K)，transB=T/C 时 ldb >= max(1, N)，Host 内存 |
| beta | 输入 | const aclblasComplex* | 标量 beta 指针，不可为 nullptr，Host 内存 |
| C | 输入/输出 | aclblasComplex* | 矩阵 C 的设备内存指针，COMPLEX64，列主序；当 K > 0 或 beta != 0 时不可为 nullptr，Device 内存 |
| ldc | 输入 | int | 矩阵 C 的主维度（列主序），ldc >= max(1, M)，Host 内存 |

#### 约束说明

- m >= 0
- n >= 0
- k >= 0
- transa 必须为 ACLBLAS_OP_N、ACLBLAS_OP_T 或 ACLBLAS_OP_C
- transb 必须为 ACLBLAS_OP_N、ACLBLAS_OP_T 或 ACLBLAS_OP_C
- transA = N 时 lda >= max(1, m)；transA = T/C 时 lda >= max(1, k)
- transB = N 时 ldb >= max(1, k)；transB = T/C 时 ldb >= max(1, n)
- ldc >= max(1, m)
- alpha 不可为 nullptr
- beta 不可为 nullptr
- k > 0 时 A 不可为 nullptr
- k > 0 时 B 不可为 nullptr
- k > 0 或 beta != 0 时 C 不可为 nullptr

边界情况处理：
- m == 0 或 n == 0 时直接返回 ACLBLAS_STATUS_SUCCESS，不执行计算
- k == 0 或 alpha == 0 时跳过矩阵乘，执行 C = beta * C（beta == 0 时置零，beta == 1 时不变，其他值逐元素缩放）
- beta == 0.0f 时 C 无需在调用前初始化

依赖限制：
- arch35（Ascend 950）的 Cube kernel 使用 tensor_api，需 asc-devkit >= 9.1（ASC_DEVKIT_MAJOR >= 9 且 ASC_DEVKIT_MINOR > 0）

#### 精度验证

采用 `MIXED_TOLERANCE` 混合容差策略（对齐[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) §2.1）：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
|----------|------|------|----------------------|-------------------|
| FLOAT32 | 2^-10 (9.77e-4) | 2^-16 (1.53e-5) | 0.99 | 1e-2 或 32·ULP |

其中，ULP（Unit in the Last Place，末位单位）表示给定浮点数与相邻可表示值之间的间距。
`max_abs_error_limit` 按输出元素逐个计算：第`i`个元素的误差上限为
`max(1e-2, 32 * ULP(abs(golden[i])))`。精度验证需同时满足以下条件：

- 至少 99% 的输出元素满足 `abs(actual[i] - golden[i]) <= atol + rtol * abs(golden[i])`。
- 所有输出元素均满足 `abs(actual[i] - golden[i]) <= max_abs_error_limit[i]`。

因此，即使 `required_matched_ratio` 已达到 0.99 或 1.0，只要任一元素超过对应的
`max_abs_error_limit`，用例仍会校验失败。当 `golden[i]` 接近 0 时，固定上限`1e-2`
起主要作用；设计大倍率`alpha`或`beta`的测试用例时，需同时考虑绝对误差的放大效应。

alpha == 0（且 alpha 非空）时使用 `EXACT` 位精确校验（C = beta * C 结果应位精确）。

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

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

struct AclMemDeleter {
    void operator()(void* p) const { aclrtFree(p); }
};

int aclblasCgemmTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-blas 句柄
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<void, aclblasStatus_t (*)(void*)> handlePtr(rawHandle, aclblasDestroy);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 2. 准备 Host 数据
    // C = alpha * op(A) * op(B) + beta * C
    // M=2, N=2, K=2, transA=N, transB=N, alpha=(1+0i), beta=(0+0i)
    int m = 2, n = 2, k = 2;
    int lda = m, ldb = k, ldc = m;
    aclblasComplex alpha = {1.0f, 0.0f};
    aclblasComplex beta = {0.0f, 0.0f};
    aclblasOperation_t transA = ACLBLAS_OP_N;
    aclblasOperation_t transB = ACLBLAS_OP_N;

    // A (M=2, K=2, column-major): [[1+1i, 3+0i], [2+0i, 4+1i]]
    std::vector<aclblasComplex> hA = {{{1.0f, 1.0f}, {2.0f, 0.0f}, {3.0f, 0.0f}, {4.0f, 1.0f}}};
    // B (K=2, N=2, column-major): [[1+0i, 0+1i], [1+0i, 1+0i]]
    std::vector<aclblasComplex> hB = {{{1.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 0.0f}}};
    std::vector<aclblasComplex> hC(static_cast<size_t>(ldc) * n, {0.0f, 0.0f});

    size_t aBytes = hA.size() * sizeof(aclblasComplex);
    size_t bBytes = hB.size() * sizeof(aclblasComplex);
    size_t cBytes = hC.size() * sizeof(aclblasComplex);

    // 3. 申请 Device 内存并拷贝数据
    aclblasComplex* rawA = nullptr;
    auto aclRet = aclrtMalloc(reinterpret_cast<void**>(&rawA), aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<aclblasComplex, AclMemDeleter> aDevicePtr(rawA);

    aclblasComplex* rawB = nullptr;
    aclRet = aclrtMalloc(reinterpret_cast<void**>(&rawB), bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for B failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<aclblasComplex, AclMemDeleter> bDevicePtr(rawB);

    aclblasComplex* rawC = nullptr;
    aclRet = aclrtMalloc(reinterpret_cast<void**>(&rawC), cBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for C failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<aclblasComplex, AclMemDeleter> cDevicePtr(rawC);

    aclRet = aclrtMemcpy(aDevicePtr.get(), aBytes, hA.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(bDevicePtr.get(), bBytes, hB.data(), bBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for B failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(cDevicePtr.get(), cBytes, hC.data(), cBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for C failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasCgemm
    blasRet = aclblasCgemm(static_cast<aclblasHandle_t>(handlePtr.get()),
                           transA, transB, m, n, k, &alpha,
                           aDevicePtr.get(), lda, bDevicePtr.get(), ldb,
                           &beta, cDevicePtr.get(), ldc);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCgemm failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    aclRet = aclrtMemcpy(hC.data(), cBytes, cDevicePtr.get(), cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", aclRet); return aclRet);

    LOG_PRINT("result C (column-major):\n");
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < m; row++) {
            const auto& val = hC[static_cast<size_t>(col) * ldc + row];
            LOG_PRINT("  C[%d][%d] = %f + %fi\n", row, col, val.real, val.imag);
        }
    }

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasCgemmTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasCgemmTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

预期输出：

```
result C (column-major):
  C[0][0] = 2.000000 + 1.000000i
  C[1][0] = 3.000000 + 2.000000i
  C[0][1] = 4.000000 + 1.000000i
  C[1][1] = 6.000000 + 3.000000i
```

### aclblasCgemmEx

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasCgemmEx(aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, const aclblasComplex* alpha, const void* A, int typeA, int lda, const void* B, int typeB, int ldb, const aclblasComplex* beta, void* C, int typeC, int ldc)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| transa | 输入 | aclblasOperation_t | 矩阵 A 的操作类型：ACLBLAS_OP_N / ACLBLAS_OP_T / ACLBLAS_OP_C，Host 内存 |
| transb | 输入 | aclblasOperation_t | 矩阵 B 的操作类型（同 transa），Host 内存 |
| m | 输入 | int | op(A) 和 C 的行数，M >= 0，Host 内存 |
| n | 输入 | int | op(B) 和 C 的列数，N >= 0，Host 内存 |
| k | 输入 | int | op(A) 的列数和 op(B) 的行数，K >= 0，Host 内存 |
| alpha | 输入 | const aclblasComplex* | 标量 alpha，不可为 nullptr，Host 内存 |
| A | 输入 | const void* | 矩阵 A，列主序，元素类型由 typeA 指定；当 m,n,k>0 且 alpha!=0 时不可为 nullptr，Device 内存 |
| typeA | 输入 | int | A 的元素类型：0=R_32，1=C_32，2=H_R_32，3=H_C_32，Host 内存 |
| lda | 输入 | int | 矩阵 A 的主维度（列主序），transA=N 时 lda >= max(1, M)，transA=T/C 时 lda >= max(1, K)，Host 内存 |
| B | 输入 | const void* | 矩阵 B，列主序，元素类型由 typeB 指定；当 m,n,k>0 且 alpha!=0 时不可为 nullptr，Device 内存 |
| typeB | 输入 | int | B 的元素类型（取值同 typeA），Host 内存 |
| ldb | 输入 | int | 矩阵 B 的主维度（列主序），transB=N 时 ldb >= max(1, K)，transB=T/C 时 ldb >= max(1, N)，Host 内存 |
| beta | 输入 | const aclblasComplex* | 标量 beta，不可为 nullptr，Host 内存 |
| C | 输入/输出 | void* | 矩阵 C，列主序，元素类型由 typeC 指定；当 m>0 且 n>0 时不可为 nullptr，Device 内存 |
| typeC | 输入 | int | C 的元素类型（取值同 typeA），Host 内存 |
| ldc | 输入 | int | 矩阵 C 的主维度（列主序），ldc >= max(1, M)，Host 内存 |

#### 约束说明

- handle 不可为 nullptr，否则返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR
- transa / transb 必须为 ACLBLAS_OP_N、ACLBLAS_OP_T 或 ACLBLAS_OP_C，否则返回 ACLBLAS_STATUS_INVALID_ENUM
- typeA / typeB / typeC 必须为 0、1、2 或 3（R_32 / C_32 / H_R_32 / H_C_32），否则返回 ACLBLAS_STATUS_INVALID_ENUM
- m >= 0，n >= 0，k >= 0，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- transA = N 时 lda >= max(1, m)；transA = T/C 时 lda >= max(1, k)
- transB = N 时 ldb >= max(1, k)；transB = T/C 时 ldb >= max(1, n)
- ldc >= max(1, m)
- alpha、beta 不可为 nullptr
- m>0 且 n>0 时 C 不可为 nullptr
- m>0、n>0、k>0 且 alpha!=0 时 A、B 不可为 nullptr
- 矩阵按列主序存储
- 仅支持 Ascend 950PR / 950DT（arch35）

边界情况处理：

- m == 0 或 n == 0 时直接返回 ACLBLAS_STATUS_SUCCESS，不访问矩阵指针、不下发 kernel
- k == 0 或 alpha == 0 时跳过矩阵乘，执行 C = beta * C
- 半精度输入会先转到 FP32/Complex32 计算，再写回 typeC

异步与同步：

- 计算 kernel 下发到 handle 绑定的 stream。读取 C 前，调用方必须对该 stream 调用 `aclrtSynchronizeStream`

#### 精度验证

与 aclblasCgemm 相同，采用 `MIXED_TOLERANCE` 混合容差策略。半精度输出按对应浮点精度校验。

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

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

struct AclMemDeleter {
    void operator()(void* p) const { aclrtFree(p); }
};

int aclblasCgemmExTest(AclContext& ctx)
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

    const int m = 2;
    const int n = 2;
    const int k = 2;
    const int lda = m;
    const int ldb = k;
    const int ldc = m;
    aclblasComplex alpha{1.0f, 0.0f};
    aclblasComplex beta{0.0f, 0.0f};

    // 列主序 2x2：A = I，B = [[1,3],[2,4]]，C = A * B
    std::vector<aclblasComplex> hA = {{1.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f}};
    std::vector<aclblasComplex> hB = {{1.0f, 0.0f}, {2.0f, 0.0f}, {3.0f, 0.0f}, {4.0f, 0.0f}};
    std::vector<aclblasComplex> hC(static_cast<size_t>(ldc) * n, {0.0f, 0.0f});
    const size_t aBytes = hA.size() * sizeof(aclblasComplex);
    const size_t bBytes = hB.size() * sizeof(aclblasComplex);
    const size_t cBytes = hC.size() * sizeof(aclblasComplex);

    void* aDevice = nullptr;
    void* bDevice = nullptr;
    void* cDevice = nullptr;
    auto aclRet = aclrtMalloc(&aDevice, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("malloc A failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, AclMemDeleter> aDevicePtr(aDevice);
    aclRet = aclrtMalloc(&bDevice, bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("malloc B failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, AclMemDeleter> bDevicePtr(bDevice);
    aclRet = aclrtMalloc(&cDevice, cBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("malloc C failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, AclMemDeleter> cDevicePtr(cDevice);

    aclRet = aclrtMemcpy(aDevicePtr.get(), aBytes, hA.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy A failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(bDevicePtr.get(), bBytes, hB.data(), bBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy B failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(cDevicePtr.get(), cBytes, hC.data(), cBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy C failed. ERROR: %d\n", aclRet); return aclRet);

    blasRet = aclblasCgemmEx(static_cast<aclblasHandle_t>(handlePtr.get()),
                           ACLBLAS_OP_N, ACLBLAS_OP_N, m, n, k, &alpha,
                           aDevicePtr.get(), 1, lda,
                           bDevicePtr.get(), 1, ldb,
                           &beta, cDevicePtr.get(), 1, ldc);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCgemmEx failed. ERROR: %d\n", blasRet);
              return blasRet);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(hC.data(), cBytes, cDevicePtr.get(), cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy result failed. ERROR: %d\n", aclRet); return aclRet);

    LOG_PRINT("result C (column-major):\n");
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < m; row++) {
            const auto& val = hC[static_cast<size_t>(col) * ldc + row];
            LOG_PRINT("  C[%d][%d] = %f + %fi\n", row, col, val.real, val.imag);
        }
    }

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasCgemmExTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasCgemmExTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

预期输出：

```
result C (column-major):
  C[0][0] = 1.000000 + 0.000000i
  C[1][0] = 2.000000 + 0.000000i
  C[0][1] = 3.000000 + 0.000000i
  C[1][1] = 4.000000 + 0.000000i
```

