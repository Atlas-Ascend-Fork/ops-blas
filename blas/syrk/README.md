# Syrk算子

## 算子概述

Syrk（Symmetric Rank-K Update）算子实现了对称矩阵的秩-k更新操作，将 alpha * op(A) * op(A)^T 加到对称矩阵 C 的指定三角区域。

数学表达式：

```
trans='N': C = alpha * A * A^T + beta * C
trans='T': C = alpha * A^T * A + beta * C
```

其中 A 为 N×K（trans='N'）或 K×N（trans='T'）矩阵，C 为 N×N 对称矩阵。uplo='U' 时仅更新上三角区域，uplo='L' 时仅更新下三角区域；另一三角的值语义不变、不参与 uplo 三角的计算（与 cuBLAS 语义一致，完整对称矩阵由调用方按对称性自行维护）。注意「值语义不变」不等于「零访问」：ssyrk 的 scale kernel 按 64×64 块整块读出/写回 C，对角跨越块的非 uplo 段会读取 C 旧值并恒等回写（语义上不修改）；csyrk 严格只访问 uplo 三角。详见各自接口章节的边界情况处理。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSsyrk | 单精度对称秩-k更新 |
| aclblasCsyrk | 单精度复数对称秩-k更新 |

## 算子执行接口

### aclblasSsyrk

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

> Ascend 950PR/Ascend 950DT 上的 ssyrk 依赖 CANN asc-devkit >= 9.1（`ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR >= 1`），低于该版本时编译与运行将跳过此算子。

#### 函数原型

```cpp
aclblasStatus_t aclblasSsyrk(aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const float* alpha, const float* A, int lda, const float* beta, float* C, int ldc)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | C 矩阵存储模式：ACLBLAS_UPPER(121) 保留上三角或 ACLBLAS_LOWER(122) 保留下三角，Host 内存 |
| trans | 输入 | aclblasOperation_t | A 矩阵转置模式：ACLBLAS_OP_N(111) 不转置、ACLBLAS_OP_T(112) 转置或 ACLBLAS_OP_C(113) 共轭转置（实数域等价于 OP_T），Host 内存 |
| n | 输入 | int | C 矩阵的阶数，n >= 0，Host 内存 |
| k | 输入 | int | 归约维度：trans='N' 时为 A 的列数（第二维度），trans='T'/'C' 时为 A 的行数（第一维度），k >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量乘数，不可为 nullptr，Device 内存 |
| A | 输入 | const float*（FP32） | 输入矩阵，trans='N' 时维度为 N×K，trans='T' 时维度为 K×N，Device 内存 |
| lda | 输入 | int | A 矩阵的主维，trans='N' 时 lda >= max(1, n)，trans='T'/'C' 时 lda >= max(1, k)，Host 内存 |
| beta | 输入 | const float*（FP32） | 标量乘数，不可为 nullptr，Device 内存 |
| C | 输入/输出 | float*（FP32） | N×N 对称矩阵，输入旧值，输出新值，Device 内存 |
| ldc | 输入 | int | C 矩阵的主维，ldc >= max(1, n)，Host 内存 |

#### 约束说明

- n >= 0, k >= 0
- uplo 为 ACLBLAS_UPPER 或 ACLBLAS_LOWER
- trans 为 ACLBLAS_OP_N、ACLBLAS_OP_T 或 ACLBLAS_OP_C（实数域 OP_C 等价于 OP_T）
- trans='N' 时：lda >= max(1, n)
- trans='T' 或 trans='C' 时：lda >= max(1, k)
- ldc >= max(1, n)
- alpha、beta 不可为 nullptr
- A 不可为 nullptr（当 n > 0 且 k > 0 时）
- C 不可为 nullptr（当 n > 0 时）。该校验为有意严于 cuBLAS 的差异：cuBLAS 仅在 beta != 0 时将 C 为 nullptr 视为错误，本实现为保证错误码口径一致，n > 0 时一律要求非空

边界情况处理：

- n = 0 时为合法 no-op，直接返回 ACLBLAS_STATUS_SUCCESS
- k = 0 或 alpha = 0 时跳过矩阵乘，仅对 uplo 指定三角执行 C = beta * C
- beta = 0 时 C 的取值无需有效（cuBLAS 契约 "If beta == 0 then C does not have to be a valid input"）：uplo 指定三角不读取 C 旧值做缩放、直接写为新值（含全零），其结果不依赖 C 旧值，C 旧值中的 NaN/Inf 不进入 uplo 三角的结果；但 C 指针仍须有效（n > 0 时非空校验不变），且实现上 C 仍被读取——对角跨越 64×64 块的非 uplo 段按 C_old 原值恒等回写（语义上不修改，该段的 NaN/Inf 原位保留、不扩散到 uplo 三角）

workspace 说明（arch35）：

- arch35 实现需约 4·n² 字节库工作区暂存 op(A)·op(A)^T 中间结果；当超出 ACLBLAS_MAX_WORKSPACE_SIZE（2 GiB）时返回 ACLBLAS_STATUS_ALLOC_FAILED

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
// 注意：示例代码使用 printf 便于演示，生产代码应使用 dlog（OP_LOGE/OP_LOGI/OP_LOGD）

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

int aclblasSsyrkTest(AclContext& ctx)
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
    // A 为 N×K 列主序矩阵 (trans='N'): A = [[1, 2], [3, 4]]
    // 列主序存储: [1, 3, 2, 4]
    constexpr int n = 2;
    constexpr int k = 2;
    constexpr int lda = n;
    constexpr int ldc = n;
    std::vector<float> hA = {1.0f, 3.0f, 2.0f, 4.0f};
    std::vector<float> hC(n * n, 0.0f);
    float alpha = 1.0f;
    float beta = 0.0f;

    // 3. 申请 Device 内存并拷贝数据
    constexpr size_t aSize = n * k * sizeof(float);
    constexpr size_t cSize = n * n * sizeof(float);

    void* rawA = nullptr;
    aclError aclRet = aclrtMalloc(&rawA, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtr(rawA, aclrtFree);

    void* rawC = nullptr;
    aclRet = aclrtMalloc(&rawC, cSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for C failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dCPtr(rawC, aclrtFree);

    void* rawAlpha = nullptr;
    aclRet = aclrtMalloc(&rawAlpha, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for alpha failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAlphaPtr(rawAlpha, aclrtFree);

    void* rawBeta = nullptr;
    aclRet = aclrtMalloc(&rawBeta, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for beta failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dBetaPtr(rawBeta, aclrtFree);

    aclRet = aclrtMemcpy(dAPtr.get(), aSize, hA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(dCPtr.get(), cSize, hC.data(), cSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for C failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(dAlphaPtr.get(), sizeof(float), &alpha, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for alpha failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(dBetaPtr.get(), sizeof(float), &beta, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for beta failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasSsyrk
    // 计算 C = 1.0 * A * A^T + 0.0 * C
    // A * A^T = [[1,2],[3,4]] * [[1,3],[2,4]] = [[5,11],[11,25]]
    blasRet = aclblasSsyrk(
        static_cast<aclblasHandle_t>(handlePtr.get()), ACLBLAS_LOWER, ACLBLAS_OP_N,
        n, k,
        static_cast<const float*>(dAlphaPtr.get()),
        static_cast<const float*>(dAPtr.get()), lda,
        static_cast<const float*>(dBetaPtr.get()),
        static_cast<float*>(dCPtr.get()), ldc);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSsyrk failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    std::vector<float> cResult(n * n, 0.0f);
    aclRet = aclrtMemcpy(cResult.data(), cSize, dCPtr.get(), cSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet); return aclRet);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            LOG_PRINT("C[%d][%d] = %f\n", i, j, cResult[j * ldc + i]);
        }
    }

    LOG_PRINT("aclblasSsyrk test passed\n");
    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSsyrkTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSsyrkTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

### aclblasCsyrk

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持（arch35 实现）
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持（arch22 实现）
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持

> 两套架构实现并存：`arch22/` 面向 Atlas A2，`arch35/` 面向 Ascend 950PR。
> 两者接口签名一致，但 `trans` 取值域不同，见下方「约束说明」。

#### 函数原型

```cpp
aclblasStatus_t aclblasCsyrk(aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha, const aclblasComplex* A, int lda, const aclblasComplex* beta, aclblasComplex* C, int ldc)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | C 矩阵存储模式：ACLBLAS_UPPER(121) 保留上三角或 ACLBLAS_LOWER(122) 保留下三角，Host 内存 |
| trans | 输入 | aclblasOperation_t | A 矩阵转置模式：ACLBLAS_OP_N(111) 不转置、ACLBLAS_OP_T(112) 普通转置；`ACLBLAS_OP_C`(113) 仅 arch35 接受（本算子为对称（非厄米特）运算，OP_C 不共轭、按 OP_T 等价处理），Host 内存 |
| n | 输入 | int | C 矩阵的阶数，n >= 0，Host 内存 |
| k | 输入 | int | A 矩阵的第二维度（trans='N' 时为 A 的列数，trans='T'/'C' 时为 A 的行数），k >= 0，Host 内存 |
| alpha | 输入 | const aclblasComplex*（complex64） | 复数标量乘数，不可为 nullptr，Device 内存 |
| A | 输入 | const aclblasComplex*（complex64） | 输入矩阵，trans='N' 时维度为 N×K，trans='T'/'C' 时维度为 K×N，Device 内存 |
| lda | 输入 | int | A 矩阵的主维，trans='N' 时 lda >= max(1, n)，trans='T'/'C' 时 lda >= max(1, k)，Host 内存 |
| beta | 输入 | const aclblasComplex*（complex64） | 复数标量乘数，不可为 nullptr，Device 内存 |
| C | 输入/输出 | aclblasComplex*（complex64） | N×N 对称（非厄米特）复数矩阵，输入旧值、输出新值，仅 uplo 引用三角被更新，另一三角不访问、由对称性隐含，Device 内存 |
| ldc | 输入 | int | C 矩阵的主维，ldc >= max(1, n)，Host 内存 |

#### 约束说明

- n >= 0, k >= 0
- uplo 为 ACLBLAS_UPPER 或 ACLBLAS_LOWER
- trans 为 ACLBLAS_OP_N、ACLBLAS_OP_T；`ACLBLAS_OP_C` 在 arch35 上等价于 OP_T（不共轭）并被接受，在 arch22 上返回 `ACLBLAS_STATUS_INVALID_ENUM`
- trans='N' 时：lda >= max(1, n)
- trans='T'/'C' 时：lda >= max(1, k)
- ldc >= max(1, n)
- alpha、beta 不可为 nullptr
- A 不可为 nullptr（当 n > 0 且 k > 0 时）
- C 不可为 nullptr（当 n > 0 时）
- n = 0 时为合法 no-op；alpha=(0,0) 或 k=0 且 beta≠(1,0) 时仅对引用三角执行 C = beta*C
- alpha、beta 均为复数，合并阶段按复数乘法缩放
- 输出 C 为**对称**（非 Hermitian）：C[i][j] = C[j][i]，**对角线元素虚部不为零**且携带有效信息——这是与 cherk 最容易混淆的一点
- A、C 为列主序 complex64（`aclblasComplex`，即 fp32 实部 + fp32 虚部）存储的 Device 内存
- 本算子内部会申请库工作区暂存拆分后的实数矩阵与 GEMM 中间结果；当超出 `ACLBLAS_MAX_WORKSPACE_SIZE`（2 GiB）时返回 `ACLBLAS_STATUS_ALLOC_FAILED` 并在日志中给出所需字节数

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
// 注意：示例代码使用 printf 便于演示，生产代码应使用 dlog（OP_LOGE/OP_LOGI/OP_LOGD）

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

struct BlasHandleDeleter {
    void operator()(aclblasHandle_t h) const { aclblasDestroy(h); }
};

int aclblasCsyrkTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-blas 句柄
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasHandle_t>::type, BlasHandleDeleter> handlePtr(rawHandle);

    blasRet = aclblasSetStream(handlePtr.get(), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 2. 准备 Host 数据
    // A 为 N×K 列主序复矩阵 (trans='N'): A = [[1+1i, 2+0i], [3+0i, 4+0i]]
    // 列主序存储: col0 = [1+1i, 3+0i], col1 = [2+0i, 4+0i]
    constexpr int n = 2;
    constexpr int k = 2;
    constexpr int lda = n;
    constexpr int ldc = n;
    std::vector<aclblasComplex> hA = {{1.0f, 1.0f}, {3.0f, 0.0f}, {2.0f, 0.0f}, {4.0f, 0.0f}};
    std::vector<aclblasComplex> hC(n * n, {0.0f, 0.0f});
    aclblasComplex alpha = {1.0f, 0.0f};  // alpha、beta 均为复数
    aclblasComplex beta = {0.0f, 0.0f};

    // 3. 申请 Device 内存并拷贝数据
    constexpr size_t aSize = n * k * sizeof(aclblasComplex);
    constexpr size_t cSize = n * n * sizeof(aclblasComplex);

    void* rawA = nullptr;
    aclError aclRet = aclrtMalloc(&rawA, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtr(rawA, aclrtFree);

    void* rawC = nullptr;
    aclRet = aclrtMalloc(&rawC, cSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for C failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dCPtr(rawC, aclrtFree);

    void* rawAlpha = nullptr;
    aclRet = aclrtMalloc(&rawAlpha, sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for alpha failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAlphaPtr(rawAlpha, aclrtFree);

    void* rawBeta = nullptr;
    aclRet = aclrtMalloc(&rawBeta, sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for beta failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dBetaPtr(rawBeta, aclrtFree);

    aclRet = aclrtMemcpy(dAPtr.get(), aSize, hA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(dCPtr.get(), cSize, hC.data(), cSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for C failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(dAlphaPtr.get(), sizeof(aclblasComplex), &alpha, sizeof(aclblasComplex), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for alpha failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(dBetaPtr.get(), sizeof(aclblasComplex), &beta, sizeof(aclblasComplex), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for beta failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasCsyrk
    // 计算 C = alpha * A * A^T + beta * C (trans='N', uplo=LOWER)
    // 注意是普通转置 A^T 而非共轭转置：A^T = [[1+1i, 3+0i], [2+0i, 4+0i]]
    // A * A^T = [[4+2i, 11+3i], [11+3i, 25+0i]]
    // 结果为对称矩阵: C[0][1] = C[1][0]（不取共轭），且 C[0][0] 的虚部为 2 而非 0
    // uplo=LOWER: 仅更新下三角 (C[0][0], C[1][0], C[1][1])
    blasRet = aclblasCsyrk(
        handlePtr.get(), ACLBLAS_LOWER, ACLBLAS_OP_N,
        n, k,
        static_cast<const aclblasComplex*>(dAlphaPtr.get()),
        static_cast<const aclblasComplex*>(dAPtr.get()), lda,
        static_cast<const aclblasComplex*>(dBetaPtr.get()),
        static_cast<aclblasComplex*>(dCPtr.get()), ldc);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCsyrk failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    std::vector<aclblasComplex> cResult(n * n, {0.0f, 0.0f});
    aclRet = aclrtMemcpy(cResult.data(), cSize, dCPtr.get(), cSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet); return aclRet);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aclblasComplex val = cResult[static_cast<size_t>(j) * ldc + i];
            LOG_PRINT("C[%d][%d] = %.4f + %.4fi\n", i, j, static_cast<double>(val.real), static_cast<double>(val.imag));
        }
    }

    LOG_PRINT("aclblasCsyrk test passed\n");
    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasCsyrkTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasCsyrkTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```
