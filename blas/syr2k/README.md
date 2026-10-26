# Syr2k算子

## 算子概述

syr2k 算子实现了对称秩2k更新运算，核心运算为 C = alpha * (op(A) * op(B)^T + op(B) * op(A)^T) + beta * C，其中 C 为对称矩阵，仅更新上三角或下三角部分。

数学表达式：

```
C = alpha * (op(A) * op(B)^T + op(B) * op(A)^T) + beta * C

其中 op(X) = X     当 trans = ACLBLAS_OP_N
      op(X) = X^T   当 trans = ACLBLAS_OP_T
      op(X) = X^T   当 trans = ACLBLAS_OP_C（Csyr2k 不执行共轭）
C 为 n×n 对称矩阵，A 和 B 为 n×k 或 k×n 矩阵
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSsyr2k | 单精度浮点对称秩2k更新 |
| aclblasCsyr2k | 单精度复数对称秩2k更新 |

## 算子执行接口

### aclblasSsyr2k

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

> Ascend 950PR/Ascend 950DT 上的 ssyr2k 依赖 CANN asc-devkit >= 9.1（`ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR >= 1`），低于该版本时编译与运行将跳过此算子。

#### 函数原型

```cpp
aclblasStatus_t aclblasSsyr2k(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    int n,
    int k,
    const float* alpha,
    const float* A,
    int lda,
    const float* B,
    int ldb,
    const float* beta,
    float* C,
    int ldc)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | 指定 C 矩阵的上三角或下三角被更新，取值 ACLBLAS_UPPER 或 ACLBLAS_LOWER，Host 内存 |
| trans | 输入 | aclblasOperation_t | 指定对矩阵 A、B 是否转置，取值 ACLBLAS_OP_N、ACLBLAS_OP_T 或 ACLBLAS_OP_C（实数域等价于 OP_T），Host 内存 |
| n | 输入 | int | 对称矩阵 C 的阶数，Host 内存 |
| k | 输入 | int | 归约维度：op(A)/op(B) 的列数，trans='N' 时为 A/B 的列数（第二维度），trans='T'/'C' 时为 A/B 的行数（第一维度），Host 内存 |
| alpha | 输入 | const float*（FP32） | 指向标量乘数 alpha 的指针，Device 内存 |
| A | 输入 | const float*（FP32） | 输入矩阵 A，列优先存储，Device 内存 |
| lda | 输入 | int | 矩阵 A 的前导维度，Host 内存 |
| B | 输入 | const float*（FP32） | 输入矩阵 B，列优先存储，Device 内存 |
| ldb | 输入 | int | 矩阵 B 的前导维度，Host 内存 |
| beta | 输入 | const float*（FP32） | 指向标量乘数 beta 的指针，Device 内存 |
| C | 输入/输出 | float*（FP32） | n×n 对称矩阵 C，列优先存储，仅 uplo 指定的三角部分被更新，Device 内存 |
| ldc | 输入 | int | 矩阵 C 的前导维度，Host 内存 |

#### 约束说明

- n >= 0
- k >= 0
- uplo 必须为 ACLBLAS_UPPER 或 ACLBLAS_LOWER
- trans 必须为 ACLBLAS_OP_N、ACLBLAS_OP_T 或 ACLBLAS_OP_C（实数域 OP_C 等价于 OP_T）
- 当 trans = ACLBLAS_OP_N 时，lda >= max(1, n)；当 trans = ACLBLAS_OP_T 或 ACLBLAS_OP_C 时，lda >= max(1, k)
- 当 trans = ACLBLAS_OP_N 时，ldb >= max(1, n)；当 trans = ACLBLAS_OP_T 或 ACLBLAS_OP_C 时，ldb >= max(1, k)
- ldc >= max(1, n)
- alpha 不能为 nullptr
- beta 不能为 nullptr
- 当 n > 0 且 k > 0 时，A 不能为 nullptr
- 当 n > 0 且 k > 0 时，B 不能为 nullptr
- C 不能为 nullptr（当 n > 0 时）。该校验为有意严于 cuBLAS 的差异：cuBLAS 仅在 beta != 0 时将 C 为 nullptr 视为错误，本实现为保证错误码口径一致，n > 0 时一律要求非空

边界情况处理：

- n = 0 时为合法 no-op，直接返回 ACLBLAS_STATUS_SUCCESS
- k = 0 或 alpha = 0 时跳过矩阵乘，仅对 uplo 指定三角执行 C = beta * C
- beta = 0 时 C 的取值无需有效（cuBLAS 契约 "If beta == 0 then C does not have to be a valid input"）：uplo 指定三角不读取 C 旧值做缩放、直接写为新值（含全零），其结果不依赖 C 旧值，C 旧值中的 NaN/Inf 不进入 uplo 三角的结果；但 C 指针仍须有效（n > 0 时非空校验不变），且实现上 C 仍被读取——对角跨越 64×64 块的非 uplo 段按 C_old 原值恒等回写（语义上不修改，该段的 NaN/Inf 原位保留、不扩散到 uplo 三角）

workspace 说明（arch35）：

- arch35 实现需约 8·n² 字节库工作区暂存 op(A)·op(B)^T 与 op(B)·op(A)^T 两份中间结果；当超出 ACLBLAS_MAX_WORKSPACE_SIZE（2 GiB）时返回 ACLBLAS_STATUS_ALLOC_FAILED

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

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

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

int aclblasSsyr2kTest(AclContext& ctx)
{
    // 参数设置：n=3, k=2, trans=N, uplo=UPPER
    constexpr int n = 3;
    constexpr int k = 2;
    constexpr int lda = n;
    constexpr int ldb = n;
    constexpr int ldc = n;
    float alpha = 1.0f;
    float beta = 0.0f;

    // 矩阵 A (3x2, 列优先): [1,2,3, 4,5,6]
    float hA[lda * k] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    // 矩阵 B (3x2, 列优先): [7,8,9, 10,11,12]
    float hB[ldb * k] = {7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    // 矩阵 C (3x3, 列优先), 初始化为 0
    float hC[ldc * n] = {0.0f};

    constexpr size_t bytesA = lda * k * sizeof(float);
    constexpr size_t bytesB = ldb * k * sizeof(float);
    constexpr size_t bytesC = ldc * n * sizeof(float);
    constexpr size_t bytesScalar = sizeof(float);

    // 1. 创建 ops-blas 句柄
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<void, aclblasStatus_t (*)(void*)> handlePtr(rawHandle, aclblasDestroy);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), ctx.Stream());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 2. 申请 Device 内存
    void* rawAlpha = nullptr;
    auto aclRet = aclrtMalloc(&rawAlpha, bytesScalar, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc alpha failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAlphaPtr(rawAlpha, aclrtFree);

    void* rawBeta = nullptr;
    aclRet = aclrtMalloc(&rawBeta, bytesScalar, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc beta failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dBetaPtr(rawBeta, aclrtFree);

    void* rawA = nullptr;
    aclRet = aclrtMalloc(&rawA, bytesA, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc A failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtr(rawA, aclrtFree);

    void* rawB = nullptr;
    aclRet = aclrtMalloc(&rawB, bytesB, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc B failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dBPtr(rawB, aclrtFree);

    void* rawC = nullptr;
    aclRet = aclrtMalloc(&rawC, bytesC, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc C failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dCPtr(rawC, aclrtFree);

    // 3. 拷贝数据到 Device
    aclRet = aclrtMemcpy(dAlphaPtr.get(), bytesScalar, &alpha, bytesScalar, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy alpha failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dBetaPtr.get(), bytesScalar, &beta, bytesScalar, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy beta failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dAPtr.get(), bytesA, hA, bytesA, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy A failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dBPtr.get(), bytesB, hB, bytesB, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy B failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dCPtr.get(), bytesC, hC, bytesC, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy C failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasSsyr2k
    blasRet = aclblasSsyr2k(
        static_cast<aclblasHandle_t>(handlePtr.get()),
        ACLBLAS_UPPER,
        ACLBLAS_OP_N,
        n, k,
        static_cast<const float*>(dAlphaPtr.get()),
        static_cast<const float*>(dAPtr.get()), lda,
        static_cast<const float*>(dBPtr.get()), ldb,
        static_cast<const float*>(dBetaPtr.get()),
        static_cast<float*>(dCPtr.get()), ldc);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSsyr2k failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(ctx.Stream());
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    aclRet = aclrtMemcpy(hC, bytesC, dCPtr.get(), bytesC, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy C back failed. ERROR: %d\n", aclRet); return aclRet);

    // 打印 C 矩阵上三角部分
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("C[%d][%d] = %f  ", i, j, hC[i + j * ldc]);
        }
        printf("\n");
    }

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSsyr2kTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSsyr2kTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

### aclblasCsyr2k

#### 产品支持情况

- Ascend 950PR：支持
- Ascend 950DT：未验证
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

> Ascend 950PR 实现位于 `blas/syr2k/arch35/`，依赖 CANN asc-devkit 9.1 或更高版本。

#### 函数原型

```cpp
aclblasStatus_t aclblasCsyr2k(
    aclblasHandle_t handle,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    int n,
    int k,
    const aclblasComplex* alpha,
    const aclblasComplex* A,
    int lda,
    const aclblasComplex* B,
    int ldb,
    const aclblasComplex* beta,
    aclblasComplex* C,
    int ldc);
```

#### 约束说明

- `uplo` 为 `ACLBLAS_UPPER` 或 `ACLBLAS_LOWER`，仅指定三角被更新；另一三角值语义不变，但并非「零访问」：uplo='L' 时对角块的非 uplo（严格上）段按 C 旧值恒等回写（语义上不修改），uplo='U' 时非 uplo 段不写出。
- `trans` 支持 `ACLBLAS_OP_N`、`ACLBLAS_OP_T` 和 `ACLBLAS_OP_C`；OP_C 按无共轭 OP_T 处理。
- trans=N 时 `lda, ldb >= max(1,n)`；trans=T/C 时 `lda, ldb >= max(1,k)`；`ldc >= max(1,n)`。
- `alpha`、`beta` 和 C 不可为空；k>0 时 A、B 不可为空。
- C 是复数对称矩阵而非 Hermitian 矩阵；非对角元素不共轭，对角元素虚部不强制为零。
- n=0 为合法 no-op；alpha=0 或 k=0 时仅按 beta 更新指定三角。

#### 编译与测试

```bash
bash build.sh --ops=csyr2k --soc=ascend950
bash build.sh --ops=csyr2k --soc=ascend950 --run --device=0
```

测试代码和官方 CSV 位于 `test/syr2k/csyr2k/`，详细步骤见该目录的 README。
