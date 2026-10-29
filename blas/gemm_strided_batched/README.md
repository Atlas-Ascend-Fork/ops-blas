# GemmStridedBatched算子

## 算子概述

GemmStridedBatched（Single-precision Strided Batched General Matrix Multiplication）算子实现了单精度浮点批量矩阵乘法，对一批规格一致的矩阵三元组 (A_i, B_i, C_i) 执行 GEMM 运算并通过固定 stride 定位每个 batch 的子矩阵，核心运算为 C_i = alpha * op(A_i) * op(B_i) + beta * C_i。

数学表达式：

```
C_i = alpha * op(A_i) * op(B_i) + beta * C_i,   i = 0, 1, ..., batchCount-1

其中：
  A_i = A + i * strideA   （A 是 batch 0 的首地址，strideA 是相邻 batch 间的元素偏移量）
  B_i = B + i * strideB
  C_i = C + i * strideC
  strideA/strideB = 0 时表示输入矩阵跨 batch 广播；C 的各 batch 不得重叠

  op(X) = X       当 trans = ACLBLAS_OP_N
  op(X) = X^T     当 trans = ACLBLAS_OP_T
  op(X) = X^H     当 trans = ACLBLAS_OP_C （实数 FP32 下 X^H == X^T）
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| [aclblasSgemmStridedBatched](#aclblassgemmstridedbatched) | 单精度浮点批量矩阵乘法（FP32） |
| [aclblasCgemmStridedBatched](#aclblascgemmstridedbatched) | 单精度复数批量矩阵乘法（COMPLEX64），支持转置与共轭转置 |

## 算子执行接口

### aclblasSgemmStridedBatched

#### 产品支持情况

| 产品 | 实现支持 | 验证状态 |
|---|---|---|
| Ascend 950PR / Ascend 950DT | 支持（arch35） | 沿用原实现 |
| Atlas A3 训练/推理系列（含 Atlas 800I A3） | 支持（arch22） | Ascend910_9382：1312 项测试、200 项性能全部通过 |
| Atlas A2 训练/推理系列（含 Atlas 800I A2） | 支持（arch22） | Ascend910B3：1312 项测试、200 项性能全部通过 |

> Ascend 950PR/Ascend 950DT 上的 aclblasSgemmStridedBatched 依赖 CANN asc-devkit >= 9.1（`ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR >= 1`），低于该版本时编译与运行将跳过此算子。

#### 函数原型

```cpp
aclblasStatus_t aclblasSgemmStridedBatched(aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k, const float* alpha, const float* A, int lda, int64_t strideA, const float* B, int ldb, int64_t strideB, const float* beta, float* C, int ldc, int64_t strideC, int batchCount)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| transA | 输入 | aclblasOperation_t | 矩阵 A 的操作类型：ACLBLAS_OP_N（不转置）、ACLBLAS_OP_T（转置）、ACLBLAS_OP_C（共轭转置，实数 FP32 下等价于 T），Host 内存 |
| transB | 输入 | aclblasOperation_t | 矩阵 B 的操作类型（取值同 transA），Host 内存 |
| m | 输入 | int | op(A_i) 与 C_i 的行数，m >= 0，Host 内存 |
| n | 输入 | int | op(B_i) 与 C_i 的列数，n >= 0，Host 内存 |
| k | 输入 | int | op(A_i) 的列数、op(B_i) 的行数，k >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | 指向标量 alpha 的 Host 指针，全批共用，不可为 nullptr，Host 内存 |
| A | 输入 | const float*（FP32） | batch 0 矩阵 A 的 Device 内存首地址，列主序存储，Device 内存 |
| lda | 输入 | int | 矩阵 A 的主维度，transA=N 时 lda >= max(1, m)，transA=T/C 时 lda >= max(1, k)，Host 内存 |
| strideA | 输入 | int64_t | 相邻两 batch 的 A 起始元素偏移（以元素个数计，非字节），允许为 0（表示 A 在所有 batch 间广播），Host 内存 |
| B | 输入 | const float*（FP32） | batch 0 矩阵 B 的 Device 内存首地址，列主序存储，Device 内存 |
| ldb | 输入 | int | 矩阵 B 的主维度，transB=N 时 ldb >= max(1, k)，transB=T/C 时 ldb >= max(1, n)，Host 内存 |
| strideB | 输入 | int64_t | 相邻两 batch 的 B 起始元素偏移（以元素个数计，非字节），允许为 0（表示 B 在所有 batch 间广播），Host 内存 |
| beta | 输入 | const float*（FP32） | 指向标量 beta 的 Host 指针，全批共用，不可为 nullptr，Host 内存 |
| C | 输入/输出 | float*（FP32） | batch 0 矩阵 C 的 Device 内存首地址，列主序存储，就地读写，Device 内存 |
| ldc | 输入 | int | 矩阵 C 的主维度，ldc >= max(1, m)，Host 内存 |
| strideC | 输入 | int64_t | 相邻两 batch 的 C 起始元素偏移（以元素个数计，非字节），Host 内存 |
| batchCount | 输入 | int | batch 数量，batchCount >= 0，Host 内存 |

#### 约束说明

- handle 不可为 nullptr，否则返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
- transA、transB 必须属于 {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}，非法取值返回 `ACLBLAS_STATUS_INVALID_ENUM`
- m >= 0、n >= 0、k >= 0、batchCount >= 0
- alpha、beta 不可为 nullptr
- transA=N 时 lda >= max(1, m)；transA=T/C 时 lda >= max(1, k)
- transB=N 时 ldb >= max(1, k)；transB=T/C 时 ldb >= max(1, n)
- ldc >= max(1, m)
- k > 0 时（且 m>0、n>0、batchCount>0），A、B 不可为 nullptr
- m>0、n>0、batchCount>0 时，arch22 实现要求 C 不可为 nullptr（beta=0 仅免去读取旧 C，输出仍会写入 C）；arch35 实现沿用上游校验，beta=0 时允许 C 为 nullptr
- 所有矩阵按列主序（Column-Major）存储，lda/ldb/ldc 语义遵循 NETLIB/CBLAS BLAS 标准
- strideA/strideB 取值无上界强制校验，允许为 0（表示对应矩阵在所有 batch 间广播复用）；调用方须保证各 batch 子矩阵在所声明的 shape/ld/stride 下不越界访问显存
- strideC == 0 且 batchCount > 1 时各 batch 写入同一 C 区域，存在写覆盖/竞争，语义未定义，由调用方负责规避
- m == 0 或 n == 0 或 batchCount == 0 时无计算，直接返回 `ACLBLAS_STATUS_SUCCESS`
- k == 0 或 alpha == 0（k>0）时跳过矩阵乘，逐 batch 执行 C_i = beta * C_i（beta==0 时置零，beta==1 时不变）

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

k == 0 或 alpha == 0（k>0）时跳过矩阵乘，逐 batch 执行 C_i = beta * C_i，结果应位精确。

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

    struct BlasHandleDeleter {
        void operator()(aclblasHandle_t h) const { aclblasDestroy(h); }
    };

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

int aclblasSgemmStridedBatchedTest(AclContext& ctx)
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

    // 2. 准备 Host 数据：batchCount=2, m=n=k=4, transA/transB=N, alpha=1.0, beta=0.0
    constexpr int m = 4;
    constexpr int n = 4;
    constexpr int k = 4;
    constexpr int lda = 4;
    constexpr int ldb = 4;
    constexpr int ldc = 4;
    constexpr int64_t strideA = static_cast<int64_t>(m) * k;
    constexpr int64_t strideB = static_cast<int64_t>(k) * n;
    constexpr int64_t strideC = static_cast<int64_t>(m) * n;
    constexpr int batchCount = 2;
    float alpha = 1.0f;
    float beta = 0.0f;

    std::vector<float> hA(static_cast<size_t>(strideA) * batchCount, 0.0f);
    std::vector<float> hB(static_cast<size_t>(strideB) * batchCount, 0.0f);
    std::vector<float> hC(static_cast<size_t>(strideC) * batchCount, 0.0f);
    for (int b = 0; b < batchCount; ++b) {
        for (int i = 0; i < m * k; ++i) {
            hA[static_cast<size_t>(b) * strideA + i] = static_cast<float>(i + b + 1);
        }
        for (int i = 0; i < k * n; ++i) {
            hB[static_cast<size_t>(b) * strideB + i] = static_cast<float>(i + b + 1);
        }
    }

    // 3. 申请 Device 内存并拷贝数据
    size_t aBytes = hA.size() * sizeof(float);
    size_t bBytes = hB.size() * sizeof(float);
    size_t cBytes = hC.size() * sizeof(float);

    void* rawA = nullptr;
    auto aclRet = aclrtMalloc(&rawA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> aDevicePtr(rawA, aclrtFree);

    void* rawB = nullptr;
    aclRet = aclrtMalloc(&rawB, bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for B failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> bDevicePtr(rawB, aclrtFree);

    void* rawC = nullptr;
    aclRet = aclrtMalloc(&rawC, cBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for C failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> cDevicePtr(rawC, aclrtFree);

    aclRet = aclrtMemcpy(aDevicePtr.get(), aBytes, hA.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(bDevicePtr.get(), bBytes, hB.data(), bBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for B failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(cDevicePtr.get(), cBytes, hC.data(), cBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for C failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasSgemmStridedBatched（alpha/beta 为 Host 指针）
    blasRet = aclblasSgemmStridedBatched(
        static_cast<aclblasHandle_t>(handlePtr.get()), ACLBLAS_OP_N, ACLBLAS_OP_N, m, n, k, &alpha,
        static_cast<float*>(aDevicePtr.get()), lda, strideA, static_cast<float*>(bDevicePtr.get()), ldb, strideB,
        &beta, static_cast<float*>(cDevicePtr.get()), ldc, strideC, batchCount);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS,
              LOG_PRINT("aclblasSgemmStridedBatched failed. ERROR: %d\n", blasRet); return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    aclRet = aclrtMemcpy(hC.data(), cBytes, cDevicePtr.get(), cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", aclRet);
              return aclRet);
    for (int b = 0; b < batchCount; ++b) {
        LOG_PRINT("batch %d result:\n", b);
        for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
                size_t idx = static_cast<size_t>(b) * strideC + static_cast<size_t>(col) * ldc + row;
                LOG_PRINT("%.1f ", hC[idx]);
            }
            LOG_PRINT("\n");
        }
    }

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSgemmStridedBatchedTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSgemmStridedBatchedTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```


### aclblasCgemmStridedBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持（arch35）
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持（arch22）
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持（arch22）

> Ascend 950PR/Ascend 950DT 上的 aclblasCgemmStridedBatched 依赖 CANN asc-devkit >= 9.1（`ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR >= 1`），低于该版本时编译与运行将跳过此算子。

#### 函数原型

```cpp
aclblasStatus_t aclblasCgemmStridedBatched(aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k, const aclblasComplex* alpha, const aclblasComplex* A, int lda, int64_t strideA, const aclblasComplex* B, int ldb, int64_t strideB, const aclblasComplex* beta, aclblasComplex* C, int ldc, int64_t strideC, int batchCount)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| transA | 输入 | aclblasOperation_t | 矩阵 A 的操作类型：ACLBLAS_OP_N（不转置）、ACLBLAS_OP_T（转置）、ACLBLAS_OP_C（共轭转置），Host 内存 |
| transB | 输入 | aclblasOperation_t | 矩阵 B 的操作类型（取值同 transA），Host 内存 |
| m | 输入 | int | op(A_i) 与 C_i 的行数，m >= 0，Host 内存 |
| n | 输入 | int | op(B_i) 与 C_i 的列数，n >= 0，Host 内存 |
| k | 输入 | int | op(A_i) 的列数、op(B_i) 的行数，k >= 0，Host 内存 |
| alpha | 输入 | const aclblasComplex* | 指向标量 alpha 的 Host 指针，全批共用，不可为 nullptr，Host 内存 |
| A | 输入 | const aclblasComplex* | batch 0 矩阵 A 的 Device 内存首地址，Complex64，列主序存储，Device 内存 |
| lda | 输入 | int | 矩阵 A 的主维度，transA=N 时 lda >= max(1, m)，transA=T/C 时 lda >= max(1, k)，Host 内存 |
| strideA | 输入 | int64_t | 相邻两 batch 的 A 起始元素偏移（以复数元素个数计，非字节），允许为 0（表示 A 在所有 batch 间广播），Host 内存 |
| B | 输入 | const aclblasComplex* | batch 0 矩阵 B 的 Device 内存首地址，Complex64，列主序存储，Device 内存 |
| ldb | 输入 | int | 矩阵 B 的主维度，transB=N 时 ldb >= max(1, k)，transB=T/C 时 ldb >= max(1, n)，Host 内存 |
| strideB | 输入 | int64_t | 相邻两 batch 的 B 起始元素偏移（以复数元素个数计，非字节），允许为 0（表示 B 在所有 batch 间广播），Host 内存 |
| beta | 输入 | const aclblasComplex* | 指向标量 beta 的 Host 指针，全批共用，不可为 nullptr，Host 内存 |
| C | 输入/输出 | aclblasComplex* | batch 0 矩阵 C 的 Device 内存首地址，Complex64，列主序存储，就地读写，Device 内存 |
| ldc | 输入 | int | 矩阵 C 的主维度，ldc >= max(1, m)，Host 内存 |
| strideC | 输入 | int64_t | 相邻两 batch 的 C 起始元素偏移（以复数元素个数计，非字节），Host 内存 |
| batchCount | 输入 | int | batch 数量，batchCount >= 0，Host 内存 |

#### Ascend 950PR / 950DT（arch35）约束说明

- handle 不可为 nullptr，否则返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
- transA、transB 必须属于 {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C}
- m >= 0、n >= 0、k >= 0、batchCount >= 0
- alpha、beta 不可为 nullptr
- transA=N 时 lda >= max(1, m)；transA=T/C 时 lda >= max(1, k)
- transB=N 时 ldb >= max(1, k)；transB=T/C 时 ldb >= max(1, n)
- ldc >= max(1, m)
- 非空计算且 k > 0 时 A、B 不可为 nullptr
- 非空计算时 C 不可为 nullptr
- 所有矩阵按列主序（Column-Major）存储，lda/ldb/ldc 语义遵循 NETLIB/CBLAS BLAS 标准
- strideA、strideB 允许为 0，表示对应矩阵在所有 batch 间广播复用；调用方须保证各 batch 子矩阵在所声明的 shape、leading dimension 和 stride 下不越界访问显存
- strideC == 0 且 batchCount > 1，或其他输出批次存储区域重叠时，执行语义未定义
- m == 0、n == 0 或 batchCount == 0 时无计算，直接返回 `ACLBLAS_STATUS_SUCCESS`
- k == 0 或 alpha == (0, 0) 时跳过矩阵乘，逐 batch 执行 C_i = beta * C_i
- beta == (0, 0) 时不读取原 C，调用方无需初始化逻辑输出区域
- kernel 在句柄 stream 上异步执行，读取结果前需同步 stream

#### Atlas A2/A3（arch22）约束说明

各矩阵按列主序存储，stride 以复数元素计。第 `i` 批使用 `A + i*strideA`、
`B + i*strideB` 和 `C + i*strideC`。`N`、`T`、`C` 分别表示不转置、转置、
共轭转置；每批执行 `C = alpha*op(A)*op(B) + beta*C`。

`strideA`、`strideB` 可为 0，以便复用输入矩阵。`strideC=0` 且 `batchCount>1`
时各批按顺序更新同一 C（beta 非零时读取前一批结果）。调用方必须保证矩阵及步长对应的显存范围有效，
且 `C` 不与 `A` 或 `B` 重叠。

`m`、`n`、`k` 或 `batchCount` 为 0 时合法并直接返回成功；`alpha=0` 时
不读取 `A`、`B`；`beta=0` 时不读取原 `C`。`alpha`、`beta` 始终需要有效的
Host 指针，非空计算需要有效的 Device `C` 指针。无效枚举、负维度及不满足
`lda >= max(1, transa==N ? m : k)`、`ldb >= max(1, transb==N ? k : n)`、
`ldc >= max(1,m)` 的参数返回 `ACLBLAS_STATUS_INVALID_VALUE`。

接口异步提交到通过 `aclblasSetStream` 绑定的 stream；读取输出前应同步该 stream。
大矩阵使用 handle 的工作空间，库管理的工作空间按需扩容；若调用方提供的
工作空间不足，则返回 `ACLBLAS_STATUS_ALLOC_FAILED`。

#### Ascend 950PR / 950DT（arch35）精度验证

实部和虚部分别采用 `MIXED_TOLERANCE` 混合容差策略：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
|----------|------|------|----------------------|-------------------|
| COMPLEX64 | 2^-10 (9.77e-4) | 2^-16 (1.53e-5) | 0.99 | 1e-2 或 32·ULP |

alpha == (0, 0) 时使用 `EXACT` 位精确校验。

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

下面片段假设已经初始化 ACL、创建并绑定 stream，且 A、B、C 的 Device 存储已经分配：

```cpp
aclblasComplex alpha{1.0f, 0.0f};
aclblasComplex beta{0.0f, 0.0f};
const int m = 256;
const int n = 256;
const int k = 256;
const int batchCount = 32;
aclblasStatus_t status = aclblasCgemmStridedBatched(
    handle, ACLBLAS_OP_N, ACLBLAS_OP_N, m, n, k,
    &alpha, deviceA, m, static_cast<int64_t>(m) * k,
    deviceB, k, static_cast<int64_t>(k) * n,
    &beta, deviceC, m, static_cast<int64_t>(m) * n, batchCount);
```

## Atlas A2/A3（arch22）实现与自测

实现位于 `arch22/`，使用 CANN 9.1.0 Ascend C kernel 直调，共用上述公开接口。
所有 kernel 提交至 handle 绑定的 stream；Host 不读取矩阵内容，也不在正常调用中同步设备。

- 乘加规模不超过 4096 且 batchCount 小于 32 的小矩阵、以及仅缩放场景使用 AIV；`beta=0` 不读取旧 C，`alpha=0`/`k=0` 不计算矩阵乘。
- 一般矩阵使用 FP32 Cube Matmul，关闭 HF32；通过 `C^T = op(B)^T op(A)^T` 处理列主序。
- NN、紧密排布、alpha=1/beta=0 的 8³ 批量矩阵使用 AIV Gather/FMA；16³/32³ 大批量使用静态 Batch Matmul，小批量使用独立 Cube 调度。
- 8³ 在 batchCount 至少为 512 时采用 32 批分组，以减少重复 Gather 初始化；较小批量保持原分组。极大值回退保留完整 48 位乘积并执行单次 FP32 舍入，避免相消时超出 ULP 误差界。
- Batch Matmul 的 AIV 分支并行检测极大输入，并在对应 Cube 完成后执行缩放回退，保持 Netlib 的溢出语义。
- 输出按 128×128 分块，并将 batch 纳入任务编号；K 超过 4096 时分段累加。
- `alpha=1,beta=0` 直接写入 C；其余场景按 workspace 容量分组处理 batch，再由 AIV 合并 alpha/beta。单个矩阵放不下时按列切条。
- workspace 至少容纳一列按 8 元素对齐的 FP32 临时输出。不满足时返回 `ACLBLAS_STATUS_ALLOC_FAILED`；调用方可使用 `aclblasSetWorkspace` 提供空间。
- 正尺寸输出必须提供可写的 Device C 指针。矩阵 padding 和 batch 间隙保持不变；负 stride 返回非法参数。

在安装了 Netlib BLAS、LAPACK、GTest 的 Ascend Linux 环境，从仓库根目录执行：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
bash build.sh --soc=ascend910b3 --ops=gemm_strided_batched
# 默认冒烟：前 50 条 CSV 用例及 3 项独立接口检查
build/test/gemm_strided_batched/gemm_strided_batched_test
# 完整验收：显式启用全量 CSV 及补充回归，再按精度/性能筛选
GSB_FULL_TEST=1 build/test/gemm_strided_batched/gemm_strided_batched_test --gtest_filter='*-*TC_PF*'
GSB_FULL_TEST=1 build/test/gemm_strided_batched/gemm_strided_batched_test --gtest_filter='*TC_PF*'
```

A3 使用与实际设备匹配的 `--soc=ascend910_9382` 等参数重新编译并复验。测试支持通过
`ASCEND_DEVICE_ID` 环境变量覆盖设备编号。

`test/gemm_strided_batched/arch22/` 保留全量 `gemm_strided_batched_test.csv`（1000 精度、200 性能）及额外回归 CSV。
默认仅加载 `gemm_strided_batched_smoke.csv`，其内容为全量 CSV 的表头及前 50 条用例，供流水线冒烟使用；更新全量 CSV 后须同步这 50 条记录。
设置 `GSB_FULL_TEST=1` 时加载全量任务 CSV 和 `gemm_strided_batched_regression.csv`。两种模式均保留独立接口检查；冒烟通过不能替代全量精度和性能验收。
额外回归覆盖跨 K 分段、NaN/Inf 短路、非对齐 padding/广播，小矩阵批尾、极大值、NaN/Inf，以及均匀、正态分布各 32 例。
正态回归用例由固定种子的 `std::normal_distribution` 生成 A/B/C，均值位于 [-5,5]，标准差位于 [0.1,2]。
全量模式共 1313 项测试，包含 110 项额外回归与 3 项独立接口检查（空 handle、负 stride、golden 非法转置枚举）。其中 7 项额外回归覆盖 32 批分组边界、尾批、循环及特殊值。
大型转置 B 会先在 CPU 参考阶段整理为连续布局，再调用同一 Netlib SGEMM；已与原调用进行 90 项逐位一致性检查。
每例输出 matched_ratio 和 max_abs_error，并逐元素检查输出区之外的数据未被修改。

性能用例先预热 20 次，再用 stream event 包围 200 次调用，输出 `[PERF] ... avg_ms=... samples=200 warmup=20`。
该计时包含整次接口提交的 kernel 序列及提交间隙，不包含 Host 数据准备、拷贝和 CPU golden。
性能重放要求用例为 `alpha=1,beta=0`；必须同时检查 GTest 成功状态。
A3 测量不能替代任务书要求的 A2/910B3 性能验收。
