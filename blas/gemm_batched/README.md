# GemmBatched算子

## 算子概述

批量矩阵乘法接口（GEMM Batched），对一批具有相同维度和转置方式的矩阵执行矩阵-矩阵乘法。所有批次共享相同的维度 (m, n, k)、leading dimensions (lda, ldb, ldc) 和转置标志 (transa, transb)，每个批次的矩阵指针通过设备侧指针数组独立指定。

数学表达式：

```
C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i],  for i ∈ [0, batchCount - 1]
```

其中 `op(X)` 根据转置标志决定：`N` 为不转置，`T` 为转置，`C` 为共轭转置。所有矩阵以列主序（Column-Major）存储。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSgemmBatched | 单精度浮点（FP32）批量矩阵乘法 |
| aclblasCgemmBatched | 单精度复数（Complex FP32）批量矩阵乘法 |

## 算子执行接口

### aclblasSgemmBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

> Ascend 950PR/Ascend 950DT 上的 sgemmBatched 依赖 CANN asc-devkit >= 9.1（`ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR >= 1`），低于该版本时编译与运行将跳过此算子。

#### 函数原型

```cpp
aclblasStatus_t aclblasSgemmBatched(aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, const float* alpha, const float* const Aarray[], int lda, const float* const Barray[], int ldb, const float* beta, float* const Carray[], int ldc, int batchCount)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，Host 内存 |
| transa | 输入 | aclblasOperation_t | 矩阵 A 的转置操作（N / T / C），Host 内存 |
| transb | 输入 | aclblasOperation_t | 矩阵 B 的转置操作（N / T / C），Host 内存 |
| m | 输入 | int | op(A[i]) 的行数，也是 C[i] 的行数，Host 内存 |
| n | 输入 | int | op(B[i]) 的列数，也是 C[i] 的列数，Host 内存 |
| k | 输入 | int | op(A[i]) 的列数，也是 op(B[i]) 的行数，Host 内存 |
| alpha | 输入 | const float* | 标量缩放因子 alpha，Host 内存 |
| Aarray | 输入 | const float* const [] | 设备侧指针数组，每个元素指向 float 类型的矩阵 A[i]，Device 内存 |
| lda | 输入 | int | 矩阵 A[i] 的 leading dimension，Host 内存 |
| Barray | 输入 | const float* const [] | 设备侧指针数组，每个元素指向 float 类型的矩阵 B[i]，Device 内存 |
| ldb | 输入 | int | 矩阵 B[i] 的 leading dimension，Host 内存 |
| beta | 输入 | const float* | 标量缩放因子 beta，Host 内存 |
| Carray | 输入/输出 | float* const [] | 设备侧指针数组，每个元素指向 float 类型的矩阵 C[i]，Device 内存 |
| ldc | 输入 | int | 矩阵 C[i] 的 leading dimension，Host 内存 |
| batchCount | 输入 | int | 批次数量，Host 内存 |

#### 约束说明

- m, n, k >= 0
- batchCount >= 0
- transa == N 时 lda >= max(1, m)；transa == T 或 C 时 lda >= max(1, k)
- transb == N 时 ldb >= max(1, k)；transb == T 或 C 时 ldb >= max(1, n)
- ldc >= max(1, m)
- 各 C[i] 矩阵之间不得重叠

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

以下示例演示列主序（Column-Major）下的批量矩阵乘法调用。计算 `C[i] = 1.0 * A[i] * B[i] + 0.0 * C[i]`，其中 A 为 M×K，B 为 K×N，C 为 M×N，共 batchCount 个批次。

```cpp
#include "acl/acl.h"
#include "cann_ops_blas.h"

// 矩阵维度（列主序）
int M = 256;
int N = 256;
int K = 128;
int batchCount = 4;

// 初始化 ACL 和 BLAS handle
aclInit(nullptr);
int32_t deviceId = 0;
aclrtSetDevice(deviceId);
aclblasHandle_t handle;
aclblasCreate(&handle);

// 在设备侧分配矩阵内存（列主序存储）
size_t sizeA = (size_t)M * K * sizeof(float);
size_t sizeB = (size_t)K * N * sizeof(float);
size_t sizeC = (size_t)M * N * sizeof(float);

void* dA[batchCount];
void* dB[batchCount];
void* dC[batchCount];
for (int i = 0; i < batchCount; i++) {
    aclrtMalloc(&dA[i], sizeA, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dB[i], sizeB, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dC[i], sizeC, ACL_MEM_MALLOC_HUGE_FIRST);
    // ... 填充数据 ...
}

// 构造设备侧指针数组（在设备内存中）
void** dAarray;
void** dBarray;
void** dCarray;
aclrtMalloc((void**)&dAarray, batchCount * sizeof(void*), ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc((void**)&dBarray, batchCount * sizeof(void*), ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc((void**)&dCarray, batchCount * sizeof(void*), ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMemcpy(dAarray, batchCount * sizeof(void*), dA, batchCount * sizeof(void*), ACL_MEMCPY_HOST_TO_DEVICE);
aclrtMemcpy(dBarray, batchCount * sizeof(void*), dB, batchCount * sizeof(void*), ACL_MEMCPY_HOST_TO_DEVICE);
aclrtMemcpy(dCarray, batchCount * sizeof(void*), dC, batchCount * sizeof(void*), ACL_MEMCPY_HOST_TO_DEVICE);

// 设置 alpha 和 beta
float alpha = 1.0f;
float beta  = 0.0f;

// 调用 aclblasSgemmBatched
// 列主序：A 为 M×K（lda=M），B 为 K×N（ldb=K），C 为 M×N（ldc=M）
aclblasStatus_t status = aclblasSgemmBatched(
    handle,
    ACLBLAS_OP_N,                          // transa: 不转置
    ACLBLAS_OP_N,                          // transb: 不转置
    M, N, K,                               // 矩阵维度
    &alpha,                                // alpha
    (const float* const*)dAarray,           // Aarray (设备侧指针数组)
    M,                                     // lda
    (const float* const*)dBarray,           // Barray (设备侧指针数组)
    K,                                     // ldb
    &beta,                                 // beta
    (float* const*)dCarray,                 // Carray (设备侧指针数组)
    M,                                     // ldc
    batchCount                             // 批次数量
);

// 同步等待完成
aclrtSynchronizeStream(nullptr);

// 清理资源
aclblasDestroy(handle);
for (int i = 0; i < batchCount; i++) {
    aclrtFree(dA[i]);
    aclrtFree(dB[i]);
    aclrtFree(dC[i]);
}
aclrtFree(dAarray);
aclrtFree(dBarray);
aclrtFree(dCarray);
aclrtResetDevice(deviceId);
aclFinalize();
```

转置调用示例：当需要对 A 进行转置时（`transa = ACLBLAS_OP_T`），A 的物理存储为 K×M（列主序），lda >= max(1, K)：

```cpp
// A^T * B: transa=T, A 物理存储为 K×M (lda=K)
aclblasStatus_t status = aclblasSgemmBatched(
    handle,
    ACLBLAS_OP_T,                          // transa: 转置 A
    ACLBLAS_OP_N,                          // transb: 不转置
    M, N, K,
    &alpha,
    (const float* const*)dAarray, K,        // lda = K（转置时）
    (const float* const*)dBarray, K,
    &beta,
    (float* const*)dCarray, M,
    batchCount
);
```

### aclblasCgemmBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

> Ascend 950PR/Ascend 950DT 上的 cgemmBatched 依赖 CANN asc-devkit >= 9.1（`ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR >= 1`），低于该版本时编译与运行将跳过此算子。

#### 函数原型

```cpp
aclblasStatus_t aclblasCgemmBatched(aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, const aclblasComplex* alpha, const aclblasComplex* const Aarray[], int lda, const aclblasComplex* const Barray[], int ldb, const aclblasComplex* beta, aclblasComplex* const Carray[], int ldc, int batchCount)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，Host 内存 |
| transa | 输入 | aclblasOperation_t | 矩阵 A 的转置操作（N / T / C），Host 内存 |
| transb | 输入 | aclblasOperation_t | 矩阵 B 的转置操作（N / T / C），Host 内存 |
| m | 输入 | int | op(A[i]) 的行数，也是 C[i] 的行数，Host 内存 |
| n | 输入 | int | op(B[i]) 的列数，也是 C[i] 的列数，Host 内存 |
| k | 输入 | int | op(A[i]) 的列数，也是 op(B[i]) 的行数，Host 内存 |
| alpha | 输入 | const aclblasComplex* | 复数标量缩放因子 alpha，Host 内存 |
| Aarray | 输入 | const aclblasComplex* const [] | 设备侧指针数组，每个元素指向 aclblasComplex 类型的矩阵 A[i]，Device 内存 |
| lda | 输入 | int | 矩阵 A[i] 的 leading dimension，Host 内存 |
| Barray | 输入 | const aclblasComplex* const [] | 设备侧指针数组，每个元素指向 aclblasComplex 类型的矩阵 B[i]，Device 内存 |
| ldb | 输入 | int | 矩阵 B[i] 的 leading dimension，Host 内存 |
| beta | 输入 | const aclblasComplex* | 复数标量缩放因子 beta，Host 内存 |
| Carray | 输入/输出 | aclblasComplex* const [] | 设备侧指针数组，每个元素指向 aclblasComplex 类型的矩阵 C[i]，Device 内存 |
| ldc | 输入 | int | 矩阵 C[i] 的 leading dimension，Host 内存 |
| batchCount | 输入 | int | 批次数量，Host 内存 |

#### 约束说明

- m, n, k >= 0
- batchCount >= 0
- transa == N 时 lda >= max(1, m)；transa == T 或 C 时 lda >= max(1, k)
- transb == N 时 ldb >= max(1, k)；transb == T 或 C 时 ldb >= max(1, n)
- ldc >= max(1, m)
- 各 C[i] 矩阵之间不得重叠
- 复数矩阵以实部/虚部交错存储（interleaved complex），即 A[i][0].real, A[i][0].imag, A[i][1].real, A[i][1].imag, ...

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

alpha == 0（即实部与虚部均为 0，且 alpha 非空）时跳过矩阵乘，逐 batch 执行 C_i = beta * C_i，结果应位精确。

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

以下示例演示列主序（Column-Major）下的复数批量矩阵乘法调用。计算 `C[i] = (1.0+0.0j) * A[i] * B[i] + (0.0+0.0j) * C[i]`，共 batchCount 个批次。

```cpp
#include "acl/acl.h"
#include "cann_ops_blas.h"

// 矩阵维度（列主序）
int M = 256;
int N = 256;
int K = 128;
int batchCount = 4;

// 初始化 ACL 和 BLAS handle
aclInit(nullptr);
int32_t deviceId = 0;
aclrtSetDevice(deviceId);
aclblasHandle_t handle;
aclblasCreate(&handle);

// 在设备侧分配复数矩阵内存（列主序存储，实部虚部交错）
size_t sizeA = (size_t)M * K * sizeof(aclblasComplex);
size_t sizeB = (size_t)K * N * sizeof(aclblasComplex);
size_t sizeC = (size_t)M * N * sizeof(aclblasComplex);

void* dA[batchCount];
void* dB[batchCount];
void* dC[batchCount];
for (int i = 0; i < batchCount; i++) {
    aclrtMalloc(&dA[i], sizeA, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dB[i], sizeB, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&dC[i], sizeC, ACL_MEM_MALLOC_HUGE_FIRST);
    // ... 填充复数数据 ...
}

// 构造设备侧指针数组（在设备内存中）
void** dAarray;
void** dBarray;
void** dCarray;
aclrtMalloc((void**)&dAarray, batchCount * sizeof(void*), ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc((void**)&dBarray, batchCount * sizeof(void*), ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc((void**)&dCarray, batchCount * sizeof(void*), ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMemcpy(dAarray, batchCount * sizeof(void*), dA, batchCount * sizeof(void*), ACL_MEMCPY_HOST_TO_DEVICE);
aclrtMemcpy(dBarray, batchCount * sizeof(void*), dB, batchCount * sizeof(void*), ACL_MEMCPY_HOST_TO_DEVICE);
aclrtMemcpy(dCarray, batchCount * sizeof(void*), dC, batchCount * sizeof(void*), ACL_MEMCPY_HOST_TO_DEVICE);

// 设置复数 alpha 和 beta
aclblasComplex alpha = {1.0f, 0.0f};
aclblasComplex beta  = {0.0f, 0.0f};

// 调用 aclblasCgemmBatched
aclblasStatus_t status = aclblasCgemmBatched(
    handle,
    ACLBLAS_OP_N,                                        // transa: 不转置
    ACLBLAS_OP_N,                                        // transb: 不转置
    M, N, K,                                             // 矩阵维度
    &alpha,                                              // alpha (复数)
    (const aclblasComplex* const*)dAarray,                // Aarray (设备侧指针数组)
    M,                                                   // lda
    (const aclblasComplex* const*)dBarray,                // Barray (设备侧指针数组)
    K,                                                   // ldb
    &beta,                                               // beta (复数)
    (aclblasComplex* const*)dCarray,                      // Carray (设备侧指针数组)
    M,                                                   // ldc
    batchCount                                           // 批次数量
);

// 同步等待完成
aclrtSynchronizeStream(nullptr);

// 清理资源
aclblasDestroy(handle);
for (int i = 0; i < batchCount; i++) {
    aclrtFree(dA[i]);
    aclrtFree(dB[i]);
    aclrtFree(dC[i]);
}
aclrtFree(dAarray);
aclrtFree(dBarray);
aclrtFree(dCarray);
aclrtResetDevice(deviceId);
aclFinalize();
```
