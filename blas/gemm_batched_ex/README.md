# GemmBatchedEx算子

## 算子概述

通用矩阵乘法扩展接口（GEMM Batched Ex），执行批量矩阵乘法运算，支持 A、B、C 矩阵使用独立数据类型。所有批次共享相同的维度 (m, n, k)、leading dimensions (lda, ldb, ldc) 和转置标志 (transa, transb)，每个批次的矩阵指针通过设备侧指针数组独立指定。

数学表达式：

```
C[i] = alpha * op(A[i]) * op(B[i]) + beta * C[i],  for i ∈ [0, batchCount - 1]
```

其中 `op(X)` 根据转置标志决定：`N` 为不转置，`T` 为转置，`C` 为共轭转置。所有矩阵以列主序（Column-Major）存储。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasGemmBatchedEx | 通用矩阵乘法批量扩展接口 |

## 算子执行接口

### aclblasGemmBatchedEx

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持（arch22，FLOAT32）
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持（arch22，FLOAT32）

#### Atlas A2/A3（arch22）支持范围

Atlas A2/A3 路径支持 Atype/Btype/Ctype 均为 ACL_FLOAT、computeType 为 ACLBLAS_COMPUTE_32F、algo 为 ACLBLAS_GEMM_DEFAULT。其余类型组合返回 NOT_SUPPORTED；下方 FP16/BF16/FP8 组合表属于 Ascend 950 路径。

alpha/beta 支持 Host 或 Device 指针；矩阵与矩阵指针数组均位于 Device。参数校验先于 no-op；m、n 或 batchCount 为零时不发射计算，k 或 alpha 为零时仅按 beta 缩放 C。实数共轭转置与转置等价。读回结果前同步 handle 绑定的 stream。

自验证平台为 Ascend910B3 / CANN9.1.0，覆盖 2609 条 CSV 精度测试及 200 条性能测试。按任务书§3.3预热5次、随后10次均值、GPU/NPU≥0.8判定，200条均通过。任务书其它章节的1.0阈值及采样口径冲突、GPU型号来源和 workspace 对比未验收单独披露，不宣称官方验收或其它款型已经实测。

测试方法与范围见 [arch22 测试说明](../../test/gemm_batched_ex/arch22/README.md)。

#### 函数原型

```cpp
aclblasStatus_t aclblasGemmBatchedEx(aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, int k, const void* alpha, const void* const Aarray[], aclDataType Atype, int lda, const void* const Barray[], aclDataType Btype, int ldb, const void* beta, void* const Carray[], aclDataType Ctype, int ldc, int batchCount, aclblasComputeType_t computeType, aclblasGemmAlgo_t algo)
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
| alpha | 输入 | const void* | 标量缩放因子，类型由 computeType 决定，Host 内存 |
| Aarray | 输入 | const void* const [] | 设备侧指针数组，每个元素指向 Atype 类型的矩阵 A[i]，Device 内存 |
| Atype | 输入 | aclDataType | 矩阵 A 的数据类型，Host 内存 |
| lda | 输入 | int | 矩阵 A[i] 的 leading dimension，Host 内存 |
| Barray | 输入 | const void* const [] | 设备侧指针数组，每个元素指向 Btype 类型的矩阵 B[i]，Device 内存 |
| Btype | 输入 | aclDataType | 矩阵 B 的数据类型，Host 内存 |
| ldb | 输入 | int | 矩阵 B[i] 的 leading dimension，Host 内存 |
| beta | 输入 | const void* | 标量缩放因子，类型由 computeType 决定，Host 内存 |
| Carray | 输入/输出 | void* const [] | 设备侧指针数组，每个元素指向 Ctype 类型的矩阵 C[i]，Device 内存 |
| Ctype | 输入 | aclDataType | 矩阵 C 的数据类型，Host 内存 |
| ldc | 输入 | int | 矩阵 C[i] 的 leading dimension，Host 内存 |
| batchCount | 输入 | int | 批次数量，Host 内存 |
| computeType | 输入 | aclblasComputeType_t | 计算精度类型，Host 内存 |
| algo | 输入 | aclblasGemmAlgo_t | GEMM 算法选择（当前仅支持 ACLBLAS_GEMM_DEFAULT），Host 内存 |

#### 约束说明

- m, n, k >= 0
- batchCount >= 0
- transa == N 时 lda >= max(1, m)；transa == T 或 C 时 lda >= max(1, k)
- transb == N 时 ldb >= max(1, k)；transb == T 或 C 时 ldb >= max(1, n)
- ldc >= max(1, m)
- algo 当前仅支持 ACLBLAS_GEMM_DEFAULT
- FP8 输入必须使用 ACLBLAS_COMPUTE_32F，否则返回 ACLBLAS_STATUS_NOT_SUPPORTED
- 不在支持表中的 Atype/Btype/Ctype/computeType 组合将返回 ACLBLAS_STATUS_NOT_SUPPORTED

支持的数据类型组合：

| 序号 | Atype | Btype | Ctype | computeType | alpha/beta 宿主类型 | 说明 |
|------|-------|-------|-------|-------------|-------------------|------|
| 1 | ACL_FLOAT16 | ACL_FLOAT16 | ACL_FLOAT16 | ACLBLAS_COMPUTE_16F | half (FP16) | FP16 纯精度计算 |
| 2 | ACL_FLOAT16 | ACL_FLOAT16 | ACL_FLOAT16 | ACLBLAS_COMPUTE_32F | float (FP32) | FP16 输入，FP32 累加 |
| 3 | ACL_BF16 | ACL_BF16 | ACL_BF16 | ACLBLAS_COMPUTE_32F | float (FP32) | BF16 输入，FP32 累加 |
| 4 | ACL_FLOAT8_E4M3FN | ACL_FLOAT8_E4M3FN | ACL_FLOAT16 | ACLBLAS_COMPUTE_32F | float (FP32) | FP8 E4M3 输入，FP16 输出 |
| 5 | ACL_FLOAT8_E5M2 | ACL_FLOAT8_E5M2 | ACL_FLOAT16 | ACLBLAS_COMPUTE_32F | float (FP32) | FP8 E5M2 输入，FP16 输出 |
| 6 | ACL_FLOAT8_E4M3FN | ACL_FLOAT8_E5M2 | ACL_FLOAT16 | ACLBLAS_COMPUTE_32F | float (FP32) | FP8 混合输入 (E4M3×E5M2) |
| 7 | ACL_FLOAT8_E5M2 | ACL_FLOAT8_E4M3FN | ACL_FLOAT16 | ACLBLAS_COMPUTE_32F | float (FP32) | FP8 混合输入 (E5M2×E4M3) |

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
size_t sizeA = (size_t)M * K * sizeof(half);
size_t sizeB = (size_t)K * N * sizeof(half);
size_t sizeC = (size_t)M * N * sizeof(half);

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

// 设置 alpha 和 beta（COMPUTE_32F 时为 float 类型）
float alpha = 1.0f;
float beta  = 0.0f;

// 调用 aclblasGemmBatchedEx
// 列主序：A 为 M×K（lda=M），B 为 K×N（ldb=K），C 为 M×N（ldc=M）
aclblasStatus_t status = aclblasGemmBatchedEx(
    handle,
    ACLBLAS_OP_N,                          // transa: 不转置
    ACLBLAS_OP_N,                          // transb: 不转置
    M, N, K,                               // 矩阵维度
    &alpha,                                // alpha (float*)
    (const void* const*)dAarray,           // Aarray (设备侧指针数组)
    ACL_FLOAT16,                           // Atype
    M,                                     // lda
    (const void* const*)dBarray,           // Barray (设备侧指针数组)
    ACL_FLOAT16,                           // Btype
    K,                                     // ldb
    &beta,                                 // beta (float*)
    (void* const*)dCarray,                 // Carray (设备侧指针数组)
    ACL_FLOAT16,                           // Ctype
    M,                                     // ldc
    batchCount,                            // 批次数量
    ACLBLAS_COMPUTE_32F,                   // FP32 累加
    ACLBLAS_GEMM_DEFAULT                   // 默认算法
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
aclblasStatus_t status = aclblasGemmBatchedEx(
    handle,
    ACLBLAS_OP_T,                          // transa: 转置 A
    ACLBLAS_OP_N,                          // transb: 不转置
    M, N, K,
    &alpha,
    (const void* const*)dAarray, ACL_FLOAT16, K,   // lda = K（转置时）
    (const void* const*)dBarray, ACL_FLOAT16, K,
    &beta,
    (void* const*)dCarray, ACL_FLOAT16, M,
    batchCount,
    ACLBLAS_COMPUTE_32F,
    ACLBLAS_GEMM_DEFAULT
);
```
