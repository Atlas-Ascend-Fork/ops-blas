# GemmGroupedBatched算子

## 算子概述

分组批量矩阵乘法（GEMM Grouped Batched），对多个分组内各批次矩阵独立执行 GEMM 运算，是 BLAS Level 3 核心算子之一。每个分组可拥有不同的 (m, n, k, transa, transb, alpha, beta, lda, ldb, ldc) 参数以及 groupSize 个批次。所有矩阵采用列主序（Column-Major）存储。

数学表达式：

```
C[i] = alpha[g] * op(A[i]) * op(B[i]) + beta[g] * C[i]
```

其中 `i` 属于分组 `g`，`op(A)` / `op(B)` 由 transa / transb 决定：`N` 为不转置，`T` 为转置。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSgemmGroupedBatched | 单精度浮点分组批量矩阵乘法 |

## 算子执行接口

### aclblasSgemmGroupedBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSgemmGroupedBatched(aclblasHandle_t handle, int groupCount, const aclblasOperation_t* transaArray, const aclblasOperation_t* transbArray, const int* mArray, const int* nArray, const int* kArray, const float* alphaArray, const float* const* Aarray, const int* ldaArray, const float* const* Barray, const int* ldbArray, const float* betaArray, float* const* Carray, const int* ldcArray, const int* groupSizeArray)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，Host 内存 |
| groupCount | 输入 | int | 分组数量，Host 内存 |
| transaArray | 输入 | const aclblasOperation_t* | 每组的 A 矩阵操作类型数组，当前支持 N/T，Host 内存 |
| transbArray | 输入 | const aclblasOperation_t* | 每组的 B 矩阵操作类型数组，当前支持 N/T，Host 内存 |
| mArray | 输入 | const int* | 每组的输出行数数组，Host 内存 |
| nArray | 输入 | const int* | 每组的输出列数数组，Host 内存 |
| kArray | 输入 | const int* | 每组的内积维度数组，Host 内存 |
| alphaArray | 输入 | const float* | 每组的标量乘数数组（float），Host 内存 |
| Aarray | 输入 | const float* const* | A 矩阵指针数组（Host 侧数组，元素为 Device 矩阵地址），长度 = sum(groupSize[g])，Host 内存 |
| ldaArray | 输入 | const int* | 每组的 A 矩阵 leading dimension 数组，Host 内存 |
| Barray | 输入 | const float* const* | B 矩阵指针数组（Host 侧数组，元素为 Device 矩阵地址），长度 = sum(groupSize[g])，Host 内存 |
| ldbArray | 输入 | const int* | 每组的 B 矩阵 leading dimension 数组，Host 内存 |
| betaArray | 输入 | const float* | 每组的标量乘数数组（类型与 alphaArray 一致），Host 内存 |
| Carray | 输入/输出 | float* const* | C 矩阵指针数组（Host 侧数组，元素为 Device 矩阵地址），长度 = sum(groupSize[g])，Host 内存 |
| ldcArray | 输入 | const int* | 每组的 C 矩阵 leading dimension 数组，Host 内存 |
| groupSizeArray | 输入 | const int* | 每组的批次数量数组，Host 内存 |

#### 约束说明

- groupCount >= 0
- m, n, k >= 0（每组）
- groupSize >= 0（每组）
- lda >= max(1, transa=N 时为 m, transa=T 时为 k)
- ldb >= max(1, transb=N 时为 k, transb=T 时为 n)
- ldc >= max(1, m)
- Aarray/Barray/Carray 为 Host 侧指针数组，元素指向 Device 上的矩阵数据；不可传入 Device 侧指针数组
- alpha=0 时跳过矩阵乘法，仅执行 C = beta * C
- k=0 时等效为 C = beta * C
- transa/transb 各支持 N/T，组合为 NN / NT / TN / TT 四种

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

```cpp
#include "acl/acl.h"
#include "cann_ops_blas.h"

// 初始化 ACL 和 handle
aclInit(nullptr);
aclrtSetDevice(0);
aclblasHandle_t handle = nullptr;
aclblasCreateHandle(&handle);

// 定义 2 个分组
int groupCount = 2;

// 分组 0: 4×4 GEMM, NN, 3 个 batch
// 分组 1: 8×8 GEMM, TN, 2 个 batch
aclblasOperation_t transaArray[] = {ACLBLAS_OP_N, ACLBLAS_OP_T};
aclblasOperation_t transbArray[] = {ACLBLAS_OP_N, ACLBLAS_OP_N};
int mArray[]      = {4, 8};
int nArray[]      = {4, 8};
int kArray[]      = {4, 4};
float alphaArray[] = {1.0f, 1.0f};
float betaArray[]  = {0.0f, 1.0f};
int ldaArray[]     = {4, 4};   // 分组0: trans=N, lda>=m=4; 分组1: trans=T, lda>=k=4
int ldbArray[]     = {4, 4};
int ldcArray[]     = {4, 8};
int groupSizeArray[] = {3, 2};

// A/B/C 指针数组（共 5 个 batch，Host 侧数组），元素指向 Device 上的列主序矩阵
const float* Aarray[5];
const float* Barray[5];
float*       Carray[5];

aclblasStatus_t ret = aclblasSgemmGroupedBatched(
    handle, groupCount,
    transaArray, transbArray, mArray, nArray, kArray,
    alphaArray, Aarray, ldaArray,
    Barray, ldbArray, betaArray,
    Carray, ldcArray, groupSizeArray);

// 清理
aclblasDestroyHandle(handle);
aclrtResetDevice(0);
aclFinalize();
```

