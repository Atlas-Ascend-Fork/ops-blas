# Trmm算子

## 算子概述

Trmm（Single-precision Triangular Matrix Multiplication）算子实现了单精度浮点三角矩阵与普通矩阵的乘法运算，结果乘以标量 alpha 后写入输出矩阵 C。

数学表达式：

```
LEFT 模式：C := alpha * op(A) * B
RIGHT 模式：C := alpha * B * op(A)
```

其中 A 为三角矩阵（仅存储上三角或下三角），B 为输入矩阵，C 为输出矩阵，op(A) 为 A、A^T 或 A^H。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasStrmm | 单精度浮点三角矩阵乘法 |

## 算子执行接口

### aclblasStrmm

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

> Ascend 950PR/Ascend 950DT 上的 strmm 依赖 CANN asc-devkit >= 9.1（`ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR >= 1`），低于该版本时编译与运行将跳过此算子。

#### 函数原型

```cpp
aclblasStatus_t aclblasStrmm(aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n, const float *alpha, const float *A, int lda, const float *B, int ldb, float *C, int ldc)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| side | 输入 | aclblasSideMode_t | A 矩阵位置：ACLBLAS_SIDE_LEFT（左侧）或 ACLBLAS_SIDE_RIGHT（右侧），Host 内存 |
| uplo | 输入 | aclblasFillMode_t | A 矩阵存储模式：ACLBLAS_UPPER（上三角）或 ACLBLAS_LOWER（下三角），Host 内存 |
| trans | 输入 | aclblasOperation_t | A 矩阵操作类型：A2/A3 支持 ACLBLAS_OP_N（不转置）和 ACLBLAS_OP_T（转置）；Ascend 950 还支持 ACLBLAS_OP_C，Host 内存 |
| diag | 输入 | aclblasDiagType_t | 对角线类型：ACLBLAS_UNIT（单位三角，对角线为 1）或 ACLBLAS_NON_UNIT（非单位三角），Host 内存 |
| m | 输入 | int | 矩阵 B/C 的行数，m >= 0，Host 内存 |
| n | 输入 | int | 矩阵 B/C 的列数，n >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量 alpha，不可为 nullptr，支持 Host 或 Device 内存 |
| A | 输入 | const float*（FP32） | 三角矩阵，列主序存储，side=LEFT 时 m×m，side=RIGHT 时 n×n，Device 内存 |
| lda | 输入 | int | 矩阵 A 的主维（列主序），side=LEFT 时 lda >= max(1, m)，side=RIGHT 时 lda >= max(1, n)，Host 内存 |
| B | 输入 | const float*（FP32） | m×n 输入矩阵，列主序存储，Device 内存 |
| ldb | 输入 | int | 矩阵 B 的主维（列主序），ldb >= max(1, m)，Host 内存 |
| C | 输出 | float*（FP32） | m×n 输出矩阵，列主序存储，Device 内存 |
| ldc | 输入 | int | 矩阵 C 的主维（列主序），ldc >= max(1, m)，Host 内存 |

#### 约束说明

- handle 不可为 nullptr，否则返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR
- side 必须为 ACLBLAS_SIDE_LEFT 或 ACLBLAS_SIDE_RIGHT，uplo 必须为 ACLBLAS_UPPER 或 ACLBLAS_LOWER，diag 必须为 ACLBLAS_UNIT 或 ACLBLAS_NON_UNIT。A2/A3 的 trans 仅支持 ACLBLAS_OP_N、ACLBLAS_OP_T，非法枚举（包括 ACLBLAS_OP_C）返回 ACLBLAS_STATUS_INVALID_VALUE；Ascend 950 还支持 ACLBLAS_OP_C，非法枚举返回 ACLBLAS_STATUS_INVALID_ENUM
- m >= 0, n >= 0，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- m==0 或 n==0 时直接返回 ACLBLAS_STATUS_SUCCESS，不访问任何指针、不校验 ld 参数（BLAS 标准）
- 以下约束仅在 m>0 且 n>0 时生效：
- side=LEFT 时：lda >= max(1, m)，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- side=RIGHT 时：lda >= max(1, n)，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- ldb >= max(1, m)，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- ldc >= max(1, m)，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- alpha 不可为 nullptr，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- C 不可为 nullptr，否则返回 ACLBLAS_STATUS_INVALID_VALUE（C 为纯输出矩阵，无 beta*C 项，必须提供有效写入目标，alpha==0 时也不例外；与 symm 不同，symm 在 beta==0 时 C 可为 nullptr）
- alpha == 0 时，A 和 B 不需要是有效的输入指针（可为 nullptr），结果 C 的 m×n 区域全为 0
- alpha != 0 时，A、B 不可为 nullptr，否则返回 ACLBLAS_STATUS_INVALID_VALUE

#### 精度验证

采用 `MIXED_TOLERANCE` 混合容差策略（对齐[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) §2.1）：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
|----------|------|------|----------------------|-------------------|
| FLOAT32（Ascend 950） | 2^-10 (9.77e-4) | 2^-16 (1.53e-5) | 0.99 | 1e-2 或 32·ULP |
| FLOAT32（Atlas A2/A3 任务验收） | 2^-13 (1.22e-4) | 2^-13 (1.22e-4) | 0.99 | 1e-2 或 32·ULP |

alpha == 0（且 alpha 非空）时使用 `EXACT` 位精确校验（结果 C 的 m×n 区域应为全零，位精确）。

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

```cpp
#include "acl/acl.h"
#include "cann_ops_blas.h"

int main()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclblasHandle_t handle = nullptr;
    aclblasCreate(&handle);

    constexpr int m = 4;
    constexpr int n = 4;
    constexpr int lda = 4;
    constexpr int ldb = 4;
    constexpr int ldc = 4;
    constexpr int dimA = m;  // side=LEFT → dimA=m
    constexpr size_t aBytes = lda * dimA * sizeof(float);
    constexpr size_t bBytes = ldb * n * sizeof(float);
    constexpr size_t cBytes = ldc * n * sizeof(float);
    float alpha = 1.0f;

    // 列主序存储：A[col*lda + row] = A[row][col]
    // 上三角 A (4x4):
    //   1  2  3  4
    //   0  5  6  7
    //   0  0  8  9
    //   0  0  0 10
    float hA[lda * dimA] = {
        1.0f, 0.0f, 0.0f, 0.0f,
        2.0f, 5.0f, 0.0f, 0.0f,
        3.0f, 6.0f, 8.0f, 0.0f,
        4.0f, 7.0f, 9.0f, 10.0f
    };
    // 列主序存储：B[col*ldb + row] = B[row][col]
    float hB[ldb * n] = {
        1.0f, 5.0f, 9.0f, 13.0f,
        2.0f, 6.0f, 10.0f, 14.0f,
        3.0f, 7.0f, 11.0f, 15.0f,
        4.0f, 8.0f, 12.0f, 16.0f
    };
    float hC[ldc * n] = {0};

    aclrtStream stream;
    aclrtCreateStream(&stream);
    aclblasSetStream(handle, stream);

    float *dA = nullptr;
    float *dB = nullptr;
    float *dC = nullptr;
    aclrtMalloc((void**)&dA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&dB, bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&dC, cBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(dA, aBytes, hA, aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dB, bBytes, hB, bBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    aclblasStatus_t status = aclblasStrmm(
        handle, ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
        m, n, &alpha, dA, lda, dB, ldb, dC, ldc);

    aclrtSynchronizeStream(stream);

    aclrtMemcpy(hC, cBytes, dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST);

    aclrtFree(dA);
    aclrtFree(dB);
    aclrtFree(dC);
    aclrtDestroyStream(stream);

    aclblasDestroy(handle);

    aclrtResetDevice(0);
    aclFinalize();

    return 0;
}
```
