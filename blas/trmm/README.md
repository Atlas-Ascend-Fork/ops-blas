# Trmm算子

## 算子概述

Trmm（Triangular Matrix Multiplication）算子族实现三角矩阵与普通矩阵的乘法运算，结果乘以标量 alpha 后写入输出矩阵 C。
族内包含两个接口：单精度浮点 `aclblasStrmm` 与单精度复数 `aclblasCtrmm`。

**`aclblasStrmm`**（单精度浮点）实现了单精度浮点三角矩阵与普通矩阵的乘法运算，结果乘以标量 alpha 后写入输出矩阵 C。

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
| aclblasCtrmm | 单精度复数三角矩阵乘法 |

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

### aclblasCtrmm

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

> Ascend 950PR/Ascend 950DT 上的 ctrmm 依赖 CANN asc-devkit >= 9.1（`ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR >= 1`），低于该版本时编译与运行将跳过此算子。

#### 函数原型

```cpp
aclblasStatus_t aclblasCtrmm(
    aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans,
    aclblasDiagType_t diag, int m, int n, const aclblasComplex* alpha, const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb, aclblasComplex* C, int ldc);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| side | 输入 | aclblasSideMode_t | A 矩阵位置：ACLBLAS_SIDE_LEFT（左侧）或 ACLBLAS_SIDE_RIGHT（右侧），Host 内存 |
| uplo | 输入 | aclblasFillMode_t | A 矩阵存储模式：ACLBLAS_UPPER（上三角）或 ACLBLAS_LOWER（下三角），Host 内存 |
| trans | 输入 | aclblasOperation_t | A 矩阵操作类型：ACLBLAS_OP_N（不转置）、ACLBLAS_OP_T（转置）、ACLBLAS_OP_C（共轭转置），Host 内存 |
| diag | 输入 | aclblasDiagType_t | 对角线类型：ACLBLAS_UNIT（单位三角，对角线视为 1 且不读取 A 对角位置）或 ACLBLAS_NON_UNIT（非单位三角，对角元照常读取），Host 内存 |
| m | 输入 | int | 矩阵 B/C 的行数，m >= 0，Host 内存 |
| n | 输入 | int | 矩阵 B/C 的列数，n >= 0，Host 内存 |
| alpha | 输入 | const aclblasComplex*（COMPLEX64） | 复数标量 alpha，不可为 nullptr，支持 Host 或 Device 内存 |
| A | 输入 | const aclblasComplex*（COMPLEX64） | 三角矩阵，列主序存储，side=LEFT 时 m×m，side=RIGHT 时 n×n，Device 内存 |
| lda | 输入 | int | 矩阵 A 的主维（列主序），side=LEFT 时 lda >= max(1, m)，side=RIGHT 时 lda >= max(1, n)，Host 内存 |
| B | 输入 | const aclblasComplex*（COMPLEX64） | m×n 输入矩阵，列主序存储，Device 内存 |
| ldb | 输入 | int | 矩阵 B 的主维（列主序），ldb >= max(1, m)，Host 内存 |
| C | 输出 | aclblasComplex*（COMPLEX64） | m×n 输出矩阵，列主序存储，Device 内存；允许与 B 为同一指针（原地计算） |
| ldc | 输入 | int | 矩阵 C 的主维（列主序），ldc >= max(1, m)，Host 内存 |

#### 约束说明

- 仅支持 COMPLEX64；仅引用 uplo 指定三角内的元素，另一侧不读取；
- `handle` 不可为 nullptr，否则返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR；
- `m < 0`、`n < 0`、枚举非法或前导维不满足约束时返回 ACLBLAS_STATUS_INVALID_VALUE；
- `alpha` 不可为 nullptr，否则返回 ACLBLAS_STATUS_INVALID_VALUE；
- `m > 0 且 n > 0` 时 `C` 不可为 nullptr，否则返回 ACLBLAS_STATUS_INVALID_VALUE（alpha == (0,0) 时也不例外）；
- `alpha != (0,0)` 时 `A`、`B` 不可为 nullptr，否则返回 ACLBLAS_STATUS_INVALID_VALUE；
- `alpha == (0,0)` 时 A 与 B 不被引用（可为 nullptr），C 的 m×n 区域全部置零；
- `m == 0` 或 `n == 0` 时为空操作，返回 ACLBLAS_STATUS_SUCCESS；
- `diag == ACLBLAS_NON_UNIT` 时对角元照常从 A 读取且**允许为零**（本算子为乘法，与求解类算子不同，不做对角非零保障）；
- 允许 `C == B`（原地计算），与 cuBLAS `cublasCtrmm` 口径一致（其他部分重叠不支持）；
- `m ≤ 4096`、`n ≤ 4096`（超限返回 `ACLBLAS_STATUS_INVALID_VALUE`）；该上限是**接口契约上限**
  （取自任务书规格与随任务用例的最大维度），**不取自矩阵乘指令的维度限制**——Phase-B 下发给
  `MmadParams` 的是 L0 分块维度（受 `PB_BASE_M/N/K = 16/16/8` 约束），全量 m/n 不会到达指令；
- 管线档（`m×n ≥ 4096`）的 workspace 随 `lda/ldb` 与问题规模线性增长（量级约为
  `(6·lda·dimA + 4·m·n)` 个 float 加影子区）；当 `lda/ldb` 显著大于维度下限（如 4096 阶
  `lda ≳ 19100`，约 4.7×m）时，所需 workspace 将超过 2 GB 平台上限，调用返回
  `ACLBLAS_STATUS_ALLOC_FAILED`——该调用在 BLAS 语义下合法，属资源上限而非参数错误；
- 含 Inf/NaN 的输入仅在 `m,n ≤ 32`（非管线档）保证与参考实现逐位一致的传播模式；
  大规格（`m×n ≥ 4096`）走 Gauss 3M 管线，非有限值传播可能与 4M 参考不同；
- 本算子当前仅支持 `device` 指针的 `A`/`B`/`C`（`alpha` 支持 Host 或 Device 内存，
  Device 指针时内部会同步回读）。
- 不支持非连续 Tensor 语义之外的额外布局（仅 lda/ldb/ldc 前导维）；不涉及广播与 dynamic shape。

#### 精度验证

采用 `MIXED_TOLERANCE` 混合容差策略（对齐[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) §2.1）；
输入输出为单精度复数，**实部、虚部分别按 FLOAT32 判定**：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
|----------|------|------|----------------------|-------------------|
| COMPLEX64（实部/虚部按 FLOAT32 分量） | 2^-10 (9.77e-4) | 2^-16 (1.53e-5) | 0.99 | 1e-2 或 32·ULP |

参考实现（golden）由 Netlib BLAS 的复数三角矩阵乘 `cblas_ctrmm` 生成，随测试工程提供。

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
    constexpr size_t aBytes = static_cast<size_t>(lda) * dimA * sizeof(aclblasComplex);
    constexpr size_t bBytes = static_cast<size_t>(ldb) * n * sizeof(aclblasComplex);
    constexpr size_t cBytes = static_cast<size_t>(ldc) * n * sizeof(aclblasComplex);

    // 复数标量 alpha = 1 + 0i
    aclblasComplex alpha{1.0f, 0.0f};

    // 列主序存储：A[col*lda + row] = A[row][col]，下三角仅存储三角内元素
    aclblasComplex hA[lda * dimA] = {
        {1.0f, 0.5f}, {2.0f, -0.5f}, {0.0f, 0.0f}, {0.0f, 0.0f},
        {3.0f, 0.25f}, {4.0f, 1.5f}, {5.0f, -0.25f}, {0.0f, 0.0f},
        {6.0f, -1.0f}, {7.0f, 0.75f}, {8.0f, 0.5f}, {9.0f, -0.75f},
        {10.0f, 1.25f}, {11.0f, -1.5f}, {12.0f, 0.25f}, {13.0f, 2.0f}
    };
    aclblasComplex hB[ldb * n] = {
        {1.0f, 1.0f}, {5.0f, -1.0f}, {9.0f, 0.5f}, {13.0f, -0.5f},
        {2.0f, -1.0f}, {6.0f, 1.0f}, {10.0f, -0.5f}, {14.0f, 0.5f},
        {3.0f, 0.5f}, {7.0f, -0.5f}, {11.0f, 1.0f}, {15.0f, -1.0f},
        {4.0f, -0.5f}, {8.0f, 0.5f}, {12.0f, -1.0f}, {16.0f, 1.0f}
    };
    aclblasComplex hC[ldc * n] = {};

    aclrtStream stream;
    aclrtCreateStream(&stream);
    aclblasSetStream(handle, stream);

    aclblasComplex *dA = nullptr;
    aclblasComplex *dB = nullptr;
    aclblasComplex *dC = nullptr;
    aclrtMalloc((void**)&dA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&dB, bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc((void**)&dC, cBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(dA, aBytes, hA, aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dB, bBytes, hB, bBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    aclblasStatus_t status = aclblasCtrmm(
        handle, ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
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
