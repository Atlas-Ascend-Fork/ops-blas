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
| aclblasCtrmm | 单精度复数三角矩阵乘法，支持 arch22（A2/A3）和 arch35（950） |

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

### aclblasCtrmm（Ascend 950PR/950DT，arch35）

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

### aclblasCtrmm（Atlas A2/A3，arch22）

#### 产品支持情况

| 产品 | 支持 |
| --- | --- |
| Atlas A2 训练/推理系列，含 Atlas 800T A2、Atlas 800I A2 | 支持 |
| Atlas A3 训练/推理系列，含 Atlas 800I A3 | 支持 |
| Ascend 950PR / 950DT | 本实现不支持 |

CANN 9.1.0；当前实机自验为 A2 910B3，A3 使用同一 arch22 实现，尚无实机结果。

#### 函数原型

```cpp
aclblasStatus_t aclblasCtrmm(aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n, const aclblasComplex* alpha, const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb, aclblasComplex* C, int ldc)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream 和 workspace，Host 内存 |
| side | 输入 | aclblasSideMode_t | A 的位置：ACLBLAS_SIDE_LEFT 对应 C=alpha·op(A)·B，ACLBLAS_SIDE_RIGHT 对应 C=alpha·B·op(A)，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | A 转置前的有效三角：ACLBLAS_UPPER（上三角）或 ACLBLAS_LOWER（下三角），Host 内存 |
| trans | 输入 | aclblasOperation_t | 对 A 的操作：ACLBLAS_OP_N（不转置）、ACLBLAS_OP_T（转置）或 ACLBLAS_OP_C（共轭转置），Host 内存 |
| diag | 输入 | aclblasDiagType_t | ACLBLAS_UNIT 将对角线视为 1+0i 且不读取对角元素；ACLBLAS_NON_UNIT 使用实际对角元素，Host 内存 |
| m | 输入 | int | 矩阵 B/C 的行数，m >= 0，Host 内存 |
| n | 输入 | int | 矩阵 B/C 的列数，n >= 0，Host 内存 |
| alpha | 输入 | const aclblasComplex*（COMPLEX64） | 复数缩放标量，不可为 nullptr，仅支持 Host 内存 |
| A | 输入 | const aclblasComplex*（COMPLEX64） | 列主序三角矩阵；side=LEFT 时为 m×m，side=RIGHT 时为 n×n，仅访问有效三角，Device 内存 |
| lda | 输入 | int | A 的列跨度，以复数元素计；非零维调用时至少为 A 的阶数，Host 内存 |
| B | 输入 | const aclblasComplex*（COMPLEX64） | m×n 列主序输入矩阵，Device 内存 |
| ldb | 输入 | int | B 的列跨度，以复数元素计；非零维调用时 ldb >= m，Host 内存 |
| C | 输出 | aclblasComplex*（COMPLEX64） | m×n 列主序输出矩阵，支持 B=C 的原地计算，Device 内存 |
| ldc | 输入 | int | C 的列跨度，以复数元素计；非零维调用时 ldc >= m，Host 内存 |

#### 约束说明

- 数据类型为 COMPLEX64；每个 `aclblasComplex` 包含 FP32 实部 `real` 和虚部 `imag`。A/B/C 均为列主序，元素 `(row, col)` 的索引为 `col * ld + row`。
- handle 不可为 nullptr，否则返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR。
- side、uplo、trans、diag 必须使用参数表中的枚举值，m/n 必须非负，alpha 不可为 nullptr；任一条件不满足返回 ACLBLAS_STATUS_INVALID_VALUE。
- 上述检查通过后，m==0 或 n==0 时返回 ACLBLAS_STATUS_SUCCESS，不读取 alpha 的值，不访问 A/B/C，也不校验 lda/ldb/ldc，允许 ld=0。
- m>0 且 n>0 时，令 q=(side==ACLBLAS_SIDE_LEFT ? m : n)，必须满足 lda>=q、ldb>=m、ldc>=m，C 不可为 nullptr；否则返回 ACLBLAS_STATUS_INVALID_VALUE。矩阵存储范围的字节数不得溢出 size_t。
- alpha 必须指向可读的 Host 标量，A/B/C 必须提供对应访问范围内的有效 Device 存储；输出仅写 C 的 m×n 有效区域，不修改列间 padding。
- alpha==0+0i 时，A/B 可为 nullptr，C 的有效区域写为 0+0i；仍需满足非零维调用的 leading dimension 和 C 存储约束。
- alpha!=0+0i 时，A/B 不可为 nullptr；C 不得与 A 的存储范围重叠，与 B 仅允许起始地址完全相同的 B=C 原地计算。B=C 时 ldb/ldc 可以不同，但共享缓冲区须同时容纳两种布局；其他 B/C 重叠返回 ACLBLAS_STATUS_INVALID_VALUE。
- 非有效三角不参与计算；diag=ACLBLAS_UNIT 时不读取 A 的对角线，并将其视为 1+0i。
- 算子所需 workspace 上限为 2 GiB，超过上限、用户提供的 workspace 容量不足或分配失败时返回 ACLBLAS_STATUS_ALLOC_FAILED。无法获取有效 AIV 核数时返回 ACLBLAS_STATUS_INTERNAL_ERROR。
- 算子在 handle 绑定的 stream 上异步执行，读回 C 或释放输入输出缓冲区前需同步。默认 workspace 扩容沿用库的同步机制，提前配置足量 workspace 后的调用无需 Host 同步。

#### 精度验证

实部/虚部分别使用atol=rtol=2^-13、匹配比例≥0.99，且最大绝对误差≤max(0.01,32*ULP(g_low))；g_low取最大误差位置的FP32 golden。日志同时保留固定0.01的判定信息；
NaN/Inf 类型及 Inf 符号另外检查。
设计说明位于独立的 cann-competitions 仓库：
`04_tasks/01_community-task-2026/tasklist/09-71-aclblasCtrmm-A2A3/Xztxzt/docs/design.md`。
复现方法见[编译、精度及性能测试步骤](../../test/trmm/ctrmm/arch22/README.md)。

#### 调用示例

以下示例计算 2×2 复数矩阵的左上三角乘法。使用 `AclContext` 和 `std::unique_ptr`
管理 ACL 上下文、stream、BLAS 句柄及 Device 内存。
编译和执行方法见[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cmath>
#include <cstddef>
#include <cstdint>
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
    explicit AclContext(int32_t deviceId) : deviceId_(deviceId) {}

    ~AclContext()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
        }
        if (aclInited_) {
            aclFinalize();
        }
    }

    int Init()
    {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, std::printf("aclInit failed: %d\n", ret); return ret);
        aclInited_ = true;
        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtSetDevice failed: %d\n", ret); return ret);
        deviceSet_ = true;
        ret = aclrtCreateStream(&stream_);
        CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtCreateStream failed: %d\n", ret); return ret);
        return ACL_SUCCESS;
    }

    aclrtStream Stream() const { return stream_; }

private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

using DevicePtr = std::unique_ptr<void, decltype(&aclrtFree)>;

int PrepareBuffer(DevicePtr& buffer, const void* source, size_t bytes)
{
    void* raw = nullptr;
    auto ret = aclrtMalloc(&raw, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtMalloc failed: %d\n", ret); return ret);
    buffer.reset(raw);
    if (source != nullptr) {
        ret = aclrtMemcpy(buffer.get(), bytes, source, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtMemcpy H2D failed: %d\n", ret); return ret);
    }
    return ACL_SUCCESS;
}

int CtrmmExample(AclContext& ctx)
{
    aclblasHandle_t rawHandle = nullptr;
    auto status = aclblasCreate(&rawHandle);
    CHECK_RET(status == ACLBLAS_STATUS_SUCCESS, std::printf("aclblasCreate failed: %d\n", status); return status);
    std::unique_ptr<_aclblas_handle, decltype(&aclblasDestroy)> handle(rawHandle, aclblasDestroy);
    status = aclblasSetStream(handle.get(), ctx.Stream());
    CHECK_RET(status == ACLBLAS_STATUS_SUCCESS, std::printf("aclblasSetStream failed: %d\n", status); return status);

    constexpr int m = 2;
    constexpr int n = 2;
    constexpr int lda = 2;
    constexpr int ldb = 2;
    constexpr int ldc = 2;
    const aclblasComplex alpha{0.5f, 0.5f};
    // 列主序：A = [[1+i, 2-i], [0, 3]]，B = [[1, 2+i], [i, 1-i]]。
    const aclblasComplex hA[lda * m] = {{1.0f, 1.0f}, {0.0f, 0.0f}, {2.0f, -1.0f}, {3.0f, 0.0f}};
    const aclblasComplex hB[ldb * n] = {{1.0f, 0.0f}, {0.0f, 1.0f}, {2.0f, 1.0f}, {1.0f, -1.0f}};
    aclblasComplex hC[ldc * n]{};
    const aclblasComplex expected[ldc * n] = {{-0.5f, 2.5f}, {-1.5f, 1.5f}, {1.0f, 1.0f}, {3.0f, 0.0f}};
    DevicePtr dA(nullptr, aclrtFree), dB(nullptr, aclrtFree), dC(nullptr, aclrtFree);
    auto ret = PrepareBuffer(dA, hA, sizeof(hA));
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = PrepareBuffer(dB, hB, sizeof(hB));
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = PrepareBuffer(dC, nullptr, sizeof(hC));
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    status = aclblasCtrmm(handle.get(), ACLBLAS_SIDE_LEFT, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
        m, n, &alpha, static_cast<const aclblasComplex*>(dA.get()), lda,
        static_cast<const aclblasComplex*>(dB.get()), ldb, static_cast<aclblasComplex*>(dC.get()), ldc);
    CHECK_RET(status == ACLBLAS_STATUS_SUCCESS, std::printf("aclblasCtrmm failed: %d\n", status); return status);
    ret = aclrtSynchronizeStream(ctx.Stream());
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtSynchronizeStream failed: %d\n", ret); return ret);
    ret = aclrtMemcpy(hC, sizeof(hC), dC.get(), sizeof(hC), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, std::printf("aclrtMemcpy D2H failed: %d\n", ret); return ret);
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            const int index = col * ldc + row;
            std::printf("C(%d,%d) = %g%+gi\n", row, col, hC[index].real, hC[index].imag);
            CHECK_RET(std::abs(hC[index].real - expected[index].real) <= 1e-6f &&
                std::abs(hC[index].imag - expected[index].imag) <= 1e-6f,
                std::printf("Unexpected result at (%d,%d)\n", row, col); return 1);
        }
    }
    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return CtrmmExample(ctx);
}
```
