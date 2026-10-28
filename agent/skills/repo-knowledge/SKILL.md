---
name: repo-knowledge
description: 仓库领域知识，提供本仓算子涉及的领域标准、概念与背景（BLAS 接口规范）。
---

# BLAS 接口规范

本技能提供 BLAS (Basic Linear Algebra Subprograms) 标准接口规范，用于指导 ops-blas 接口开发。

## 1. 接口命名规范

### 1.1 命名格式

BLAS 接口采用统一的命名格式：

```
<精度前缀><操作名>
```

### 1.2 精度前缀

| 前缀 | 数据类型 | 说明 |
|------|---------|------|
| `s` | `float` | 单精度实数 |
| `d` | `double` | 双精度实数 |
| `c` | `complex<float>` | 单精度复数 |
| `z` | `complex<double>` | 双精度复数 |

特殊前缀组合：
- `sc` / `cs`: 单精度实数操作单精度复数（如 `csscal`）
- `dz` / `zd`: 双精度实数操作双精度复数（如 `zdscal`）
- `sd` / `ds`: 混合精度（如 `sdsdot`）

### 1.3 操作名分类

#### Level 1 BLAS（向量-向量操作）

| 操作名 | 功能 | 示例 |
|--------|------|------|
| `amax` | 返回绝对值最大元素的索引 | `isamax`, `idamax`, `icamax`, `izamax` |
| `amin` | 返回绝对值最小元素的索引 | `isamin`, `idamin`, `icamin`, `izamin` |
| `asum` | 计算绝对值之和 | `sasum`, `dasum`, `scasum`, `dzasum` |
| `axpy` | 计算 y = αx + y | `saxpy`, `daxpy`, `caxpy`, `zaxpy` |
| `copy` | 复制向量 y = x | `scopy`, `dcopy`, `ccopy`, `zcopy` |
| `dot` | 点积 | `sdot`, `ddot`, `cdotu`, `zdotu`, `cdotc`, `zdotc` |
| `nrm2` | 计算欧几里得范数 | `snrm2`, `dnrm2`, `scnrm2`, `dznrm2` |
| `rot` | 应用 Givens 旋转 | `srot`, `drot`, `csrot`, `zdrot` |
| `rotg` | 构造 Givens 旋转 | `srotg`, `drotg`, `crotg`, `zrotg` |
| `rotm` | 应用修正的 Givens 旋转 | `srotm`, `drotm` |
| `rotmg` | 构造修正的 Givens 旋转 | `srotmg`, `drotmg` |
| `scal` | 缩放向量 x = αx | `sscal`, `dscal`, `cscal`, `zscal`, `csscal`, `zdscal` |
| `swap` | 交换向量 | `sswap`, `dswap`, `cswap`, `zswap` |

#### Level 2 BLAS（矩阵-向量操作）

| 操作名 | 功能 | 示例 |
|--------|------|------|
| `gemv` | 通用矩阵-向量乘法 y = αAx + βy | `sgemv`, `dgemv`, `cgemv`, `zgemv` |
| `gbmv` | 带状矩阵-向量乘法 | `sgbmv`, `dgbmv`, `cgbmv`, `zgbmv` |
| `symv` | 对称矩阵-向量乘法 | `ssymv`, `dsymv` |
| `hemv` | 埃尔米特矩阵-向量乘法 | `chemv`, `zhemv` |
| `sbmv` | 对称带状矩阵-向量乘法 | `ssbmv`, `dsbmv` |
| `hbmv` | 埃尔米特带状矩阵-向量乘法 | `chbmv`, `zhbmv` |
| `spmv` | 对称打包矩阵-向量乘法 | `sspmv`, `dspmv` |
| `hpmv` | 埃尔米特打包矩阵-向量乘法 | `chpmv`, `zhpmv` |
| `trmv` | 三角矩阵-向量乘法 | `strmv`, `dtrmv`, `ctrmv`, `ztrmv` |
| `tbmv` | 三角带状矩阵-向量乘法 | `stbmv`, `dtbmv`, `ctbmv`, `ztbmv` |
| `tpmv` | 三角打包矩阵-向量乘法 | `stpmv`, `dtpmv`, `ctpmv`, `ztpmv` |
| `trsv` | 三角矩阵求解 | `strsv`, `dtrsv`, `ctrsv`, `ztrsv` |
| `tbsv` | 三角带状矩阵求解 | `stbsv`, `dtbsv`, `ctbsv`, `ztbsv` |
| `tpsv` | 三角打包矩阵求解 | `stpsv`, `dtpsv`, `ctpsv`, `ztpsv` |
| `ger` | 秩-1 更新 A = αxy^T + A | `sger`, `dger` |
| `geru` | 秩-1 更新（无共轭） | `cgeru`, `zgeru` |
| `gerc` | 秩-1 更新（共轭） | `cgerc`, `zgerc` |
| `syr` | 对称秩-1 更新 | `ssyr`, `dsyr` |
| `her` | 埃尔米特秩-1 更新 | `cher`, `zher` |
| `spr` | 对称打包秩-1 更新 | `sspr`, `dspr` |
| `hpr` | 埃尔米特打包秩-1 更新 | `chpr`, `zhpr` |
| `syr2` | 对称秩-2 更新 | `ssyr2`, `dsyr2` |
| `her2` | 埃尔米特秩-2 更新 | `cher2`, `zher2` |
| `spr2` | 对称打包秩-2 更新 | `sspr2`, `dspr2` |
| `hpr2` | 埃尔米特打包秩-2 更新 | `chpr2`, `zhpr2` |

#### Level 3 BLAS（矩阵-矩阵操作）

| 操作名 | 功能 | 示例 |
|--------|------|------|
| `gemm` | 通用矩阵乘法 C = αAB + βC | `sgemm`, `dgemm`, `cgemm`, `zgemm` |
| `symm` | 对称矩阵乘法 | `ssymm`, `dsymm`, `csymm`, `zsymm` |
| `hemm` | 埃尔米特矩阵乘法 | `chemm`, `zhemm` |
| `syrk` | 对称秩-k 更新 | `ssyrk`, `dsyrk`, `csyrk`, `zsyrk` |
| `herk` | 埃尔米特秩-k 更新 | `cherk`, `zherk` |
| `syr2k` | 对称秩-2k 更新 | `ssyr2k`, `dsyr2k`, `csyr2k`, `zsyr2k` |
| `her2k` | 埃尔米特秩-2k 更新 | `cher2k`, `zher2k` |
| `trmm` | 三角矩阵乘法 | `strmm`, `dtrmm`, `ctrmm`, `ztrmm` |
| `trsm` | 三角矩阵求解 | `strsm`, `dtrsm`, `ctrsm`, `ztrsm` |

## 2. 参数规范

### 2.1 参数顺序

BLAS 接口遵循严格的参数顺序：

1. **枚举参数类型**（以下仅列举常见类型；具体类型及位置以各 API 的正式签名为准）：
   - `order`（行主序/列主序）
   - `trans`（转置类型）
   - `uplo`（上三角/下三角）
   - `diag`（单位对角线/非单位对角线）
   - `side`（左乘/右乘）

2. **维度参数**：
   - 矩阵维度：`m`, `n`, `k`
   - 向量长度：`n`
   - 带宽参数：`kl`, `ku`, `k`

3. **标量参数**：
   - `alpha`, `beta`（缩放因子）

4. **数组参数**：
   - 输入数组：`A`, `B`, `X`, `Y`
   - 输出数组：`C`, `Y`（当同时作为输入时）
   - 打包数组：`AP`

5. **步长参数**：
   - `lda`, `ldb`, `ldc`（leading dimension）
   - `incx`, `incy`（向量增量）

### 2.2 典型参数顺序示例

#### Level 1 BLAS

```c
// 向量操作
void cblas_saxpy(const int n, const float alpha, const float *x, 
                 const int incx, float *y, const int incy);

// 点积
float cblas_sdot(const int n, const float *x, const int incx, 
                 const float *y, const int incy);

// 范数
float cblas_snrm2(const int n, const float *x, const int incx);

// 缩放
void cblas_sscal(const int n, const float alpha, float *x, const int incx);
```

#### Level 2 BLAS

```c
// 通用矩阵-向量乘法
void cblas_sgemv(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE trans, 
                 const int m, const int n,
                 const float alpha, const float *A, const int lda,
                 const float *x, const int incx,
                 const float beta, float *y, const int incy);

// 三角矩阵-向量乘法
void cblas_strmv(const enum CBLAS_ORDER order, 
                 const enum CBLAS_UPLO uplo,
                 const enum CBLAS_TRANSPOSE trans, 
                 const enum CBLAS_DIAG diag,
                 const int n, const float *A, const int lda,
                 float *x, const int incx);
```

#### Level 3 BLAS

```c
// 通用矩阵乘法
void cblas_sgemm(const enum CBLAS_ORDER order,
                 const enum CBLAS_TRANSPOSE transA,
                 const enum CBLAS_TRANSPOSE transB,
                 const int m, const int n, const int k,
                 const float alpha, const float *A, const int lda,
                 const float *B, const int ldb,
                 const float beta, float *C, const int ldc);
```

### 2.3 枚举类型定义

```c
enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
enum CBLAS_UPLO {CblasUpper=121, CblasLower=122};
enum CBLAS_DIAG {CblasNonUnit=131, CblasUnit=132};
enum CBLAS_SIDE {CblasLeft=141, CblasRight=142};
```

### 2.4 参数类型规范

#### 输入参数（只读）

- 使用 `const` 修饰
- 示例：`const float *x`, `const float alpha`

#### 输出参数（可写）

- 不使用 `const` 修饰
- 示例：`float *y`, `float *C`

#### 输入/输出参数

- 不使用 `const` 修饰
- 示例：`float *x`（在 `scal` 中既是输入也是输出）

### 2.5 标量参数传递

BLAS 标准中，标量参数（`alpha`, `beta`）的传递方式：

- **CBLAS 接口**：按值传递
  ```c
  void cblas_saxpy(const int n, const float alpha, ...);
  ```

- **Fortran BLAS 接口**：按引用传递
  ```fortran
  SUBROUTINE SAXPY(N, ALPHA, X, INCX, Y, INCY)
  ```

## 3. ops-blas 接口映射

前面的 CBLAS 说明是背景知识。本仓公开 ABI 以 [`include/cann_ops_blas.h`](../../../include/cann_ops_blas.h) 和 [`include/cann_ops_blas_common.h`](../../../include/cann_ops_blas_common.h) 为准，不能只重命名 CBLAS 函数并保留原实参列表。

### 3.1 命名与调用方式

`cblas_sgemm`、`cblas_sgemv`、`cblas_scopy` 可分别对应本仓 `aclblasSgemm`、`aclblasSgemv`、`aclblasScopy` 的计算功能；可用接口及具体声明需查公共头。

- 本仓这些接口的首参为 `aclblasHandle_t`。
- `alpha`、`beta` 等标量在这些接口中通过指针传递，不是 CBLAS 的标量值参数。
- 返回值为 `aclblasStatus_t`；例如 `aclblasSdot` 将数值结果写到 `result`，不直接返回点积值。
- 没有 CBLAS `order` 对应参数。矩阵布局应遵循具体接口文档，不能自行添加 `aclblasOrder_t`。

### 3.2 枚举类型对应

| CBLAS 概念 | 本仓公共类型 |
| --- | --- |
| transpose | `aclblasOperation_t` |
| uplo | `aclblasFillMode_t` |
| diag | `aclblasDiagType_t` |
| side | `aclblasSideMode_t` |

上述实际定义位于 `cann_ops_blas_common.h`；不要使用并未定义的 `aclblasTranspose_t`、`aclblasUplo_t`、`aclblasDiag_t` 或 `aclblasSide_t`。

### 3.3 真实声明示例

以下声明逐字取自公共头，用于说明实参合同，不表示接口在所有芯片上均有实现或已经执行验证。

```cpp
aclblasStatus_t aclblasSgemm(
    aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb,
    int m, int n, int k, const float* alpha, const float* A, int lda,
    const float* B, int ldb, const float* beta, float* C, int ldc);

aclblasStatus_t aclblasSgemv(
    aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float* alpha, const float* A, int lda,
    const float* x, int incx, const float* beta, float* y, int incy);

aclblasStatus_t aclblasSdot(
    aclblasHandle_t handle, int n, const float* x, int incx, const float* y, int incy, float* result);
```

## 4. 常见错误与规范检查

### 4.1 命名错误

❌ 错误示例：`aclblas_gemm`、`aclblasGEMM`、`aclblasgemm`。

✅ 正确示例：`aclblasSgemm`、`aclblasCgemv`、`aclblasScopy`；仍需核对接口是否已声明及支持范围。

### 4.2 ABI 实参错误

不要把 CBLAS 的 `order` 作为本仓首参，不要省略 `handle`，也不要把真实指针标量改成值传参。参数顺序必须与具体公共声明一致，不能要求所有接口沿用 CBLAS 原实参列表。

### 4.3 const 与输入输出检查

新接口应按输入/输出语义考虑 `const`；使用既有接口时保留其真实 ABI，不自行改写公共签名。比如本仓只读 Scopy 输入与可写输出声明为：

```cpp
aclblasStatus_t aclblasScopy(aclblasHandle_t handle, int n, const float* x, int incx, float* y, int incy);
```

### 4.4 步长参数检查

按真实声明保留 `lda`、`ldb`、`ldc`、`incx`、`incy` 等相关参数。适用维度、布局和合法取值按具体接口文档与实现检查，不能仅依据通用命名猜测签名。

## 5. 参考资源

- **BLAS 标准文档**：http://www.netlib.org/blas/
- **CBLAS 参考实现**：http://www.netlib.org/blas/blast-forum/cblas.tgz
- **LAPACK 文档**：http://www.netlib.org/lapack/

## 6. 使用场景

本技能适用于以下场景：

1. **新接口开发**：开发新的 BLAS 接口时，确保命名和参数符合标准
2. **接口审查**：审查现有接口是否符合 BLAS 规范
3. **接口迁移**：从其他 BLAS 库（如 MKL、OpenBLAS）迁移接口时，保持一致性
4. **文档编写**：编写接口文档时，参考标准命名和参数说明
