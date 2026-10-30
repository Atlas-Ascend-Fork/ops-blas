# Ger算子

## 算子概述

Ger（Rank-1 Update）算子实现了矩阵的秩-1更新操作。

数学表达式：

```
A = A + alpha * x * y^T
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSger | 单精度浮点矩阵秩-1更新 |
| aclblasCgeru | 单精度复数无共轭矩阵秩-1更新 |

## 算子执行接口

### aclblasSger

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSger(aclblasHandle_t handle, int m, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| m | 输入 | int | 矩阵 A 的行数，Host 内存 |
| n | 输入 | int | 矩阵 A 的列数，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量乘数，Host 内存 |
| x | 输入 | const float*（FP32） | 长度为 m 的列向量，Device 内存 |
| incx | 输入 | int | 向量 x 的步长，Host 内存 |
| y | 输入 | const float*（FP32） | 长度为 n 的列向量，Device 内存 |
| incy | 输入 | int | 向量 y 的步长，Host 内存 |
| A | 输入/输出 | float*（FP32） | m x n 矩阵，原地更新，Device 内存 |
| lda | 输入 | int | 矩阵 A 的主维，Host 内存 |

#### 约束说明

- m >= 0, n >= 0
- incx != 0, incy != 0
- lda >= max(1, m)

### aclblasCgeru

#### 产品支持情况

| 产品 | 支持情况 |
|------|----------|
| Ascend 950PR | 支持 |
| Ascend 950DT | 支持 |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | 支持 |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | 支持 |

#### 函数原型

```cpp
aclblasStatus_t aclblasCgeru(aclblasHandle_t handle, int m, int n, const aclblasComplex* alpha, const aclblasComplex* x, int incx, const aclblasComplex* y, int incy, aclblasComplex* A, int lda)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| m | 输入 | int | 矩阵 A 的行数，Host 内存 |
| n | 输入 | int | 矩阵 A 的列数，Host 内存 |
| alpha | 输入 | const aclblasComplex* | Complex64 标量乘数，Host 内存 |
| x | 输入 | const aclblasComplex* | 长度为 m 的 Complex64 列向量，Device 内存 |
| incx | 输入 | int | 向量 x 的非零步长，Host 内存 |
| y | 输入 | const aclblasComplex* | 长度为 n 的 Complex64 列向量，Device 内存 |
| incy | 输入 | int | 向量 y 的非零步长，Host 内存 |
| A | 输入/输出 | aclblasComplex* | m x n 列主序 Complex64 矩阵，原地更新，Device 内存 |
| lda | 输入 | int | 矩阵 A 的主维，Host 内存 |

`aclblasCgeru` 计算 `A = A + alpha * x * y^T`，不对 y 做共轭。支持正、负非零步长；
`m == 0`、`n == 0` 或 alpha 为复零时不访问 Device 数据。

负步长的基指针约定与 Netlib 折算一致：`incx < 0`（或 `incy < 0`）时，`x`/`y` 需传入物理存储
低地址端的指针，逻辑元素 `j` 位于 `ptr[(count-1-j)*|inc|]`（x 的 count 为 m，y 的 count 为 n），
物理存储长度需为 `1+(count-1)*|inc|` 个元素，且越界偏移不被校验。例如 `m=4、incx=-2` 时
逻辑 `x[0]` 位于 `ptr[6]`、`x[3]` 位于 `ptr[0]`。按 C-BLAS 常见的"元素 `j` 位于 `x+j*incx`、
允许负偏移"直觉传参会得到错误结果。

#### 约束说明

- m >= 0，n >= 0
- incx != 0，incy != 0
- 列主序矩阵的 lda >= max(1, m)，不以列数 n 作为下限
- handle 和 alpha 不为空；非快速返回时 x、y、A 不为空
- 额外 GM 工作区需求为 0 字节；各架构的全部执行路径均仅使用核内 UB 或寄存器作为临时存储，不分配或使用 handle 工作区

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

int main()
{
    constexpr int deviceId = 0;
    constexpr int m = 2;
    constexpr int n = 2;
    constexpr int incx = 1;
    constexpr int incy = 1;
    constexpr int lda = m;
    const aclblasComplex alpha{1.0f, 0.0f};
    const std::vector<aclblasComplex> hostX{{1.0f, 1.0f}, {2.0f, -1.0f}};
    const std::vector<aclblasComplex> hostY{{3.0f, 0.0f}, {0.0f, 2.0f}};
    std::vector<aclblasComplex> hostA(lda * n, {0.0f, 0.0f});

    aclInit(nullptr);
    aclrtSetDevice(deviceId);
    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    aclblasHandle_t handle = nullptr;
    aclblasCreate(&handle);
    aclblasSetStream(handle, stream);

    aclblasComplex* deviceX = nullptr;
    aclblasComplex* deviceY = nullptr;
    aclblasComplex* deviceA = nullptr;
    const size_t xBytes = hostX.size() * sizeof(aclblasComplex);
    const size_t yBytes = hostY.size() * sizeof(aclblasComplex);
    const size_t aBytes = hostA.size() * sizeof(aclblasComplex);
    aclrtMalloc(reinterpret_cast<void**>(&deviceX), xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&deviceY), yBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&deviceA), aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(deviceX, xBytes, hostX.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(deviceY, yBytes, hostY.data(), yBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(deviceA, aBytes, hostA.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    const aclblasStatus_t status =
        aclblasCgeru(handle, m, n, &alpha, deviceX, incx, deviceY, incy, deviceA, lda);
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(hostA.data(), aBytes, deviceA, aBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    std::printf("aclblasCgeru status: %d\n", static_cast<int>(status));

    aclrtFree(deviceX);
    aclrtFree(deviceY);
    aclrtFree(deviceA);
    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(deviceId);
    aclFinalize();
    return status == ACLBLAS_STATUS_SUCCESS ? 0 : 1;
}
```

负步长调用时按上述约定构造物理缓冲，其余步骤与上例相同：

```cpp
// m = 4, incx = -2：逻辑 x = {x0, x1, x2, x3}
std::vector<aclblasComplex> xbuf(1 + (m - 1) * 2);  // 物理长度 1+(m-1)*|incx| = 7
xbuf[6] = x0;  // 逻辑 x[0] 位于 ptr[(m-1-j)*|incx|]
xbuf[4] = x1;
xbuf[2] = x2;
xbuf[0] = x3;  // 逻辑 x[3] 位于 ptr[0]，即物理低地址端
aclblasCgeru(handle, m, n, &alpha, xbuf.data(), incx, ybuf.data(), incy, deviceA, lda);
```

#### 执行路径与浮点语义

Host 只做参数校验、负步长起点折算（`xStart/yStart`）和一维列切分：
`colsPerCore = ceil(n / min(n, aivCoreNum))`，每核处理一段连续列；不计算行分块。
行方向由 kernel 以固定块 `CGERU_MAX_CHUNK_ROWS = 1024` 遍历（通用路径）。
m == 0、n == 0 或 alpha 为复零时快速返回，不 launch。

Ascend 950PR/950DT（arch35，SIMT/VF 编程模型）：连续布局 `incx == 1 && incy == 1 && lda == m`、
`m <= 4096` 且 `m * n > 32768` 时选择 RegBase 路径；其余布局在每核 tile 的行、列均不超过
4096 时选择 SIMT Cache，否则选择 DirectGM。这些条件是布局、核内容量与工作量门槛，不依赖测试编号。

RegBase 路径不对行向量切片，仅按列分核；每核拷入完整 x，按 UB 容量滑动处理列块。
容量规划使用 64 位中间值，将 x、矩阵列块、参数缓冲及其对齐填充的总占用限定在 240 KiB 内。
若无法容纳一列，在申请 UB 缓冲前回退为同一 tile 的 DirectGM 计算。

SIMT 路径根据 FP32 指数判断复乘的任一乘积是否可能溢出；对可能溢出或包含非有限操作数的复乘
保留独立乘积求值，以维持 Netlib BLAS 的 Inf/NaN 语义。该保护仅作用于特殊值分支，不在整个文件
禁用 FMA；常规分支保持原有优化。
RegBase 中 alpha 的 UB 标量写入使用 `volatile` 保留实际存储，并通过 S→V 事件建立流水线顺序，
与上述防乘加收缩的用途不同。

Atlas A2/A3（arch22，经典向量编程模型）：单 kernel，按列分核；每核内按 1024 行块处理。

- **快路径**（`incx == 1 && lda == m`，且 `m*2` 为 8 的倍数、UB 预算足够）：
  将整列 x 一次搬入 UB，构造 `xKeep = [xr, xi]` 与 `xSwap = [-xi, xr]` 两个交错向量；
  每列 j 直接从 GM 读 y[j] 合成 `coeff = alpha * y[j]`，随后两次融合 Axpy 完成
  `A[:, j] += cr * xKeep + ci * xSwap`；A 列按批连续搬入/搬出。快路径 UB 预算 176 KiB。
- **通用路径**（`incx != 1` 或 `lda > m`，或快路径预算不足）：
  x 按 1024 行块装入 UB；`incx == 1` 时整块连续 DataCopyPad，`incx != 1` 时逐元素
  `GetValue` 折算；y 标量始终从 GM 逐元素读取（`incy` 只影响地址步长，不做突发/带隙拷贝）。
  每列用 GatherMask 解交织 x、Gather 重交织乘积，再写回 A；UB 总预算 192 KiB，额外 GM workspace 0 字节。

浮点语义：`CgeruCoeff` 刻意不做“纯实/纯虚 shortcut”，与 cblas golden 的 `0*Inf→NaN`
保持一致（TC_FL_103）；唯一整列跳过是 `y[j] == (0, 0)`（对应 Netlib cgeru 的
`IF (Y(J).NE.ZERO)` 守卫）。
