# Hpr算子

## 算子概述

hpr (Hermitian Packed Rank-1 Update) 实现 Hermitian 矩阵 packed 格式的秩-1 更新。该算子将 `alpha * x * x^H` 加到 packed Hermitian 矩阵的指定三角区域，并将对角线虚部置零。

数学表达式：

```
A := alpha * x * x^H + A
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasChpr | 单精度复数 Hermitian packed 秩-1 更新 |

## 算子执行接口

### aclblasChpr

#### 产品支持情况

- Ascend 950PR：支持
- Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasChpr(aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float *alpha, const aclblasComplex *x, int incx, aclblasComplex *ap)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | ACLBLAS_UPPER(121) 或 ACLBLAS_LOWER(122)，指定更新的三角区域，Host 内存 |
| n | 输入 | int | 矩阵阶数，n >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | 实数标量乘数指针，Host 内存 |
| x | 输入 | const aclblasComplex* | 输入向量指针，长度至少 1 + (n-1) * abs(incx)，Device 内存 |
| incx | 输入 | int | x 的元素间步长，incx != 0 且 incx != INT_MIN，Host 内存 |
| ap | 输入/输出 | aclblasComplex* | packed Hermitian 矩阵指针，长度 n*(n+1)/2，Device 内存 |

#### 约束说明

- n >= 0
- incx != 0 且 incx != INT_MIN；负步长从物理偏移 `(n-1)*abs(incx)` 开始反向遍历。
- alpha 为实数标量
- `ap` 按列压缩存储：上三角 `A(i,j)` 的偏移为 `j*(j+1)/2+i`，下三角为 `j*(2*n-j+1)/2+i-j`。
- 更新时忽略对角元素的输入虚部，输出对角虚部为零；`x` 保持只读。
- arch35 按 handle、`n<0`、`n=0`、uplo、incx、alpha/x/ap 的顺序处理参数。空 handle 优先返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；handle 有效且 `n=0` 时直接成功且不访问其余参数；`n>0` 的 `alpha=0` 在参数校验成功后直接返回，`ap` 保持原样。
- 接口异步执行于 handle 绑定的 stream；读回结果或释放输入输出内存前应同步该 stream。

#### arch35 实现

实现与 arch22 沿用同一接口和文件布局，位于 `blas/hpr/arch35/`：`chpr_host.cpp` 负责校验与下发，
`chpr_kernel.cpp` 实现 Ascend C 内核，`chpr_kernel.h` 和 `chpr_tiling_data.h` 定义内部下发接口与参数。
SIMT 相邻线程处理 packed 列中的连续元素，各线程组循环分配列；可容纳的向量缓存于 UB，较大向量使用 GM 路径。
计算不申请 workspace，不展开完整矩阵。测试与复现步骤见 [Chpr 测试说明](../../test/hpr/chpr/README.md)。

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
#include "acl/acl.h"
#include "cann_ops_blas.h"

int main()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclblasHandle_t handle = nullptr;
    aclblasCreate(&handle);

    const int n = 4;
    float alpha = 1.0f;
    aclblasComplex x[4] = {{1.0f, 2.0f}, {3.0f, 4.0f}, {5.0f, 6.0f}, {7.0f, 8.0f}};
    aclblasComplex ap[10] = {};

    aclblasComplex* dX = nullptr;
    aclblasComplex* dAp = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&dX), sizeof(x), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&dAp), sizeof(ap), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(dX, sizeof(x), x, sizeof(x), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(dAp, sizeof(ap), ap, sizeof(ap), ACL_MEMCPY_HOST_TO_DEVICE);

    aclblasChpr(handle, ACLBLAS_UPPER, n, &alpha, dX, 1, dAp);
    aclrtSynchronizeDevice();

    aclrtFree(dX);
    aclrtFree(dAp);
    aclblasDestroy(handle);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
```
