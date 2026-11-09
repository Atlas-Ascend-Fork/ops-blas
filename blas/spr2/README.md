# Sspr2算子

## 算子概述

Sspr2 算子实现了对称矩阵 packed 格式的秩-2更新操作，核心运算为：`A := alpha * x * y^T + alpha * y * x^T + A`，其中 A 为 n×n 对称矩阵，以 packed 列优先格式存储，仅上三角或下三角区域被引用和更新。

数学表达式：

```text
A := alpha * x * y^T + alpha * y * x^T + A
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSspr2 | 单精度对称矩阵 packed 格式秩-2更新 |
| aclblasChpr2 | 单精度复数 Hermitian 矩阵 packed 格式秩-2更新 |

## 算子执行接口

### aclblasSspr2

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持（arch35 既有实现，与 arch22 公共算法解耦）
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持（arch22，Ascend 910_93，与 A2 共享同一份实现）
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持（arch22，Ascend 910B3/B4，已在 910B3 真机验证）

#### 函数原型

```cpp
aclblasStatus_t aclblasSspr2(aclblasHandle_t handle,
                              aclblasFillMode_t uplo,
                              int n,
                              const float *alpha,
                              const float *x, int incx,
                              const float *y, int incy,
                              float *ap);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | 指定更新的三角区域：ACLBLAS_UPPER(121) 或 ACLBLAS_LOWER(122)，Host 内存 |
| n | 输入 | int | 矩阵阶数，n >= 0，Host 内存 |
| alpha | 输入 | const float\*（FP32） | 标量乘数指针，Host 内存 |
| x | 输入 | const float\*（FP32） | 输入向量，长度至少 1 + (n-1)\*\|incx\|，Device 内存 |
| incx | 输入 | int | x 的元素间步长，incx != 0 且 incx != INT_MIN，Host 内存 |
| y | 输入 | const float\*（FP32） | 输入向量，长度至少 1 + (n-1)\*\|incy\|，Device 内存 |
| incy | 输入 | int | y 的元素间步长，incy != 0 且 incy != INT_MIN，Host 内存 |
| ap | 输入/输出 | float\*（FP32） | packed 对称矩阵，长度 n*(n+1)/2，Device 内存 |

#### 约束说明

- n >= 0 且 n <= 4096，n == 0 时直接返回成功
- incx != 0 且 incx != INT_MIN
- incy != 0 且 incy != INT_MIN
- x/y 向量长度上限：1 + (n-1)*|incx| <= 12288 且 1 + (n-1)*|incy| <= 12288，超出返回 ACLBLAS_STATUS_INVALID_VALUE
- 算子输入 shape 为 [n]、[n]，输出 shape 为 [n*(n+1)/2]
- Host 侧不做流同步，调用方需自行管理同步

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
#include "acl/acl.h"
#include "cann_ops_blas.h"

int main()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    aclblasHandle_t handle = nullptr;
    aclblasCreate(&handle);
    aclblasSetStream(handle, stream);

    int n = 4;
    float alpha = 1.0f;
    int incx = 1, incy = 1;

    float xHost[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float yHost[] = {1.0f, 1.0f, 1.0f, 1.0f};

    size_t size = n * (n + 1) / 2 * sizeof(float);
    void *xDev = nullptr, *yDev = nullptr, *apDev = nullptr;
    aclrtMalloc(&xDev, n * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&yDev, n * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&apDev, size, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(xDev, n * sizeof(float), xHost, n * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(yDev, n * sizeof(float), yHost, n * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemsetAsync(apDev, size, 0, size, stream);

    aclblasSspr2(handle, ACLBLAS_UPPER, n, &alpha,
                 static_cast<const float *>(xDev), incx,
                 static_cast<const float *>(yDev), incy,
                 static_cast<float *>(apDev));

    aclrtSynchronizeStream(stream);

    aclrtFree(xDev);
    aclrtFree(yDev);
    aclrtFree(apDev);

    aclblasDestroy(handle);
    aclrtDestroyStream(stream);

    aclrtResetDevice(0);
    aclFinalize();

    return 0;
}
```

### aclblasChpr2

#### 产品支持情况

| 产品 | 支持情况 |
| --- | --- |
| Ascend 950PR | 支持（arch35） |
| Ascend 950DT | 支持（arch35） |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | 不支持 |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | 不支持 |

#### 函数原型

```cpp
aclblasStatus_t aclblasChpr2(aclblasHandle_t handle, aclblasFillMode_t uplo, const int n, const aclblasComplex* alpha, const aclblasComplex* x, const int incx, const aclblasComplex* y, const int incy, aclblasComplex* ap)
```

计算 `A = A + alpha * x * y^H + conj(alpha) * y * x^H`，其中 `H` 表示共轭转置。
`aclblasComplex` 的实部和虚部均为 float32。仅更新指定三角的 packed 数据，结果原地写入 `ap`。

#### 参数说明

| 参数 | 位置 | 说明 |
| --- | --- | --- |
| handle | Host | 有效库句柄，使用其绑定的 stream |
| uplo | Host | `ACLBLAS_UPPER` 或 `ACLBLAS_LOWER` |
| n | Host | 矩阵阶数，`n >= 0` |
| alpha | Host | 指向一个 `aclblasComplex` 标量，不支持 Device 标量指针 |
| x、y | Device | 只读复数向量，分配长度分别至少为 `1 + (n-1)*abs(incx)`、`1 + (n-1)*abs(incy)` |
| incx、incy | Host | 复数元素单位的步长，非零且不为 `INT_MIN` |
| ap | Device | 输入/输出复数 packed 数组，长度至少为 `n*(n+1)/2` |

#### 约束说明

- 使用 0-based 下标，上三角 `A(i,j)` 的 packed 下标为 `i+j*(j+1)/2`（`i<=j`）；下三角为 `i+(2*n-j-1)*j/2`（`i>=j`）。没有 `lda` 参数。
- 负步长时，逻辑元素 `k` 从物理下标 `(1-n)*inc+k*inc` 读取；正步长时下标为 `k*inc`。
- 非 quick return 时，对角线的输入虚部不参与计算，输出虚部置零。`n=0` 或合法零 alpha 调用完全不修改 `ap`。
- 检查顺序为：空 handle → 负 n → n=0 quick return → 空 alpha → 非法 uplo → 非法 incx → 非法 incy → 空 x/y/ap → 零 alpha quick return。空 handle 返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`，其他非法参数返回 `ACLBLAS_STATUS_INVALID_VALUE`。特别地，`n>0` 时零 alpha 不能掩盖空 x/y/ap。
- 不支持广播。调用方保证 Device 缓冲区长度有效且 `ap` 不与输入向量发生有害重叠。
- 接口异步执行；复制结果到 Host 或释放 Device 缓冲区前须同步 handle 绑定的 stream。

#### 调用示例

以下示例使用上三角 packed 存储，编译和运行方法参见[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

```cpp
#include "acl/acl.h"
#include "cann_ops_blas.h"

int main()
{
    aclInit(nullptr);
    aclrtSetDevice(0);
    aclrtStream stream;
    aclrtCreateStream(&stream);
    aclblasHandle_t handle = nullptr;
    aclblasCreate(&handle);
    aclblasSetStream(handle, stream);

    const int n = 2;
    const aclblasComplex alpha = {1.0f, 0.0f};
    aclblasComplex x[] = {{1.0f, 0.0f}, {2.0f, 0.0f}};
    aclblasComplex y[] = {{0.0f, 1.0f}, {1.0f, 0.0f}};
    aclblasComplex ap[] = {{1.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f}};
    void *xDev = nullptr, *yDev = nullptr, *apDev = nullptr;
    aclrtMalloc(&xDev, sizeof(x), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&yDev, sizeof(y), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&apDev, sizeof(ap), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(xDev, sizeof(x), x, sizeof(x), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(yDev, sizeof(y), y, sizeof(y), ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(apDev, sizeof(ap), ap, sizeof(ap), ACL_MEMCPY_HOST_TO_DEVICE);
    aclblasChpr2(handle, ACLBLAS_UPPER, n, &alpha, static_cast<const aclblasComplex *>(xDev), 1,
                 static_cast<const aclblasComplex *>(yDev), 1, static_cast<aclblasComplex *>(apDev));
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(ap, sizeof(ap), apDev, sizeof(ap), ACL_MEMCPY_DEVICE_TO_HOST);
    aclrtFree(xDev);
    aclrtFree(yDev);
    aclrtFree(apDev);
    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    aclrtResetDevice(0);
    aclFinalize();
    return 0;
}
```

#### 实现与测试

实现位于 `blas/spr2/arch35/chpr2_host.cpp` 和 `chpr2_kernel.cpp`。各核处理对齐的连续 packed 区间，SIMT 各线程更新独立复数元素。整数列定位与 64 位偏移支持上/下三角和正/负步长；尾块使用精确元素访问，不额外读写分配范围外的数据。

编译和运行方法见 `test/spr2/chpr2/arch35/README.md`。精度对标 Netlib `cblas_chpr2`，实部、虚部分别按任务 FLOAT32 混合容差判定；性能以 msprof op 的 task duration 与任务 GPU 基线比较。
