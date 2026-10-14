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
