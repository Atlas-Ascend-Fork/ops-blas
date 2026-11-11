# Symv算子

## 算子概述

symv (Symmetric Matrix-Vector Multiplication) 实现对称矩阵与向量的乘法运算。该算子针对对称矩阵的存储特性进行优化，仅存储上三角或下三角部分，通过对称性推断未存储部分，高效完成矩阵与向量的乘加运算。

数学表达式：

```
y = alpha * A * x + beta * y
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSsymv | 单精度对称矩阵-向量乘法 |
| aclblasCsymv | 单精度复数对称矩阵-向量乘法 |

## 算子执行接口

### aclblasSsymv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSsymv(aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const float* A, int lda, const float* x, int incx, const float* beta, float* y, int incy)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | 指定矩阵 A 存储上三角（ACLBLAS_UPPER）或下三角（ACLBLAS_LOWER），Host 内存 |
| n | 输入 | int | 对称矩阵 A 的行数和列数，Host 内存 |
| alpha | 输入 | const float*（FP32） | 用于乘法的 float 标量，Host/Device 内存 |
| A | 输入 | const float*（FP32） | 对称矩阵 float 数组，维度为 lda x n，仅存储 uplo 指定的三角部分，Device 内存 |
| lda | 输入 | int | 用于存储矩阵 A 的二维数组的主维，lda >= max(1, n)，Host 内存 |
| x | 输入 | const float*（FP32） | float 向量，包含 n 个元素，Device 内存 |
| incx | 输入 | int | x 中连续元素之间的步长，incx != 0，Host 内存 |
| beta | 输入 | const float*（FP32） | 用于乘法的 float 标量。如果 beta == 0，则 y 不必是有效输入，Host/Device 内存 |
| y | 输入/输出 | float*（FP32） | float 向量，包含 n 个元素。输入为初始 y 值，输出为计算结果，Device 内存 |
| incy | 输入 | int | y 中连续元素之间的步长，incy != 0，Host 内存 |

#### 约束说明

- n >= 0
- incx != 0
- incy != 0
- lda >= max(1, n)

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
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
    explicit AclContext(int deviceId) : deviceId_(deviceId) {}

    ~AclContext()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
            deviceSet_ = false;
        }
        if (aclInited_) {
            aclFinalize();
            aclInited_ = false;
        }
    }

    int Init()
    {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        aclInited_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(&stream_);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        return ACL_SUCCESS;
    }

    aclrtStream Stream() const { return stream_; }

private:
    int deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

struct AclrtMemDeleter {
    void operator()(void* ptr) const
    {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
};

struct AclblasHandleDeleter {
    void operator()(aclblasHandle_t handle) const
    {
        if (handle != nullptr) {
            aclblasDestroy(handle);
        }
    }
};

int aclblasSsymvTest(AclContext& ctx)
{
    constexpr int n = 3;
    constexpr int lda = 3;
    constexpr int incx = 1;
    constexpr int incy = 1;
    constexpr size_t aSize = lda * n * sizeof(float);
    constexpr size_t xSize = n * sizeof(float);
    constexpr size_t ySize = n * sizeof(float);
    float alpha = 1.0f;
    float beta = 1.0f;

    // A 按列主序存储，此处存储上三角部分。
    float hA[lda * n] = {
        1.0f, 0.0f, 0.0f,
        2.0f, 5.0f, 0.0f,
        3.0f, 6.0f, 9.0f
    };
    float hX[n] = {1.0f, 2.0f, 3.0f};
    float hY[n] = {10.0f, 20.0f, 30.0f};

    void *rawA = nullptr;
    auto aclRet = aclrtMalloc(&rawA, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dA(rawA);

    void *rawX = nullptr;
    aclRet = aclrtMalloc(&rawX, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dX(rawX);

    void *rawY = nullptr;
    aclRet = aclrtMalloc(&rawY, ySize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dY(rawY);

    aclRet = aclrtMemcpy(dA.get(), aSize, hA, aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dX.get(), xSize, hX, xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dY.get(), ySize, hY, ySize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);
    std::unique_ptr<void, AclblasHandleDeleter> handle(rawHandle);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handle.get()), ctx.Stream());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    blasRet = aclblasSsymv(
        static_cast<aclblasHandle_t>(handle.get()), ACLBLAS_UPPER, n, &alpha,
        static_cast<float*>(dA.get()), lda, static_cast<float*>(dX.get()), incx, &beta,
        static_cast<float*>(dY.get()), incy);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    aclRet = aclrtSynchronizeStream(ctx.Stream());
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclRet = aclrtMemcpy(hY, ySize, dY.get(), ySize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    return 0;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSsymvTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}
```

### aclblasCsymv

#### 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasCsymv(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda, const aclblasComplex* x, int incx,
    const aclblasComplex* beta, aclblasComplex* y, int incy);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 上下文句柄，携带执行 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | `ACLBLAS_UPPER` 使用 A 的上三角，`ACLBLAS_LOWER` 使用下三角，Host 内存 |
| n | 输入 | int | A 的行列数以及 x、y 的逻辑长度，`n>=0`，Host 内存 |
| alpha | 输入 | const aclblasComplex* | 复数乘数，Host/Device 内存 |
| A | 输入 | const aclblasComplex* | 列主序复数对称矩阵，数组大小 `lda*n`，Device 内存 |
| lda | 输入 | int | A 的前导维度，`lda>=max(1,n)`，Host 内存 |
| x | 输入 | const aclblasComplex* | 逻辑长度为 n 的复数向量，Device 内存 |
| incx | 输入 | int | x 的元素步长，支持正负值，`incx!=0`，Host 内存 |
| beta | 输入 | const aclblasComplex* | 复数乘数，Host/Device 内存 |
| y | 输入/输出 | aclblasComplex* | 逻辑长度为 n 的复数向量，原地写回，Device 内存 |
| incy | 输入 | int | y 的元素步长，支持正负值，`incy!=0`，Host 内存 |

`aclblasComplex` 包含 FP32 的 `real`、`imag` 两个分量，数据类型为 `COMPLEX64`。

#### 计算与存储

计算 `y = alpha * A * x + beta * y`，A 满足 `A=A^T`。只读取 `uplo` 指定的三角，另一半按转置关系使用，保留虚部符号。列主序元素地址为 `A[row+col*lda]`；`lda` 超出 n 的部分不参与计算。

`n>0` 时，向量存储长度至少为 `1+(n-1)*abs(inc)` 个复数。传入指针始终指向存储基址；`inc<0` 时，逻辑第一个元素位于 `(n-1)*abs(inc)`，之后按 inc 遍历。输出仅改写 y 的逻辑元素，步长空隙保持原值。

#### A2/A3 workspace

接口直接使用原始矩阵和向量，内部根据输入布局和尺寸选择执行路径。分块计算使用 handle 的现有 workspace 保存中间结果；空间不足时使用 Owner 路径。

#### 参数校验与标量语义

- `handle` 必须有效。`n=0` 直接返回成功，不访问 alpha、beta、A、x、y。
- `n>0` 时校验 uplo、lda、incx、incy，alpha 和 beta 均须有效。
- `alpha=(0,0),beta=(1,0)` 时不访问 A/x/y，三者可为空。
- `alpha=(0,0)` 时不读取 A/x，二者可为空；需要写结果时 y 必须有效。
- `beta=(0,0)` 时不读取 y 的输入值；`beta=(1,0)` 时直接使用旧 y，保留含 Inf 分量的语义。

Host 标量随调用参数传入，Device 标量在 kernel 中读取。数据指针齐全的调用保持异步；任一数据指针为空且标量位于 Device 时，Host 在 handle stream 上读回标量并同步，确认该空指针是否符合上述条件。异步计算完成后，调用方通过同一 stream 同步或建立后续依赖。

#### 返回码

| 场景 | 返回码 |
|---|---|
| `handle==nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| uplo 非法 | `ACLBLAS_STATUS_INVALID_ENUM` |
| n、lda 或步长非法；缺少必要指针；标量地址查询失败 | `ACLBLAS_STATUS_INVALID_VALUE` |
| Device 标量读回或同步失败 | `ACLBLAS_STATUS_INTERNAL_ERROR` |
| 获取计算资源或提交 kernel 失败 | `ACLBLAS_STATUS_EXECUTION_FAILED` |
| 参数合法且完成 quick return 或提交计算 | `ACLBLAS_STATUS_SUCCESS` |

#### 调用示例

编译和执行方法见[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <memory>
#include <type_traits>

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
    explicit AclContext(int deviceId) : deviceId_(deviceId) {}

    ~AclContext()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
            stream_ = nullptr;
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
            deviceSet_ = false;
        }
        if (aclInited_) {
            aclFinalize();
            aclInited_ = false;
        }
    }

    int Init()
    {
        auto ret = aclInit(nullptr);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        aclInited_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(&stream_);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        return ACL_SUCCESS;
    }

    aclrtStream Stream() const { return stream_; }

private:
    int deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

struct AclrtMemDeleter {
    void operator()(void* ptr) const
    {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
};

struct AclblasHandleDeleter {
    void operator()(aclblasHandle_t handle) const
    {
        if (handle != nullptr) {
            aclblasDestroy(handle);
        }
    }
};

int aclblasCsymvTest(AclContext& ctx)
{
    constexpr int n = 3;
    constexpr int lda = 3;
    constexpr int incx = 1;
    constexpr int incy = 1;
    constexpr size_t aSize = lda * n * sizeof(aclblasComplex);
    constexpr size_t xSize = n * sizeof(aclblasComplex);
    constexpr size_t ySize = n * sizeof(aclblasComplex);
    aclblasComplex alpha = {1.0f, 0.0f};
    aclblasComplex beta = {1.0f, 0.0f};

    // A 按列主序存储，此处存储上三角部分。
    aclblasComplex hA[lda * n] = {
        {1.0f, 1.0f}, {0.0f, 0.0f}, {0.0f, 0.0f},
        {2.0f, 1.0f}, {5.0f, 1.0f}, {0.0f, 0.0f},
        {3.0f, 1.0f}, {6.0f, 1.0f}, {9.0f, 1.0f}
    };
    aclblasComplex hX[n] = {{1.0f, 0.0f}, {2.0f, 0.0f}, {3.0f, 0.0f}};
    aclblasComplex hY[n] = {{10.0f, 0.0f}, {20.0f, 0.0f}, {30.0f, 0.0f}};

    void *rawA = nullptr;
    auto aclRet = aclrtMalloc(&rawA, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dA(rawA);

    void *rawX = nullptr;
    aclRet = aclrtMalloc(&rawX, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dX(rawX);

    void *rawY = nullptr;
    aclRet = aclrtMalloc(&rawY, ySize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dY(rawY);

    aclRet = aclrtMemcpy(dA.get(), aSize, hA, aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dX.get(), xSize, hX, xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dY.get(), ySize, hY, ySize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);
    std::unique_ptr<std::remove_pointer_t<aclblasHandle_t>, AclblasHandleDeleter> handle(rawHandle);

    blasRet = aclblasSetStream(handle.get(), ctx.Stream());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    blasRet = aclblasCsymv(
        handle.get(), ACLBLAS_UPPER, n, &alpha,
        static_cast<aclblasComplex*>(dA.get()), lda, static_cast<aclblasComplex*>(dX.get()), incx, &beta,
        static_cast<aclblasComplex*>(dY.get()), incy);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    aclRet = aclrtSynchronizeStream(ctx.Stream());
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclRet = aclrtMemcpy(hY, ySize, dY.get(), ySize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    return 0;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasCsymvTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}
```
