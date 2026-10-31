# Swap算子

## 算子概述

Swap 算子实现了两个向量对应元素的交换操作，属于纯数据搬运类算子，不涉及任何数值计算。

数学表达式：

```
for i = 0 ... n-1:
    x[kx + i * incx] <-> y[ky + i * incy]
其中 kx = (incx < 0) ? (1 - n) * incx : 0，ky = (incy < 0) ? (1 - n) * incy : 0。
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSswap | 实数向量交换 |
| aclblasCswap | 复数向量交换 |

## 算子执行接口

### aclblasSswap

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSswap(aclblasHandle_t handle, int n, float* x, int incx, float* y, int incy)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 向量中参与交换的元素个数，Host 内存 |
| x | 输入/输出 | float*（FP32） | 指向 float 向量的 device 指针，交换后包含原 y 的元素，Device 内存 |
| incx | 输入 | int | 向量 x 中相邻元素之间的步长，Host 内存 |
| y | 输入/输出 | float*（FP32） | 指向 float 向量的 device 指针，交换后包含原 x 的元素，Device 内存 |
| incy | 输入 | int | 向量 y 中相邻元素之间的步长，Host 内存 |

#### 约束说明

- n == 0 时直接返回成功，不执行任何操作；n < 0 返回 ACLBLAS_STATUS_INVALID_VALUE
- handle 不可为 nullptr
- x、y 不可为 nullptr
- incx != 0, incy != 0

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
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

int aclblasSswapTest(AclContext& ctx)
{
    constexpr int n = 4;
    constexpr int incx = 1;
    constexpr int incy = 1;
    constexpr size_t bytes = n * sizeof(float);

    float hX[n] = {1.0f, 2.0f, 3.0f, 4.0f};
    float hY[n] = {10.0f, 20.0f, 30.0f, 40.0f};

    void *rawX = nullptr;
    auto aclRet = aclrtMalloc(&rawX, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dX(rawX);

    void *rawY = nullptr;
    aclRet = aclrtMalloc(&rawY, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dY(rawY);

    aclRet = aclrtMemcpy(dX.get(), bytes, hX, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dY.get(), bytes, hY, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);
    std::unique_ptr<void, AclblasHandleDeleter> handle(rawHandle);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handle.get()), ctx.Stream());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    blasRet = aclblasSswap(
        static_cast<aclblasHandle_t>(handle.get()), n,
        static_cast<float*>(dX.get()), incx,
        static_cast<float*>(dY.get()), incy);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    aclRet = aclrtSynchronizeStream(ctx.Stream());
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclRet = aclrtMemcpy(hX, bytes, dX.get(), bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(hY, bytes, dY.get(), bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 预期结果：x = {10, 20, 30, 40}，y = {1, 2, 3, 4}
    for (int i = 0; i < n; i++) {
        printf("x[%d] = %f, y[%d] = %f\n", i, hX[i], i, hY[i]);
    }

    return 0;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSswapTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}
```

### aclblasCswap

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasCswap(aclblasHandle_t handle, int n, aclblasComplex* x, int incx, aclblasComplex* y, int incy)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 向量中的复数元素个数，Host 内存 |
| x | 输入/输出 | aclblasComplex* | 指向复数向量的 device 指针，交换后包含原 y 的元素，Device 内存 |
| incx | 输入 | int | x 中连续元素之间的步长，Host 内存，可正可负 |
| y | 输入/输出 | aclblasComplex* | 指向复数向量的 device 指针，交换后包含原 x 的元素，Device 内存 |
| incy | 输入 | int | y 中连续元素之间的步长，Host 内存，可正可负 |

#### 约束说明

- n == 0 时直接返回成功，不执行任何操作；n < 0 返回 ACLBLAS_STATUS_INVALID_VALUE
- n > 0 时 handle、x、y 均不可为 nullptr
- incx != 0, incy != 0
- 负步长按 Netlib BLAS 语义从物理向量尾部开始访问
- 复数实部和虚部按原始位模式交换，不进行浮点计算或类型转换
- 算子在 handle 绑定的 stream 上异步执行，读取结果前需同步该 stream

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
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

int aclblasCswapTest(AclContext& ctx)
{
    constexpr int n = 4;
    constexpr int incx = 1;
    constexpr int incy = 1;
    constexpr size_t bytes = static_cast<size_t>(n) * sizeof(aclblasComplex);

    aclblasComplex hX[n] = {{1.0f, 0.0f}, {2.0f, 0.0f}, {3.0f, 0.0f}, {4.0f, 0.0f}};
    aclblasComplex hY[n] = {{10.0f, 0.0f}, {20.0f, 0.0f}, {30.0f, 0.0f}, {40.0f, 0.0f}};

    void *rawX = nullptr;
    auto aclRet = aclrtMalloc(&rawX, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dX(rawX);

    void *rawY = nullptr;
    aclRet = aclrtMalloc(&rawY, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dY(rawY);

    aclRet = aclrtMemcpy(dX.get(), bytes, hX, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dY.get(), bytes, hY, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);
    std::unique_ptr<void, AclblasHandleDeleter> handle(rawHandle);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handle.get()), ctx.Stream());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    blasRet = aclblasCswap(
        static_cast<aclblasHandle_t>(handle.get()), n,
        static_cast<aclblasComplex*>(dX.get()), incx,
        static_cast<aclblasComplex*>(dY.get()), incy);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    aclRet = aclrtSynchronizeStream(ctx.Stream());
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclRet = aclrtMemcpy(hX, bytes, dX.get(), bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(hY, bytes, dY.get(), bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 预期结果：x = {10,20,30,40}，y = {1,2,3,4}
    for (int i = 0; i < n; i++) {
        printf("x[%d] = (%f, %f), y[%d] = (%f, %f)\n",
               i, hX[i].real, hX[i].imag, i, hY[i].real, hY[i].imag);
    }

    return 0;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasCswapTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}
```
