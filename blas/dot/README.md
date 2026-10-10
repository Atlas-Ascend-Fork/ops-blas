# Dot算子

## 算子概述

Dot 算子实现了向量点积运算，核心运算为两个向量的内积累加。包含实数点积（Sdot）和复数点积（Cdotu/Cdotc）两类接口，广泛应用于信号处理、统计学、量子计算和线性代数等领域。

数学表达式：

```
Sdot:   result = x · y = Σ(x[i] * y[i])  for i = 0 to n-1

Cdotu:  result = x · y = Σ(x[i] * y[i])  for i = 0 to n-1        （无共轭）

Cdotc:  result = conj(x) · y = Σ(conj(x[i]) * y[i])  for i = 0 to n-1  （共轭）
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSdot | 实数向量点积 |
| aclblasCdotu | 无共轭复数点积 |
| aclblasCdotc | 共轭复数点积 |

## 算子执行接口

### aclblasSdot

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSdot(aclblasHandle_t handle, int64_t n, const float* x, int64_t incx, const float* y, int64_t incy, float* result);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，内部携带 stream 和 workspace，Host 内存 |
| n | 输入 | int64_t | 向量元素个数，n >= 0，Host 内存 |
| x | 输入 | const float*（FP32） | 实数向量，包含 n 个 float 元素，Device 内存 |
| incx | 输入 | int64_t | 向量 x 的步长，当前仅支持 incx = 1，Host 内存 |
| y | 输入 | const float*（FP32） | 实数向量，包含 n 个 float 元素，Device 内存 |
| incy | 输入 | int64_t | 向量 y 的步长，当前仅支持 incy = 1，Host 内存 |
| result | 输出 | float*（FP32） | 点积结果，包含 1 个 float 元素，Device 内存 |

#### 约束说明

- n >= 0；n == 0 时直接返回成功，result 置 0
- incx != 0，incy != 0
- 当前实现仅支持 incx = 1 且 incy = 1，非单位步长尚未支持
- n > 0 时，x、y、result 不能为 nullptr


#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cstdio>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

class AclContext {
public:
    explicit AclContext(int32_t deviceId) : deviceId_(deviceId) {}

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
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        aclInited_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(&stream_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        return ACL_SUCCESS;
    }

    aclrtStream Stream() const { return stream_; }

private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

int aclblasSdotTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-blas 句柄
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<void, aclblasStatus_t (*)(void*)> handlePtr(rawHandle, aclblasDestroy);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 2. 准备 Host 数据
    int64_t n = 8;
    int64_t incx = 1;
    int64_t incy = 1;
    std::vector<float> hX = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> hY = {8.0f, 7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f};
    size_t bytes = static_cast<size_t>(n) * sizeof(float);

    // 3. 申请 Device 内存并拷贝数据
    void *dX = nullptr, *dY = nullptr, *dResult = nullptr;
    auto aclRet = aclrtMalloc(&dX, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc dX failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dXPtr(dX, aclrtFree);

    aclRet = aclrtMalloc(&dY, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc dY failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dYPtr(dY, aclrtFree);

    aclRet = aclrtMalloc(&dResult, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc dResult failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dResultPtr(dResult, aclrtFree);

    aclRet = aclrtMemcpy(dX, bytes, hX.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D dX failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dY, bytes, hY.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D dY failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasSdot
    blasRet = aclblasSdot(
        static_cast<aclblasHandle_t>(handlePtr.get()), n,
        static_cast<const float*>(dX), incx,
        static_cast<const float*>(dY), incy,
        static_cast<float*>(dResult));
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS,
              LOG_PRINT("aclblasSdot failed. ERROR: %d\n", blasRet); return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    float hResult = 0.0f;
    aclRet = aclrtMemcpy(&hResult, sizeof(float), dResult, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy D2H result failed. ERROR: %d\n", aclRet); return aclRet);
    LOG_PRINT("sdot result = %.4f\n", static_cast<double>(hResult));

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSdotTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSdotTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

### aclblasCdotu

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasCdotu(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, const aclblasComplex* y, int incy, aclblasComplex* result);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，内部携带 stream 和 workspace，Host 内存 |
| n | 输入 | int | 复数向量元素个数，n >= 0，Host 内存 |
| x | 输入 | const aclblasComplex*（复数 FP32） | 复数向量，物理长度 1+(n-1)*\|incx\|，Device 内存，本算子对 x 不取共轭 |
| incx | 输入 | int | 向量 x 的步长（复数元素单位），可正可负，incx != 0，Host 内存 |
| y | 输入 | const aclblasComplex*（复数 FP32） | 复数向量，物理长度 1+(n-1)*\|incy\|，Device 内存 |
| incy | 输入 | int | 向量 y 的步长（复数元素单位），可正可负，incy != 0，Host 内存 |
| result | 输出 | aclblasComplex*（复数 FP32） | 复数点积结果，包含 1 个 aclblasComplex 元素，Device 内存 |

#### 约束说明

> aclblasCdotu 的约束随架构实现不同而有所差异，请按目标产品对应阅读：

**Ascend 950PR / Ascend 950DT（arch35 实现）**

- handle 为 nullptr 时返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR
- n >= 0；n == 0 时直接返回成功，result 置 (0, 0)
- incx == 0 或 incy == 0 时返回 ACLBLAS_STATUS_INVALID_VALUE；支持任意非零步长（含负步长，负步长语义对齐 Netlib cdotu，不视为 no-op）
- result 不能为 nullptr（任意 n，含 n == 0）；n > 0 时，x、y 不能为 nullptr
- incx == 1 且 incy == 1 时，x、y 的 Device 基地址必须 16 字节对齐（由 aclrtMalloc 分配的缓冲区天然满足；不得传入 8 字节对齐的内部指针，如 base + 1 个复数元素），否则返回 ACLBLAS_STATUS_INVALID_VALUE
- 复数乘法语义：result = Σ(x[k] × y[j])，k = 1+(i-1)*incx，j = 1+(i-1)*incy（无共轭）
- 归约累加顺序不保证跨次一致（无确定性计算要求）

**Atlas A2 训练系列产品 / Atlas A2 推理系列产品（arch22 实现）**

- handle 为 nullptr 时返回 ACLBLAS_STATUS_NOT_INITIALIZED
- n <= 0 时视为空操作，向 result 写 (0, 0) 后返回成功
- 仅支持 incx == 1 且 incy == 1，非单位步长返回 ACLBLAS_STATUS_INVALID_VALUE（不支持负步长）
- result 不能为 nullptr；x、y 在 n > 0 时不能为 nullptr（为空返回 ACLBLAS_STATUS_INVALID_VALUE）


#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cstdio>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

#define CHECK_RET(cond, return_expr) \
    do {                             \
        if (!(cond)) {               \
            return_expr;             \
        }                            \
    } while (0)

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

class AclContext {
public:
    explicit AclContext(int32_t deviceId) : deviceId_(deviceId) {}

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
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
        aclInited_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(&stream_);
        CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
        return ACL_SUCCESS;
    }

    aclrtStream Stream() const { return stream_; }

private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

int aclblasCdotuTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-blas 句柄
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<void, aclblasStatus_t (*)(void*)> handlePtr(rawHandle, aclblasDestroy);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 2. 准备 Host 数据
    int n = 4;  // 复数元素个数
    int incx = 1;
    int incy = 1;
    // x = [1+0.5i, 2+1i, 3+1.5i, 4+2i]
    std::vector<aclblasComplex> hX = {{1.0f, 0.5f}, {2.0f, 1.0f}, {3.0f, 1.5f}, {4.0f, 2.0f}};
    // y = [3+2i, 1+0.5i, 2+1i, 1+3i]
    std::vector<aclblasComplex> hY = {{3.0f, 2.0f}, {1.0f, 0.5f}, {2.0f, 1.0f}, {1.0f, 3.0f}};
    size_t inputBytes = static_cast<size_t>(n) * sizeof(aclblasComplex);
    size_t outputBytes = sizeof(aclblasComplex);

    // 3. 申请 Device 内存并拷贝数据
    void *dX = nullptr, *dY = nullptr, *dResult = nullptr;
    auto aclRet = aclrtMalloc(&dX, inputBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc dX failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dXPtr(dX, aclrtFree);

    aclRet = aclrtMalloc(&dY, inputBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc dY failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dYPtr(dY, aclrtFree);

    aclRet = aclrtMalloc(&dResult, outputBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc dResult failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dResultPtr(dResult, aclrtFree);

    aclRet = aclrtMemcpy(dX, inputBytes, hX.data(), inputBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D dX failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dY, inputBytes, hY.data(), inputBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D dY failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasCdotu
    blasRet = aclblasCdotu(
        static_cast<aclblasHandle_t>(handlePtr.get()), n,
        static_cast<const aclblasComplex*>(dX), incx,
        static_cast<const aclblasComplex*>(dY), incy,
        static_cast<aclblasComplex*>(dResult));
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS,
              LOG_PRINT("aclblasCdotu failed. ERROR: %d\n", blasRet); return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    aclblasComplex hResult = {0.0f, 0.0f};
    aclRet = aclrtMemcpy(&hResult, outputBytes, dResult, outputBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy D2H result failed. ERROR: %d\n", aclRet); return aclRet);
    LOG_PRINT("cdotu result = %.4f + %.4fi\n", static_cast<double>(hResult.real), static_cast<double>(hResult.imag));

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasCdotuTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasCdotuTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

### aclblasCdotc

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasCdotc(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, const aclblasComplex* y, int incy, aclblasComplex* result);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，内部携带 stream 和 workspace，Host 内存 |
| n | 输入 | int | 复数向量元素个数，n >= 0，Host 内存 |
| x | 输入 | const aclblasComplex*（复数 FP32） | 复数向量，包含 n 个 aclblasComplex 元素，Device 内存 |
| incx | 输入 | int | 向量 x 的步长（复数元素单位），支持任意非 0 正负步长，Host 内存 |
| y | 输入 | const aclblasComplex*（复数 FP32） | 复数向量，包含 n 个 aclblasComplex 元素，Device 内存 |
| incy | 输入 | int | 向量 y 的步长（复数元素单位），支持任意非 0 正负步长，Host 内存 |
| result | 输出 | aclblasComplex*（复数 FP32） | 复数点积结果，包含 1 个 aclblasComplex 元素，Device 内存 |

#### 约束说明

- n >= 0；n == 0 时直接返回成功，result 置 0
- incx != 0，incy != 0
- Ascend 950 实现支持任意非 0 的 incx/incy（含负步长）
- n > 0 时，x、y、result 不能为 nullptr

#### Ascend 950 实现说明

- 正向单位步长输入采用多核 RegBase 计算，最多使用 48 个 AIV，先计算各核部分和，再完成最终归约。
- 其他支持的步长组合由一个 AIV 分块搬运，并使用 RegBase 向量指令完成复数乘加，支持正、负及混合步长。
- 分块搬运使用双缓冲，并为 RegBase 计算预留必要的 LocalMemory 安全空间。


#### 调用示例

调用方式与 aclblasCdotu 一致，仅需将调用 `aclblasCdotu` 替换为 `aclblasCdotc`。Cdotc 对 x 向量取共轭后再做点积。完整 RAII 框架代码请参考上方 aclblasCdotu 的调用示例。
