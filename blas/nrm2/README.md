# Nrm2算子

## 算子概述

向量欧几里得范数（2-范数）算子，计算向量的尺度规范化值，常用于向量长度计算、归一化和误差估计。

数学表达式：

```text
result = sqrt(Σ |x[i]|²), i = 0 to n-1
```

复数向量（Scnrm2）：

```text
result = sqrt(Σ (|real(z[i])|² + |imag(z[i])|²)), i = 0 to n-1
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSnrm2 | 实数向量欧几里得范数 |
| aclblasScnrm2 | 复数向量欧几里得范数 |
| aclblasSnrm2Ex | 扩展精度欧几里得范数，支持 FP16/FP32 输入 |

## 算子执行接口

### aclblasSnrm2

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSnrm2(aclblasHandle_t handle, int n, const float* x, int incx, float* result);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，内部携带 stream 和 workspace，Host 内存 |
| n | 输入 | int | 向量元素个数，n >= 0（n <= 0 时直接返回 0.0），Host 内存 |
| x | 输入 | const float\*（FP32） | 输入向量，当 n > 0 时不可为 nullptr，Device 内存 |
| incx | 输入 | int | x 中连续元素之间的步长，incx != 0，Host 内存 |
| result | 输出 | float\*（FP32） | 欧几里得范数结果，不可为 nullptr，Device 内存 |

#### 约束说明

- n >= 0
- incx != 0

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

struct AclMemDeleter {
    void operator()(void* p) const { aclrtFree(p); }
};
struct BlasHandleDeleter {
    void operator()(aclblasHandle_t h) const { aclblasDestroy(h); }
};

int aclblasSnrm2Test(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-blas 句柄
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasHandle_t>::type, BlasHandleDeleter> handlePtr(rawHandle);

    blasRet = aclblasSetStream(handlePtr.get(), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 2. 准备 Host 数据
    int n = 5, incx = 1;
    std::vector<float> xHostData = {3.0f, 4.0f, 0.0f, 0.0f, 0.0f};  // sqrt(9+16) = 5
    size_t xBytes = n * sizeof(float);

    // 3. 申请 Device 内存并拷贝数据
    void* rawMemX = nullptr;
    auto aclRet = aclrtMalloc(&rawMemX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for x failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<float, AclMemDeleter> xDevicePtr(static_cast<float*>(rawMemX));

    void* rawMemResult = nullptr;
    aclRet = aclrtMalloc(&rawMemResult, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for result failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<float, AclMemDeleter> resultDevicePtr(static_cast<float*>(rawMemResult));

    aclRet = aclrtMemcpy(xDevicePtr.get(), xBytes, xHostData.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for x failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasSnrm2
    blasRet = aclblasSnrm2(handlePtr.get(), n, xDevicePtr.get(), incx, resultDevicePtr.get());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSnrm2 failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    float result = 0.0f;
    aclRet = aclrtMemcpy(&result, sizeof(float), resultDevicePtr.get(), sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", aclRet);
              return aclRet);
    LOG_PRINT("nrm2 result is: %f\n", result);  // 期望 5.0

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSnrm2Test(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSnrm2Test failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

### aclblasScnrm2

#### 产品支持情况

- Ascend 950PR：支持
- Ascend 950DT：不支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasScnrm2(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, float* result);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，内部携带 stream，Host 内存 |
| n | 输入 | int | 复数元素个数（kernel 内部处理 2\*n 个 float 元素），Host 内存 |
| x | 输入 | const aclblasComplex\* | 复数向量（交错实部/虚部存储，实际为 2\*n 个 float），Device 内存 |
| incx | 输入 | int | x 中连续元素之间的步长，支持正负步长，Host 内存 |
| result | 输出 | float\* | 复数向量的欧几里得范数（FP32 结果），Device 内存 |

#### 约束说明

- n <= 0 时 quick return，result 写 0.0f
- incx != 0

Ascend 950PR 的补充约束与执行行为：

- handle 和 result 不可为空；n > 0 时 x 不可为空，incx 不可为 INT32_MIN。
- 正维度计算使用 handle 绑定的 stream，读取结果前应调用 `aclrtSynchronizeStream`。
- 连续大向量优先使用 `aclnnNorm`；描述符创建、工作区查询或执行失败时记录错误并尝试自定义 kernel。
- `aclnnNorm` 工作区不足时回退到自定义 kernel，不会在未执行计算时返回成功；自定义 kernel 工作区也不足时返回 `ACLBLAS_STATUS_ALLOC_FAILED`。
- 自定义 kernel 采用缩放平方和归约，避免有限输入在中间平方时溢出或下溢；Inf/NaN 单独传播。
- `aclnnNorm` 之后在同一 stream 上检查结果，对零值、极小值和溢出结果进行缩放归约重算，无需 Host 同步。
- n <= 0 的 quick return 使用 Host 到 Device 的零值拷贝，不启动计算 kernel。

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

struct BlasHandleDeleter {
    void operator()(aclblasHandle_t h) const { aclblasDestroy(h); }
};

int aclblasScnrm2Test(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-blas 句柄
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasHandle_t>::type, BlasHandleDeleter> handlePtr(rawHandle);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 2. 准备 Host 数据：复数向量 (3+4i, 0+0i)，期望结果 = sqrt(9+16) = 5
    int64_t n = 2, incx = 1;
    std::vector<float> xHostData = {3.0f, 4.0f, 0.0f, 0.0f};
    size_t xBytes = 2 * n * sizeof(float);

    // 3. 申请 Device 内存并拷贝数据
    void* rawMemX = nullptr;
    auto aclRet = aclrtMalloc(&rawMemX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for x failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> xDevicePtr(rawMemX, aclrtFree);

    void* rawMemResult = nullptr;
    aclRet = aclrtMalloc(&rawMemResult, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for result failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> resultDevicePtr(rawMemResult, aclrtFree);

    aclRet = aclrtMemcpy(xDevicePtr.get(), xBytes, xHostData.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for x failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasScnrm2
    blasRet = aclblasScnrm2(
        static_cast<aclblasHandle_t>(handlePtr.get()), n,
        static_cast<aclblasComplex*>(xDevicePtr.get()), incx,
        static_cast<float*>(resultDevicePtr.get()));
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasScnrm2 failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    float result = 0.0f;
    aclRet = aclrtMemcpy(&result, sizeof(float), resultDevicePtr.get(), sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", aclRet);
              return aclRet);
    LOG_PRINT("cnrm2 result is: %f\n", result);  // 期望 5.0

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasScnrm2Test(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasScnrm2Test failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

### aclblasSnrm2Ex

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSnrm2Ex(aclblasHandle_t handle, aclDataType xtype,
                                const void* x, const int64_t n, const int64_t incx, void* result);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，内部携带 stream 和 workspace，Host 内存 |
| xtype | 输入 | aclDataType | 输入数据类型：ACL_FLOAT 或 ACL_FLOAT16，Host 内存 |
| x | 输入 | const void\* | 输入向量，类型由 xtype 指定，Device 内存 |
| n | 输入 | int64_t | 向量元素个数，0 <= n <= UINT32_MAX，Host 内存 |
| incx | 输入 | int64_t | x 中连续元素之间的步长，incx != 0，Host 内存 |
| result | 输出 | void\* | 欧几里得范数结果（固定 FP32），不可为 nullptr，Device 内存 |

#### 约束说明

- n == 0 时直接返回 result = 0.0f，不触发 kernel
- n > 0 时 x 不可为 nullptr
- incx 不能为 0、INT32_MIN 或 INT64_MIN
- (n-1)\*\|incx\| 不超过 UINT64_MAX
- result 不可为 nullptr
- handle 不可为 nullptr
- 当前仅支持 FP32 和 FP16 输入，输出固定 FP32

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

struct BlasHandleDeleter {
    void operator()(aclblasHandle_t h) const { aclblasDestroy(h); }
};

int aclblasSnrm2ExTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-blas 句柄
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasHandle_t>::type, BlasHandleDeleter> handlePtr(rawHandle);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 2. 准备 Host 数据
    int64_t n = 1024, incx = 1;
    std::vector<float> xHostData(n, 1.0f);  // 全 1.0，期望结果 = sqrt(1024) = 32.0
    size_t xBytes = n * sizeof(float);

    // 3. 申请 Device 内存并拷贝数据
    void* rawMemX = nullptr;
    auto aclRet = aclrtMalloc(&rawMemX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for x failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> xDevicePtr(rawMemX, aclrtFree);

    void* rawMemResult = nullptr;
    aclRet = aclrtMalloc(&rawMemResult, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for result failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> resultDevicePtr(rawMemResult, aclrtFree);

    aclRet = aclrtMemcpy(xDevicePtr.get(), xBytes, xHostData.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy x from host to device failed. ERROR: %d\n", aclRet);
              return aclRet);

    // 4. 调用 aclblasSnrm2Ex（FP32 输入）
    blasRet = aclblasSnrm2Ex(static_cast<aclblasHandle_t>(handlePtr.get()), ACL_FLOAT,
                              xDevicePtr.get(), n, incx, resultDevicePtr.get());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSnrm2Ex failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 5. 同步并取回结果
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    float result = 0.0f;
    aclRet = aclrtMemcpy(&result, sizeof(float), resultDevicePtr.get(), sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", aclRet);
              return aclRet);
    LOG_PRINT("nrm2Ex result is: %f\n", result);  // 期望 32.0

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSnrm2ExTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSnrm2ExTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```
