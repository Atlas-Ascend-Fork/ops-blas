# Asum算子

## 算子概述

向量运算算子，计算向量元素绝对值之和（L1 范数 / 曼哈顿范数），常用于向量稀疏度度量和误差估计。

数学表达式：

```
result = sum(|x[i]|) for i = 0 to n-1
```

正常计算路径（`n > 0` 且 `incx > 0`）遵循 IEEE 754 非有限值传播：被选元素中任一输入为 NaN 时结果为 NaN；没有 NaN 但包含无穷值，或有限绝对值之和溢出时，结果为正无穷。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSasum | 实数向量绝对值之和 |
| aclblasScasum | 复数（complex64）向量绝对值分量之和 |

## 算子执行接口

### aclblasSasum

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSasum(aclblasHandle_t handle, int n, const float *x, int incx, float *result)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 向量元素个数，Host 内存 |
| x | 输入 | const float*（FP32） | float 向量，包含 n 个元素，Device 内存 |
| incx | 输入 | int | x 中连续元素之间的步长，Host 内存 |
| result | 输出 | float*（FP32） | 向量元素绝对值之和，Device 内存 |

- n <= 0 或 incx <= 0 时，result 被置为 0 并返回成功

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

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

int aclblasSasumTest(AclContext& ctx)
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
    int n = 5;
    int incx = 1;
    std::vector<float> xHostData = {1.0f, -2.0f, 3.0f, -4.0f, 5.0f};  // |x| 之和 = 15
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

    // 4. 调用 aclblasSasum
    blasRet = aclblasSasum(handlePtr.get(), n, xDevicePtr.get(), incx, resultDevicePtr.get());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSasum failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    float result = 0.0f;
    aclRet = aclrtMemcpy(&result, sizeof(float), resultDevicePtr.get(), sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", aclRet);
              return aclRet);
    LOG_PRINT("asum result is: %f\n", result);  // 期望 15.0

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSasumTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSasumTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

**aclblasScasum 调用示例：**

```cpp
#include <cstdio>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

int aclblasScasumTestDemo()
{
    aclblasHandle_t handle = nullptr;
    aclblasStatus_t ret = aclblasCreate(&handle);
    if (ret != ACLBLAS_STATUS_SUCCESS) return ret;

    aclrtStream stream = nullptr;
    aclrtCreateStream(&stream);
    aclblasSetStream(handle, stream);

    int n = 16; int incx = 1;
    std::vector<aclblasComplex> xHostData(n);
    for (int k = 0; k < n; ++k) {
        xHostData[k].real = static_cast<float>(k + 1);
        xHostData[k].imag = static_cast<float>(-(k + 1));
    }
    size_t xBytes = n * sizeof(aclblasComplex);

    aclblasComplex* dX = nullptr;
    float* dR = nullptr;
    aclrtMalloc(reinterpret_cast<void**>(&dX), xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void**>(&dR), sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(dX, xBytes, xHostData.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    ret = aclblasScasum(handle, n, dX, incx, dR);

    float result = 0.0f;
    aclrtSynchronizeStream(stream);
    aclrtMemcpy(&result, sizeof(float), dR, sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    printf("scasum result: %f\n", result);

    aclrtFree(dX); aclrtFree(dR); aclrtDestroyStream(stream); aclblasDestroy(handle);
    return ACL_SUCCESS;
}
```

### aclblasScasum

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasScasum(aclblasHandle_t handle, int n, const aclblasComplex *x, int incx, float *result)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 复数向量元素个数，Host 内存 |
| x | 输入 | const aclblasComplex*（COMPLEX64） | 复数向量，实部虚部交错存储，包含 n 个元素，Device 内存 |
| incx | 输入 | int | x 中相邻复数元素之间的步长（以复数元素为单位），Host 内存 |
| result | 输出 | float*（FP32） | 各元素绝对值分量之和 Σ(\|Re\|+\|Im\|)，Device 内存 |

- Atlas A2/A3（arch22）与 Ascend 950PR/950DT（arch35）：`n <= 0` 或 `incx <= 0` 时，`result` 置为 0.0f 并返回成功（本算子实现遵循 cuBLAS/Netlib `scasum` 语义）。
- 跨度上限：`2 × (n-1) × |incx| > UINT32_MAX` 时返回 `ACLBLAS_STATUS_INVALID_VALUE`
- `n > 0` 且 `incx > 0` 时，计算异步提交到 handle 绑定的 stream。调用方应在其他 stream 消费 result 前建立依赖；从 Host 读取 result 前应等待 handle stream 完成。
