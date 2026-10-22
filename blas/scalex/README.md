# Scalex算子

## 算子概述

Scalex（混合精度向量标量乘）算子实现 `x[j] = alpha * x[j]`，其中 `j = (i - 1) * incx`，`i = 1, 2, ..., n`，支持混合精度计算。

数学表达式：

```
x[j] = alpha * x[j], j = (i - 1) * incx, i = 1, 2, ..., n
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasScalex | 混合精度向量标量乘（alpha=FP32, x in {FP16/BF16/FP32}, execution=FP32） |

## 算子执行接口

### aclblasScalex

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasScalex(aclblasHandle_t handle, int n, const void *alpha, aclDataType alphaType, void *x, aclDataType xType, int incx, aclDataType executionType)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 向量元素个数，Host 内存 |
| alpha | 输入 | const void* | 标量因子指针，实际类型由 alphaType 指定，Host/Device 内存 |
| alphaType | 输入 | aclDataType | alpha 数据类型，固定为 ACL_FLOAT，Host 内存 |
| x | 输入/输出 | void* | Device 端向量指针，类型由 xType 指定，Device 内存 |
| xType | 输入 | aclDataType | 向量 x 的数据类型，Host 内存 |
| incx | 输入 | int | x 中相邻元素的步长，Host 内存 |
| executionType | 输入 | aclDataType | 计算精度类型，固定为 ACL_FLOAT，Host 内存 |

#### 约束说明

- n <= 0 时为 no-op（直接返回 ACLBLAS_STATUS_SUCCESS，不修改 x）
- incx 为整数；incx <= 0 时为 no-op（不修改 x，直接返回 ACLBLAS_STATUS_SUCCESS，对齐 aclblasSscal / 参考 BLAS cblas_sscal 的 IF (INCX.LE.0) RETURN 语义）
- alphaType 固定为 ACL_FLOAT
- executionType 固定为 ACL_FLOAT
- xType 必须为 ACL_FLOAT、ACL_FLOAT16 或 ACL_BF16


#### 调用示例

示例代码如下（以 `xType = ACL_FLOAT` 为例；同样适用于 `ACL_FLOAT16` / `ACL_BF16`），仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

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

int aclblasScalexTest(AclContext& ctx)
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
    float alpha = 2.0f;
    aclDataType alphaType = ACL_FLOAT;
    aclDataType xType = ACL_FLOAT;          // x 可为 ACL_FLOAT / ACL_FLOAT16 / ACL_BF16
    aclDataType executionType = ACL_FLOAT;  // executionType 固定为 ACL_FLOAT
    std::vector<float> xHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};  // 缩放后期望 {2,4,6,8,10}
    size_t xBytes = n * sizeof(float);

    // 3. 申请 Device 内存并拷贝数据
    void* rawMem = nullptr;
    auto aclRet = aclrtMalloc(&rawMem, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for x failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, AclMemDeleter> xDevicePtr(rawMem);

    aclRet = aclrtMemcpy(xDevicePtr.get(), xBytes, xHostData.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for x failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasScalex（原地缩放）
    blasRet = aclblasScalex(handlePtr.get(), n, &alpha, alphaType, xDevicePtr.get(), xType, incx, executionType);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasScalex failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    std::vector<float> resultData(n, 0);
    aclRet = aclrtMemcpy(resultData.data(), xBytes, xDevicePtr.get(), xBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", aclRet);
              return aclRet);
    for (int i = 0; i < n; i++) {
        LOG_PRINT("result[%d] is: %f\n", i, resultData[i]);
    }

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasScalexTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasScalexTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```
