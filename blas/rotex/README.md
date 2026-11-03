# RotEx算子

## 算子概述

RotEx (Givens Rotation) 算子实现了向量的 Givens 旋转，对向量 x 和 y 的对应元素对应用旋转矩阵。算子只支持对称类型（xType==yType==csType），x/y 可为 FP32/BF16/FP16，计算在 FP32 精度下执行。根据步长模式自动选择 SIMD 连续路径或 SIMT 离散路径。

数学表达式（Fortran 1-based 索引）：

```
x[k] = c * x[k] + s * y[j]
y[j] = -s * x[k] + c * y[j]

k = 1 + (i-1) * incx
j = 1 + (i-1) * incy
  for i = 1, 2, ..., n
```

其中 `c = cos(theta)`、`s = sin(theta)` 为 Givens 旋转的标量参数。

第二个等式中 `y[j]` 的计算使用旋转前的原始 `x[k]` 值（即第一条公式更新前的值），而非更新后的值。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasRotEx | Givens 旋转，支持混合精度（S 组） |

## 算子执行接口

### aclblasRotEx

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasRotEx(
    aclblasHandle_t handle,
    int n,
    void *x,
    aclDataType xType,
    int incx,
    void *y,
    aclDataType yType,
    int incy,
    const void *c,
    const void *s,
    aclDataType csType,
    aclDataType executionType);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 向量 x 和 y 中需旋转的元素数量，n >= 0，Host 内存 |
| x | 输入/输出 | void* | 输入/输出向量 x 的设备指针，按 xType 类型解释，Device 内存 |
| xType | 输入 | aclDataType | 向量 x 的数据类型枚举，S 组支持 FP32 / FP16 / BF16，Host 内存 |
| incx | 输入 | int | x 元素的步长（正数或负数，不可为零），Host 内存 |
| y | 输入/输出 | void* | 输入/输出向量 y 的设备指针，按 yType 类型解释，Device 内存 |
| yType | 输入 | aclDataType | 向量 y 的数据类型枚举，S 组支持 FP32 / FP16 / BF16，Host 内存 |
| incy | 输入 | int | y 元素的步长（正数或负数，不可为零），Host 内存 |
| c | 输入 | const void* | Givens 旋转的余弦值标量，按 csType 解释，Host 或 Device 内存 |
| s | 输入 | const void* | Givens 旋转的正弦值标量，按 csType 解释，Host 或 Device 内存 |
| csType | 输入 | aclDataType | c 和 s 的数据类型枚举，S 组支持 FP32 / FP16 / BF16，Host 内存 |
| executionType | 输入 | aclDataType | 计算精度类型枚举，arch35 仅支持 FP32，Host 内存 |

#### 支持数据类型

| executionType | xType / yType | csType |
|---------------|---------------|--------|
| FP32 | FP32 | FP32 |
| FP32 | FP16 | FP16 |
| FP32 | BF16 | BF16 |

> arch35 芯片仅支持 S 组（executionType=FP32），D 组（FP64）、C 组（C64）和 Z 组（C128）当前在 arch35 上不提供支持，传入对应 executionType 将返回 `ACLBLAS_STATUS_NOT_SUPPORTED`。

#### 约束说明

- `handle != nullptr`，否则返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
- `n >= 0`，否则返回 `ACLBLAS_STATUS_INVALID_VALUE`
- `n == 0` 时直接返回 `ACLBLAS_STATUS_SUCCESS`，不启动 Kernel
- `n > 0` 时 `x != nullptr`、`y != nullptr`、`c != nullptr`、`s != nullptr`，否则返回 `ACLBLAS_STATUS_INVALID_VALUE`
- `incx != 0`，否则返回 `ACLBLAS_STATUS_INVALID_VALUE`
- `incy != 0`，否则返回 `ACLBLAS_STATUS_INVALID_VALUE`
- `executionType` 必须为 FP32（arch35 仅支持 S 组），否则返回 `ACLBLAS_STATUS_NOT_SUPPORTED`
- `xType`、`yType` 和 `csType` 必须相同（仅对称类型），且必须为 FP32 / FP16 / BF16 之一，否则返回 `ACLBLAS_STATUS_NOT_SUPPORTED`
- c/s 标量指针可为 Host 或 Device 内存，运行时自动检测并处理

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

int aclblasRotExTest(AclContext& ctx)
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
    int n = 5;
    int incx = 1;
    int incy = 1;
    aclDataType xType = ACL_FLOAT;
    aclDataType yType = ACL_FLOAT;
    aclDataType csType = ACL_FLOAT;
    aclDataType executionType = ACL_FLOAT;

    std::vector<float> hX = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> hY = {6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    float c = 0.8f;
    float s = 0.6f;
    size_t dataBytes = n * sizeof(float);

    // 3. 申请 Device 内存并拷贝数据
    void* rawMemX = nullptr;
    auto aclRet = aclrtMalloc(&rawMemX, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for x failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> xDevicePtr(rawMemX, aclrtFree);

    void* rawMemY = nullptr;
    aclRet = aclrtMalloc(&rawMemY, dataBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for y failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> yDevicePtr(rawMemY, aclrtFree);

    aclRet = aclrtMemcpy(xDevicePtr.get(), dataBytes, hX.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for x failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(yDevicePtr.get(), dataBytes, hY.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for y failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasRotEx
    blasRet = aclblasRotEx(
        static_cast<aclblasHandle_t>(handlePtr.get()), n,
        xDevicePtr.get(), xType, incx,
        yDevicePtr.get(), yType, incy,
        &c, &s, csType, executionType);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasRotEx failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    std::vector<float> resultX(n, 0);
    std::vector<float> resultY(n, 0);
    aclRet = aclrtMemcpy(resultX.data(), dataBytes, xDevicePtr.get(), dataBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy x result from device to host failed. ERROR: %d\n", aclRet);
              return aclRet);
    aclRet = aclrtMemcpy(resultY.data(), dataBytes, yDevicePtr.get(), dataBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy y result from device to host failed. ERROR: %d\n", aclRet);
              return aclRet);
    for (int i = 0; i < n; i++) {
        LOG_PRINT("resultX[%d] is: %f, resultY[%d] is: %f\n", i, resultX[i], i, resultY[i]);
    }

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasRotExTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasRotExTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```
