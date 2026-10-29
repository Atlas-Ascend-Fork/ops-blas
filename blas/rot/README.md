# Rot算子

## 算子概述

向量旋转算子，实现对两个等长向量的平面旋转（Givens 旋转），常用于 QR 分解、求解线性方程组和特征值计算等数值算法中。rot 算子族支持实数（FP32）与复数（FP32 complex）两种数据类型的 Givens 旋转。

数学表达式：

```
x[i] = c * x[i] + s * y[i]
y[i] = c * y[i] - s * x[i] (使用原始 x[i])
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasCsrot | 复数（FP32 complex）向量平面旋转 |
| aclblasSrot | 单精度（FP32）实数向量平面旋转 |

## 算子执行接口

### aclblasCsrot

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：不支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasCsrot(aclblasHandle_t handle, int n, aclblasComplex* x, int incx, aclblasComplex* y, int incy, const float* c, const float* s)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 向量元素个数，Host 内存 |
| x | 输入/输出 | aclblasComplex*（FP32 complex） | 向量，包含 n 个元素，原地修改，Device 内存 |
| incx | 输入 | int | x 中连续元素之间的步长，Host 内存 |
| y | 输入/输出 | aclblasComplex*（FP32 complex） | 向量，包含 n 个元素，原地修改，Device 内存 |
| incy | 输入 | int | y 中连续元素之间的步长，Host 内存 |
| c | 输入 | const float* | 指向旋转角度余弦值的指针，Host 内存 |
| s | 输入 | const float* | 指向旋转角度正弦值的指针，Host 内存 |

#### 约束说明

- n >= 0

### aclblasSrot

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSrot(
    aclblasHandle_t handle, int n, float* x, int incx, float* y, int incy, const float* c, const float* s);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 参与旋转的元素个数，Host 内存 |
| x | 输入/输出 | float*（FP32） | FP32 向量，in-place 修改，按 incx 步长访问，Device 内存 |
| incx | 输入 | int | 向量 x 中连续元素之间的步长，可为正/负/零，Host 内存 |
| y | 输入/输出 | float*（FP32） | FP32 向量，in-place 修改，按 incy 步长访问，Device 内存 |
| incy | 输入 | int | 向量 y 中连续元素之间的步长，可为正/负/零，Host 内存 |
| c | 输入 | const float* | 指向 Givens 旋转参数 cos(θ) 的标量指针，Host 内存或 Device 内存，不可为 nullptr；算子运行时通过 aclrtPointerGetAttributes 自动判定其内存位置 |
| s | 输入 | const float* | 指向 Givens 旋转参数 sin(θ) 的标量指针，Host 内存或 Device 内存，不可为 nullptr；算子运行时通过 aclrtPointerGetAttributes 自动判定其内存位置 |

#### 约束说明

**Ascend 950PR / Ascend 950DT（arch35）：**

- handle 不能为 nullptr，否则返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR
- n 为整数；n <= 0 时为 no-op（直接返回 ACLBLAS_STATUS_SUCCESS，不修改 x、y，对齐参考 BLAS 的 `IF (N.LE.0) RETURN` 语义）
- n > 0 时 x、y 不能为 nullptr，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- incx、incy 为整数，可为正、负、零，均不拦截：
  - incx == 1 且 incy == 1 时走连续路径
  - 其余 stride 组合（含正非 1、负、零及其混合）走 stride 路径；零 stride 时按参考实现语义对同一元素反复旋转，负 stride 时从向量尾端起算沿负方向步进
- 当 (n-1) * stride 的乘积超出 int32 表示范围时，返回 ACLBLAS_STATUS_INVALID_VALUE，避免 kernel 侧地址偏移溢出
- c、s 可为 host 内存指针或 device 内存指针，指向 cos/sin 标量值，由算子运行时通过 aclrtPointerGetAttributes 自动判定内存位置；c 或 s 为 nullptr 时返回 ACLBLAS_STATUS_INVALID_VALUE
- c、s 指向的标量值可取任意 FP32 值（包括 c == 1 且 s == 0 的单位旋转，不做短路，正常执行旋转公式）
- 精度标准：FP32 单标杆（MARE ≤ 10·2⁻¹³，MERE ≤ 2⁻¹³）

**Atlas A2 / Atlas A3 系列产品：**

- 不支持

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

int aclblasSrotTest(AclContext& ctx)
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
    int n = 4;
    int incx = 1;
    int incy = 1;
    float c = 0.8f;
    float s = 0.6f;
    std::vector<float> xHostData = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> yHostData = {5.0f, 6.0f, 7.0f, 8.0f};
    size_t xBytes = n * sizeof(float);
    size_t yBytes = n * sizeof(float);

    // 3. 申请 Device 内存并拷贝数据
    void* rawXMem = nullptr;
    auto aclRet = aclrtMalloc(&rawXMem, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for x failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<float, AclMemDeleter> xDevicePtr(static_cast<float*>(rawXMem));

    void* rawYMem = nullptr;
    aclRet = aclrtMalloc(&rawYMem, yBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for y failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<float, AclMemDeleter> yDevicePtr(static_cast<float*>(rawYMem));

    aclRet = aclrtMemcpy(xDevicePtr.get(), xBytes, xHostData.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for x failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(yDevicePtr.get(), yBytes, yHostData.data(), yBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for y failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasSrot（c、s 为 Host 指针，原地旋转）
    blasRet = aclblasSrot(handlePtr.get(), n, xDevicePtr.get(), incx, yDevicePtr.get(), incy, &c, &s);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSrot failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    std::vector<float> resultX(n, 0);
    aclRet = aclrtMemcpy(resultX.data(), xBytes, xDevicePtr.get(), xBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy x from device to host failed. ERROR: %d\n", aclRet);
              return aclRet);
    for (int i = 0; i < n; i++) {
        LOG_PRINT("x[%d] is: %f\n", i, resultX[i]);
    }

    std::vector<float> resultY(n, 0);
    aclRet = aclrtMemcpy(resultY.data(), yBytes, yDevicePtr.get(), yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy y from device to host failed. ERROR: %d\n", aclRet);
              return aclRet);
    for (int i = 0; i < n; i++) {
        LOG_PRINT("y[%d] is: %f\n", i, resultY[i]);
    }

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSrotTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSrotTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```