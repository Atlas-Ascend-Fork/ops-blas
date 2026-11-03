# Copy算子

## 算子概述

向量拷贝算子，实现 `Y = X` 的向量数据搬移。

数学表达式：

```
Y[i] = X[i],   for i = 0, 1, ..., N-1
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasScopy | 单精度浮点向量拷贝 |
| aclblasCcopy | 单精度复数向量逐位拷贝 |

## 算子执行接口

### aclblasCcopy

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasCcopy(aclblasHandle_t handle, int n, const aclblasComplex* x, int incx, aclblasComplex* y, int incy)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 复数元素个数，Host 内存 |
| x | 输入 | const aclblasComplex*（FP32 complex） | 源向量，只读，Device 内存 |
| incx | 输入 | int | X 的复数元素步长，可正可负，Host 内存 |
| y | 输出 | aclblasComplex*（FP32 complex） | 目标向量；仅逻辑目标元素会被覆盖，Device 内存 |
| incy | 输入 | int | Y 的复数元素步长，可正可负，Host 内存 |

#### 约束说明

- `n >= 0`；`n == 0` 为合法 no-op，此时不访问 handle、x 和 y。
- `n > 0` 时 handle、x 和 y 不可为 nullptr。
- `incx != 0`，`incy != 0`。
- x 和 y 的物理长度分别至少为 `1 + (n - 1) * abs(incx)` 和
  `1 + (n - 1) * abs(incy)` 个复数元素。负步长遵循 BLAS 语义：从物理 span 的高地址逻辑首元素
  向低地址遍历，调用方传入的仍是物理 span 基址。

`aclblasCcopy` 将每个 complex64 的两个 32 位分量按原始位模式复制，因此包括 NaN payload、Inf、
有符号零在内均保持 bit-exact。跨步写入时，`y` 的步长空洞保持调用前内容不变。

函数通过 `handle` 绑定的 stream 异步下发。读取输出前，调用方必须同步对应 stream。

### aclblasScopy

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasScopy(aclblasHandle_t handle, int n, const float *x, int incx, float *y, int incy)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 向量长度（元素个数），Host 内存 |
| x | 输入 | const float*（FP32） | 源向量 X，只读，Device 内存 |
| incx | 输入 | int | X 元素的步长（以 float 元素为单位），不可为 0，Host 内存 |
| y | 输出 | float*（FP32） | 目标向量 Y，可写，Device 内存 |
| incy | 输入 | int | Y 元素的步长（以 float 元素为单位），不可为 0，Host 内存 |

#### 约束说明

- n >= 0（n < 0 时返回 INVALID_VALUE）
- incx != 0
- incy != 0
- x、y 不可为 nullptr

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

struct BlasHandleDeleter {
    void operator()(aclblasHandle_t h) const { aclblasDestroy(h); }
};

int aclblasScopyTest(AclContext& ctx)
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
    std::vector<float> xHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
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

    aclRet = aclrtMemcpy(xDevicePtr.get(), dataBytes, xHostData.data(), dataBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for x failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasScopy
    blasRet = aclblasScopy(
        static_cast<aclblasHandle_t>(handlePtr.get()), n,
        static_cast<const float*>(xDevicePtr.get()), incx,
        static_cast<float*>(yDevicePtr.get()), incy);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasScopy failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    std::vector<float> resultData(n, 0);
    aclRet = aclrtMemcpy(resultData.data(), dataBytes, yDevicePtr.get(), dataBytes, ACL_MEMCPY_DEVICE_TO_HOST);
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

    ret = aclblasScopyTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasScopyTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```
