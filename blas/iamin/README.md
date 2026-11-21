# Iamin算子

## 算子概述

Iamin 算子实现了查找向量中绝对值最小元素的索引，核心运算为遍历向量取绝对值并比较大小。该算子返回 1-based 索引，遵循 BLAS 惯例，常用于稀疏矩阵预处理和数值优化中。

数学表达式：

```
result = 1 + argmin_i |x[i]|
       （返回 1-based 索引；复数接口按 |Re(x[i])| + |Im(x[i])| 比较）
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasIsamin | 查找 FP32 向量中绝对值最小元素的 1-based 索引 |
| aclblasIcamin | 查找 COMPLEX64 向量中 1-范数模（\|Re\|+\|Im\|）最小元素的 1-based 索引 |

## 算子执行接口

### aclblasIsamin

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasIsamin(aclblasHandle_t handle, int n, const float *x, int incx, int *result)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 向量元素个数，Host 内存 |
| x | 输入 | const float* | 指向 float 向量的 device 指针，Device 内存 |
| incx | 输入 | int | 向量 x 中相邻元素之间的步长，Host 内存 |
| result | 输出 | int* | 绝对值最小元素的 1-based 索引，Device 内存 |

#### 约束说明

- n < 0 时返回 ACLBLAS_STATUS_INVALID_VALUE
- n = 0 或 incx < 1 时 result 写 0，返回 ACLBLAS_STATUS_SUCCESS
- 对 Ascend 950PR / 950DT 的正常计算（n > 0 且 incx >= 1），要求 `(n - 1) * incx <= INT32_MAX`；超过时返回 `ACLBLAS_STATUS_INVALID_VALUE`，不会下发 kernel
- handle 不可为 nullptr
- x、result 不可为 nullptr
- 当多个元素绝对值相同时，返回索引最小的元素

### aclblasIcamin

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasIcamin(aclblasHandle_t handle, int n, const aclblasComplex *x, int incx, int *result)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 向量元素个数，Host 内存 |
| x | 输入 | const aclblasComplex* | 指向 COMPLEX64 向量的 device 指针，Device 内存 |
| incx | 输入 | int | 向量 x 中相邻元素之间的步长，Host 内存 |
| result | 输出 | int* | \|Re\|+\|Im\| 最小元素的 1-based 索引，Device 内存 |

#### 约束说明

- n < 0 时返回 ACLBLAS_STATUS_INVALID_VALUE
- n = 0 或 incx < 1 时 result 写 0，返回 ACLBLAS_STATUS_SUCCESS
- handle 不可为 nullptr
- x、result 不可为 nullptr
- 当 `(n-1)*incx` 超出 int32 可表示范围时返回 ACLBLAS_STATUS_INVALID_VALUE
- 模值比较采用严格小于（与 cblas `icamin` 一致）：NaN 模永不更新结果；首元素模为 NaN 时结果为 1
- 当多个元素模值相同时，返回索引最小的元素

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

int aclblasIsaminTest(AclContext& ctx)
{
    constexpr int n = 5;
    constexpr int incx = 1;
    constexpr size_t xBytes = n * sizeof(float);

    float hX[n] = {1.0f, -5.0f, 3.0f, -4.0f, 2.0f};

    void *rawX = nullptr;
    auto aclRet = aclrtMalloc(&rawX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dX(rawX);

    void *rawResult = nullptr;
    aclRet = aclrtMalloc(&rawResult, sizeof(int), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dResult(rawResult);

    aclRet = aclrtMemcpy(dX.get(), xBytes, hX, xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);
    std::unique_ptr<void, AclblasHandleDeleter> handle(rawHandle);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handle.get()), ctx.Stream());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    blasRet = aclblasIsamin(
        static_cast<aclblasHandle_t>(handle.get()), n,
        static_cast<const float*>(dX.get()), incx,
        static_cast<int*>(dResult.get()));
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    aclRet = aclrtSynchronizeStream(ctx.Stream());
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    int result = 0;
    aclRet = aclrtMemcpy(&result, sizeof(int), dResult.get(), sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 预期结果：1（1-based 索引，|1| 最小）
    printf("isamin result = %d\n", result);

    return 0;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasIsaminTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}
```
