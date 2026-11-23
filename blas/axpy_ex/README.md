# AxpyEx算子

## 算子概述

AxpyEx（混合精度向量 AXPY）算子实现 BLAS Level-1 向量运算 `y = alpha * x + y` 的扩展版本，支持 FP16/BF16/FP32 数据类型与 FP32 计算精度，并支持跨步（stride）访问与负步长（反向遍历）语义。

数学表达式：

```
y[j] = alpha * x[i] + y[j],    k = 0, 1, ..., n-1

其中：
  i = ix_0 + k * incx
  j = iy_0 + k * incy
  ix_0 = (incx >= 0) ? 0 : (1 - n) * incx
  iy_0 = (incy >= 0) ? 0 : (1 - n) * incy
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasAxpyEx | 混合精度向量 AXPY（alpha=FP32, x/y ∈ {FP16/BF16/FP32}, execution=FP32） |

## 算子执行接口

### aclblasAxpyEx

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasAxpyEx(aclblasHandle_t handle, int n, const void *alpha, aclDataType alphaType, const void *x, aclDataType xType, int incx, void *y, aclDataType yType, int incy, aclDataType executionType)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 向量元素个数，Host 内存 |
| alpha | 输入 | const void* | 标量乘数指针，实际类型由 alphaType 指定（固定 FP32），Host/Device 内存 |
| alphaType | 输入 | aclDataType | alpha 数据类型，固定为 ACL_FLOAT，Host 内存 |
| x | 输入 | const void* | 输入向量 x 指针，类型由 xType 指定，Device 内存 |
| xType | 输入 | aclDataType | 向量 x 的数据类型，支持 ACL_FLOAT、ACL_FLOAT16、ACL_BF16，Host 内存 |
| incx | 输入 | int | 向量 x 中相邻元素的步长（元素间隔，非字节），负值表示反向遍历，Host 内存 |
| y | 输入/输出 | void* | 输入/输出向量 y 指针，类型由 yType 指定，Device 内存 |
| yType | 输入 | aclDataType | 向量 y 的数据类型，支持 ACL_FLOAT、ACL_FLOAT16、ACL_BF16，Host 内存 |
| incy | 输入 | int | 向量 y 中相邻元素的步长（元素间隔，非字节），负值表示反向遍历，Host 内存 |
| executionType | 输入 | aclDataType | 计算精度类型，固定为 ACL_FLOAT（FP32 精度计算），Host 内存 |

#### 约束说明

- n >= 0（n < 0 返回 ACLBLAS_STATUS_INVALID_VALUE；n == 0 为 no-op，直接返回 ACLBLAS_STATUS_SUCCESS）
- incx != 0（incx == 0 返回 ACLBLAS_STATUS_INVALID_VALUE；支持负步长，表示反向遍历）
- incy != 0（incy == 0 返回 ACLBLAS_STATUS_INVALID_VALUE；支持负步长，表示反向遍历）
- alphaType 固定为 ACL_FLOAT（否则返回 ACLBLAS_STATUS_NOT_SUPPORTED）
- executionType 固定为 ACL_FLOAT（否则返回 ACLBLAS_STATUS_NOT_SUPPORTED）
- xType 必须为 ACL_FLOAT、ACL_FLOAT16 或 ACL_BF16（否则返回 ACLBLAS_STATUS_NOT_SUPPORTED）
- yType 必须为 ACL_FLOAT、ACL_FLOAT16 或 ACL_BF16（否则返回 ACLBLAS_STATUS_NOT_SUPPORTED）
- xType == yType（xType != yType 返回 ACLBLAS_STATUS_NOT_SUPPORTED）
- n > 0 时 alpha、x、y 指针均不能为 nullptr（否则返回 ACLBLAS_STATUS_INVALID_VALUE）
- alpha 可位于 Host 或 Device 内存，由算子内部通过 aclrtPointerGetAttributes 自动判断

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

int aclblasAxpyExTest(AclContext& ctx)
{
    constexpr int n = 4;
    constexpr int incx = 1;
    constexpr int incy = 1;
    constexpr size_t bytes = n * sizeof(float);
    float alpha = 2.0f;

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

    blasRet = aclblasAxpyEx(
        static_cast<aclblasHandle_t>(handle.get()), n,
        static_cast<const void*>(&alpha), ACL_FLOAT,
        static_cast<const void*>(dX.get()), ACL_FLOAT, incx,
        static_cast<void*>(dY.get()), ACL_FLOAT, incy,
        ACL_FLOAT);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    aclRet = aclrtSynchronizeStream(ctx.Stream());
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclRet = aclrtMemcpy(hY, bytes, dY.get(), bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 预期结果：y = alpha*x + y = {12, 24, 36, 48}
    for (int i = 0; i < n; i++) {
        printf("y[%d] = %f\n", i, hY[i]);
    }

    return 0;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasAxpyExTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}
```
