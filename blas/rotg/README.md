# rotg算子

## 算子概述

rotg 算子实现了 Givens 旋转参数的构造，根据输入的两个标量 a、b 计算平面旋转矩阵的余弦 c 和正弦 s，使得旋转后 2×1 向量 (a, b)ᵀ 的第二个元素置零。输出时标量 a 被覆写为旋转后的范数 r，b 被覆写为恢复参数 z，常用于 QR 分解、最小二乘求解等数值算法中，其输出 c、s 可直接作为下游 rot 算子的输入。

数学表达式：

```
| c   s |   | a |   | r |
| -s  c | * | b | = | 0 |
```

其中 r = ±√(a² + b²)，算法采用 scale = |a| + |b| 进行缩放以避免溢出。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSrotg | 单精度构造 Givens 旋转参数 |
| aclblasCrotg | 单精度复数构造 Givens 旋转参数 |

## 目录结构介绍

```
blas/rotg/
├── README.md                       // 说明文档
├── arch22/
│   ├── srotg_host.cpp              // Host 侧入口：参数校验 + Host/Device 双路径判别 + CPU 计算
│   ├── srotg_kernel.cpp            // Device kernel：标量 Givens 旋转参数生成
│   ├── srotg_tiling_data.h         // Tiling 数据结构
│   ├── crotg_host.cpp              // Host 侧入口：复数 Givens 旋转参数构造
│   ├── crotg_kernel.cpp            // Device kernel：复数 Givens 旋转参数生成
│   └── crotg_tiling_data.h         // Tiling 数据结构
└── arch35/
    ├── srotg_host.cpp              // Host 侧入口：参数校验 + Host/Device 双路径判别 + CPU 计算
    ├── srotg_kernel.cpp            // Device SIMT kernel：标量 Givens 旋转参数生成
    ├── srotg_kernel.h              // srotg_kernel_do 声明
    ├── crotg_common.h              // CROTG_ALGO 宏：host/SIMT 单一算法源
    ├── crotg_host.cpp              // Host 侧入口：复数 Givens 旋转参数构造
    ├── crotg_kernel.cpp            // Device SIMT kernel：复数 Givens 旋转参数生成
    └── crotg_kernel.h              // crotg_kernel_do 声明

test/rotg/srotg/
├── CMakeLists.txt
├── srotg_param.h                   // CSV 用例参数结构
├── srotg_golden.h                  // CPU golden（cblas_srotg）
├── arch22/
│   ├── srotg_test.cpp              // GTest：arch22 测试用例
│   ├── srotg_test.csv              // 用例数据
│   └── srotg_npu_wrapper.h         // NPU device 指针路径 wrapper
└── arch35/
    ├── srotg_test.cpp              // GTest：错误路径 + Device/Host 双路径 CSV 驱动
    ├── srotg_test.csv              // 用例数据（含 NaN/Inf/denormal）
    └── srotg_npu_wrapper.h         // NPU device 指针路径 wrapper

test/rotg/crotg/
├── CMakeLists.txt
├── crotg_param.h                   // CSV 用例参数结构（复数 a/b + iters）
├── crotg_golden.h                  // CPU golden（Netlib crotg，double 高精度）
├── arch22/
│   ├── CrotgArch22Test.cpp         // GTest：arch22 测试用例
│   ├── crotg_test.csv              // 用例数据
│   └── crotg_npu_wrapper.h         // NPU device 指针路径 wrapper
└── arch35/
    ├── crotg_test.cpp              // GTest：错误路径 + Device/Host 双路径 + 数学性质验证 + 性能
    ├── crotg_test.csv              // 用例数据（CI 冒烟加载前 50 条）
    ├── crotg_test_full.csv         // 全量用例数据归档（1202 条）
    ├── gpu_baseline.csv            // GPU 性能基线数据
    └── crotg_npu_wrapper.h         // NPU device 指针路径 wrapper
```

## 算子执行接口

### aclblasSrotg

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSrotg(aclblasHandle_t handle, float *a, float *b, float *c, float *s);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| a | 输入/输出 | float*（FP32） | 输入标量 a，运算后覆写为 r = ±√(a²+b²)，Device 内存 |
| b | 输入/输出 | float*（FP32） | 输入标量 b，运算后覆写为恢复参数 z，Device 内存 |
| c | 输出 | float*（FP32） | Givens 旋转矩阵余弦元素，Device 内存 |
| s | 输出 | float*（FP32） | Givens 旋转矩阵正弦元素，Device 内存 |

#### 约束说明

- a、b、c、s 不能为 nullptr，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- a、b、c、s 必须全部位于 Host 侧或全部位于 Device 侧；混合 Host/Device 指针将返回 ACLBLAS_STATUS_INVALID_VALUE
- Host 路径：直接 CPU 计算，无 kernel 开销，无数据搬运
- Device 路径：启动 device kernel（1 block SIMT），计算和结果都留在 device 上
- 算子输入 shape 为标量（各 1 个元素），输出 shape 同为标量
- Host 侧不做流同步，调用方需自行管理同步

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。


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

int aclblasSrotgTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-blas 句柄并绑定 stream
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasHandle_t>::type, BlasHandleDeleter> handlePtr(rawHandle);

    blasRet = aclblasSetStream(handlePtr.get(), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 2. 准备 Host 数据（输入标量 a、b；rotg 会把 a 覆写为 r、b 覆写为恢复参数 z）
    float aHost = 3.0f;
    float bHost = 4.0f;  // 期望 r=5.0, c=0.6, s=0.8, z≈1.6667
    size_t scalarBytes = sizeof(float);

    // 3. 申请 Device 内存（a/b/c/s 四个标量指针必须同侧，此处全部位于 Device）
    void* rawMemA = nullptr;
    void* rawMemB = nullptr;
    void* rawMemC = nullptr;
    void* rawMemS = nullptr;
    auto aclRet = aclrtMalloc(&rawMemA, scalarBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for a failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<float, AclMemDeleter> aDevicePtr(static_cast<float*>(rawMemA));
    aclRet = aclrtMalloc(&rawMemB, scalarBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for b failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<float, AclMemDeleter> bDevicePtr(static_cast<float*>(rawMemB));
    aclRet = aclrtMalloc(&rawMemC, scalarBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for c failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<float, AclMemDeleter> cDevicePtr(static_cast<float*>(rawMemC));
    aclRet = aclrtMalloc(&rawMemS, scalarBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for s failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<float, AclMemDeleter> sDevicePtr(static_cast<float*>(rawMemS));

    aclRet = aclrtMemcpy(aDevicePtr.get(), scalarBytes, &aHost, scalarBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for a failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(bDevicePtr.get(), scalarBytes, &bHost, scalarBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for b failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasSrotg（a/b 原地覆写为 r/z，c/s 输出 Givens 旋转矩阵的余弦/正弦）
    blasRet = aclblasSrotg(handlePtr.get(), aDevicePtr.get(), bDevicePtr.get(),
                           cDevicePtr.get(), sDevicePtr.get());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSrotg failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    float rHost = 0.0f;
    float zHost = 0.0f;
    float cHost = 0.0f;
    float sHost = 0.0f;
    aclRet = aclrtMemcpy(&rHost, scalarBytes, aDevicePtr.get(), scalarBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy r from device to host failed. ERROR: %d\n", aclRet);
              return aclRet);
    aclRet = aclrtMemcpy(&zHost, scalarBytes, bDevicePtr.get(), scalarBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy z from device to host failed. ERROR: %d\n", aclRet);
              return aclRet);
    aclRet = aclrtMemcpy(&cHost, scalarBytes, cDevicePtr.get(), scalarBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy c from device to host failed. ERROR: %d\n", aclRet);
              return aclRet);
    aclRet = aclrtMemcpy(&sHost, scalarBytes, sDevicePtr.get(), scalarBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy s from device to host failed. ERROR: %d\n", aclRet);
              return aclRet);
    LOG_PRINT("rotg result: r=%f, z=%f, c=%f, s=%f\n", rHost, zHost, cHost, sHost);
    // 期望：r=5.000000, z≈1.666667, c=0.600000, s=0.800000

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSrotgTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSrotgTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

### aclblasCrotg

#### 产品支持情况

- Ascend 950PR：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasCrotg(
    aclblasHandle_t handle, aclblasComplex* a, aclblasComplex* b, float* c, aclblasComplex* s);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| a | 输入/输出 | aclblasComplex*（COMPLEX64） | 输入复数标量 a，运算后覆写为 r = sgn(a)·sqrt(\|a\|²+\|b\|²) |
| b | 输入 | aclblasComplex*（COMPLEX64） | 输入复数标量 b，只读不覆写 |
| c | 输出 | float*（FP32） | Givens 旋转矩阵余弦元素，实数 |
| s | 输出 | aclblasComplex*（COMPLEX64） | Givens 旋转矩阵正弦元素，复数 |

#### 数学定义

```
[  c         s ] [ a ] = [ r ]
[ -conjg(s)  c ] [ b ]   [ 0 ]
c = |a|/sqrt(|a|²+|b|²)（实数）
s = sgn(a)·conjg(b)/sqrt(|a|²+|b|²)
r = sgn(a)·sqrt(|a|²+|b|²)，sgn(a) = a/|a|（a=0 时取 1）
```

复数模 |a|、|b| 以及 sqrt(|a|²+|b|²) 均由 hypotf 计算（hypotf 内部带防溢出缩放），保证大数不溢出、小数不下溢；而 c = |a|/sqrt(...)、sgn(a) = a/|a| 等是在上述模基础上的归一化（浮点/复数除法），不属于复数模计算，故不使用 hypotf。

#### 约束说明

- a、b、c、s 均不可为 nullptr，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- handle 为 nullptr 返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR
- a、b、c、s 须全部位于 Host 侧或全部位于 Device 侧，混合返回 ACLBLAS_STATUS_INVALID_VALUE
- 数据类型：a、b、s 为 COMPLEX64，c 为 FLOAT32

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

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

int aclblasCrotgTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-blas 句柄并绑定 stream
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasHandle_t>::type, BlasHandleDeleter> handlePtr(rawHandle);

    blasRet = aclblasSetStream(handlePtr.get(), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 2. 准备 Host 数据（输入复数标量 a、b；rotg 会把 a 覆写为 r，c/s 输出 Givens 旋转矩阵的余弦/正弦）
    aclblasComplex aHost = {3.0f, 4.0f};
    aclblasComplex bHost = {1.0f, 2.0f};
    size_t complexBytes = sizeof(aclblasComplex);
    size_t floatBytes = sizeof(float);

    // 3. 申请 Device 内存（a/b/c/s 四个标量指针必须同侧，此处全部位于 Device）
    void* rawMemA = nullptr;
    void* rawMemB = nullptr;
    void* rawMemC = nullptr;
    void* rawMemS = nullptr;
    auto aclRet = aclrtMalloc(&rawMemA, complexBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for a failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<aclblasComplex, AclMemDeleter> aDevicePtr(static_cast<aclblasComplex*>(rawMemA));
    aclRet = aclrtMalloc(&rawMemB, complexBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for b failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<aclblasComplex, AclMemDeleter> bDevicePtr(static_cast<aclblasComplex*>(rawMemB));
    aclRet = aclrtMalloc(&rawMemC, floatBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for c failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<float, AclMemDeleter> cDevicePtr(static_cast<float*>(rawMemC));
    aclRet = aclrtMalloc(&rawMemS, complexBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for s failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<aclblasComplex, AclMemDeleter> sDevicePtr(static_cast<aclblasComplex*>(rawMemS));

    aclRet = aclrtMemcpy(aDevicePtr.get(), complexBytes, &aHost, complexBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for a failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(bDevicePtr.get(), complexBytes, &bHost, complexBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for b failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasCrotg（a 原地覆写为 r，c/s 输出 Givens 旋转矩阵的余弦/正弦）
    blasRet = aclblasCrotg(handlePtr.get(), aDevicePtr.get(),
                           bDevicePtr.get(),
                           cDevicePtr.get(),
                           sDevicePtr.get());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCrotg failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    aclblasComplex rHost = {0.0f, 0.0f};
    aclblasComplex sHost = {0.0f, 0.0f};
    float cHost = 0.0f;
    aclRet = aclrtMemcpy(&rHost, complexBytes, aDevicePtr.get(), complexBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy r from device to host failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(&cHost, floatBytes, cDevicePtr.get(), floatBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy c from device to host failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(&sHost, complexBytes, sDevicePtr.get(), complexBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy s from device to host failed. ERROR: %d\n", aclRet); return aclRet);
    LOG_PRINT("rotg result: r=(%f,%f), c=%f, s=(%f,%f)\n", rHost.real, rHost.imag, cHost, sHost.real, sHost.imag);

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasCrotgTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasCrotgTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

## 编译运行

在仓库根目录下执行如下步骤，编译并运行算子测试。

- 配置环境变量
  ```bash
  source /usr/local/Ascend/cann/set_env.sh
  ```

- Srotg 编译并执行测试
  ```bash
  bash build.sh --ops=srotg --soc=ascend950 --run
  ```

  执行结果如下，说明所有测试用例通过。
  ```
  [PASS] srotg_test

  ========================================
  Test Summary:
    Passed:  1 - srotg
    Skipped: 0 -  (not supported on ascend950)
    Failed:  0 -
  ========================================
  ```

- Crotg 编译并执行测试
  ```bash
  bash build.sh --ops=crotg --soc=ascend950 --run
  ```

  执行结果如下，说明所有测试用例通过。
  ```
  [PASS] crotg_test

  ========================================
  Test Summary:
    Passed:  1 - crotg
    Skipped: 0 -  (not supported on ascend950)
    Failed:  0 -
  ========================================
  ```
