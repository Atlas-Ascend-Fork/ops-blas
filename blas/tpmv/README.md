# Tpmv算子

## 算子概述

tpmv (Triangular Packed Matrix-Vector Multiplication) 实现三角压缩矩阵与向量的乘法运算。该算子针对三角矩阵的 packed 存储特性进行优化，采用压缩存储格式以节省内存空间，高效完成矩阵与向量的乘加运算。

数学表达式：

```
x = A * x
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasStpmv | 单精度三角压缩矩阵-向量乘法（标准接口） |
| aclblasStpmv_legacy | 单精度三角压缩矩阵-向量乘法（早期接口） |
| aclblasCtpmv | 单精度复数三角压缩矩阵-向量乘法 |

## 算子执行接口

### aclblasStpmv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasStpmv(aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, const float *AP, float *x, int incx)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | 矩阵填充类型：ACLBLAS_UPPER(上三角)、ACLBLAS_LOWER(下三角)，Host 内存 |
| trans | 输入 | aclblasOperation_t | 矩阵操作类型：ACLBLAS_OP_N(不转置)、ACLBLAS_OP_T(转置)、ACLBLAS_OP_C(共轭转置)，Host 内存 |
| diag | 输入 | aclblasDiagType_t | 对角线类型：ACLBLAS_NON_UNIT(非单位对角线)、ACLBLAS_UNIT(单位对角线)，Host 内存 |
| n | 输入 | int | 三角压缩矩阵的行数和列数，Host 内存 |
| AP | 输入 | const float*（FP32） | 三角压缩矩阵 float 数组，维度为 n*(n+1)/2，Device 内存 |
| x | 输入/输出 | float*（FP32） | 输入/输出向量，包含 n 个元素，Device 内存 |
| incx | 输入 | int | x 中连续元素之间的步长，Host 内存 |

#### 约束说明

- n >= 0
- incx != 0

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

int aclblasStpmvTest(AclContext& ctx)
{
    constexpr int n = 3;
    constexpr int incx = 1;
    constexpr size_t apSize = n * (n + 1) / 2 * sizeof(float);
    constexpr size_t xSize = n * sizeof(float);

    // 上三角压缩矩阵（按列打包）：[[1,2,3],[0,5,6],[0,0,9]]
    float hAP[n * (n + 1) / 2] = {1.0f, 2.0f, 5.0f, 3.0f, 6.0f, 9.0f};
    float hX[n] = {1.0f, 1.0f, 1.0f};

    void *rawAP = nullptr;
    auto aclRet = aclrtMalloc(&rawAP, apSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dAP(rawAP);

    void *rawX = nullptr;
    aclRet = aclrtMalloc(&rawX, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dX(rawX);

    aclRet = aclrtMemcpy(dAP.get(), apSize, hAP, apSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dX.get(), xSize, hX, xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);
    std::unique_ptr<void, AclblasHandleDeleter> handle(rawHandle);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handle.get()), ctx.Stream());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    blasRet = aclblasStpmv(
        static_cast<aclblasHandle_t>(handle.get()), ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, n,
        static_cast<const float*>(dAP.get()), static_cast<float*>(dX.get()), incx);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    aclRet = aclrtSynchronizeStream(ctx.Stream());
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclRet = aclrtMemcpy(hX, xSize, dX.get(), xSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 预期结果：x = A*x = {6, 11, 9}
    for (int i = 0; i < n; i++) {
        printf("x[%d] = %f\n", i, hX[i]);
    }

    return 0;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasStpmvTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}
```

### aclblasCtpmv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持（arch35）
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持（arch22）
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持（arch22）

以下 workspace 分配说明针对 arch35；arch22 的范围与测试说明见 [A2/A3 测试说明](../../test/tpmv/ctpmv/README.md)。

#### 函数原型

```cpp
aclblasStatus_t aclblasCtpmv(aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, const aclblasComplex *AP, aclblasComplex *x, int incx)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | 矩阵填充类型：ACLBLAS_UPPER(上三角)、ACLBLAS_LOWER(下三角)，Host 内存 |
| trans | 输入 | aclblasOperation_t | 矩阵操作类型：ACLBLAS_OP_N(不转置)、ACLBLAS_OP_T(转置)、ACLBLAS_OP_C(共轭转置)，Host 内存 |
| diag | 输入 | aclblasDiagType_t | 对角线类型：ACLBLAS_NON_UNIT(非单位对角线)、ACLBLAS_UNIT(单位对角线，对角假定为 1，不读取 AP 对角位置)，Host 内存 |
| n | 输入 | int | 三角压缩矩阵的行数和列数，Host 内存 |
| AP | 输入 | const aclblasComplex*（COMPLEX64） | 三角压缩矩阵 packed 数组（列优先紧凑，无 lda），维度为 n*(n+1)/2，Device 内存 |
| x | 输入/输出 | aclblasComplex*（COMPLEX64） | 输入/输出向量，原地覆写，包含 n 个元素（步长展开后 1+(n-1)*\|incx\|），Device 内存 |
| incx | 输入 | int | x 中连续元素之间的步长（支持负步长反向遍历），Host 内存 |

#### 约束说明

- n >= 0（n = 0 为合法 no-op，直接返回成功）
- incx != 0
- diag = ACLBLAS_UNIT 时不读取 AP 的对角位置（对角元素按 1 参与计算）
- 接口在 `handle` 绑定的 stream 上异步提交清零、拷贝和计算，返回成功不表示 Device 计算已完成。读取或释放 `AP`、`x` 前须同步该 stream，并检查同步返回值。
- 临时存储复用 handle 的 workspace，算子内不分配、不释放，也不隐式同步。连续调用须在同一 stream 上有序提交；跨 stream 共享数据须由调用方建立依赖。
- 令 `S = n * sizeof(aclblasComplex)`：T/C 且 `incx=1、n>=8` 需要 `3*S+32768` 字节；N 且 `n<=32` 需要 `2*S`；N 且 `32<n<=128` 需要 `33*S`；其余路径需要 `(B+9)*S`，其中 `n<=8` 时 `B=1`，否则 `B=min(n, AIV核数)`。
- 创建 handle 时预分配 32 MiB 默认 workspace。容量不足返回 `ACLBLAS_STATUS_EXECUTION_FAILED`，可先用 `aclblasSetWorkspace` 配置足够大的设备缓冲区后重试。用户 workspace 在关联 stream 完成前必须保持有效，库不会释放它。

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cstdio>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blas.h"

int main()
{
    int32_t deviceId = 0;
    auto ret = aclInit(nullptr);
    ret = aclrtSetDevice(deviceId);
    aclrtStream stream = nullptr;
    ret = aclrtCreateStream(&stream);
    aclblasHandle_t handle = nullptr;
    ret = aclblasCreate(&handle);
    ret = aclblasSetStream(handle, stream);

    // n=4 上三角复数 packed 矩阵，op(A)=A，非单位对角，x 原地更新
    constexpr int n = 4;
    std::vector<aclblasComplex> ap(n * (n + 1) / 2);
    std::vector<aclblasComplex> x(n);
    // ... 填充 ap（packed 列优先）与 x ...
    aclblasComplex *apDev = nullptr;
    aclblasComplex *xDev = nullptr;
    aclrtMalloc(reinterpret_cast<void **>(&apDev), ap.size() * sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(reinterpret_cast<void **>(&xDev), x.size() * sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMemcpy(apDev, ap.size() * sizeof(aclblasComplex), ap.data(), ap.size() * sizeof(aclblasComplex),
                ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(xDev, x.size() * sizeof(aclblasComplex), x.data(), x.size() * sizeof(aclblasComplex),
                ACL_MEMCPY_HOST_TO_DEVICE);

    ret = aclblasCtpmv(handle, ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, n, apDev, xDev, 1);
    if (ret != ACLBLAS_STATUS_SUCCESS) {
        printf("aclblasCtpmv failed, ret=%d\n", static_cast<int>(ret));
    }

    // AP、x 和 workspace 必须存活到该 stream 的异步任务完成。
    ret = aclrtSynchronizeStream(stream);
    if (ret != ACL_SUCCESS) {
        printf("stream synchronization failed, ret=%d\n", static_cast<int>(ret));
        return 1;
    }

    aclrtFree(apDev);
    aclrtFree(xDev);
    aclblasDestroy(handle);
    aclrtDestroyStream(stream);
    ret = aclrtResetDevice(deviceId);
    ret = aclFinalize();
    return 0;
}
```

### aclblasStpmv_legacy

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：不支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasStpmv_legacy(aclblasHandle_t handle, aclblasFillMode uplo, aclblasOperation trans, aclblasDiagType diag, int64_t n, const float *aPacked, const float *x, float *y, int64_t incx)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode | 矩阵填充类型：ACLBLAS_UPPER 或 ACLBLAS_LOWER，Host 内存 |
| trans | 输入 | aclblasOperation | 矩阵操作类型，Host 内存 |
| diag | 输入 | aclblasDiagType | 对角线类型，Host 内存 |
| n | 输入 | int64_t | 三角压缩矩阵 A 的行数和列数，Host 内存 |
| aPacked | 输入 | const float*（FP32） | 三角压缩矩阵 float 数组，Device 内存 |
| x | 输入 | const float*（FP32） | float 输入向量，包含 n 个元素，Device 内存 |
| y | 输出 | float*（FP32） | float 输出向量，包含 n 个元素，Device 内存 |
| incx | 输入 | int64_t | x 中连续元素之间的步长，Host 内存 |

#### 约束说明

- n >= 0
- incx != 0

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

int aclblasStpmvLegacyTest(AclContext& ctx)
{
    constexpr int64_t n = 3;
    constexpr int64_t incx = 1;
    constexpr size_t apSize = static_cast<size_t>(n) * (n + 1) / 2 * sizeof(float);
    constexpr size_t xSize = static_cast<size_t>(n) * sizeof(float);

    // 上三角压缩矩阵（按列打包）：[[1,2,3],[0,5,6],[0,0,9]]
    float hAP[n * (n + 1) / 2] = {1.0f, 2.0f, 5.0f, 3.0f, 6.0f, 9.0f};
    float hX[n] = {1.0f, 1.0f, 1.0f};
    float hY[n] = {0.0f, 0.0f, 0.0f};

    void *rawAP = nullptr;
    auto aclRet = aclrtMalloc(&rawAP, apSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dAP(rawAP);

    void *rawX = nullptr;
    aclRet = aclrtMalloc(&rawX, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dX(rawX);

    void *rawY = nullptr;
    aclRet = aclrtMalloc(&rawY, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dY(rawY);

    aclRet = aclrtMemcpy(dAP.get(), apSize, hAP, apSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dX.get(), xSize, hX, xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);
    std::unique_ptr<void, AclblasHandleDeleter> handle(rawHandle);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handle.get()), ctx.Stream());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    blasRet = aclblasStpmv_legacy(
        static_cast<aclblasHandle_t>(handle.get()), ACLBLAS_UPPER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT, n,
        static_cast<const float*>(dAP.get()), static_cast<const float*>(dX.get()),
        static_cast<float*>(dY.get()), incx);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    aclRet = aclrtSynchronizeStream(ctx.Stream());
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclRet = aclrtMemcpy(hY, xSize, dY.get(), xSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 预期结果：y = A*x = {6, 11, 9}
    for (int64_t i = 0; i < n; i++) {
        printf("y[%lld] = %f\n", static_cast<long long>(i), hY[i]);
    }

    return 0;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasStpmvLegacyTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}
```