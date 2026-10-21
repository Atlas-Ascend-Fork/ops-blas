# Tpsv算子

## 算子概述

tpsv (Triangular Packed matrix Solve) 求解三角线性系统。该算子针对三角矩阵的 packed 存储格式进行优化，支持前向和后向求解。

数学表达式：

```
op(A) * x = b
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasStpsv | 单精度三角 packed 矩阵求解 |
| aclblasCtpsv | 单精度复数三角 packed 矩阵求解 |

## 算子执行接口

### aclblasStpsv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasStpsv(aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, const float *AP, float *x, int incx)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | ACLBLAS_UPPER(121) — A 为上三角矩阵；ACLBLAS_LOWER(122) — A 为下三角矩阵，Host 内存 |
| trans | 输入 | aclblasOperation_t | ACLBLAS_OP_N(111) — op(A) = A；ACLBLAS_OP_T(112) — op(A) = A^T；ACLBLAS_OP_C(113) — op(A) = A^H（FP32 下与 T 等价），Host 内存 |
| diag | 输入 | aclblasDiagType_t | ACLBLAS_NON_UNIT(131) — 对角元从 AP 读取；ACLBLAS_UNIT(132) — 对角元固定为 1，Host 内存 |
| n | 输入 | int | 矩阵阶数，n >= 0。n == 0 时为空操作直接返回成功，Host 内存 |
| AP | 输入 | const float*（FP32） | packed 三角矩阵指针，共 n*(n+1)/2 个元素，Device 内存 |
| x | 输入/输出 | float*（FP32） | 输入时存储右端向量 b，输出时原地覆盖为解向量 x，Device 内存 |
| incx | 输入 | int | x 的存储增量，incx != 0（可正可负）。incx < 0 时 x 反向存储，Host 内存 |

#### 约束说明

- n >= 0，n == 0 时为空操作直接返回成功
- uplo 必须为 ACLBLAS_UPPER 或 ACLBLAS_LOWER
- trans 必须为 ACLBLAS_OP_N、ACLBLAS_OP_T 或 ACLBLAS_OP_C
- diag 必须为 ACLBLAS_NON_UNIT 或 ACLBLAS_UNIT
- incx != 0（可正可负）
- n > 0 时 AP、x 不可为 nullptr
- 不做奇异/近奇异检测：diag = ACLBLAS_NON_UNIT 时调用方须保证对角元非零

#### 返回值

| 场景 | 返回值 |
|--------|---------|
| 正常执行 / n == 0 空操作 | `ACLBLAS_STATUS_SUCCESS` |
| handle 为 nullptr | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| n < 0 / incx == 0 / uplo、trans、diag 非法枚举值 / n > 0 时 AP 或 x 为 nullptr | `ACLBLAS_STATUS_INVALID_VALUE` |
| arch22 实现：n > 32768（整解向量 UB 常驻策略的容量上界） | `ACLBLAS_STATUS_NOT_SUPPORTED` |

> arch22（Atlas A2 / Atlas A3）实现采用「整解向量 UB 常驻」策略，受 UB 容量约束存在阶数上界
> `n <= 32768`；超出时 host 侧直接返回 `ACLBLAS_STATUS_NOT_SUPPORTED`，不产生 UB 溢出。

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
int aclblasStpsvTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<void, aclblasStatus_t (*)(void*)> handlePtr(rawHandle, aclblasDestroy);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    constexpr int n = 2;
    constexpr int incx = 1;
    constexpr int apSize = n * (n + 1) / 2;
    constexpr size_t apBytes = apSize * sizeof(float);
    constexpr size_t xBytes = n * sizeof(float);

    std::vector<float> hAP = {1.0f, 2.0f, 3.0f};
    std::vector<float> hX = {1.0f, 8.0f};

    void* rawAP = nullptr;
    aclError aclRet;
    aclRet = aclrtMalloc(&rawAP, apBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for AP failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtr(rawAP, aclrtFree);

    void* rawX = nullptr;
    aclRet = aclrtMalloc(&rawX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for x failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dXPtr(rawX, aclrtFree);

    aclRet = aclrtMemcpy(dAPtr.get(), apBytes, hAP.data(), apBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for AP failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(dXPtr.get(), xBytes, hX.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for x failed. ERROR: %d\n", aclRet); return aclRet);

    blasRet = aclblasStpsv(
        static_cast<aclblasHandle_t>(handlePtr.get()), ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
        n, static_cast<const float*>(dAPtr.get()), static_cast<float*>(dXPtr.get()), incx);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasStpsv failed. ERROR: %d\n", blasRet);
              return blasRet);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    std::vector<float> xResult(n, 0.0f);
    aclRet = aclrtMemcpy(xResult.data(), xBytes, dXPtr.get(), xBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet); return aclRet);
    for (int i = 0; i < n; i++) {
        LOG_PRINT("x[%d] = %f\n", i, xResult[i]);
    }

    LOG_PRINT("aclblasStpsv test passed\n");
    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasStpsvTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasStpsvTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

### aclblasCtpsv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持（arch35）
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持（arch22）
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持（arch22）

- Ascend 950PR / 950DT 由 `blas/tpsv/arch35/` 实现支持；Atlas A2/A3 系列产品由 `blas/tpsv/arch22/` 实现支持（与已合入的实数版 aclblasStpsv 同族同目录），arch22 精度/性能验收环境为 Atlas 800I A2（Ascend 910B3）+ CANN 9.1.0。

#### 函数原型

```cpp
aclblasStatus_t aclblasCtpsv(aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, const aclblasComplex *AP, aclblasComplex *x, int incx)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | ACLBLAS_UPPER(121) — A 为上三角矩阵；ACLBLAS_LOWER(122) — A 为下三角矩阵，Host 内存 |
| trans | 输入 | aclblasOperation_t | ACLBLAS_OP_N(111) — op(A) = A；ACLBLAS_OP_T(112) — op(A) = A^T；ACLBLAS_OP_C(113) — op(A) = A^H，Host 内存 |
| diag | 输入 | aclblasDiagType_t | ACLBLAS_NON_UNIT(131) — 对角元从 AP 读取；ACLBLAS_UNIT(132) — 对角元固定为 1，Host 内存 |
| n | 输入 | int | 矩阵阶数，n >= 0。n == 0 时为空操作直接返回成功，Host 内存 |
| AP | 输入 | const aclblasComplex*（COMPLEX64） | packed 三角矩阵指针，共 n*(n+1)/2 个复数元素，Device 内存 |
| x | 输入/输出 | aclblasComplex*（COMPLEX64） | 输入时存储右端向量 b，输出时原地覆盖为解向量 x，Device 内存 |
| incx | 输入 | int | x 的存储增量，incx != 0（可正可负）。incx < 0 时 x 反向存储，Host 内存 |

#### 约束说明

- n >= 0，n == 0 时为空操作直接返回成功（n = 0 先于 AP/x 空指针校验，此时 AP、x 允许为 nullptr）
- uplo 必须为 ACLBLAS_UPPER 或 ACLBLAS_LOWER
- trans 必须为 ACLBLAS_OP_N、ACLBLAS_OP_T 或 ACLBLAS_OP_C（复数下 OP_T 纯转置与 OP_C 共轭转置为两个不同语义，host 侧不折叠）
- diag 必须为 ACLBLAS_NON_UNIT 或 ACLBLAS_UNIT
- incx != 0（可正可负）
- AP、x 不可为 nullptr
- packed 存储索引（0 基）：uplo = ACLBLAS_UPPER 时 A(i,j)（i <= j）存于 AP[i + j*(j+1)/2]；uplo = ACLBLAS_LOWER 时 A(i,j)（i >= j）存于 AP[i + (2*n-j-1)*j/2]
- diag = ACLBLAS_UNIT 时不访问对角元素，假定对角恒为 (1,0)
- 本算子不做奇异或近奇异检测，diag = ACLBLAS_NON_UNIT 时要求调用方保证对角元素非零；Inf/NaN 输入不提前拦截，按 IEEE 浮点规则沿依赖链传播
- 非法 uplo/trans/diag 枚举、n < 0、incx == 0、n > 0 时 AP/x 为 nullptr 返回 ACLBLAS_STATUS_INVALID_VALUE；handle 为 nullptr 返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR
- **arch22（Atlas A2/A3）支持域双层语义**：
  - **索引安全域 n ∈ [1, 65535]**：n(n+1)/2 < 2^31 的 uint32 索引域上限（n > 65535 返回 ACLBLAS_STATUS_INVALID_VALUE）；
  - **UB 能力域**：超向量 Unified Buffer 常驻预算的档位（约 n > 10K，随 trans/incx 组合浮动，host 侧守卫式联合不等式为权威判定，与 kernel 侧同式）返回 ACLBLAS_STATUS_NOT_SUPPORTED；
  - (4096, ~10K] 为功能正确、未做性能优化的窗口（x 平面分段驻留路径）；测试与精度/性能验收覆盖口径为 n ≤ 4096（精度 [0,2048] / 性能 [1,4096]）

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
int aclblasCtpsvTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<void, aclblasStatus_t (*)(void*)> handlePtr(rawHandle, aclblasDestroy);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    constexpr int n = 2;
    constexpr int incx = 1;
    constexpr int apSize = n * (n + 1) / 2;
    constexpr size_t apBytes = apSize * sizeof(aclblasComplex);
    constexpr size_t xBytes = n * sizeof(aclblasComplex);

    // lower packed: A = [[(1,0), 0], [(2,1), (3,0)]]
    std::vector<aclblasComplex> hAP = {{1.0f, 0.0f}, {2.0f, 1.0f}, {3.0f, 0.0f}};
    std::vector<aclblasComplex> hX = {{1.0f, 0.0f}, {8.0f, 3.0f}};

    void* rawAP = nullptr;
    aclError aclRet;
    aclRet = aclrtMalloc(&rawAP, apBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for AP failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtr(rawAP, aclrtFree);

    void* rawX = nullptr;
    aclRet = aclrtMalloc(&rawX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for x failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dXPtr(rawX, aclrtFree);

    aclRet = aclrtMemcpy(dAPtr.get(), apBytes, hAP.data(), apBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for AP failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(dXPtr.get(), xBytes, hX.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for x failed. ERROR: %d\n", aclRet); return aclRet);

    blasRet = aclblasCtpsv(
        static_cast<aclblasHandle_t>(handlePtr.get()), ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
        n, static_cast<const aclblasComplex*>(dAPtr.get()), static_cast<aclblasComplex*>(dXPtr.get()), incx);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCtpsv failed. ERROR: %d\n", blasRet);
              return blasRet);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    std::vector<aclblasComplex> xResult(n, {0.0f, 0.0f});
    aclRet = aclrtMemcpy(xResult.data(), xBytes, dXPtr.get(), xBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet); return aclRet);
    for (int i = 0; i < n; i++) {
        LOG_PRINT("x[%d] = (%f, %f)\n", i, xResult[i].real, xResult[i].imag);
    }

    LOG_PRINT("aclblasCtpsv test passed\n");
    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasCtpsvTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasCtpsvTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```
