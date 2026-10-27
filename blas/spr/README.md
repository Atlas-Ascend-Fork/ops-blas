# Spr算子

## 算子概述

spr (Symmetric Packed Rank-1 Update) 实现对称矩阵 packed 格式的秩-1更新操作。该算子将 `alpha * x * x^T` 加到 packed 对称矩阵的指定三角区域。

数学表达式：

```
A := alpha * x * x^T + A
```

- `x` 为 n 元素 float32 向量，`alpha` 为 float32 标量，`A` 为 n×n 对称实数矩阵。
- **打包存储（Packed Storage）**：A 以一维数组 `ap` 存储，长度 `n*(n+1)/2`，按**列主序**堆叠 `uplo` 指定三角的各列（含对角），**无前导维 lda、无 beta**。仅 `uplo` 指定三角被引用与更新，另一三角不存于 ap、由对称性 A = A^T 隐含。勿与全存储同族算子 `aclblasSsyr`（含 lda，A 为 n×n 二维数组）混淆。
- 0 基索引映射：UPPER 时 A(i,j)（i ≤ j）→ `ap[j*(j+1)/2 + i]`；LOWER 时 A(i,j)（j ≤ i）→ `ap[j*(2n-j+1)/2 + (i-j)]`。
- `n = 0` 或 `alpha = 0` 为合法 no-op（quick return，ap 不变，alpha=0 时 x 内容不被读取）。
- 支持任意非零 incx（含负步长，负步长按 Netlib `sspr` 语义从 x 尾部反向遍历）。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSspr | 单精度对称 packed 秩-1更新 |

## 算子执行接口

### aclblasSspr

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持（arch35）
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持（arch22）
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持（arch22）

> arch22 实现依赖 CANN 9.1.0；接口声明位于仓 `include/cann_ops_blas.h`，供各产品线共用，无产品私有平行接口。

#### 函数原型

```cpp
aclblasStatus_t aclblasSspr(aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, float *ap)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存；为 nullptr 时返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| uplo | 输入 | aclblasFillMode_t | ACLBLAS_UPPER(121) 或 ACLBLAS_LOWER(122)，指定引用/更新的三角区域，Host 内存；取值不在上述枚举时返回 `ACLBLAS_STATUS_INVALID_ENUM` |
| n | 输入 | int | 矩阵阶数，n >= 0，Host 内存；n=0 为合法 no-op（quick return），n<0 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| alpha | 输入 | const float*（FP32） | 标量乘数指针，Host 内存；为 nullptr 时返回 `ACLBLAS_STATUS_INVALID_VALUE`；alpha=0 时为合法 no-op（不读取 x、ap 不变） |
| x | 输入 | const float*（FP32） | 输入向量指针，只读，ND 排布，逻辑长度 n，物理长度至少 1 + (n-1) * abs(incx)，Device 内存；n>0 时为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE` |
| incx | 输入 | int | x 的元素间步长，incx != 0（=0 返回 `ACLBLAS_STATUS_INVALID_VALUE`），Host 内存；支持负步长（反向遍历） |
| ap | 输入/输出 | float*（FP32） | packed 对称矩阵指针，一维数组长度 n*(n+1)/2，列主序堆叠 uplo 三角（含对角），输入旧值、原地覆写输出，Device 内存；n>0 时为 nullptr 返回 `ACLBLAS_STATUS_INVALID_VALUE` |

返回值 `aclblasStatus_t`：`ACLBLAS_STATUS_SUCCESS`（含 n=0 / alpha=0 合法 no-op）、`ACLBLAS_STATUS_INVALID_VALUE`（n<0 / incx=0 / alpha,x,ap 为 nullptr（n>0 时））、`ACLBLAS_STATUS_INVALID_ENUM`（uplo 非法）、`ACLBLAS_STATUS_HANDLE_IS_NULLPTR`（handle 为 nullptr）。校验序：handle → n<0 → n=0 → uplo 枚举 → incx → 指针 → alpha=0。

#### 约束说明

- n >= 0
- incx != 0
- alpha 指针不可为 nullptr；n>0 时 x / ap 指针不可为 nullptr（指针校验先于 alpha=0 quick return 执行；alpha=0 时 x 内容不被读取，含 NaN/Inf 输入不影响结果）
- 负 incx 语义（Netlib `sspr`）：逻辑元素 x(i)（0 基）取自 `x[(n-1-i)*|incx|]`，x 物理长度 ≥ 1+(n-1)×|incx|
- 数据类型仅支持 FLOAT32；x 与 ap 均为 ND 连续排布；n / incx 为运行时入参（不要求 dynamic shape 编译）
- 异步执行：依赖 `aclblasSetStream` 绑定 stream；读回 Device 结果前须同步 stream

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cmath>
#include <cstdio>
#include <cstring>
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
    void operator()(void *ptr) const
    {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
};

struct AclblasHandleDeleter {
    void operator()(void *handle) const
    {
        if (handle != nullptr) {
            aclblasDestroy(static_cast<aclblasHandle_t>(handle));
        }
    }
};

// packed 索引（0 基）：UPPER：ap[j*(j+1)/2+i]；LOWER：ap[j*(2n-j+1)/2+(i-j)]
static inline size_t PackedIndex(aclblasFillMode_t uplo, int n, int i, int j)
{
    if (uplo == ACLBLAS_UPPER) {
        return static_cast<size_t>(j) * (j + 1) / 2 + i;
    }
    return static_cast<size_t>(j) * (2 * n - j + 1) / 2 + (i - j);
}

// 单用例：设备内存准备 -> aclblasSspr -> 同步 -> 结果校验（CPU golden 对拍）
// 用例覆盖：UPPER / LOWER、正步长 / 负步长 / 单位步长
struct SsprCase {
    const char *name;
    aclblasFillMode_t uplo;
    int n;
    int incx;
    float alpha;
};

int RunSsprCase(AclContext &ctx, aclblasHandle_t handle, const SsprCase &c)
{
    const size_t apLen = static_cast<size_t>(c.n) * (c.n + 1) / 2;
    const size_t apBytes = apLen * sizeof(float);
    const size_t xPhysLen = 1 + static_cast<size_t>(c.n - 1) * static_cast<size_t>(std::abs(c.incx));
    const size_t xBytes = xPhysLen * sizeof(float);

    std::vector<float> hAp(apLen);
    std::vector<float> hX(xPhysLen);
    for (size_t k = 0; k < apLen; ++k) {
        hAp[k] = 0.5f + 0.25f * static_cast<float>(k % 11);
    }
    for (size_t k = 0; k < xPhysLen; ++k) {
        hX[k] = 0.1f * static_cast<float>(k % 7) - 0.3f;
    }

    // CPU golden：ap_out = alpha * xL * xL^T + ap_in
    // 正 incx：xL(i) = x[i*incx]；负 incx（Netlib sspr 语义）：xL(i) = x[(n-1-i)*|incx|]
    const int absInc = std::abs(c.incx);
    std::vector<float> xL(c.n);
    for (int i = 0; i < c.n; ++i) {
        xL[i] = (c.incx > 0) ? hX[static_cast<size_t>(i) * absInc]
                             : hX[static_cast<size_t>(c.n - 1 - i) * absInc];
    }
    std::vector<float> golden = hAp;
    for (int j = 0; j < c.n; ++j) {
        const int iBegin = (c.uplo == ACLBLAS_UPPER) ? 0 : j;      // UPPER: i <= j；LOWER: i >= j
        const int iEnd = (c.uplo == ACLBLAS_UPPER) ? j : c.n - 1;
        for (int i = iBegin; i <= iEnd; ++i) {
            golden[PackedIndex(c.uplo, c.n, i, j)] += c.alpha * xL[i] * xL[j];
        }
    }

    // 设备内存准备
    void *rawAp = nullptr;
    auto aclRet = aclrtMalloc(&rawAp, apBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dAp(rawAp);

    void *rawX = nullptr;
    aclRet = aclrtMalloc(&rawX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dX(rawX);

    aclRet = aclrtMemcpy(dAp.get(), apBytes, hAp.data(), apBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dX.get(), xBytes, hX.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // 调用 aclblasSspr（alpha 为 Host 指针；x/ap 为 Device 指针）
    float alpha = c.alpha;
    auto blasRet = aclblasSspr(handle, c.uplo, c.n, &alpha, static_cast<const float *>(dX.get()), c.incx,
        static_cast<float *>(dAp.get()));
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    // 同步 + 结果读回 + 校验
    aclRet = aclrtSynchronizeStream(ctx.Stream());
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::vector<float> hOut(apLen);
    aclRet = aclrtMemcpy(hOut.data(), apBytes, dAp.get(), apBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    float maxAbsErr = 0.0f;
    for (size_t k = 0; k < apLen; ++k) {
        maxAbsErr = std::fmaxf(maxAbsErr, std::fabs(hOut[k] - golden[k]));
    }
    const bool pass = std::isfinite(maxAbsErr) && maxAbsErr <= 1e-4f;
    std::printf("[sspr-example] %-16s n=%-3d incx=%-3d maxAbsErr=%.3e %s\n", c.name, c.n, c.incx,
        static_cast<double>(maxAbsErr), pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    // handle 创建（aclblasCreate）+ stream 绑定（aclblasSetStream），RAII 释放
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);
    std::unique_ptr<void, AclblasHandleDeleter> handle(rawHandle);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handle.get()), ctx.Stream());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    const std::vector<SsprCase> cases = {
        {"UPPER_incx_1", ACLBLAS_UPPER, 6, 1, 2.5f},   // 上三角，单位步长
        {"LOWER_incx_2", ACLBLAS_LOWER, 6, 2, -1.5f},  // 下三角，正步长 > 1
        {"UPPER_incx_-2", ACLBLAS_UPPER, 6, -2, 1.0f}, // 上三角，负步长（反向遍历）
        {"LOWER_incx_-3", ACLBLAS_LOWER, 8, -3, 0.0f}, // 下三角，负步长 + alpha=0 quick return
    };
    for (const auto &c : cases) {
        ret = RunSsprCase(ctx, static_cast<aclblasHandle_t>(handle.get()), c);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
    }

    std::printf("[sspr-example] all %zu cases passed\n", cases.size());
    return 0;
}
```
