# aclblasChemm（复数厄米特矩阵乘）

## 功能简介

`aclblasChemm` 计算单精度复数（complex64）厄米特矩阵与普通矩阵的乘积。厄米特矩阵满足 `A = A^H`，仅需存储并引用单侧三角，对角线元素的虚部按 0 处理。

按 `side` 取值，计算式为：

```
side = LEFT ：C := alpha * A * B + beta * C
side = RIGHT：C := alpha * B * A + beta * C
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasChemm | 单精度复数厄米特矩阵乘法 |

## 算子执行接口

### aclblasChemm

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：不支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasChemm(aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int64_t m, int64_t n, const aclblasComplex *alpha, const aclblasComplex *A, int64_t lda, const aclblasComplex *B, int64_t ldb, const aclblasComplex *beta, aclblasComplex *C, int64_t ldc)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ACL-BLAS 句柄，Host 内存 |
| side | 输入 | aclblasSideMode_t | A 位于左侧或右侧，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | 指定引用 A 的上三角或下三角，Host 内存 |
| m | 输入 | int64_t | B、C 的行数，Host 内存 |
| n | 输入 | int64_t | B、C 的列数，Host 内存 |
| alpha | 输入 | const aclblasComplex* | 复数乘数，Host 内存 |
| A | 输入 | const aclblasComplex* | 厄米特矩阵，Device 内存 |
| lda | 输入 | int64_t | A 的前导维度，Host 内存 |
| B | 输入 | const aclblasComplex* | 普通矩阵，Device 内存 |
| ldb | 输入 | int64_t | B 的前导维度，Host 内存 |
| beta | 输入 | const aclblasComplex* | 复数乘数，Host 内存 |
| C | 输入/输出 | aclblasComplex* | 输入旧值并原地写回结果，Device 内存 |
| ldc | 输入 | int64_t | C 的前导维度，Host 内存 |

#### 约束说明

- A、B、C 均为列主序存储，`aclblasComplex` 的实部与虚部都是 FP32。
- `side=ACLBLAS_SIDE_LEFT`：A 为 `m×m` 的厄米特矩阵，结果为 `C = alpha*A*B + beta*C`。
- `side=ACLBLAS_SIDE_RIGHT`：A 为 `n×n` 的厄米特矩阵，结果为 `C = alpha*B*A + beta*C`。
- 仅访问 `uplo`（UPPER/LOWER）所指的 A 三角；A 对角元素的虚部一律当作 0。
- `side`/`uplo` 取值非法时返回 `ACLBLAS_STATUS_INVALID_ENUM`。
- 当 `m<0`、`n<0`、`alpha`/`beta` 为空指针或前导维不满足下界时，返回 `ACLBLAS_STATUS_INVALID_VALUE`。
- `m==0` 或 `n==0` 属合法的空操作，直接返回 `ACLBLAS_STATUS_SUCCESS`（先于前导维校验）。
- 前导维下界：`lda >= max(1, side==LEFT ? m : n)`、`ldb >= max(1, m)`、`ldc >= max(1, m)`。
- 当 `beta=(0,0)` 时不读取 C 的旧值，但 C 仍作为输出缓冲。
- 计算下发到 handle 所绑定的 stream，读取输出前需先同步该 stream。

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

class AclContext {
public:
    explicit AclContext(int deviceId) : deviceId_(deviceId) {}

    ~AclContext()
    {
        if (stream_ != nullptr) {
            aclrtDestroyStream(stream_);
        }
        if (deviceSet_) {
            aclrtResetDevice(deviceId_);
        }
        if (aclInited_) {
            aclFinalize();
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
    void operator()(void* handle) const
    {
        if (handle != nullptr) {
            aclblasDestroy(static_cast<aclblasHandle_t>(handle));
        }
    }
};

int RunChemm(AclContext& ctx)
{
    constexpr int64_t m = 2;
    constexpr int64_t n = 2;
    constexpr int64_t ld = 2;
    constexpr size_t bytes = 4 * sizeof(aclblasComplex);
    const aclblasComplex alpha{1.0f, 0.0f};
    const aclblasComplex beta{0.0f, 0.0f};
    // Column-major A = [[2, 1+2i], [1-2i, 3]], B = I, C = 0.
    aclblasComplex hA[4] = {
        {2.0f, 0.0f}, {1.0f, -2.0f},
        {1.0f, 2.0f}, {3.0f, 0.0f}};
    aclblasComplex hB[4] = {
        {1.0f, 0.0f}, {0.0f, 0.0f},
        {0.0f, 0.0f}, {1.0f, 0.0f}};
    aclblasComplex hC[4] = {};

    void* rawA = nullptr;
    void* rawB = nullptr;
    void* rawC = nullptr;
    auto ret = aclrtMalloc(&rawA, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<void, AclrtMemDeleter> dA(rawA);
    ret = aclrtMalloc(&rawB, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<void, AclrtMemDeleter> dB(rawB);
    ret = aclrtMalloc(&rawC, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    std::unique_ptr<void, AclrtMemDeleter> dC(rawC);

    ret = aclrtMemcpy(dA.get(), bytes, hA, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMemcpy(dB.get(), bytes, hB, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMemcpy(dC.get(), bytes, hC, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);
    std::unique_ptr<void, AclblasHandleDeleter> handle(rawHandle);
    blasRet = aclblasSetStream(
        static_cast<aclblasHandle_t>(handle.get()), ctx.Stream());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    blasRet = aclblasChemm(
        static_cast<aclblasHandle_t>(handle.get()), ACLBLAS_SIDE_LEFT,
        ACLBLAS_UPPER, m, n, &alpha,
        static_cast<const aclblasComplex*>(dA.get()), ld,
        static_cast<const aclblasComplex*>(dB.get()), ld, &beta,
        static_cast<aclblasComplex*>(dC.get()), ld);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);
    ret = aclrtSynchronizeStream(ctx.Stream());
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    ret = aclrtMemcpy(hC, bytes, dC.get(), bytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    for (int64_t col = 0; col < n; ++col) {
        for (int64_t row = 0; row < m; ++row) {
            const auto& value = hC[row + col * ld];
            std::printf("C[%lld,%lld] = (%f, %f)\n",
                static_cast<long long>(row), static_cast<long long>(col),
                value.real, value.imag);
        }
    }
    return 0;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return RunChemm(ctx);
}
```
