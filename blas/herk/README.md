# Herk算子

## 算子概述

Herk（Single-precision Complex Hermitian Rank-K Update）算子实现了单精度复数 Hermitian 矩阵的秩-k更新操作，将 alpha * op(A) * op(A)^H 加到 Hermitian 矩阵 C 的指定三角区域。其中 alpha、beta 为实数标量，A 为复数矩阵，C 为 Hermitian 复矩阵。

数学表达式：

```
trans='N': C = alpha * A * A^H + beta * C
trans='C': C = alpha * A^H * A + beta * C
```

其中 A^H 为 A 的共轭转置（conjugate transpose），即 A^H[i][j] = conj(A[j][i])；A 为 N×K（trans='N'）或 K×N（trans='C'）复矩阵，C 为 N×N Hermitian 复矩阵。uplo='U' 时仅更新上三角区域，uplo='L' 时仅更新下三角区域，另一半保持不变。C 满足 Hermitian 性质：C[i][j] = conj(C[j][i])，对角线元素虚部为零。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasCherk | 单精度复数 Hermitian 秩-k更新 |

## 算子执行接口

### aclblasCherk

#### 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Ascend 950PR / Ascend 950DT | 是 |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | 是 |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | 是 |

Ascend 950PR/Ascend 950DT 使用 `arch35` 实现；Atlas A2（Ascend 910B）与
Atlas A3（Ascend 910_93）使用 `arch22` 实现。arch35 依赖 CANN asc-devkit >= 9.1
（`ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR >= 1`）。

#### 函数原型

```cpp
aclblasStatus_t aclblasCherk(aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const float* alpha, const aclblasComplex* A, int lda, const float* beta, aclblasComplex* C, int ldc)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | C 矩阵存储模式：ACLBLAS_UPPER(121) 仅更新上三角或 ACLBLAS_LOWER(122) 仅更新下三角，Host 内存 |
| trans | 输入 | aclblasOperation_t | A 矩阵转置模式：ACLBLAS_OP_N(111) 不转置或 ACLBLAS_OP_C(113) 共轭转置（计算 A^H），Host 内存 |
| n | 输入 | int | C 矩阵的阶数，n >= 0，Host 内存 |
| k | 输入 | int | A 矩阵的第二维度，k >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | 实数标量乘数，不可为 nullptr，Device 内存 |
| A | 输入 | const aclblasComplex*（复数 FP32） | 输入复矩阵，trans='N' 时维度为 N×K，trans='C' 时维度为 K×N，Device 内存 |
| lda | 输入 | int | A 矩阵的主维，trans='N' 时 lda >= max(1, n)，trans='C' 时 lda >= max(1, k)，Host 内存 |
| beta | 输入 | const float*（FP32） | 实数标量乘数，不可为 nullptr，Device 内存 |
| C | 输入/输出 | aclblasComplex*（复数 FP32） | N×N Hermitian 复矩阵，输入旧值，输出新值，仅 uplo 指定三角区域被更新，Device 内存 |
| ldc | 输入 | int | C 矩阵的主维，ldc >= max(1, n)，Host 内存 |

#### 约束说明

- n >= 0, k >= 0
- uplo 为 ACLBLAS_UPPER 或 ACLBLAS_LOWER
- trans 为 ACLBLAS_OP_N 或 ACLBLAS_OP_C
- trans='N' 时：lda >= max(1, n)
- trans='C' 时：lda >= max(1, k)
- ldc >= max(1, n)
- alpha、beta 不可为 nullptr
- A 不可为 nullptr（当 n > 0 且 k > 0 时）
- C 不可为 nullptr（当 n > 0 时）
- alpha、beta 为实数（float），分别缩放复数 C 的实部与虚部
- 仅更新指定三角，对角线虚部置零。另一三角和 leading-dimension padding 保持原值
- 当 alpha=0 或 k=0，且 beta=1 时直接返回，C 全部保持原值
- 接口在 handle 绑定的 stream 上异步执行；库接口自身不执行 Device 或 stream 同步

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
// 注意：示例代码使用 printf 便于演示，生产代码应使用 dlog（OP_LOGE/OP_LOGI/OP_LOGD）

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

int aclblasCherkTest(AclContext& ctx)
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
    // A 为 N×K 列主序复矩阵 (trans='N'): A = [[1+1i, 2+0i], [3+0i, 4+0i]]
    // 列主序存储: col0 = [1+1i, 3+0i], col1 = [2+0i, 4+0i]
    constexpr int n = 2;
    constexpr int k = 2;
    constexpr int lda = n;
    constexpr int ldc = n;
    std::vector<aclblasComplex> hA = {{1.0f, 1.0f}, {3.0f, 0.0f}, {2.0f, 0.0f}, {4.0f, 0.0f}};
    std::vector<aclblasComplex> hC(n * n, {0.0f, 0.0f});
    float alpha = 1.0f;
    float beta = 0.0f;

    // 3. 申请 Device 内存并拷贝数据
    constexpr size_t aSize = n * k * sizeof(aclblasComplex);
    constexpr size_t cSize = n * n * sizeof(aclblasComplex);

    void* rawA = nullptr;
    aclError aclRet = aclrtMalloc(&rawA, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtr(rawA, aclrtFree);

    void* rawC = nullptr;
    aclRet = aclrtMalloc(&rawC, cSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for C failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dCPtr(rawC, aclrtFree);

    void* rawAlpha = nullptr;
    aclRet = aclrtMalloc(&rawAlpha, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for alpha failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAlphaPtr(rawAlpha, aclrtFree);

    void* rawBeta = nullptr;
    aclRet = aclrtMalloc(&rawBeta, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for beta failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dBetaPtr(rawBeta, aclrtFree);

    aclRet = aclrtMemcpy(dAPtr.get(), aSize, hA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(dCPtr.get(), cSize, hC.data(), cSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for C failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(dAlphaPtr.get(), sizeof(float), &alpha, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for alpha failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(dBetaPtr.get(), sizeof(float), &beta, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for beta failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasCherk
    // 计算 C = 1.0 * A * A^H + 0.0 * C (trans='N', uplo=LOWER)
    // A^H = [[1-1i, 3+0i], [2+0i, 4+0i]]
    // A * A^H = [[6, 11+3i], [11-3i, 25]]
    // C 为 Hermitian 矩阵: C[0][1] = conj(C[1][0]), 对角线虚部为零
    // uplo=LOWER: 仅更新下三角 (C[0][0], C[1][0], C[1][1])
    blasRet = aclblasCherk(
        handlePtr.get(), ACLBLAS_LOWER, ACLBLAS_OP_N,
        n, k,
        static_cast<const float*>(dAlphaPtr.get()),
        static_cast<const aclblasComplex*>(dAPtr.get()), lda,
        static_cast<const float*>(dBetaPtr.get()),
        static_cast<aclblasComplex*>(dCPtr.get()), ldc);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCherk failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    std::vector<aclblasComplex> cResult(n * n, {0.0f, 0.0f});
    aclRet = aclrtMemcpy(cResult.data(), cSize, dCPtr.get(), cSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet); return aclRet);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aclblasComplex val = cResult[static_cast<size_t>(j) * ldc + i];
            LOG_PRINT("C[%d][%d] = %.4f + %.4fi\n", i, j, static_cast<double>(val.real), static_cast<double>(val.imag));
        }
    }

    LOG_PRINT("aclblasCherk test passed\n");
    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasCherkTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasCherkTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```
