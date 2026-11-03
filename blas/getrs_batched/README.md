# GetrsBatched算子

## 算子概述

GetrsBatched（批量线性方程组求解）算子对一批已经由 aclblasSgetrfBatched 完成 LU 分解的 n×n 方阵，批量求解线性方程组。属于 LAPACK 风格的批量求解算子，接口对齐 LAPACK sgetrs 标准。

数学表达式：

```
op(A[i]) * X[i] = B[i],   i = 0, 1, ..., batchCount - 1
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSgetrsBatched | 单精度批量线性方程组求解 |

## 算子执行接口

### aclblasSgetrsBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSgetrsBatched(aclblasHandle_t handle, aclblasOperation_t trans, int n, int nrhs, const float *const Aarray[], int lda, const int *devIpiv, float *const Barray[], int ldb, int *info, int batchCount)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| trans | 输入 | aclblasOperation_t | 转置操作类型：ACLBLAS_OP_N（不转置）、ACLBLAS_OP_T（转置）、ACLBLAS_OP_C（共轭转置，实数域等价于转置），Host 内存 |
| n | 输入 | int | 每个矩阵 Aarray[i] 的行数和列数（方阵边长），n >= 0，Host 内存 |
| nrhs | 输入 | int | 每个矩阵 Barray[i] 的列数（右端项数量），nrhs >= 0，nrhs <= 256，Host 内存 |
| Aarray | 输入 | const float *const []（FP32） | Device 侧指针数组，每个元素指向 Device 内存中已 LU 分解的 n×n float 矩阵（列主序），Device 内存 |
| lda | 输入 | int | 每个矩阵 Aarray[i] 的 leading dimension，lda >= max(1, n)，Host 内存 |
| devIpiv | 输入 | const int* | 大小为 n × batchCount 的数组，存储每个矩阵的主元序列（来自 aclblasSgetrfBatched 输出），可为 NULL，Device 内存 |
| Barray | 输入/输出 | float *const []（FP32） | Device 侧指针数组，每个元素指向 Device 内存中 n×nrhs float 矩阵（列主序），输入时为右端矩阵 B，输出时被覆盖为解矩阵 X，Device 内存 |
| ldb | 输入 | int | 每个矩阵 Barray[i] 的 leading dimension，ldb >= max(1, n)，Host 内存 |
| info | 输出 | int* | 仅反映参数级错误，*info == 0 表示参数校验通过；*info = -j 表示第 j 个参数非法，Host 内存 |
| batchCount | 输入 | int | 指针数组中包含的矩阵数量，batchCount >= 0，Host 内存 |

#### 约束说明

- n >= 0, nrhs >= 0, batchCount >= 0
- nrhs <= 256
- lda >= max(1, n), ldb >= max(1, n)
- n == 0 或 nrhs == 0 或 batchCount == 0 时直接返回成功，不启动 Kernel
- 矩阵以列主序（Column-major）存储，与 LAPACK 标准一致
- 调用前必须先使用 aclblasSgetrfBatched 对 Aarray[i] 完成 LU 分解

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
struct BlasHandleDeleter {
    void operator()(aclblasHandle_t h) const { aclblasDestroy(h); }
};

int aclblasSgetrsBatchedTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasHandle_t>::type, BlasHandleDeleter> handlePtr(rawHandle);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    constexpr int n = 2;
    constexpr int nrhs = 1;
    constexpr int lda = n;
    constexpr int ldb = n;
    constexpr int batchCount = 1;
    constexpr size_t aSize = lda * n * sizeof(float);
    constexpr size_t bSize = ldb * nrhs * sizeof(float);
    constexpr size_t ipivSize = n * sizeof(int);

    std::vector<float> hA = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> hB = {3.0f, 7.0f};
    std::vector<int> hIpiv = {0, 1};
    int info = 0;

    void* rawA = nullptr;
    aclError aclRet;
    aclRet = aclrtMalloc(&rawA, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtr(rawA, aclrtFree);

    void* rawB = nullptr;
    aclRet = aclrtMalloc(&rawB, bSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for B failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dBPtr(rawB, aclrtFree);

    void* rawIpiv = nullptr;
    aclRet = aclrtMalloc(&rawIpiv, ipivSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for ipiv failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dIpivPtr(rawIpiv, aclrtFree);

    void* rawAPtrs = nullptr;
    aclRet = aclrtMalloc(&rawAPtrs, sizeof(float*), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for APtrs failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtrs(rawAPtrs, aclrtFree);

    void* rawBPtrs = nullptr;
    aclRet = aclrtMalloc(&rawBPtrs, sizeof(float*), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for BPtrs failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dBPtrs(rawBPtrs, aclrtFree);

    aclRet = aclrtMemcpy(dAPtr.get(), aSize, hA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(dBPtr.get(), bSize, hB.data(), bSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for B failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(dIpivPtr.get(), ipivSize, hIpiv.data(), ipivSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for ipiv failed. ERROR: %d\n", aclRet); return aclRet);

    float* hAPtrHost = static_cast<float*>(dAPtr.get());
    float* hBPtrHost = static_cast<float*>(dBPtr.get());
    aclRet = aclrtMemcpy(dAPtrs.get(), sizeof(float*), &hAPtrHost, sizeof(float*), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for APtrs failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dBPtrs.get(), sizeof(float*), &hBPtrHost, sizeof(float*), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for BPtrs failed. ERROR: %d\n", aclRet); return aclRet);

    blasRet = aclblasSgetrsBatched(
        static_cast<aclblasHandle_t>(handlePtr.get()), ACLBLAS_OP_N, n, nrhs,
        reinterpret_cast<const float* const*>(dAPtrs.get()), lda,
        static_cast<const int*>(dIpivPtr.get()),
        reinterpret_cast<float* const*>(dBPtrs.get()), ldb,
        &info, batchCount);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSgetrsBatched failed. ERROR: %d\n", blasRet);
              return blasRet);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    std::vector<float> bResult(ldb * nrhs, 0.0f);
    aclRet = aclrtMemcpy(bResult.data(), bSize, dBPtr.get(), bSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet); return aclRet);

    LOG_PRINT("info = %d\n", info);
    for (int i = 0; i < n * nrhs; i++) {
        LOG_PRINT("B[%d] = %f\n", i, bResult[i]);
    }

    LOG_PRINT("aclblasSgetrsBatched test passed\n");
    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSgetrsBatchedTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSgetrsBatchedTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```
