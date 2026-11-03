# MatinvBatched算子

## 算子概述

批量矩阵求逆算子针对一组 n×n 方阵，分别计算其逆矩阵。核心运算为：

```
Ainv[i] = A[i]⁻¹,  i = 0, 1, ..., batchSize - 1
```

内部通过 LU 分解（PA = LU）加上三角求逆两步完成。支持 n ≤ 32 的小方阵批量求逆。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSmatinvBatched | 单精度批量矩阵求逆（FP32） |

## 算子执行接口

### aclblasSmatinvBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSmatinvBatched(aclblasHandle_t handle, int n, const float* const A[], int lda, float* const Ainv[], int lda_inv, int* info, int batchSize)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 每个方阵的行和列数，0 ≤ n ≤ 32，Host 内存 |
| A | 输入 | const float* const[] | Device 侧指针数组，每个元素 `A[i]` 指向第 i 个输入矩阵（FP32，列主序，n×n），Device 内存 |
| lda | 输入 | int | 输入矩阵 A[i] 的 leading dimension，Host 内存 |
| Ainv | 输出 | float* const[] | Device 侧指针数组，每个元素 `Ainv[i]` 指向第 i 个输出矩阵（FP32，列主序，n×n），Device 内存 |
| lda_inv | 输入 | int | 输出矩阵 Ainv[i] 的 leading dimension，Host 内存 |
| info | 输出 | int* | 长度为 batchSize 的数组，`info[i] = 0` 表示求逆成功；`info[i] = k`（k > 0）表示 U(k,k) == 0（矩阵奇异），Device 内存 |
| batchSize | 输入 | int | 矩阵数量，batchSize ≥ 0，Host 内存 |

#### 约束说明

- n >= 0 且 n <= 32，方阵边长
- batchSize >= 0
- lda >= max(1, n)
- lda_inv >= max(1, n)
- 当 n == 0 或 batchSize == 0 时，函数直接返回成功，不访问 A / Ainv / info 指针（允许传 nullptr）
- 当 n > 0 且 batchSize > 0 时，A、Ainv、info 不得为 nullptr
- A[i] 与 Ainv[i] 的内存空间不可重叠
- 矩阵以列主序（Column-major）存储
- 本函数为异步执行，用户需自行通过 `aclrtSynchronizeStream` 同步获取结果
- Workspace：函数内部使用 handle workspace，需求大小为 `batchSize × 8 + n² × batchSize × 4 + n × batchSize × 4` 字节；workspace 不足时返回 `ACLBLAS_STATUS_EXECUTION_FAILED`，用户需通过 `aclblasSetWorkspace()` 扩容后重试

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cstdio>
#include <functional>
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

int aclblasSmatinvBatchedTest(AclContext& ctx)
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

    constexpr int n = 16;
    constexpr int batchSize = 4;
    constexpr int lda = n;
    constexpr int lda_inv = n;
    constexpr size_t aSize = static_cast<size_t>(lda) * n * sizeof(float);
    constexpr size_t invSize = static_cast<size_t>(lda_inv) * n * sizeof(float);
    constexpr size_t infoSize = batchSize * sizeof(int);

    std::vector<std::vector<float>> hostA(batchSize);
    std::vector<std::vector<float>> hostAinv(batchSize);
    std::vector<int> hostInfo(batchSize, 0);
    for (int b = 0; b < batchSize; b++) {
        hostA[b].resize(static_cast<size_t>(lda) * n, 0.0f);
        hostAinv[b].resize(static_cast<size_t>(lda_inv) * n, 0.0f);
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < n; i++) {
                hostA[b][j * lda + i] = (i == j) ? 2.0f : 0.1f;
            }
        }
    }

    std::vector<void*> dAMatrices(batchSize, nullptr);
    std::vector<void*> dAinvMatrices(batchSize, nullptr);
    auto cleanupMatrices = [&]() {
        for (int i = 0; i < batchSize; i++) {
            if (dAMatrices[i]) aclrtFree(dAMatrices[i]);
            if (dAinvMatrices[i]) aclrtFree(dAinvMatrices[i]);
        }
    };
    struct MatrixGuard {
        std::function<void()> cleanup;
        ~MatrixGuard() { cleanup(); }
    };
    MatrixGuard guard{cleanupMatrices};

    aclError aclRet;
    for (int b = 0; b < batchSize; b++) {
        aclRet = aclrtMalloc(&dAMatrices[b], aSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A[%d] failed. ERROR: %d\n", b, aclRet);
                  return aclRet);
        aclRet = aclrtMemcpy(dAMatrices[b], aSize, hostA[b].data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A[%d] failed. ERROR: %d\n", b, aclRet);
                  return aclRet);
        aclRet = aclrtMalloc(&dAinvMatrices[b], invSize, ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for Ainv[%d] failed. ERROR: %d\n", b, aclRet);
                  return aclRet);
    }

    constexpr size_t ptrArraySize = static_cast<size_t>(batchSize) * sizeof(float*);
    std::vector<float*> hAPtrs(batchSize);
    std::vector<float*> hAinvPtrs(batchSize);
    for (int b = 0; b < batchSize; b++) {
        hAPtrs[b] = static_cast<float*>(dAMatrices[b]);
        hAinvPtrs[b] = static_cast<float*>(dAinvMatrices[b]);
    }

    void* rawAPtrArray = nullptr;
    aclRet = aclrtMalloc(&rawAPtrArray, ptrArraySize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for APtrArray failed. ERROR: %d\n", aclRet);
              return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtrArray(rawAPtrArray, aclrtFree);

    void* rawAinvPtrArray = nullptr;
    aclRet = aclrtMalloc(&rawAinvPtrArray, ptrArraySize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for AinvPtrArray failed. ERROR: %d\n", aclRet);
              return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAinvPtrArray(rawAinvPtrArray, aclrtFree);

    void* rawInfo = nullptr;
    aclRet = aclrtMalloc(&rawInfo, infoSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for info failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dInfoPtr(rawInfo, aclrtFree);

    aclRet = aclrtMemcpy(dAPtrArray.get(), ptrArraySize, hAPtrs.data(), ptrArraySize,
                         ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for APtrArray failed. ERROR: %d\n", aclRet);
              return aclRet);
    aclRet = aclrtMemcpy(dAinvPtrArray.get(), ptrArraySize, hAinvPtrs.data(), ptrArraySize,
                         ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for AinvPtrArray failed. ERROR: %d\n", aclRet);
              return aclRet);

    blasRet = aclblasSmatinvBatched(
        static_cast<aclblasHandle_t>(handlePtr.get()), n,
        reinterpret_cast<const float* const*>(dAPtrArray.get()), lda,
        reinterpret_cast<float* const*>(dAinvPtrArray.get()), lda_inv,
        static_cast<int*>(dInfoPtr.get()), batchSize);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSmatinvBatched failed. ERROR: %d\n", blasRet);
              return blasRet);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet);
              return aclRet);

    for (int b = 0; b < batchSize; b++) {
        aclRet = aclrtMemcpy(hostAinv[b].data(), invSize, dAinvMatrices[b], invSize,
                             ACL_MEMCPY_DEVICE_TO_HOST);
        CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for Ainv[%d] failed. ERROR: %d\n", b, aclRet);
                  return aclRet);
    }
    aclRet = aclrtMemcpy(hostInfo.data(), infoSize, dInfoPtr.get(), infoSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy info failed. ERROR: %d\n", aclRet); return aclRet);

    for (int b = 0; b < batchSize; b++) {
        if (hostInfo[b] == 0) {
            LOG_PRINT("Batch %d: inversion succeeded\n", b);
        } else {
            LOG_PRINT("Batch %d: singular (info=%d)\n", b, hostInfo[b]);
        }
    }

    LOG_PRINT("aclblasSmatinvBatched test passed\n");
    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSmatinvBatchedTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSmatinvBatchedTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```
