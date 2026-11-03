# GetrfBatched算子

## 算子概述

GetrfBatched（批量 LU 分解）算子对一批 n x n 方阵分别执行带部分主元选取的 LU 分解（LU factorization with partial pivoting）。属于 LAPACK 风格的批量分解算子，接口对齐 cuBLAS `cublasSgetrfBatched`，算法参考 NETLIB LAPACK sgetrf。

数学表达式：

```
P * A[i] = L * U,   i = 0, 1, ..., batchSize - 1
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSgetrfBatched | 单精度批量 LU 分解（带部分主元选取） |

## 算子执行接口

### aclblasSgetrfBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSgetrfBatched(aclblasHandle_t handle, int n, float *const Aarray[], int lda, int *PivotArray, int *infoArray, int batchSize)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 每个矩阵 Aarray[i] 的行数和列数（方阵边长），n >= 0，Host 内存 |
| Aarray | 输入/输出 | float *const []（FP32） | Device 侧指针数组，每个元素指向 Device 内存中 n x n float 矩阵，矩阵以列主序存储，Device 内存 |
| lda | 输入 | int | 每个矩阵 Aarray[i] 的 leading dimension，lda >= max(1, n)，Host 内存 |
| PivotArray | 输出 | int* | 大小为 n x batchSize 的数组，存储每个矩阵的主元序列（1-indexed，LAPACK 约定），可为 NULL，Device 内存 |
| infoArray | 输出 | int* | 大小为 batchSize 的数组，infoArray[i] = 0 表示分解成功；= k > 0 表示 U(k,k) == 0，Device 内存 |
| batchSize | 输入 | int | 指针数组中包含的矩阵数量，batchSize >= 0，Host 内存 |

#### 约束说明

- n >= 0, batchSize >= 0
- lda >= max(1, n)
- n == 0 或 batchSize == 0 时直接返回成功，不启动 Kernel
- PivotArray != NULL 时 infoArray 不可为 NULL
- PivotArray == NULL 合法（禁用主元选取，执行非主元 LU 分解）

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

int aclblasSgetrfBatchedTest(AclContext& ctx)
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
    constexpr int lda = 2;
    constexpr int batchSize = 1;
    constexpr size_t aSize = static_cast<size_t>(lda) * n * sizeof(float);
    constexpr size_t ipivSize = static_cast<size_t>(n) * batchSize * sizeof(int);
    constexpr size_t infoSize = sizeof(int);

    std::vector<float> hA = {4.0f, 2.0f,
                             1.0f, 3.0f};
    std::vector<int> hIpiv(n * batchSize, 0);
    std::vector<int> hInfo(batchSize, 0);

    void* rawA = nullptr;
    aclError aclRet;
    aclRet = aclrtMalloc(&rawA, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtr(rawA, aclrtFree);

    void* rawIpiv = nullptr;
    aclRet = aclrtMalloc(&rawIpiv, ipivSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for ipiv failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dIpivPtr(rawIpiv, aclrtFree);

    void* rawInfo = nullptr;
    aclRet = aclrtMalloc(&rawInfo, infoSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for info failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dInfoPtr(rawInfo, aclrtFree);

    aclRet = aclrtMemcpy(dAPtr.get(), aSize, hA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A failed. ERROR: %d\n", aclRet); return aclRet);

    float* hAPtrHost = static_cast<float*>(dAPtr.get());
    void* rawAPtrs = nullptr;
    aclRet = aclrtMalloc(&rawAPtrs, sizeof(float*), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for APtrs failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtrs(rawAPtrs, aclrtFree);

    aclRet = aclrtMemcpy(dAPtrs.get(), sizeof(float*), &hAPtrHost, sizeof(float*), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for APtrs failed. ERROR: %d\n", aclRet); return aclRet);

    blasRet = aclblasSgetrfBatched(
        static_cast<aclblasHandle_t>(handlePtr.get()), n,
        static_cast<float* const*>(dAPtrs.get()), lda,
        static_cast<int*>(dIpivPtr.get()),
        static_cast<int*>(dInfoPtr.get()), batchSize);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSgetrfBatched failed. ERROR: %d\n", blasRet);
              return blasRet);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(hIpiv.data(), ipivSize, dIpivPtr.get(), ipivSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for ipiv failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(hInfo.data(), infoSize, dInfoPtr.get(), infoSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for info failed. ERROR: %d\n", aclRet); return aclRet);

    LOG_PRINT("info[0] = %d\n", hInfo[0]);
    if (hInfo[0] == 0) {
        LOG_PRINT("ipiv = [%d, %d]\n", hIpiv[0], hIpiv[1]);
        LOG_PRINT("Success! Factorization completed.\n");
    } else {
        LOG_PRINT("Failed! U(%d,%d) is exactly zero.\n", hInfo[0], hInfo[0]);
    }

    LOG_PRINT("aclblasSgetrfBatched test passed\n");
    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSgetrfBatchedTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSgetrfBatchedTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```
