# GetriBatched算子

## 算子概述

GetriBatched（批量矩阵求逆）算子对一批已经完成 LU 分解的 n x n 方阵批量计算逆矩阵。属于 LAPACK 风格的批量求逆算子，支持单精度实数和单精度复数输入。

数学表达式：

```
inv(A[i]) = inv(U[i]) * inv(L[i]) * P[i],   i = 0, 1, ..., batchSize - 1
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSgetriBatched | 单精度批量矩阵求逆 |
| aclblasCgetriBatched | 单精度复数批量矩阵求逆 |

## 算子执行接口

### aclblasCgetriBatched

#### 产品支持情况

| 产品 | 是否支持 |
|------|----------|
| Ascend 950PR | 支持 |
| Ascend 950DT | 不支持 |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | 不支持 |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | 不支持 |

#### 函数原型

```cpp
aclblasStatus_t aclblasCgetriBatched(
    aclblasHandle_t handle, int n,
    const aclblasComplex* const Aarray[], int lda,
    const int* PivotArray,
    aclblasComplex* const Carray[], int ldc,
    int* infoArray, int batchSize);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 每个方阵的边长，n >= 0，Host 内存 |
| Aarray | 输入 | const aclblasComplex *const [] | Device 侧指针数组；每个元素指向列主序 COMPLEX64 LU 因子矩阵 |
| lda | 输入 | int | 输入矩阵 leading dimension，lda >= max(1, n) |
| PivotArray | 输入 | const int* | Device 侧 n x batchSize 个 1-based 主元索引；可为 nullptr，表示无主元模式 |
| Carray | 输出 | aclblasComplex *const [] | Device 侧指针数组；每个元素指向独立的列主序逆矩阵输出区 |
| ldc | 输入 | int | 输出矩阵 leading dimension，ldc >= max(1, n) |
| infoArray | 输出 | int* | Device 侧长度为 batchSize 的数组；info[i]=0 表示成功；info[i]=k>0 表示 U(k,k)==0，接口仍返回成功 |
| batchSize | 输入 | int | 矩阵数量，batchSize >= 0 |

#### 约束说明

- 调用前，输入必须是与本接口 n、lda、batchSize 一致的复数 LU 分解结果；
- Aarray[i] 与 Carray[i] 不可重叠；
- n == 0 或 batchSize == 0 时直接返回成功，不启动 Kernel；
- PivotArray == nullptr 合法，表示无主元 LU；
- 接口异步执行，读取 Carray/infoArray 前必须同步 handle 绑定的 stream；
- n >= 321 时采用 blocked AIV+Cube 路径，并由 handle 的默认 workspace 管理中间实部/虚部平面；workspace 分配失败时返回对应错误码；
- 当前实现仅面向 Ascend 950PR 的 arch35 路径。

#### 调用示例

下面的函数提交一个 2 x 2、无主元的复数 LU 求逆任务。调用前应创建 handle 并绑定 stream，在 Device 上准备 4 个 COMPLEX64 元素的 dLu、4 个 COMPLEX64 元素的 dInverse、一个 int 的 dInfo，以及各容纳一个指针的 dInputArray 和 dOutputArray。dLu 中必须是已完成 LU 分解的数据，输入和输出矩阵使用独立存储。

```cpp
#include <cstdio>

#include "acl/acl.h"
#include "cann_ops_blas.h"

aclblasStatus_t SubmitComplexInverseExample(
    aclblasHandle_t handle, const aclblasComplex* dLu, aclblasComplex* dInverse,
    const aclblasComplex** dInputArray, aclblasComplex** dOutputArray, int* dInfo)
{
    // 指针数组本身也必须位于 Device；这里将 Host 上的指针值复制过去。
    const aclblasComplex* inputPointers[] = {dLu};
    aclblasComplex* outputPointers[] = {dInverse};
    aclError ret = aclrtMemcpy(
        dInputArray, sizeof(inputPointers), inputPointers, sizeof(inputPointers), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "copy input pointer array failed: %d\n", ret);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    ret = aclrtMemcpy(
        dOutputArray, sizeof(outputPointers), outputPointers, sizeof(outputPointers), ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        std::fprintf(stderr, "copy output pointer array failed: %d\n", ret);
        return ACLBLAS_STATUS_EXECUTION_FAILED;
    }
    // nullptr 表示无主元；n、lda、ldc 均为 2，batchSize 为 1。
    return aclblasCgetriBatched(handle, 2, dInputArray, 2, nullptr, dOutputArray, 2, dInfo, 1);
}
```

返回成功表示任务已提交。调用方在读取结果、复用或释放上述 Device 内存前，需要同步 handle 绑定的 stream，再将 dInfo 和 dInverse 拷回 Host 检查。算子实现本身不执行流同步。

#### 计算路径

- n<8：SIMT 以 warp 处理矩阵，在 UB 中复用 LU，每个 lane 求解一列。
- 8<=n<=128：SIMD 在 64 个向量 lane 上并行求解 RHS 列，使用复数前代和回代；n<=32 根据尺寸将 8/4/2 个独立矩阵交错到 lane 中，主元置换和零对角检测在向量寄存器中完成。65<=n<=96 的输出暂存使用 LU 平面尾部的空闲区域，在后续 RHS 列块复用 LU、对角倒数、奇异状态和预先生成的主元置换；先扫描主元，在确认不交换行时省去恒等置换的执行。65<=n<=72 的尾部 1..8 列压紧为每行 8 个 float，通过 LU 系数和 RHS 行的块广播一次更新 8 行，完成后还原输出布局。n>64 使用向量转置与 DMA 写回输出，并在 SIMT 状态写回与 SIMD 转置之间同步。
- 129<=n<=320：64 列 RHS 驻留 UB，n<=256 按 16 列、257<=n<=288 按 8 列、289<=n<=320 按 4 列读取 LU 因子，在同一个 kernel 内完成前代和回代。每个 LU 平面按 64 个 float 向上补齐，容纳最后一条向量指令的访问范围。
- n>=321：以 64 行为块，AIV 将 64 列 RHS 和三角块搬入 UB 并使用 SIMD 求解，Cube 的实数 GEMM 组合完成复数块间更新。初始化、更新合并和输出使用 DMA 与向量操作。

完整 LU 和分片路径根据实际主元置换计算 RHS 第一个非零行，省略此前全零行的前代更新；该判断适用于任意合法主元序列。

各路径均保留列主序及独立矩阵指针语义，按照实际 LU 和主元数据计算，不要求矩阵对角或批次数据相同。自测覆盖范围、Release 构建和完整调用性能统计见 [复数版自测说明](../../test/getri_batched/cgetri_batched/README.md)。

#### 构建与测试

```bash
bash build.sh --pkg --soc=ascend950 --ops=cgetri_batched
bash build.sh --soc=ascend950 --ops=cgetri_batched --run
```

### aclblasSgetriBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSgetriBatched(aclblasHandle_t handle, int n, const float *const Aarray[], int lda, const int *PivotArray, float *const Carray[], int ldc, int *infoArray, int batchSize)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 每个矩阵 Aarray[i] 的行数和列数（方阵边长），n >= 0，Host 内存 |
| Aarray | 输入 | const float *const []（FP32） | Device 侧指针数组，每个元素指向 Device 内存中已 LU 分解的 n x n float 矩阵（列主序），Device 内存 |
| lda | 输入 | int | 每个矩阵 Aarray[i] 的 leading dimension，lda >= max(1, n)，Host 内存 |
| PivotArray | 输入 | const int* | 大小为 n x batchSize 的数组，存储每个矩阵的主元序列（来自 aclblasSgetrfBatched 输出），可为 NULL，Device 内存 |
| Carray | 输出 | float *const []（FP32） | Device 侧指针数组，每个元素指向 Device 内存中 n x n float 矩阵（列主序），用于存储逆矩阵，Device 内存 |
| ldc | 输入 | int | 每个矩阵 Carray[i] 的 leading dimension，ldc >= max(1, n)，Host 内存 |
| infoArray | 输出 | int* | 大小为 batchSize 的数组，infoArray[i] = 0 表示求逆成功；= k > 0 表示 U(k,k) == 0（求逆失败），Device 内存 |
| batchSize | 输入 | int | 指针数组中包含的矩阵数量，batchSize >= 0，Host 内存 |

#### 约束说明

- n >= 0, batchSize >= 0
- lda >= max(1, n), ldc >= max(1, n)
- n == 0 或 batchSize == 0 时直接返回成功，不启动 Kernel
- Carray[i] 的内存空间不可与 Aarray[i] 重叠
- 调用前必须先使用 aclblasSgetrfBatched 完成 LU 分解

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
int aclblasSgetriBatchedTest(AclContext& ctx)
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
    constexpr int lda = n;
    constexpr int ldc = n;
    constexpr int batchSize = 1;
    constexpr size_t aSize = lda * n * sizeof(float);
    constexpr size_t cSize = ldc * n * sizeof(float);
    constexpr size_t pivotSize = n * sizeof(int);
    constexpr size_t infoSize = batchSize * sizeof(int);

    std::vector<float> hA = {1.0f, 0.0f, 0.0f, 1.0f};
    std::vector<float> hC(n * n, 0.0f);
    std::vector<int> hPivot = {1, 2};
    std::vector<int> hInfo(batchSize, 0);

    void* rawA = nullptr;
    aclError aclRet;
    aclRet = aclrtMalloc(&rawA, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtr(rawA, aclrtFree);

    void* rawC = nullptr;
    aclRet = aclrtMalloc(&rawC, cSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for C failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dCPtr(rawC, aclrtFree);

    void* rawPivot = nullptr;
    aclRet = aclrtMalloc(&rawPivot, pivotSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for pivot failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dPivotPtr(rawPivot, aclrtFree);

    void* rawInfo = nullptr;
    aclRet = aclrtMalloc(&rawInfo, infoSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for info failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dInfoPtr(rawInfo, aclrtFree);

    void* rawAPtrs = nullptr;
    aclRet = aclrtMalloc(&rawAPtrs, sizeof(float*), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for APtrs failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtrs(rawAPtrs, aclrtFree);

    void* rawCPtrs = nullptr;
    aclRet = aclrtMalloc(&rawCPtrs, sizeof(float*), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for CPtrs failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dCPtrs(rawCPtrs, aclrtFree);

    aclRet = aclrtMemcpy(dAPtr.get(), aSize, hA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(dCPtr.get(), cSize, hC.data(), cSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for C failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(dPivotPtr.get(), pivotSize, hPivot.data(), pivotSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for pivot failed. ERROR: %d\n", aclRet); return aclRet);

    float* hAPtrHost = static_cast<float*>(dAPtr.get());
    float* hCPtrHost = static_cast<float*>(dCPtr.get());
    aclRet = aclrtMemcpy(dAPtrs.get(), sizeof(float*), &hAPtrHost, sizeof(float*), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for APtrs failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dCPtrs.get(), sizeof(float*), &hCPtrHost, sizeof(float*), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for CPtrs failed. ERROR: %d\n", aclRet); return aclRet);

    blasRet = aclblasSgetriBatched(
        static_cast<aclblasHandle_t>(handlePtr.get()), n,
        reinterpret_cast<const float* const*>(dAPtrs.get()), lda,
        static_cast<const int*>(dPivotPtr.get()),
        reinterpret_cast<float* const*>(dCPtrs.get()), ldc,
        static_cast<int*>(dInfoPtr.get()), batchSize);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSgetriBatched failed. ERROR: %d\n", blasRet);
              return blasRet);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(hC.data(), cSize, dCPtr.get(), cSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(hInfo.data(), infoSize, dInfoPtr.get(), infoSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy info failed. ERROR: %d\n", aclRet); return aclRet);

    LOG_PRINT("info[0] = %d\n", hInfo[0]);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            LOG_PRINT("C[%d][%d] = %f\n", i, j, hC[j * ldc + i]);
        }
    }

    LOG_PRINT("aclblasSgetriBatched test passed\n");
    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSgetriBatchedTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSgetriBatchedTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```
