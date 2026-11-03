# GemvStridedBatched算子

## 算子概述

GemvStridedBatched 算子实现了批量矩阵-向量乘法的跨步批处理运算。对 batchCount 个独立矩阵-向量对执行 GEMV 运算，每个 batch 通过显式 stride 参数定位向量，对齐 cuBLAS 的 `cublasSgemvStridedBatched` 系列接口。支持 FP32、FP16、BF16 及其混合精度变体。

数学表达式：

```
对每个 batch b = 0, 1, ..., batchCount-1:
  若 trans == N:
    y_b[i] = alpha * Σ_{j=0}^{n-1} A_b[i + j*lda] * x_b[j*incx] + beta * y_b[i*incy]
  若 trans == T (或 C):
    y_b[j] = alpha * Σ_{i=0}^{m-1} A_b[i + j*lda] * x_b[i*incx] + beta * y_b[j*incy]

  其中 A_b = A + b * strideA    (strideA 为显式参数)
       x_b = x + b * stridex    (stridex 为显式参数)
       y_b = y + b * stridey    (stridey 为显式参数)
```

矩阵 A 采用列主序（column-major）存储，A(i,j) = A[i + j*lda]。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSgemvStridedBatched | FP32 批量矩阵-向量跨步乘法 |
| aclblasHSHgemvStridedBatched | FP16 输入输出，FP32 计算 |
| aclblasHSSgemvStridedBatched | FP16 输入，FP32 输出 |
| aclblasTSTgemvStridedBatched | BF16 输入输出，FP32 计算 |
| aclblasTSSgemvStridedBatched | BF16 输入，FP32 输出 |

## 算子执行接口

### aclblasSgemvStridedBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSgemvStridedBatched(aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float *alpha, const float *A, int lda, int64_t strideA, const float *x, int incx, int64_t stridex, const float *beta, float *y, int incy, int64_t stridey, int batchCount)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| trans | 输入 | aclblasOperation_t | 矩阵运算类型：ACLBLAS_OP_N / ACLBLAS_OP_T / ACLBLAS_OP_C，Host 内存 |
| m | 输入 | int | 矩阵 A 的行数，m >= 0，Host 内存 |
| n | 输入 | int | 矩阵 A 的列数，n >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量缩放因子指针，Host 内存 |
| A | 输入 | const float*（FP32） | 一组列主序矩阵，第 b 个矩阵起始于 A + b * strideA，Device 内存 |
| lda | 输入 | int | 矩阵 A 的前导维度，lda >= max(1, m)，Host 内存 |
| strideA | 输入 | int64_t | batch 间 A 的步长（元素数），strideA > 0，Host 内存 |
| x | 输入 | const float*（FP32） | 一组向量，第 b 个向量起始于 x + b * stridex，Device 内存 |
| incx | 输入 | int | x 向量内元素步长，incx != 0 且 incx != INT_MIN，可正可负，Host 内存 |
| stridex | 输入 | int64_t | batch 间 x 的步长（元素数），stridex > 0，Host 内存 |
| beta | 输入 | const float*（FP32） | 标量缩放因子指针，Host 内存 |
| y | 输入/输出 | float*（FP32） | 一组向量，第 b 个向量起始于 y + b * stridey，被结果覆盖，Device 内存 |
| incy | 输入 | int | y 向量内元素步长，incy != 0 且 incy != INT_MIN，可正可负，Host 内存 |
| stridey | 输入 | int64_t | batch 间 y 的步长（元素数），stridey > 0，Host 内存 |
| batchCount | 输入 | int | 批量大小，batchCount >= 0，Host 内存 |

#### 约束说明

- m >= 0，n >= 0，batchCount >= 0；m == 0 或 n == 0 或 batchCount == 0 时直接返回成功（空操作）
- lda >= max(1, m)
- strideA > 0
- incx != 0 且 incx != INT_MIN
- incy != 0 且 incy != INT_MIN
- stridex > 0，stridey > 0
- alpha、beta 不能为 nullptr
- 若 m > 0 且 n > 0，A、x、y 不能为 nullptr
- trans = N 时使用 SIMT 编程模型；trans = T/C 且 incx = 1 且 incy = 1 时使用 AIV SIMD 加速路径；否则回退 SIMT
- 矩阵 A 为列主序存储，strideA >= lda * n 通常成立
- 算子输入 shape：A 为 [batchCount, lda, n]，x 为 [batchCount, *]，输出 shape：y 为 [batchCount, *]
- Host 侧不做流同步，调用方需自行管理同步

#### 调用示例

示例代码如下。在项目根目录执行以下命令编译和运行aclblasSgemvStridedBatched算子示例：

```bash
bash build.sh --pkg --soc=${soc_version} --ops=gemv_strided_batched --run
```

具体编译和执行细节请参见[编译与运行示例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

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

int aclblasSgemvStridedBatchedTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-blas 句柄
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasHandle_t>::type, BlasHandleDeleter> handlePtr(rawHandle);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 2. 准备 Host 数据
    int m = 8, n = 4, batchCount = 2;
    int lda = m, incx = 1, incy = 1;
    // trans=T: A^T is n×m, so x has m elements and y has n elements
    int64_t strideA = static_cast<int64_t>(lda) * n;
    int64_t stridex = m, stridey = n;
    float alpha = 1.0f, beta = 0.0f;

    size_t aBytes = static_cast<size_t>(batchCount) * strideA * sizeof(float);
    size_t xBytes = static_cast<size_t>(batchCount) * stridex * sizeof(float);
    size_t yBytes = static_cast<size_t>(batchCount) * stridey * sizeof(float);

    std::vector<float> hA(aBytes / sizeof(float), 1.0f);
    std::vector<float> hX(xBytes / sizeof(float), 2.0f);
    std::vector<float> hY(yBytes / sizeof(float), 0.0f);

    // 3. 申请 Device 内存并拷贝数据
    void *dA = nullptr, *dx = nullptr, *dy = nullptr;
    auto aclRet = aclrtMalloc(&dA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc dA failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtr(dA, aclrtFree);

    aclRet = aclrtMalloc(&dx, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc dx failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dxPtr(dx, aclrtFree);

    aclRet = aclrtMalloc(&dy, yBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc dy failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dyPtr(dy, aclrtFree);

    aclRet = aclrtMemcpy(dA, aBytes, hA.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D dA failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(dx, xBytes, hX.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D dx failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(dy, yBytes, hY.data(), yBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D dy failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasSgemvStridedBatched
    blasRet = aclblasSgemvStridedBatched(
        static_cast<aclblasHandle_t>(handlePtr.get()),
        ACLBLAS_OP_T, m, n, &alpha,
        static_cast<const float*>(dA), lda, strideA,
        static_cast<const float*>(dx), incx, stridex,
        &beta,
        static_cast<float*>(dy), incy, stridey,
        batchCount);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS,
              LOG_PRINT("aclblasSgemvStridedBatched failed. ERROR: %d\n", blasRet); return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    std::vector<float> hYOut(static_cast<size_t>(batchCount) * stridey);
    aclRet = aclrtMemcpy(hYOut.data(), yBytes, dy, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy D2H dy failed. ERROR: %d\n", aclRet); return aclRet);

    // trans=T: output y has n elements per batch
    for (int b = 0; b < batchCount; b++) {
        LOG_PRINT("Batch %d:", b);
        for (int i = 0; i < n; i++) {
            LOG_PRINT(" %.4f", static_cast<double>(hYOut[b * stridey + i * incy]));
        }
        LOG_PRINT("\n");
    }

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSgemvStridedBatchedTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSgemvStridedBatchedTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

### aclblasHSHgemvStridedBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasHSHgemvStridedBatched(aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float *alpha, const uint16_t *A, int lda, int64_t strideA, const uint16_t *x, int incx, int64_t stridex, const float *beta, uint16_t *y, int incy, int64_t stridey, int batchCount)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| trans | 输入 | aclblasOperation_t | 矩阵运算类型：ACLBLAS_OP_N / ACLBLAS_OP_T / ACLBLAS_OP_C，Host 内存 |
| m | 输入 | int | 矩阵 A 的行数，m >= 0，Host 内存 |
| n | 输入 | int | 矩阵 A 的列数，n >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量缩放因子指针，Host 内存 |
| A | 输入 | const uint16_t*（FP16） | 一组列主序矩阵，第 b 个矩阵起始于 A + b * strideA，Device 内存 |
| lda | 输入 | int | 矩阵 A 的前导维度，lda >= max(1, m)，Host 内存 |
| strideA | 输入 | int64_t | batch 间 A 的步长（元素数），strideA > 0，Host 内存 |
| x | 输入 | const uint16_t*（FP16） | 一组向量，第 b 个向量起始于 x + b * stridex，Device 内存 |
| incx | 输入 | int | x 向量内元素步长，incx != 0 且 incx != INT_MIN，Host 内存 |
| stridex | 输入 | int64_t | batch 间 x 的步长（元素数），stridex > 0，Host 内存 |
| beta | 输入 | const float*（FP32） | 标量缩放因子指针，Host 内存 |
| y | 输入/输出 | uint16_t*（FP16） | 一组向量，第 b 个向量起始于 y + b * stridey，被结果覆盖，Device 内存 |
| incy | 输入 | int | y 向量内元素步长，incy != 0 且 incy != INT_MIN，Host 内存 |
| stridey | 输入 | int64_t | batch 间 y 的步长（元素数），stridey > 0，Host 内存 |
| batchCount | 输入 | int | 批量大小，batchCount >= 0，Host 内存 |

#### 约束说明

与 aclblasSgemvStridedBatched 相同。A 和 x 为 FP16 数据（用 uint16_t 表示），y 为 FP16 输出，alpha/beta 始终为 FP32 标量。内部使用 FP32 精度计算，结果转换为 FP16 输出。

#### 调用示例

调用方式与 aclblasSgemvStridedBatched 类似，仅将 A、x、y 数据类型从 `float` 替换为 `uint16_t`（FP16）。完整示例代码如下：

```cpp
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

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

class AclContext {
public:
    explicit AclContext(int32_t deviceId) : deviceId_(deviceId) {}
    ~AclContext()
    {
        if (stream_ != nullptr) { aclrtDestroyStream(stream_); stream_ = nullptr; }
        if (deviceSet_) { aclrtResetDevice(deviceId_); deviceSet_ = false; }
        if (aclInited_) { aclFinalize(); aclInited_ = false; }
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
    bool aclInited_ = false, deviceSet_ = false;
};

static float fp16_to_float(uint16_t h)
{
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        bits = sign;
    } else if (exp == 0x1F) {
        bits = sign | 0x7F800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

struct BlasHandleDeleter {
    void operator()(aclblasHandle_t h) const { aclblasDestroy(h); }
};

int aclblasHSHgemvStridedBatchedTest(AclContext& ctx)
{
    // 1. 创建 ops-blas 句柄
    aclrtStream stream = ctx.Stream();
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS,
              LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet); return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasHandle_t>::type, BlasHandleDeleter> handlePtr(rawHandle);
    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS,
              LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet); return blasRet);

    // 2. 准备 Host 数据
    int m = 8, n = 4, batchCount = 2;
    LOG_PRINT("[INFO] m=%d, n=%d, batchCount=%d, trans=T\n", m, n, batchCount);
    int lda = m, incx = 1, incy = 1;
    int64_t strideA = static_cast<int64_t>(lda) * n;
    int64_t stridex = m, stridey = n;
    float alpha = 1.0f, beta = 0.0f;  // FP16 接口的 alpha/beta 仍为 FP32
    LOG_PRINT("[INFO] alpha=%.1f, beta=%.1f, lda=%d, incx=%d, incy=%d\n", alpha, beta, lda, incx, incy);

    size_t elemSize = sizeof(uint16_t);
    size_t aBytes = static_cast<size_t>(batchCount) * strideA * elemSize;
    size_t xBytes = static_cast<size_t>(batchCount) * stridex * elemSize;
    size_t yBytes = static_cast<size_t>(batchCount) * stridey * elemSize;
    if (batchCount == 0) {
        LOG_PRINT("[INFO] batchCount=0, skip computation\n");
        return ACL_SUCCESS;
    }

    // FP16 数据: 1.0f = 0x3C00, 2.0f = 0x4000
    std::vector<uint16_t> hA(batchCount * strideA, 0x3C00);
    std::vector<uint16_t> hX(batchCount * stridex, 0x4000);
    std::vector<uint16_t> hY(batchCount * stridey, 0);

    // 3. 申请 Device 内存并拷贝数据
    void *dA = nullptr, *dx = nullptr, *dy = nullptr;
    auto aclRet = aclrtMalloc(&dA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc dA failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtr(dA, aclrtFree);
    aclRet = aclrtMalloc(&dx, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc dx failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dxPtr(dx, aclrtFree);
    aclRet = aclrtMalloc(&dy, yBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc dy failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dyPtr(dy, aclrtFree);

    aclRet = aclrtMemcpy(dA, aBytes, hA.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D dA failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dx, xBytes, hX.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D dx failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasHSHgemvStridedBatched
    LOG_PRINT("[INFO] Calling aclblasHSHgemvStridedBatched...\n");
    blasRet = aclblasHSHgemvStridedBatched(
        static_cast<aclblasHandle_t>(handlePtr.get()),
        ACLBLAS_OP_T, m, n, &alpha,
        static_cast<const uint16_t*>(dA), lda, strideA,
        static_cast<const uint16_t*>(dx), incx, stridex,
        &beta,
        static_cast<uint16_t*>(dy), incy, stridey,
        batchCount);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS,
              LOG_PRINT("aclblasHSHgemvStridedBatched failed. ERROR: %d\n", blasRet); return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    std::vector<uint16_t> hYOut(batchCount * stridey);
    aclRet = aclrtMemcpy(hYOut.data(), yBytes, dy, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy D2H dy failed. ERROR: %d\n", aclRet); return aclRet);

    for (int b = 0; b < batchCount; b++) {
        LOG_PRINT("Batch %d:", b);
        for (int i = 0; i < n; i++) {
            LOG_PRINT(" %.4f", static_cast<double>(fp16_to_float(hYOut[b * stridey + i * incy])));
        }
        LOG_PRINT("\n");
    }
    LOG_PRINT("[INFO] aclblasHSHgemvStridedBatched test PASSED\n");
    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("AclContext Init failed. ERROR: %d\n", ret); return ret);
    ret = aclblasHSHgemvStridedBatchedTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasHSHgemvStridedBatchedTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

### aclblasHSSgemvStridedBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasHSSgemvStridedBatched(aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float *alpha, const uint16_t *A, int lda, int64_t strideA, const uint16_t *x, int incx, int64_t stridex, const float *beta, float *y, int incy, int64_t stridey, int batchCount)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| trans | 输入 | aclblasOperation_t | 矩阵运算类型：ACLBLAS_OP_N / ACLBLAS_OP_T / ACLBLAS_OP_C，Host 内存 |
| m | 输入 | int | 矩阵 A 的行数，m >= 0，Host 内存 |
| n | 输入 | int | 矩阵 A 的列数，n >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量缩放因子指针，Host 内存 |
| A | 输入 | const uint16_t*（FP16） | 一组列主序矩阵，第 b 个矩阵起始于 A + b * strideA，Device 内存 |
| lda | 输入 | int | 矩阵 A 的前导维度，lda >= max(1, m)，Host 内存 |
| strideA | 输入 | int64_t | batch 间 A 的步长（元素数），strideA > 0，Host 内存 |
| x | 输入 | const uint16_t*（FP16） | 一组向量，第 b 个向量起始于 x + b * stridex，Device 内存 |
| incx | 输入 | int | x 向量内元素步长，incx != 0 且 incx != INT_MIN，Host 内存 |
| stridex | 输入 | int64_t | batch 间 x 的步长（元素数），stridex > 0，Host 内存 |
| beta | 输入 | const float*（FP32） | 标量缩放因子指针，Host 内存 |
| y | 输入/输出 | float*（FP32） | 一组向量，第 b 个向量起始于 y + b * stridey，被结果覆盖，Device 内存 |
| incy | 输入 | int | y 向量内元素步长，incy != 0 且 incy != INT_MIN，Host 内存 |
| stridey | 输入 | int64_t | batch 间 y 的步长（元素数），stridey > 0，Host 内存 |
| batchCount | 输入 | int | 批量大小，batchCount >= 0，Host 内存 |

#### 约束说明

与 aclblasSgemvStridedBatched 相同。A 和 x 为 FP16 数据（用 uint16_t 表示），y 为 FP32 输出，alpha/beta 始终为 FP32 标量。内部使用 FP32 精度计算，结果直接以 FP32 输出（无精度损失）。

#### 调用示例

调用方式与 aclblasSgemvStridedBatched 类似，A 和 x 使用 `uint16_t`（FP16），y 使用 `float`（FP32）。完整示例代码如下：

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
        if (stream_ != nullptr) { aclrtDestroyStream(stream_); stream_ = nullptr; }
        if (deviceSet_) { aclrtResetDevice(deviceId_); deviceSet_ = false; }
        if (aclInited_) { aclFinalize(); aclInited_ = false; }
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
    bool aclInited_ = false, deviceSet_ = false;
};

struct BlasHandleDeleter {
    void operator()(aclblasHandle_t h) const { aclblasDestroy(h); }
};

int aclblasHSSgemvStridedBatchedTest(AclContext& ctx)
{
    // 1. 创建 ops-blas 句柄
    aclrtStream stream = ctx.Stream();
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS,
              LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet); return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasHandle_t>::type, BlasHandleDeleter> handlePtr(rawHandle);
    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS,
              LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet); return blasRet);

    // 2. 准备 Host 数据
    int m = 8, n = 4, batchCount = 2;
    LOG_PRINT("[INFO] m=%d, n=%d, batchCount=%d, trans=T\n", m, n, batchCount);
    int lda = m, incx = 1, incy = 1;
    int64_t strideA = static_cast<int64_t>(lda) * n;
    int64_t stridex = m, stridey = n;
    float alpha = 1.0f, beta = 0.0f;  // alpha/beta 为 FP32
    LOG_PRINT("[INFO] alpha=%.1f, beta=%.1f, lda=%d, incx=%d, incy=%d\n", alpha, beta, lda, incx, incy);

    size_t aBytes = static_cast<size_t>(batchCount) * strideA * sizeof(uint16_t);
    size_t xBytes = static_cast<size_t>(batchCount) * stridex * sizeof(uint16_t);
    size_t yBytes = static_cast<size_t>(batchCount) * stridey * sizeof(float);  // y: FP32
    if (batchCount == 0) {
        LOG_PRINT("[INFO] batchCount=0, skip computation\n");
        return ACL_SUCCESS;
    }

    // FP16 数据: 1.0f = 0x3C00, 2.0f = 0x4000
    std::vector<uint16_t> hA(batchCount * strideA, 0x3C00);
    std::vector<uint16_t> hX(batchCount * stridex, 0x4000);
    std::vector<float> hY(batchCount * stridey, 0.0f);

    // 3. 申请 Device 内存并拷贝数据
    void *dA = nullptr, *dx = nullptr, *dy = nullptr;
    auto aclRet = aclrtMalloc(&dA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc dA failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtr(dA, aclrtFree);
    aclRet = aclrtMalloc(&dx, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc dx failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dxPtr(dx, aclrtFree);
    aclRet = aclrtMalloc(&dy, yBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc dy failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dyPtr(dy, aclrtFree);

    aclRet = aclrtMemcpy(dA, aBytes, hA.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D dA failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dx, xBytes, hX.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D dx failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasHSSgemvStridedBatched
    LOG_PRINT("[INFO] Calling aclblasHSSgemvStridedBatched...\n");
    blasRet = aclblasHSSgemvStridedBatched(
        static_cast<aclblasHandle_t>(handlePtr.get()),
        ACLBLAS_OP_T, m, n, &alpha,
        static_cast<const uint16_t*>(dA), lda, strideA,
        static_cast<const uint16_t*>(dx), incx, stridex,
        &beta,
        static_cast<float*>(dy), incy, stridey,
        batchCount);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS,
              LOG_PRINT("aclblasHSSgemvStridedBatched failed. ERROR: %d\n", blasRet); return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印（y 为 FP32，直接打印）
    std::vector<float> hYOut(batchCount * stridey);
    aclRet = aclrtMemcpy(hYOut.data(), yBytes, dy, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy D2H dy failed. ERROR: %d\n", aclRet); return aclRet);

    for (int b = 0; b < batchCount; b++) {
        LOG_PRINT("Batch %d:", b);
        for (int i = 0; i < n; i++) {
            LOG_PRINT(" %.4f", static_cast<double>(hYOut[b * stridey + i * incy]));
        }
        LOG_PRINT("\n");
    }
    LOG_PRINT("[INFO] aclblasHSSgemvStridedBatched test PASSED\n");
    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("AclContext Init failed. ERROR: %d\n", ret); return ret);
    ret = aclblasHSSgemvStridedBatchedTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasHSSgemvStridedBatchedTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

### aclblasTSTgemvStridedBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasTSTgemvStridedBatched(aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float *alpha, const uint16_t *A, int lda, int64_t strideA, const uint16_t *x, int incx, int64_t stridex, const float *beta, uint16_t *y, int incy, int64_t stridey, int batchCount)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| trans | 输入 | aclblasOperation_t | 矩阵运算类型：ACLBLAS_OP_N / ACLBLAS_OP_T / ACLBLAS_OP_C，Host 内存 |
| m | 输入 | int | 矩阵 A 的行数，m >= 0，Host 内存 |
| n | 输入 | int | 矩阵 A 的列数，n >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量缩放因子指针，Host 内存 |
| A | 输入 | const uint16_t*（BF16） | 一组列主序矩阵，第 b 个矩阵起始于 A + b * strideA，Device 内存 |
| lda | 输入 | int | 矩阵 A 的前导维度，lda >= max(1, m)，Host 内存 |
| strideA | 输入 | int64_t | batch 间 A 的步长（元素数），strideA > 0，Host 内存 |
| x | 输入 | const uint16_t*（BF16） | 一组向量，第 b 个向量起始于 x + b * stridex，Device 内存 |
| incx | 输入 | int | x 向量内元素步长，incx != 0 且 incx != INT_MIN，Host 内存 |
| stridex | 输入 | int64_t | batch 间 x 的步长（元素数），stridex > 0，Host 内存 |
| beta | 输入 | const float*（FP32） | 标量缩放因子指针，Host 内存 |
| y | 输入/输出 | uint16_t*（BF16） | 一组向量，第 b 个向量起始于 y + b * stridey，被结果覆盖，Device 内存 |
| incy | 输入 | int | y 向量内元素步长，incy != 0 且 incy != INT_MIN，Host 内存 |
| stridey | 输入 | int64_t | batch 间 y 的步长（元素数），stridey > 0，Host 内存 |
| batchCount | 输入 | int | 批量大小，batchCount >= 0，Host 内存 |

#### 约束说明

与 aclblasSgemvStridedBatched 相同。A 和 x 为 BF16 数据（用 uint16_t 表示），y 为 BF16 输出，alpha/beta 始终为 FP32 标量。内部使用 FP32 精度计算，结果转换为 BF16 输出。

#### 调用示例

调用方式与 aclblasHSHgemvStridedBatched 类似，仅数据类型从 FP16 替换为 BF16（同样用 uint16_t 表示）。完整示例代码如下：

```cpp
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

#define LOG_PRINT(message, ...)         \
    do {                                \
        printf(message, ##__VA_ARGS__); \
    } while (0)

class AclContext {
public:
    explicit AclContext(int32_t deviceId) : deviceId_(deviceId) {}
    ~AclContext()
    {
        if (stream_ != nullptr) { aclrtDestroyStream(stream_); stream_ = nullptr; }
        if (deviceSet_) { aclrtResetDevice(deviceId_); deviceSet_ = false; }
        if (aclInited_) { aclFinalize(); aclInited_ = false; }
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
    bool aclInited_ = false, deviceSet_ = false;
};

static float bf16_to_float(uint16_t bf)
{
    uint32_t bits = (uint32_t)bf << 16;
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

struct BlasHandleDeleter {
    void operator()(aclblasHandle_t h) const { aclblasDestroy(h); }
};

int aclblasTSTgemvStridedBatchedTest(AclContext& ctx)
{
    // 1. 创建 ops-blas 句柄
    aclrtStream stream = ctx.Stream();
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS,
              LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet); return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasHandle_t>::type, BlasHandleDeleter> handlePtr(rawHandle);
    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS,
              LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet); return blasRet);

    // 2. 准备 Host 数据
    int m = 8, n = 4, batchCount = 2;
    LOG_PRINT("[INFO] m=%d, n=%d, batchCount=%d, trans=T\n", m, n, batchCount);
    int lda = m, incx = 1, incy = 1;
    int64_t strideA = static_cast<int64_t>(lda) * n;
    int64_t stridex = m, stridey = n;
    float alpha = 1.0f, beta = 0.0f;  // BF16 接口的 alpha/beta 为 FP32
    LOG_PRINT("[INFO] alpha=%.1f, beta=%.1f, lda=%d, incx=%d, incy=%d\n", alpha, beta, lda, incx, incy);

    size_t elemSize = sizeof(uint16_t);  // BF16 用 uint16_t 表示
    size_t aBytes = static_cast<size_t>(batchCount) * strideA * elemSize;
    size_t xBytes = static_cast<size_t>(batchCount) * stridex * elemSize;
    size_t yBytes = static_cast<size_t>(batchCount) * stridey * elemSize;
    if (batchCount == 0) {
        LOG_PRINT("[INFO] batchCount=0, skip computation\n");
        return ACL_SUCCESS;
    }

    // BF16 数据: 1.0f = 0x3F80, 2.0f = 0x4000
    std::vector<uint16_t> hA(batchCount * strideA, 0x3F80);
    std::vector<uint16_t> hX(batchCount * stridex, 0x4000);
    std::vector<uint16_t> hY(batchCount * stridey, 0);

    // 3. 申请 Device 内存并拷贝数据
    void *dA = nullptr, *dx = nullptr, *dy = nullptr;
    auto aclRet = aclrtMalloc(&dA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc dA failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtr(dA, aclrtFree);
    aclRet = aclrtMalloc(&dx, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc dx failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dxPtr(dx, aclrtFree);
    aclRet = aclrtMalloc(&dy, yBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc dy failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dyPtr(dy, aclrtFree);

    aclRet = aclrtMemcpy(dA, aBytes, hA.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D dA failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dx, xBytes, hX.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D dx failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasTSTgemvStridedBatched
    LOG_PRINT("[INFO] Calling aclblasTSTgemvStridedBatched...\n");
    blasRet = aclblasTSTgemvStridedBatched(
        static_cast<aclblasHandle_t>(handlePtr.get()),
        ACLBLAS_OP_T, m, n, &alpha,
        static_cast<const uint16_t*>(dA), lda, strideA,
        static_cast<const uint16_t*>(dx), incx, stridex,
        &beta,
        static_cast<uint16_t*>(dy), incy, stridey,
        batchCount);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS,
              LOG_PRINT("aclblasTSTgemvStridedBatched failed. ERROR: %d\n", blasRet); return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    std::vector<uint16_t> hYOut(batchCount * stridey);
    aclRet = aclrtMemcpy(hYOut.data(), yBytes, dy, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy D2H dy failed. ERROR: %d\n", aclRet); return aclRet);

    for (int b = 0; b < batchCount; b++) {
        LOG_PRINT("Batch %d:", b);
        for (int i = 0; i < n; i++) {
            LOG_PRINT(" %.4f", static_cast<double>(bf16_to_float(hYOut[b * stridey + i * incy])));
        }
        LOG_PRINT("\n");
    }
    LOG_PRINT("[INFO] aclblasTSTgemvStridedBatched test PASSED\n");
    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("AclContext Init failed. ERROR: %d\n", ret); return ret);
    ret = aclblasTSTgemvStridedBatchedTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasTSTgemvStridedBatchedTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

### aclblasTSSgemvStridedBatched

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasTSSgemvStridedBatched(aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, const float *alpha, const uint16_t *A, int lda, int64_t strideA, const uint16_t *x, int incx, int64_t stridex, const float *beta, float *y, int incy, int64_t stridey, int batchCount)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| trans | 输入 | aclblasOperation_t | 矩阵运算类型：ACLBLAS_OP_N / ACLBLAS_OP_T / ACLBLAS_OP_C，Host 内存 |
| m | 输入 | int | 矩阵 A 的行数，m >= 0，Host 内存 |
| n | 输入 | int | 矩阵 A 的列数，n >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量缩放因子指针，Host 内存 |
| A | 输入 | const uint16_t*（BF16） | 一组列主序矩阵，第 b 个矩阵起始于 A + b * strideA，Device 内存 |
| lda | 输入 | int | 矩阵 A 的前导维度，lda >= max(1, m)，Host 内存 |
| strideA | 输入 | int64_t | batch 间 A 的步长（元素数），strideA > 0，Host 内存 |
| x | 输入 | const uint16_t*（BF16） | 一组向量，第 b 个向量起始于 x + b * stridex，Device 内存 |
| incx | 输入 | int | x 向量内元素步长，incx != 0 且 incx != INT_MIN，Host 内存 |
| stridex | 输入 | int64_t | batch 间 x 的步长（元素数），stridex > 0，Host 内存 |
| beta | 输入 | const float*（FP32） | 标量缩放因子指针，Host 内存 |
| y | 输入/输出 | float*（FP32） | 一组向量，第 b 个向量起始于 y + b * stridey，被结果覆盖，Device 内存 |
| incy | 输入 | int | y 向量内元素步长，incy != 0 且 incy != INT_MIN，Host 内存 |
| stridey | 输入 | int64_t | batch 间 y 的步长（元素数），stridey > 0，Host 内存 |
| batchCount | 输入 | int | 批量大小，batchCount >= 0，Host 内存 |

#### 约束说明

与 aclblasSgemvStridedBatched 相同。A 和 x 为 BF16 数据（用 uint16_t 表示），y 为 FP32 输出，alpha/beta 始终为 FP32 标量。内部使用 FP32 精度计算，结果直接以 FP32 输出（无精度损失）。

#### 调用示例

调用方式与 aclblasHSSgemvStridedBatched 类似，A/x 数据类型从 FP16 替换为 BF16（同样用 uint16_t 表示），y 保持 float（FP32）。完整示例代码如下：

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
        if (stream_ != nullptr) { aclrtDestroyStream(stream_); stream_ = nullptr; }
        if (deviceSet_) { aclrtResetDevice(deviceId_); deviceSet_ = false; }
        if (aclInited_) { aclFinalize(); aclInited_ = false; }
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
    bool aclInited_ = false, deviceSet_ = false;
};

struct BlasHandleDeleter {
    void operator()(aclblasHandle_t h) const { aclblasDestroy(h); }
};

int aclblasTSSgemvStridedBatchedTest(AclContext& ctx)
{
    // 1. 创建 ops-blas 句柄
    aclrtStream stream = ctx.Stream();
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS,
              LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet); return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasHandle_t>::type, BlasHandleDeleter> handlePtr(rawHandle);
    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS,
              LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet); return blasRet);

    // 2. 准备 Host 数据
    int m = 8, n = 4, batchCount = 2;
    LOG_PRINT("[INFO] m=%d, n=%d, batchCount=%d, trans=T\n", m, n, batchCount);
    int lda = m, incx = 1, incy = 1;
    int64_t strideA = static_cast<int64_t>(lda) * n;
    int64_t stridex = m, stridey = n;
    float alpha = 1.0f, beta = 0.0f;
    LOG_PRINT("[INFO] alpha=%.1f, beta=%.1f, lda=%d, incx=%d, incy=%d\n", alpha, beta, lda, incx, incy);

    size_t aBytes = static_cast<size_t>(batchCount) * strideA * sizeof(uint16_t);  // A: BF16
    size_t xBytes = static_cast<size_t>(batchCount) * stridex * sizeof(uint16_t);  // x: BF16
    size_t yBytes = static_cast<size_t>(batchCount) * stridey * sizeof(float);     // y: FP32
    if (batchCount == 0) {
        LOG_PRINT("[INFO] batchCount=0, skip computation\n");
        return ACL_SUCCESS;
    }

    // BF16 数据: 1.0f = 0x3F80, 2.0f = 0x4000
    std::vector<uint16_t> hA(batchCount * strideA, 0x3F80);
    std::vector<uint16_t> hX(batchCount * stridex, 0x4000);
    std::vector<float> hY(batchCount * stridey, 0.0f);

    // 3. 申请 Device 内存并拷贝数据
    void *dA = nullptr, *dx = nullptr, *dy = nullptr;
    auto aclRet = aclrtMalloc(&dA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc dA failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtr(dA, aclrtFree);
    aclRet = aclrtMalloc(&dx, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc dx failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dxPtr(dx, aclrtFree);
    aclRet = aclrtMalloc(&dy, yBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc dy failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dyPtr(dy, aclrtFree);

    aclRet = aclrtMemcpy(dA, aBytes, hA.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D dA failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dx, xBytes, hX.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D dx failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasTSSgemvStridedBatched
    LOG_PRINT("[INFO] Calling aclblasTSSgemvStridedBatched...\n");
    blasRet = aclblasTSSgemvStridedBatched(
        static_cast<aclblasHandle_t>(handlePtr.get()),
        ACLBLAS_OP_T, m, n, &alpha,
        static_cast<const uint16_t*>(dA), lda, strideA,
        static_cast<const uint16_t*>(dx), incx, stridex,
        &beta,
        static_cast<float*>(dy), incy, stridey,
        batchCount);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS,
              LOG_PRINT("aclblasTSSgemvStridedBatched failed. ERROR: %d\n", blasRet); return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印（y 为 FP32，直接打印）
    std::vector<float> hYOut(batchCount * stridey);
    aclRet = aclrtMemcpy(hYOut.data(), yBytes, dy, yBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy D2H dy failed. ERROR: %d\n", aclRet); return aclRet);

    for (int b = 0; b < batchCount; b++) {
        LOG_PRINT("Batch %d:", b);
        for (int i = 0; i < n; i++) {
            LOG_PRINT(" %.4f", static_cast<double>(hYOut[b * stridey + i * incy]));
        }
        LOG_PRINT("\n");
    }
    LOG_PRINT("[INFO] aclblasTSSgemvStridedBatched test PASSED\n");
    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("AclContext Init failed. ERROR: %d\n", ret); return ret);
    ret = aclblasTSSgemvStridedBatchedTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasTSSgemvStridedBatchedTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```
