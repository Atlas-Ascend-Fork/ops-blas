# GemmEx算子

## 算子概述

通用矩阵乘法扩展接口（GEMM Ex），支持 A、B、C 矩阵使用独立数据类型。本目录包含两个接口：

- **aclblasGemmEx**：通用矩阵乘法扩展接口，支持 A/B/C 矩阵使用不同数据类型（FP16/BF16/FP32）
- **aclblasSgemmEx**：单精度（FP32）通用矩阵乘法扩展接口，A/B/C 及 alpha/beta 均为 FP32，保留算法选择参数 `algo`，与 `aclblasGemmEx` 接口对齐

数学表达式：

```
C = alpha * op(A) * op(B) + beta * C
```

其中：
- op(A) = A（transA = ACLBLAS_OP_N）、A^T（transA = ACLBLAS_OP_T）或 A^H（transA = ACLBLAS_OP_C，FP32 实数场景下等价于 A^T）
- op(B) = B（transB = ACLBLAS_OP_N）、B^T（transB = ACLBLAS_OP_T）或 B^H（transB = ACLBLAS_OP_C，FP32 实数场景下等价于 B^T）
- A 为 M×K 矩阵，B 为 K×N 矩阵，C 为 M×N 矩阵
- 矩阵采用 BLAS 标准列主序存储

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasGemmEx | 通用矩阵乘法扩展接口，支持 A/B/C 矩阵使用独立数据类型 |
| aclblasSgemmEx | 单精度（FP32）通用矩阵乘法，支持矩阵转置和 alpha/beta 缩放 |

## 算子执行接口

### aclblasGemmEx

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasGemmEx(aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k, const void* alpha, const void* A, aclDataType Atype, int lda, const void* B, aclDataType Btype, int ldb, const void* beta, void* C, aclDataType Ctype, int ldc, aclblasComputeType_t computeType, aclblasGemmAlgo_t algo)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| transA | 输入 | aclblasOperation_t | 矩阵 A 的操作类型：ACLBLAS_OP_N（不转置）、ACLBLAS_OP_T（转置）、ACLBLAS_OP_C（共轭转置），Host 内存 |
| transB | 输入 | aclblasOperation_t | 矩阵 B 的操作类型（同 transA），Host 内存 |
| m | 输入 | int | op(A) 和 C 的行数，M >= 0，Host 内存 |
| n | 输入 | int | op(B) 和 C 的列数，N >= 0，Host 内存 |
| k | 输入 | int | op(A) 的列数和 op(B) 的行数，K >= 0，Host 内存 |
| alpha | 输入 | const void* | 标量 alpha 指针，指向 FP32 标量，不可为 nullptr，Host 内存 |
| A | 输入 | const void* | 矩阵 A 的设备内存指针，列主序；当 K > 0 时不可为 nullptr，Device 内存 |
| Atype | 输入 | aclDataType | 矩阵 A 的数据类型，Host 内存 |
| lda | 输入 | int | 矩阵 A 的主维度（列主序），transA=N 时 lda >= max(1, M)，transA=T/C 时 lda >= max(1, K)，Host 内存 |
| B | 输入 | const void* | 矩阵 B 的设备内存指针，列主序；当 K > 0 时不可为 nullptr，Device 内存 |
| Btype | 输入 | aclDataType | 矩阵 B 的数据类型，Host 内存 |
| ldb | 输入 | int | 矩阵 B 的主维度（列主序），transB=N 时 ldb >= max(1, K)，transB=T/C 时 ldb >= max(1, N)，Host 内存 |
| beta | 输入 | const void* | 标量 beta 指针，指向 FP32 标量，不可为 nullptr，Host 内存 |
| C | 输入/输出 | void* | 矩阵 C 的设备内存指针，列主序；当 beta != 0 时不可为 nullptr，Device 内存 |
| Ctype | 输入 | aclDataType | 矩阵 C 的数据类型，Host 内存 |
| ldc | 输入 | int | 矩阵 C 的主维度（列主序），ldc >= max(1, M)，Host 内存 |
| computeType | 输入 | aclblasComputeType_t | 计算精度类型，Host 内存 |
| algo | 输入 | aclblasGemmAlgo_t | 算法选择，当前仅支持 ACLBLAS_GEMM_DEFAULT，Host 内存 |

#### 数据类型支持

| Atype | Btype | Ctype | computeType |
|-------|-------|-------|-------------|
| ACL_FLOAT16 | ACL_FLOAT16 | ACL_FLOAT16 | ACL_COMPUTE_HIGH_PRECISION |
| ACL_BF16 | ACL_BF16 | ACL_BF16 | ACL_COMPUTE_HIGH_PRECISION |
| ACL_FLOAT | ACL_FLOAT | ACL_FLOAT | ACL_COMPUTE_HIGH_PRECISION |

#### 约束说明

- m >= 0
- n >= 0
- k >= 0
- transA 必须为 ACLBLAS_OP_N、ACLBLAS_OP_T 或 ACLBLAS_OP_C
- transB 必须为 ACLBLAS_OP_N、ACLBLAS_OP_T 或 ACLBLAS_OP_C
- transA = N 时 lda >= max(1, m)；transA = T/C 时 lda >= max(1, k)
- transB = N 时 ldb >= max(1, k)；transB = T/C 时 ldb >= max(1, n)
- ldc >= max(1, m)
- alpha 不可为 nullptr
- beta 不可为 nullptr
- algo 当前仅支持 ACLBLAS_GEMM_DEFAULT
- k > 0 时 A 不可为 nullptr
- k > 0 时 B 不可为 nullptr
- beta != 0.0f 时 C 不可为 nullptr
- Atype、Btype、Ctype 必须为上表所列的有效组合

边界情况处理：
- m == 0 或 n == 0 时直接返回 ACLBLAS_STATUS_SUCCESS，不执行计算
- k == 0 或 alpha == 0.0f 时跳过矩阵乘，执行 C = beta * C（beta == 0 时置零，beta == 1 时不变，其他值逐元素缩放）
- C == nullptr 且 beta == 0.0f 时直接返回 ACLBLAS_STATUS_SUCCESS

### aclblasSgemmEx

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSgemmEx(
    aclblasHandle_t handle, aclblasOperation_t transA, aclblasOperation_t transB,
    int m, int n, int k, const float* alpha,
    const float* A, int lda, const float* B, int ldb,
    const float* beta, float* C, int ldc, aclblasGemmAlgo_t algo)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| transA | 输入 | aclblasOperation_t | 矩阵 A 的操作类型：ACLBLAS_OP_N（不转置）、ACLBLAS_OP_T（转置）、ACLBLAS_OP_C（共轭转置，FP32 实数等价于转置），Host 内存 |
| transB | 输入 | aclblasOperation_t | 矩阵 B 的操作类型（同 transA），Host 内存 |
| m | 输入 | int | op(A) 和 C 的行数，M >= 0，Host 内存 |
| n | 输入 | int | op(B) 和 C 的列数，N >= 0，Host 内存 |
| k | 输入 | int | op(A) 的列数和 op(B) 的行数，K >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量 alpha 指针，指向 FP32 标量，不可为 nullptr，Host 内存 |
| A | 输入 | const float*（FP32） | 矩阵 A 的设备内存指针，FP32，列主序；当 K > 0 时不可为 nullptr，Device 内存 |
| lda | 输入 | int | 矩阵 A 的主维度（列主序），transA=N 时 lda >= max(1, M)，transA=T/C 时 lda >= max(1, K)，Host 内存 |
| B | 输入 | const float*（FP32） | 矩阵 B 的设备内存指针，FP32，列主序；当 K > 0 时不可为 nullptr，Device 内存 |
| ldb | 输入 | int | 矩阵 B 的主维度（列主序），transB=N 时 ldb >= max(1, K)，transB=T/C 时 ldb >= max(1, N)，Host 内存 |
| beta | 输入 | const float*（FP32） | 标量 beta 指针，指向 FP32 标量，不可为 nullptr，Host 内存 |
| C | 输入/输出 | float*（FP32） | 矩阵 C 的设备内存指针，FP32，列主序；当 beta != 0 时不可为 nullptr，Device 内存 |
| ldc | 输入 | int | 矩阵 C 的主维度（列主序），ldc >= max(1, M)，Host 内存 |
| algo | 输入 | aclblasGemmAlgo_t | 算法选择，当前仅支持 ACLBLAS_GEMM_DEFAULT，Host 内存 |

#### 约束说明

- m >= 0
- n >= 0
- k >= 0
- transA 必须为 ACLBLAS_OP_N、ACLBLAS_OP_T 或 ACLBLAS_OP_C
- transB 必须为 ACLBLAS_OP_N、ACLBLAS_OP_T 或 ACLBLAS_OP_C
- transA = N 时 lda >= max(1, m)；transA = T/C 时 lda >= max(1, k)
- transB = N 时 ldb >= max(1, k)；transB = T/C 时 ldb >= max(1, n)
- ldc >= max(1, m)
- alpha 不可为 nullptr
- beta 不可为 nullptr
- algo 当前仅支持 ACLBLAS_GEMM_DEFAULT
- k > 0 时 A 不可为 nullptr
- k > 0 时 B 不可为 nullptr
- beta != 0.0f 时 C 不可为 nullptr

边界情况处理：
- m == 0 或 n == 0 时直接返回 ACLBLAS_STATUS_SUCCESS，不执行计算
- k == 0 或 alpha == 0.0f 时跳过矩阵乘，执行 C = beta * C（beta == 0 时置零，beta == 1 时不变，其他值逐元素缩放）
- C == nullptr 且 beta == 0.0f 时直接返回 ACLBLAS_STATUS_SUCCESS

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

struct AclMemDeleter {
    void operator()(void* p) const { aclrtFree(p); }
};

struct BlasHandleDeleter {
    void operator()(aclblasHandle_t h) const { aclblasDestroy(h); }
};

int aclblasSgemmExTest(AclContext& ctx)
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
    // C = alpha * op(A) * op(B) + beta * C
    // M=2, N=2, K=3, transA=N, transB=N, alpha=1.0, beta=0.0
    int m = 2, n = 2, k = 3;
    int lda = m, ldb = k, ldc = m;
    float alpha = 1.0f;
    float beta = 0.0f;
    aclblasOperation_t transA = ACLBLAS_OP_N;
    aclblasOperation_t transB = ACLBLAS_OP_N;
    aclblasGemmAlgo_t algo = ACLBLAS_GEMM_DEFAULT;

    // A (M=2, K=3, column-major, lda=2): [[1,2,3],[4,5,6]]
    std::vector<float> hA = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
    // B (K=3, N=2, column-major, ldb=3): [[1,0],[0,1],[1,1]]
    std::vector<float> hB = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f};
    // C (M=2, N=2, column-major, ldc=2): expected result [[4,5],[10,11]]
    std::vector<float> hC(static_cast<size_t>(ldc) * n, 0.0f);

    size_t aBytes = hA.size() * sizeof(float);
    size_t bBytes = hB.size() * sizeof(float);
    size_t cBytes = hC.size() * sizeof(float);

    // 3. 申请 Device 内存并拷贝数据
    void* rawA = nullptr;
    auto aclRet = aclrtMalloc(&rawA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<float, AclMemDeleter> aDevicePtr(static_cast<float*>(rawA));

    void* rawB = nullptr;
    aclRet = aclrtMalloc(&rawB, bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for B failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<float, AclMemDeleter> bDevicePtr(static_cast<float*>(rawB));

    void* rawC = nullptr;
    aclRet = aclrtMalloc(&rawC, cBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for C failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<float, AclMemDeleter> cDevicePtr(static_cast<float*>(rawC));

    aclRet = aclrtMemcpy(aDevicePtr.get(), aBytes, hA.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(bDevicePtr.get(), bBytes, hB.data(), bBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for B failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(cDevicePtr.get(), cBytes, hC.data(), cBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for C failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasSgemmEx
    blasRet = aclblasSgemmEx(static_cast<aclblasHandle_t>(handlePtr.get()),
                             transA, transB, m, n, k, &alpha,
                             aDevicePtr.get(), lda, bDevicePtr.get(), ldb,
                             &beta, cDevicePtr.get(), ldc, algo);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSgemmEx failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    aclRet = aclrtMemcpy(hC.data(), cBytes, cDevicePtr.get(), cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", aclRet);
              return aclRet);

    // 打印结果（列主序存储：hC[col * ldc + row] = C[row][col]）
    LOG_PRINT("result C (column-major):\n");
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < m; row++) {
            LOG_PRINT("  C[%d][%d] = %f\n", row, col, hC[static_cast<size_t>(col) * ldc + row]);
        }
    }

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSgemmExTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSgemmExTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

预期输出：

```
result C (column-major):
  C[0][0] = 4.000000
  C[1][0] = 10.000000
  C[0][1] = 5.000000
  C[1][1] = 11.000000
```

### aclblasGemmEx 调用示例

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

struct AclMemDeleter {
    void operator()(void* p) const { aclrtFree(p); }
};

struct BlasHandleDeleter {
    void operator()(aclblasHandle_t h) const { aclblasDestroy(h); }
};

int aclblasGemmExTest(AclContext& ctx)
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
    // C = alpha * op(A) * op(B) + beta * C
    // M=2, N=2, K=3, transA=N, transB=N, alpha=1.0, beta=0.0
    // 使用 FP16 输入、FP32 输出
    int m = 2, n = 2, k = 3;
    int lda = m, ldb = k, ldc = m;
    float alpha = 1.0f;
    float beta = 0.0f;
    aclblasOperation_t transA = ACLBLAS_OP_N;
    aclblasOperation_t transB = ACLBLAS_OP_N;
    aclblasComputeType_t computeType = ACLBLAS_COMPUTE_HIGH_PRECISION;
    aclblasGemmAlgo_t algo = ACLBLAS_GEMM_DEFAULT;

    // A (M=2, K=3, column-major, lda=2, FP16): [[1,2,3],[4,5,6]]
    std::vector<__fp16> hA = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
    // B (K=3, N=2, column-major, ldb=3, FP16): [[1,0],[0,1],[1,1]]
    std::vector<__fp16> hB = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f};
    // C (M=2, N=2, column-major, ldc=2, FP32): expected result [[4,5],[10,11]]
    std::vector<float> hC(static_cast<size_t>(ldc) * n, 0.0f);

    size_t aBytes = hA.size() * sizeof(__fp16);
    size_t bBytes = hB.size() * sizeof(__fp16);
    size_t cBytes = hC.size() * sizeof(float);

    // 3. 申请 Device 内存并拷贝数据
    void* rawA = nullptr;
    auto aclRet = aclrtMalloc(&rawA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, AclMemDeleter> aDevicePtr(rawA);

    void* rawB = nullptr;
    aclRet = aclrtMalloc(&rawB, bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for B failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, AclMemDeleter> bDevicePtr(rawB);

    void* rawC = nullptr;
    aclRet = aclrtMalloc(&rawC, cBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for C failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, AclMemDeleter> cDevicePtr(rawC);

    aclRet = aclrtMemcpy(aDevicePtr.get(), aBytes, hA.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(bDevicePtr.get(), bBytes, hB.data(), bBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for B failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(cDevicePtr.get(), cBytes, hC.data(), cBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for C failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasGemmEx
    blasRet = aclblasGemmEx(static_cast<aclblasHandle_t>(handlePtr.get()),
                            transA, transB, m, n, k, &alpha,
                            aDevicePtr.get(), ACL_FLOAT16, lda,
                            bDevicePtr.get(), ACL_FLOAT16, ldb,
                            &beta, cDevicePtr.get(), ACL_FLOAT, ldc,
                            computeType, algo);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasGemmEx failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    aclRet = aclrtMemcpy(hC.data(), cBytes, cDevicePtr.get(), cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", aclRet);
              return aclRet);

    // 打印结果（列主序存储：hC[col * ldc + row] = C[row][col]）
    LOG_PRINT("result C (column-major):\n");
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < m; row++) {
            LOG_PRINT("  C[%d][%d] = %f\n", row, col, hC[static_cast<size_t>(col) * ldc + row]);
        }
    }

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasGemmExTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasGemmExTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

预期输出：

```
result C (column-major):
  C[0][0] = 4.000000
  C[1][0] = 10.000000
  C[0][1] = 5.000000
  C[1][1] = 11.000000
```
