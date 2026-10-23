# Geam算子

## 算子概述

Geam（General Matrix Add）算子执行带标量缩放的矩阵加法运算，支持对输入矩阵 A、B 分别施加可选的转置（Transpose）或共轭转置（ConjTrans）操作。数学定义为：

```txt
C[i, j] = alpha * op(A)[i, j] + beta * op(B)[i, j],  i ∈ [0, m), j ∈ [0, n)
```

其中 `op(X)` 根据转置参数取 `X`（NoTrans）、`X^T`（Trans）或 `X^H`（ConjTrans，仅复数有意义，浮点数等价于 Trans）。矩阵采用列主序（Column-Major）存储，`X[i, j]` 的物理地址为 `X[i + j * ldX]`。

本算子提供两个接口，覆盖实数与复数两类矩阵加法场景：

| 接口名 | 功能简述 |
|--------|----------|
| aclblasSgeam | 单精度浮点 GEAM（General Matrix Add） |
| aclblasCgeam | 单精度复数 GEAM（General Matrix Add） |

## 算子执行接口

### aclblasSgeam

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSgeam(aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, const float* alpha, const float* A, int lda, const float* beta, const float* B, int ldb, float* C, int ldc)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| transa | 输入 | aclblasOperation_t | 矩阵 A 的转置操作：ACLBLAS_OP_N（不转置）、ACLBLAS_OP_T（转置）、ACLBLAS_OP_C（共轭转置，FP32 实数等价于转置），Host 内存 |
| transb | 输入 | aclblasOperation_t | 矩阵 B 的转置操作（同 transa），Host 内存 |
| m | 输入 | int | 输出矩阵 C（及 op(A)）的行数，m >= 0，Host 内存 |
| n | 输入 | int | 输出矩阵 C（及 op(B)）的列数，n >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | A 的缩放因子指针，指向 Host 侧单个标量；不可为 nullptr，Host 内存 |
| A | 输入 | const float*（FP32） | 输入矩阵 A，列主序存储；当 *alpha != 0 时不可为 nullptr，Device 内存 |
| lda | 输入 | int | 矩阵 A 的主维度，transa=N 时 lda >= max(1, m)，transa=T/C 时 lda >= max(1, n)，Host 内存 |
| beta | 输入 | const float*（FP32） | B 的缩放因子指针，指向 Host 侧单个标量；不可为 nullptr，Host 内存 |
| B | 输入 | const float*（FP32） | 输入矩阵 B，列主序存储；当 *beta != 0 时不可为 nullptr，Device 内存 |
| ldb | 输入 | int | 矩阵 B 的主维度，transb=N 时 ldb >= max(1, m)，transb=T/C 时 ldb >= max(1, n)，Host 内存 |
| C | 输出 | float*（FP32） | 输出矩阵 C，列主序存储；m > 0 且 n > 0 时不可为 nullptr，Device 内存 |
| ldc | 输入 | int | 矩阵 C 的主维度，ldc >= max(1, m)，Host 内存 |

#### 约束说明

- handle 不能为 nullptr，否则返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR
- transa / transb 必须为 ACLBLAS_OP_N、ACLBLAS_OP_T 或 ACLBLAS_OP_C，否则返回 ACLBLAS_STATUS_INVALID_ENUM
- m >= 0，n >= 0，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- transa = N 时 lda >= max(1, m)；transa = T/C 时 lda >= max(1, n)
- transb = N 时 ldb >= max(1, m)；transb = T/C 时 ldb >= max(1, n)
- ldc >= max(1, m)
- alpha 不允许为 nullptr，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- beta 不允许为 nullptr，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- *alpha == 0 时，A 不被引用，可为 nullptr
- *beta == 0 时，B 不被引用，可为 nullptr
- *alpha != 0 时，A 不能为 nullptr
- *beta != 0 时，B 不能为 nullptr
- m > 0 且 n > 0 时，C 不能为 nullptr
- 支持 in-place：C == A 时要求 transa == N 且 lda == ldc；C == B 时要求 transb == N 且 ldb == ldc
- m == 0 或 n == 0 时直接返回 ACLBLAS_STATUS_SUCCESS，不执行计算

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://www.hiascend.com/document/detail/zh/CANN/community/8.2.RC1/quickstart/quickstart_18_0041.html)。

以下示例演示 NN 模式（NoTrans/NoTrans），取 m=4, n=3, alpha=1.0, beta=0.5，计算 C(4x3) = 1.0 * A(4x3) + 0.5 * B(4x3)：

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

int aclblasSgeamTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-blas 句柄
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<void, aclblasStatus_t (*)(void*)> handlePtr(rawHandle, aclblasDestroy);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 2. 准备 Host 数据
    // C(4x3) = 1.0 * A(4x3) + 0.5 * B(4x3)，NN 模式
    int m = 4, n = 3;
    int lda = m, ldb = m, ldc = m;
    float alpha = 1.0f;
    float beta = 0.5f;

    // A (m=4, n=3, column-major): 列0=[1,2,3,4], 列1=[5,6,7,8], 列2=[9,10,11,12]
    std::vector<float> hA = {1.0f, 2.0f, 3.0f, 4.0f,
                              5.0f, 6.0f, 7.0f, 8.0f,
                              9.0f, 10.0f, 11.0f, 12.0f};
    // B (m=4, n=3, column-major): 列0=[2,4,6,8], 列1=[10,12,14,16], 列2=[18,20,22,24]
    std::vector<float> hB = {2.0f, 4.0f, 6.0f, 8.0f,
                              10.0f, 12.0f, 14.0f, 16.0f,
                              18.0f, 20.0f, 22.0f, 24.0f};

    size_t aBytes = hA.size() * sizeof(float);
    size_t bBytes = hB.size() * sizeof(float);
    size_t cBytes = static_cast<size_t>(ldc) * static_cast<size_t>(n) * sizeof(float);

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

    // 4. 调用 aclblasSgeam
    blasRet = aclblasSgeam(static_cast<aclblasHandle_t>(handlePtr.get()),
                            ACLBLAS_OP_N, ACLBLAS_OP_N, m, n,
                            &alpha, aDevicePtr.get(), lda,
                            &beta, bDevicePtr.get(), ldb,
                            static_cast<float*>(cDevicePtr.get()), ldc);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSgeam failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    // 预期 C = alpha*A + beta*B (列主序)
    //   列0=[1+1, 2+2, 3+3, 4+4]=[2,4,6,8]
    //   列1=[5+5, 6+6, 7+7, 8+8]=[10,12,14,16]
    //   列2=[9+9, 10+10, 11+11, 12+12]=[18,20,22,24]
    std::vector<float> hC(static_cast<size_t>(ldc) * static_cast<size_t>(n), 0.0f);
    aclRet = aclrtMemcpy(hC.data(), cBytes, cDevicePtr.get(), cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", aclRet); return aclRet);

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

    ret = aclblasSgeamTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSgeamTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

预期输出：

```
result C (column-major):
  C[0][0] = 2.000000
  C[1][0] = 4.000000
  C[2][0] = 6.000000
  C[3][0] = 8.000000
  C[0][1] = 10.000000
  C[1][1] = 12.000000
  C[2][1] = 14.000000
  C[3][1] = 16.000000
  C[0][2] = 18.000000
  C[1][2] = 20.000000
  C[2][2] = 22.000000
  C[3][2] = 24.000000
```

### aclblasCgeam

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasCgeam(aclblasHandle_t handle, aclblasOperation_t transa, aclblasOperation_t transb, int m, int n, const aclblasComplex* alpha, const aclblasComplex* A, int lda, const aclblasComplex* beta, const aclblasComplex* B, int ldb, aclblasComplex* C, int ldc)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| transa | 输入 | aclblasOperation_t | 矩阵 A 的转置操作：ACLBLAS_OP_N（不转置）、ACLBLAS_OP_T（转置）、ACLBLAS_OP_C（共轭转置），Host 内存 |
| transb | 输入 | aclblasOperation_t | 矩阵 B 的转置操作（同 transa），Host 内存 |
| m | 输入 | int | 输出矩阵 C（及 op(A)）的行数，m >= 0，Host 内存 |
| n | 输入 | int | 输出矩阵 C（及 op(B)）的列数，n >= 0，Host 内存 |
| alpha | 输入 | const aclblasComplex* | A 的缩放因子指针，指向 Host 侧单个标量；不可为 nullptr，Host 内存 |
| A | 输入 | const aclblasComplex* | 输入矩阵 A，列主序存储；当 *alpha != 0 时不可为 nullptr，Device 内存 |
| lda | 输入 | int | 矩阵 A 的主维度，transa=N 时 lda >= max(1, m)，transa=T/C 时 lda >= max(1, n)，Host 内存 |
| beta | 输入 | const aclblasComplex* | B 的缩放因子指针，指向 Host 侧单个标量；不可为 nullptr，Host 内存 |
| B | 输入 | const aclblasComplex* | 输入矩阵 B，列主序存储；当 *beta != 0 时不可为 nullptr，Device 内存 |
| ldb | 输入 | int | 矩阵 B 的主维度，transb=N 时 ldb >= max(1, m)，transb=T/C 时 ldb >= max(1, n)，Host 内存 |
| C | 输出 | aclblasComplex* | 输出矩阵 C，列主序存储；m > 0 且 n > 0 时不可为 nullptr，Device 内存 |
| ldc | 输入 | int | 矩阵 C 的主维度，ldc >= max(1, m)，Host 内存 |

#### 约束说明

- handle 不能为 nullptr，否则返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR
- transa / transb 必须为 ACLBLAS_OP_N、ACLBLAS_OP_T 或 ACLBLAS_OP_C，否则返回 ACLBLAS_STATUS_INVALID_ENUM
- m >= 0，n >= 0，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- transa = N 时 lda >= max(1, m)；transa = T/C 时 lda >= max(1, n)
- transb = N 时 ldb >= max(1, m)；transb = T/C 时 ldb >= max(1, n)
- ldc >= max(1, m)
- alpha 不允许为 nullptr，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- beta 不允许为 nullptr，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- *alpha == 0 时，A 不被引用，可为 nullptr
- *beta == 0 时，B 不被引用，可为 nullptr
- *alpha != 0 时，A 不能为 nullptr
- *beta != 0 时，B 不能为 nullptr
- m > 0 且 n > 0 时，C 不能为 nullptr
- 支持 in-place：C == A 时要求 transa == N 且 lda == ldc；C == B 时要求 transb == N 且 ldb == ldc
- m == 0 或 n == 0 时直接返回 ACLBLAS_STATUS_SUCCESS，不执行计算
- ACLBLAS_OP_C 对复数执行共轭转置，实部不变，虚部取反

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://www.hiascend.com/document/detail/zh/CANN/community/8.2.RC1/quickstart/quickstart_18_0041.html)。

以下示例演示 NC 模式（NoTrans/ConjTrans），取 m=2, n=3, alpha=(1+1i), beta=(0.5+0.5i)，计算 C(2x3) = (1+i) * A(2x3) + (0.5+0.5i) * B^H(2x3)：

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

int aclblasCgeamTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-blas 句柄
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<void, aclblasStatus_t (*)(void*)> handlePtr(rawHandle, aclblasDestroy);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 2. 准备 Host 数据
    // C(2x3) = (1+i) * A(2x3) + (0.5+0.5i) * B^H(2x3)，NC 模式
    int m = 2, n = 3;
    int lda = m, ldb = n, ldc = m;  // transb=C 时 ldb >= max(1, n=3)
    aclblasComplex alpha = {1.0f, 1.0f};   // 1+i
    aclblasComplex beta  = {0.5f, 0.5f};   // 0.5+0.5i

    // A (m=2, n=3, column-major): 列0=[(1+i),(2+0i)], 列1=[(3+0i),(4+i)], 列2=[(0+2i),(1+1i)]
    std::vector<aclblasComplex> hA = {{{1.0f, 1.0f}, {2.0f, 0.0f},
                                         {3.0f, 0.0f}, {4.0f, 1.0f},
                                         {0.0f, 2.0f}, {1.0f, 1.0f}}};
    // B 原始 (n=3, m=2, column-major)，转置前 B 形状为 (3,2)
    //   列0=[(1+0i),(0+1i),(1+1i)], 列1=[(2+0i),(0+0i),(1+0i)]
    // B^H 将 B 共轭转置为 (m=2, n=3)
    std::vector<aclblasComplex> hB = {{{1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f},
                                         {2.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f}}};

    size_t aBytes = hA.size() * sizeof(aclblasComplex);
    size_t bBytes = hB.size() * sizeof(aclblasComplex);
    size_t cBytes = static_cast<size_t>(ldc) * static_cast<size_t>(n) * sizeof(aclblasComplex);

    // 3. 申请 Device 内存并拷贝数据
    aclblasComplex* rawA = nullptr;
    auto aclRet = aclrtMalloc(reinterpret_cast<void**>(&rawA), aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<aclblasComplex, AclMemDeleter> aDevicePtr(rawA);

    aclblasComplex* rawB = nullptr;
    aclRet = aclrtMalloc(reinterpret_cast<void**>(&rawB), bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for B failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<aclblasComplex, AclMemDeleter> bDevicePtr(rawB);

    aclblasComplex* rawC = nullptr;
    aclRet = aclrtMalloc(reinterpret_cast<void**>(&rawC), cBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for C failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<aclblasComplex, AclMemDeleter> cDevicePtr(rawC);

    aclRet = aclrtMemcpy(aDevicePtr.get(), aBytes, hA.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(bDevicePtr.get(), bBytes, hB.data(), bBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for B failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasCgeam
    blasRet = aclblasCgeam(static_cast<aclblasHandle_t>(handlePtr.get()),
                            ACLBLAS_OP_N, ACLBLAS_OP_C, m, n,
                            &alpha, aDevicePtr.get(), lda,
                            &beta, bDevicePtr.get(), ldb,
                            cDevicePtr.get(), ldc);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCgeam failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    std::vector<aclblasComplex> hC(static_cast<size_t>(ldc) * static_cast<size_t>(n), {0.0f, 0.0f});
    aclRet = aclrtMemcpy(hC.data(), cBytes, cDevicePtr.get(), cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", aclRet); return aclRet);

    LOG_PRINT("result C (column-major):\n");
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < m; row++) {
            const auto& val = hC[static_cast<size_t>(col) * ldc + row];
            LOG_PRINT("  C[%d][%d] = %f + %fi\n", row, col, val.real, val.imag);
        }
    }

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasCgeamTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasCgeamTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

预期输出：

```
result C (column-major):
  C[0][0] = 0.500000 + 2.500000i
  C[1][0] = 3.000000 + 3.000000i
  C[0][1] = 3.500000 + 2.500000i
  C[1][1] = 3.000000 + 5.000000i
  C[0][2] = -1.000000 + 2.000000i
  C[1][2] = 0.500000 + 2.500000i
```

## 支持的数据类型

| 接口 | 数据类型 | 说明 |
|------|----------|------|
| `aclblasSgeam` | `float` (FP32) | 单精度实数 |
| `aclblasCgeam` | `aclblasComplex` | 单精度复数（实部 + 虚部各为 float） |

## 返回值 / 错误码

| 返回值 | 含义 |
|--------|------|
| `ACLBLAS_STATUS_SUCCESS` | 执行成功 |
| `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` | handle 为空指针 |
| `ACLBLAS_STATUS_INVALID_VALUE` | 维度非法（m < 0、n < 0）、leading dimension 不满足约束、必需指针为空（alpha、beta、C、A/B 在对应标量非零时为 nullptr） |
| `ACLBLAS_STATUS_INVALID_ENUM` | transa / transb 取值不在 N/T/C 范围内 |
| `ACLBLAS_STATUS_INTERNAL_ERROR` | 内部错误（如获取核心数失败） |

## 精度标准

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
|----------|------|------|----------------------|-------------------|
| `float` (FP32) | 2^-10 (≈ 9.77e-4) | 2^-16 (≈ 1.53e-5) | 0.99 | 1e-2 或 32×ULP |
| `aclblasComplex` | 实部 / 虚部分别按 float 标准校验 | 同上 | 同上 | 同上 |

## 支持芯片

| 芯片 | 架构 | 支持状态 |
|------|------|----------|
| Ascend 950PR | arch35 (DAV_3510) | 支持 |
| Ascend 950DT | arch35 (DAV_3510) | 支持 |
| Atlas A3 | — | 不支持 |
| Atlas A2 | — | 不支持 |
