# Trsm算子

## 算子概述

trsm (Triangular Solve with Multiple Right-hand sides) 算子实现了三角矩阵线性方程组求解，核心运算为 op(A) * X = alpha * B（side=LEFT）或 X * op(A) = alpha * B（side=RIGHT），其中 A 为三角矩阵，B 为右端矩阵，X 为解矩阵（原地覆盖 B）。

数学表达式：

```
side == LEFT:  op(A) * X = alpha * B
side == RIGHT: X * op(A) = alpha * B
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasStrsm | 单精度三角矩阵线性方程组求解 |
| aclblasCtrsm | 单精度复数三角矩阵线性方程组求解 |

## 算子执行接口

### aclblasStrsm

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasStrsm(aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n, const float* alpha, const float* A, int lda, float* B, int ldb)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| side | 输入 | aclblasSideMode_t | A 在等式左侧(ACLBLAS_SIDE_LEFT)或右侧(ACLBLAS_SIDE_RIGHT)，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | A 为上三角(ACLBLAS_UPPER)或下三角(ACLBLAS_LOWER)，Host 内存 |
| trans | 输入 | aclblasOperation_t | op(A) 类型：ACLBLAS_OP_N(不转置)、ACLBLAS_OP_T(转置)、ACLBLAS_OP_C(共轭转置，实数下同 T)，Host 内存 |
| diag | 输入 | aclblasDiagType_t | 对角线类型：ACLBLAS_NON_UNIT(非单位对角线)、ACLBLAS_UNIT(单位对角线)，Host 内存 |
| m | 输入 | int | B 的行数，m >= 0，Host 内存 |
| n | 输入 | int | B 的列数，n >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量系数，仅支持 Host 指针，alpha == 0 时 B 直接置零，Host 内存 |
| A | 输入 | const float*（FP32） | 三角矩阵，side==LEFT 时维度 m×m，side==RIGHT 时维度 n×n，Device 内存 |
| lda | 输入 | int | A 的 leading dimension，side==LEFT 时 lda >= max(1, m)，side==RIGHT 时 lda >= max(1, n)，Host 内存 |
| B | 输入/输出 | float*（FP32） | m×n 矩阵，输入为右端矩阵，输出原地覆盖为解矩阵 X，Device 内存 |
| ldb | 输入 | int | B 的 leading dimension，ldb >= max(1, m)，Host 内存 |

#### 约束说明

- m >= 0, n >= 0
- side 取值必须为 ACLBLAS_SIDE_LEFT 或 ACLBLAS_SIDE_RIGHT
- uplo 取值必须为 ACLBLAS_UPPER 或 ACLBLAS_LOWER
- trans 取值必须为 ACLBLAS_OP_N、ACLBLAS_OP_T 或 ACLBLAS_OP_C
- diag 取值必须为 ACLBLAS_NON_UNIT 或 ACLBLAS_UNIT
- alpha 不能为 nullptr，且仅支持 Host 指针
- side == LEFT 时 lda >= max(1, m)；side == RIGHT 时 lda >= max(1, n)
- ldb >= max(1, m)
- B 不能为 nullptr；alpha != 0 时 A 不能为 nullptr（m==0 或 n==0 时按 BLAS quick-return 直接返回 SUCCESS）

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

int aclblasStrsmTest(AclContext& ctx)
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
    constexpr int m = 4;
    constexpr int n = 2;
    constexpr int lda = 4;
    constexpr int ldb = 4;
    constexpr size_t aSize = static_cast<size_t>(lda) * m * sizeof(float);
    constexpr size_t bSize = static_cast<size_t>(ldb) * n * sizeof(float);

    // 下三角矩阵 A（列主序）：
    // [[2, 0, 0, 0],
    //  [1, 3, 0, 0],
    //  [0.5, 1.5, 4, 0],
    //  [0.25, 0.75, 2, 5]]
    float hA[lda * m] = {
        2.0f, 1.0f, 0.5f, 0.25f,
        0.0f, 3.0f, 1.5f, 0.75f,
        0.0f, 0.0f, 4.0f, 2.0f,
        0.0f, 0.0f, 0.0f, 5.0f
    };
    // 右端矩阵 B（列主序）：
    // [[4, 8],
    //  [7, 14],
    //  [10, 20],
    //  [13, 26]]
    float hB[ldb * n] = {
        4.0f, 7.0f, 10.0f, 13.0f,
        8.0f, 14.0f, 20.0f, 26.0f
    };
    float alpha = 1.0f;

    // 3. 申请 Device 内存并拷贝数据
    void *rawA = nullptr;
    auto aclRet = aclrtMalloc(&rawA, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc A failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, void (*)(void*)> dA(rawA, [](void* p) { aclrtFree(p); });

    void *rawB = nullptr;
    aclRet = aclrtMalloc(&rawB, bSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc B failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, void (*)(void*)> dB(rawB, [](void* p) { aclrtFree(p); });

    aclRet = aclrtMemcpy(dA.get(), aSize, hA, aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D A failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dB.get(), bSize, hB, bSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D B failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasStrsm
    blasRet = aclblasStrsm(
        static_cast<aclblasHandle_t>(handlePtr.get()), ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER, ACLBLAS_OP_N,
        ACLBLAS_NON_UNIT, m, n, &alpha, static_cast<const float*>(dA.get()), lda,
        static_cast<float*>(dB.get()), ldb);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasStrsm failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    aclRet = aclrtMemcpy(hB, bSize, dB.get(), bSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy D2H B failed. ERROR: %d\n", aclRet); return aclRet);

    // 预期结果：X = A^{-1} * B = [[2, 4], [1.6667, 3.3333], [1.625, 3.25], [1.6, 3.2]]
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            LOG_PRINT("X[%d][%d] = %f\n", i, j, hB[i + j * ldb]);
        }
    }

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasStrsmTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasStrsmTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

### aclblasCtrsm

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasCtrsm(aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int m, int n, const aclblasComplex* alpha, const aclblasComplex* A, int lda, aclblasComplex* B, int ldb)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| side | 输入 | aclblasSideMode_t | `ACLBLAS_SIDE_LEFT` 求解 `op(A) * X = alpha * B`；`ACLBLAS_SIDE_RIGHT` 求解 `X * op(A) = alpha * B` |
| uplo | 输入 | aclblasFillMode_t | `ACLBLAS_UPPER` 或 `ACLBLAS_LOWER`，仅引用指定的三角部分 |
| trans | 输入 | aclblasOperation_t | `ACLBLAS_OP_N`、`ACLBLAS_OP_T` 或 `ACLBLAS_OP_C`，分别表示不转置、转置和共轭转置 |
| diag | 输入 | aclblasDiagType_t | `ACLBLAS_NON_UNIT` 或 `ACLBLAS_UNIT`；UNIT 模式不读取 A 的对角元 |
| m | 输入 | int | B 的行数，`m >= 0` |
| n | 输入 | int | B 的列数，`n >= 0` |
| alpha | 输入 | const aclblasComplex* | Host 侧 COMPLEX64 标量；不可为 nullptr |
| A | 输入 | const aclblasComplex* | Device 侧 COMPLEX64 三角矩阵，列主序；`alpha != 0` 时不可为 nullptr |
| lda | 输入 | int | A 的前导维；LEFT 时 `lda >= max(1,m)`，RIGHT 时 `lda >= max(1,n)` |
| B | 输入/输出 | aclblasComplex* | Device 侧 COMPLEX64 矩阵，列主序；输出 X 原地覆盖 B |
| ldb | 输入 | int | B 的前导维，`ldb >= max(1,m)` |

#### 约束说明

- handle 为 nullptr 时返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
- side / uplo / trans / diag 非法时返回 `ACLBLAS_STATUS_INVALID_VALUE`
- m 或 n 为负数、alpha 为 nullptr 时返回 `ACLBLAS_STATUS_INVALID_VALUE`
- 上述检查通过后，`m == 0` 或 `n == 0` 时直接返回 `ACLBLAS_STATUS_SUCCESS`，不检查 lda / ldb / A / B
- 非空矩阵的 lda / ldb 不满足约束、B 为 nullptr，或 alpha 非零且 A 为 nullptr 时返回 `ACLBLAS_STATUS_INVALID_VALUE`
- `alpha == (0,0)` 时不引用 A，将 B 的有效 `m * n` 区域置零
- `diag == ACLBLAS_NON_UNIT` 时由调用方保证 A 的对角元非零，本接口不做奇异性检测
- 算子在 handle 绑定的 stream 上异步执行，读回结果前需要同步 stream
- 派生矩阵字节大小、工作区大小及行距采用 64 位计算；无法表示的存储大小或超出 Matmul 接口有符号整数行距范围时，返回 `ACLBLAS_STATUS_INVALID_VALUE`，不启动 kernel

#### 调用示例

以下完整示例使用 RAII 管理资源；示例矩阵的虚部为零，预期解的虚部也为零。
编译方式见[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)，精度和性能验证见[测试指导](../../test/trsm/ctrsm/README.md)。

```cpp
#include <cstdio>
#include <memory>
#include <type_traits>

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

int aclblasCtrsmTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-blas 句柄
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<std::remove_pointer_t<aclblasHandle_t>, decltype(&aclblasDestroy)> handlePtr(
        rawHandle, aclblasDestroy);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 2. 准备 Host 数据
    constexpr int m = 4;
    constexpr int n = 2;
    constexpr int lda = 4;
    constexpr int ldb = 4;
    constexpr size_t aSize = static_cast<size_t>(lda) * m * sizeof(aclblasComplex);
    constexpr size_t bSize = static_cast<size_t>(ldb) * n * sizeof(aclblasComplex);

    // 下三角矩阵 A（列主序）：
    // [[2, 0, 0, 0],
    //  [1, 3, 0, 0],
    //  [0.5, 1.5, 4, 0],
    //  [0.25, 0.75, 2, 5]]
    aclblasComplex hA[lda * m] = {
        {2.0f, 0.0f}, {1.0f, 0.0f}, {0.5f, 0.0f}, {0.25f, 0.0f},
        {0.0f, 0.0f}, {3.0f, 0.0f}, {1.5f, 0.0f}, {0.75f, 0.0f},
        {0.0f, 0.0f}, {0.0f, 0.0f}, {4.0f, 0.0f}, {2.0f, 0.0f},
        {0.0f, 0.0f}, {0.0f, 0.0f}, {0.0f, 0.0f}, {5.0f, 0.0f}
    };
    // 右端矩阵 B（列主序）：
    // [[4, 8],
    //  [7, 14],
    //  [10, 20],
    //  [13, 26]]
    aclblasComplex hB[ldb * n] = {
        {4.0f, 0.0f}, {7.0f, 0.0f}, {10.0f, 0.0f}, {13.0f, 0.0f},
        {8.0f, 0.0f}, {14.0f, 0.0f}, {20.0f, 0.0f}, {26.0f, 0.0f}
    };
    aclblasComplex alpha{1.0f, 0.0f};

    // 3. 申请 Device 内存并拷贝数据
    void *rawA = nullptr;
    auto aclRet = aclrtMalloc(&rawA, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc A failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, void (*)(void*)> dA(rawA, [](void* p) { aclrtFree(p); });

    void *rawB = nullptr;
    aclRet = aclrtMalloc(&rawB, bSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc B failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, void (*)(void*)> dB(rawB, [](void* p) { aclrtFree(p); });

    aclRet = aclrtMemcpy(dA.get(), aSize, hA, aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D A failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dB.get(), bSize, hB, bSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy H2D B failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasCtrsm
    blasRet = aclblasCtrsm(
        static_cast<aclblasHandle_t>(handlePtr.get()), ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER, ACLBLAS_OP_N,
        ACLBLAS_NON_UNIT, m, n, &alpha, static_cast<const aclblasComplex*>(dA.get()), lda,
        static_cast<aclblasComplex*>(dB.get()), ldb);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCtrsm failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    aclRet = aclrtMemcpy(hB, bSize, dB.get(), bSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy D2H B failed. ERROR: %d\n", aclRet); return aclRet);

    // 预期结果：X = A^{-1} * B = [[2, 4], [1.6667, 3.3333], [1.625, 3.25], [1.6, 3.2]]
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            LOG_PRINT("X[%d][%d] = (%f, %f)\n", i, j, hB[i + j * ldb].real, hB[i + j * ldb].imag);
        }
    }

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasCtrsmTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasCtrsmTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```
