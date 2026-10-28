# Tbsv算子

## 算子概述

tbsv (Triangular Banded matrix Solve) 求解三角带状方程组，核心运算为：`op(A) * x = b`，其中 A 为 n×n 三角带状矩阵（带宽 k），结果原地覆盖到输入向量 x 中。

数学表达式：

```text
op(A) * x = b    (x 原地更新为解)
```

其中 `op(A)` 由 trans 参数决定：`A`（N）、`A^T`（T）或 `A^H`（C）。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasStbsv | 单精度三角带状方程组求解 |
| aclblasCtbsv | 单精度复数三角带状方程组求解 |

## 算子执行接口

### aclblasStbsv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasStbsv(aclblasHandle_t handle,
                              aclblasFillMode_t uplo,
                              aclblasOperation_t trans,
                              aclblasDiagType_t diag,
                              int n, int k,
                              const float *A, int lda,
                              float *x, int incx);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | 指定三角区域：ACLBLAS_UPPER(121) 或 ACLBLAS_LOWER(122)，Host 内存 |
| trans | 输入 | aclblasOperation_t | 矩阵操作类型：ACLBLAS_OP_N(111)、ACLBLAS_OP_T(112) 或 ACLBLAS_OP_C(113)，Host 内存 |
| diag | 输入 | aclblasDiagType_t | 对角线类型：ACLBLAS_NON_UNIT(131) 或 ACLBLAS_UNIT(132)，Host 内存 |
| n | 输入 | int | 矩阵阶数，n >= 0，Host 内存 |
| k | 输入 | int | 带宽（super/sub-diagonal 数量），k >= 0，Host 内存 |
| A | 输入 | const float\*（FP32） | 带状矩阵，列主序存储，维度 (k+1)×n，Device 内存 |
| lda | 输入 | int | A 的 leading dimension，lda > k，Host 内存 |
| x | 输入/输出 | float\*（FP32） | 解向量，长度至少 1 + (n-1)\*\|incx\|，Device 内存 |
| incx | 输入 | int | x 的元素间步长，incx != 0 且 incx != INT_MIN，Host 内存 |

#### 约束说明

- n >= 0，参数校验通过后 n == 0 时直接返回成功
- k >= 0
- lda > k
- incx != 0 且 incx != INT_MIN
- k == 0 且 diag == UNIT 时直接返回成功（x 不变）
- 算子输入 shape 为 [(k+1)×n]、[n]，输出 shape 为 [n]
- Host 侧不做流同步，调用方需自行管理同步

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
#include "acl/acl.h"
#include "cann_ops_blas.h"

int main()
{
    aclInit(nullptr);
    aclrtSetDevice(0);

    aclrtStream stream;
    aclrtCreateStream(&stream);

    aclblasHandle_t handle = nullptr;
    aclblasCreate(&handle);
    aclblasSetStream(handle, stream);

    int n = 4, k = 1, lda = 2, incx = 1;

    // 下三角单位对角带状矩阵 A (lda x n, 列主序)，解 A x = b
    float aHost[lda * n] = {1.0f, 1.0f, 0.0f, 1.0f,
                            0.0f, 1.0f, 0.0f, 1.0f};
    float xHost[n] = {1.0f, 1.0f, 1.0f, 1.0f};

    size_t aBytes = lda * n * sizeof(float);
    size_t xBytes = n * sizeof(float);
    void *aDev = nullptr, *xDev = nullptr;
    aclrtMalloc(&aDev, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    aclrtMalloc(&xDev, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);

    aclrtMemcpy(aDev, aBytes, aHost, aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    aclrtMemcpy(xDev, xBytes, xHost, xBytes, ACL_MEMCPY_HOST_TO_DEVICE);

    aclblasStbsv(handle, ACLBLAS_LOWER, ACLBLAS_OP_N, ACLBLAS_NON_UNIT,
                 n, k, static_cast<const float *>(aDev), lda,
                 static_cast<float *>(xDev), incx);

    aclrtSynchronizeStream(stream);

    aclrtFree(aDev);
    aclrtFree(xDev);

    aclblasDestroy(handle);
    aclrtDestroyStream(stream);

    aclrtResetDevice(0);
    aclFinalize();

    return 0;
}
```

### aclblasCtbsv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品（含 Atlas 800I A3）：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品（含 Atlas 800I A2）：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasCtbsv(aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, aclblasDiagType_t diag, int n, int k, const aclblasComplex *A, int lda, aclblasComplex *x, int incx)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | 三角存储模式：ACLBLAS_UPPER(121) 或 ACLBLAS_LOWER(122)，Host 内存 |
| trans | 输入 | aclblasOperation_t | 矩阵操作：ACLBLAS_OP_N(111)、ACLBLAS_OP_T(112) 或 ACLBLAS_OP_C(113，共轭转置)，Host 内存 |
| diag | 输入 | aclblasDiagType_t | 对角线类型：ACLBLAS_NON_UNIT(131) 或 ACLBLAS_UNIT(132)，Host 内存 |
| n | 输入 | int | 矩阵阶数，n >= 0，Host 内存 |
| k | 输入 | int | 带宽（超对角 / 次对角数），k >= 0，Host 内存 |
| A | 输入 | const aclblasComplex\*（COMPLEX64） | 三角带状矩阵，列主序，数组 lda×n，Device 内存 |
| lda | 输入 | int | A 的 leading dimension，lda ≥ k+1，Host 内存 |
| x | 输入/输出 | aclblasComplex\*（COMPLEX64） | 入口为右端 b、出口为解，长度至少 1+(n-1)·\|incx\|，Device 内存 |
| incx | 输入 | int | x 的元素间步长，incx != 0 且 incx != INT_MIN，Host 内存 |

#### 约束说明

- handle 为 nullptr 时返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR（最先校验）
- uplo / trans / diag 取值非法时，Atlas A2/A3（arch22）返回 ACLBLAS_STATUS_INVALID_VALUE，Ascend 950（arch35）返回 ACLBLAS_STATUS_INVALID_ENUM
- n >= 0，k >= 0，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- 参数校验通过后 n == 0 时直接返回 ACLBLAS_STATUS_SUCCESS（空操作）
- lda ≥ k+1（带状存储，不与 n 比较），否则返回 ACLBLAS_STATUS_INVALID_VALUE
- incx != 0 且 incx != INT_MIN，否则返回 ACLBLAS_STATUS_INVALID_VALUE；负步长合法，按 Netlib 语义反向遍历，不是 no-op
- n > 0 时 A、x 不能为 nullptr，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- k == 0 且 diag == UNIT 时直接返回成功（x 不变）
- 仅引用 uplo 指定的三角带；trans=C 时按 A^H 计算
- 不做奇异性检查；diag=NON_UNIT 时对角必须非零
- Host 侧不做流同步，调用方需自行管理同步

Atlas A2/A3 实现位于 `blas/tbsv/arch22/`。`n<=4096` 且 `k<=2048` 时使用 UB 快速路径，超过该范围时使用 GM 兜底路径；`n<=2` 使用独立 AIC kernel。算子不申请 workspace。编译和精度、性能复现步骤见 [Ctbsv 测试说明](../../test/tbsv/ctbsv/README.md)。

#### Atlas A2/A3 的并行与标量路径

当前接口一次求解一个右端项。对于一般的非零带宽三角系统，前代或回代的相邻步骤有数据依赖，下一步必须使用此前已求得的 x；因此使用一个计算核保留求解链，列内的带状更新和点积分别使用 `Axpy`、`MulAddDst`、`ReduceSum` 等向量接口。直接增加启动核数会使多个核重复求解并竞争原地输出；多核分块方案还需要另行设计块间数据交换和同步，不能只调整 blockDim。`k=0` 的对角系统没有这类依赖，当前仍复用单核派发和 Smith 复数除法，没有单独实现多核对角求解。

连续 x 的输入使用批量搬运及 `Gather` 拆分实部、虚部。长度达到向量打包阈值时，输出通过整数向量索引和 `Gather` 交错写回；该步骤不改变浮点数的位模式。短向量保留标量打包，以避免固定的索引准备开销。

仍保留标量操作的位置包括逐步求解的主元、短带/非对齐前缀、非连续步长、大规格 GM 兜底以及数值敏感的宽带 UNIT 参考顺序路径。这些路径分别用于处理串行依赖、保持 stride hole 不变或保持与 CBLAS 一致的浮点求值顺序，并非所有列内计算都采用标量实现。

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

int aclblasCtbsvTest(AclContext& ctx)
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

    int n = 4, k = 1, lda = 2, incx = 1;
    std::vector<aclblasComplex> aHost(static_cast<size_t>(lda) * n, {0.0f, 0.0f});
    std::vector<aclblasComplex> xHost(static_cast<size_t>(n), {1.0f, 0.0f});
    for (int j = 0; j < n; ++j) {
        aHost[static_cast<size_t>(j * lda)] = {1.0f, 0.0f};
        if (j + 1 < n) {
            aHost[static_cast<size_t>(1 + j * lda)] = {0.5f, 0.0f};
        }
    }

    size_t aBytes = aHost.size() * sizeof(aclblasComplex);
    size_t xBytes = xHost.size() * sizeof(aclblasComplex);
    void* aDev = nullptr;
    void* xDev = nullptr;
    auto aclRet = aclrtMalloc(&aDev, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc A failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> aGuard(aDev, aclrtFree);
    aclRet = aclrtMalloc(&xDev, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc x failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> xGuard(xDev, aclrtFree);

    aclRet = aclrtMemcpy(aDev, aBytes, aHost.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy A failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(xDev, xBytes, xHost.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy x failed. ERROR: %d\n", aclRet); return aclRet);

    blasRet = aclblasCtbsv(static_cast<aclblasHandle_t>(handlePtr.get()), ACLBLAS_LOWER, ACLBLAS_OP_N,
                           ACLBLAS_NON_UNIT, n, k, static_cast<const aclblasComplex*>(aDev), lda,
                           static_cast<aclblasComplex*>(xDev), incx);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCtbsv failed. ERROR: %d\n", blasRet);
              return blasRet);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);
    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasCtbsvTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasCtbsvTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

