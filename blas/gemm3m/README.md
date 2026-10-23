# Gemm3m算子

## 算子概述

3M 方法矩阵乘法算子（gemm3m），用于将复数矩阵乘法分解为 3 次实数矩阵乘法后在单次 Kernel 调用内完成累加。算子接收 2 个合并后的实数矩阵（A 包含 A1/A2/A3 沿 K 维拼接，B 包含 B1/B2/B3 沿 K 维堆叠），计算三组矩阵乘积之和并叠加缩放后的原始 C，相比 3 次独立 GEMM 调用减少中间结果的设备内存往返。所有矩阵均为 FP32（float）、列主序。

数学表达式：

```
C = alpha * (op(A1)*op(B1) + op(A2)*op(B2) + op(A3)*op(B3)) + beta * C
```

其中 op(X) 为 ACLBLAS_OP_N（不转置）、ACLBLAS_OP_T（转置）或 ACLBLAS_OP_C（共轭转置，实数矩阵等价于 T）；A 包含 A1/A2/A3 沿 K 维拼接，共享 transA 与 lda；B 包含 B1/B2/B3 沿 K 维堆叠，共享 transB 与 ldb。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSgemm3m | 单精度（FP32）3M 方法矩阵乘法，计算三组实数 GEMM 的加权和 |

## 算子执行接口

### aclblasSgemm3m

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

> Ascend 950PR/Ascend 950DT 上的 sgemm3m 依赖 CANN asc-devkit >= 9.1（`ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR >= 1`），低于该版本时编译与运行将跳过此算子。

#### 函数原型

```cpp
aclblasStatus_t aclblasSgemm3m(
    aclblasHandle handle, aclblasOperation_t transA, aclblasOperation_t transB, int m, int n, int k,
    const float* alpha, const float* A, int lda,
    const float* B, int ldb,
    const float* beta, float* C, int ldc)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| transA | 输入 | aclblasOperation_t | A 的操作类型：ACLBLAS_OP_N（不转置）、ACLBLAS_OP_T（转置）、ACLBLAS_OP_C（共轭转置，实数等价于 T），Host 内存 |
| transB | 输入 | aclblasOperation_t | B 的操作类型（同 transA），Host 内存 |
| m | 输入 | int | op(Ai) 和 C 的行数，M >= 0，Host 内存 |
| n | 输入 | int | op(Bi) 和 C 的列数，N >= 0，Host 内存 |
| k | 输入 | int | op(Ai) 的列数 / op(Bi) 的行数（单个子矩阵），K >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | 指向标量 alpha 的指针，Host 内存 |
| A | 输入 | const float*（FP32） | 合并矩阵 A 设备内存指针，包含 A1/A2/A3 沿 K 维拼接。transA=N 时为 M×(3K)，lda>=M；transA=T/C 时为 (3K)×M，lda>=3K，列主序，Device 内存 |
| lda | 输入 | int | A 的主维度（列主序），Host 内存 |
| B | 输入 | const float*（FP32） | 合并矩阵 B 设备内存指针，包含 B1/B2/B3 沿 K 维堆叠。transB=N 时为 (3K)×N，ldb>=3K；transB=T/C 时为 N×(3K)，ldb>=N，列主序，Device 内存 |
| ldb | 输入 | int | B 的主维度（列主序），Host 内存 |
| beta | 输入 | const float*（FP32） | 指向标量 beta 的指针，Host 内存 |
| C | 输入/输出 | float*（FP32） | 矩阵 C 设备内存指针，M×N，列主序，Device 内存 |
| ldc | 输入 | int | C 的主维度（列主序），ldc >= max(1, m)，Host 内存 |

#### 约束说明

- m, n, k >= 0
- handle、alpha、beta 不能为 nullptr
- transA=N 时 lda >= max(1, m)；transA=T/C 时 lda >= max(1, 3*k)
- transB=N 时 ldb >= max(1, 3*k)；transB=T/C 时 ldb >= max(1, n)
- ldc >= max(1, m)
- m>0 且 n>0 且 k>0 且 alpha!=0 时，A 和 B 不能为 nullptr
- 与 gemm 不同（gemm 仅在 k == 0 且 beta == 0 时允许 C 为 nullptr），gemm3m 在 m>0 且 n>0 时无论 beta、k 取值均要求 C 非 nullptr
- 仅支持 FP32 数据类型
- transA=C（共轭转置）对实数矩阵等价于 transA=T

边界情况处理：
- m == 0 或 n == 0 时直接返回 ACLBLAS_STATUS_SUCCESS，不执行计算
- k == 0 或 alpha == 0 时跳过矩阵乘，执行 C = beta * C（beta == 0 时置零，beta == 1 时不变，其他值逐元素缩放）
- beta == 0.0f 时 C 不会被读取，无需在调用前初始化（但 C 指针本身仍不能为 nullptr）

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
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        aclInited_ = true;

        ret = aclrtSetDevice(deviceId_);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        deviceSet_ = true;

        ret = aclrtCreateStream(&stream_);
        CHECK_RET(ret == ACL_SUCCESS, return ret);
        return ACL_SUCCESS;
    }

    aclrtStream Stream() const { return stream_; }

private:
    int32_t deviceId_;
    aclrtStream stream_ = nullptr;
    bool aclInited_ = false;
    bool deviceSet_ = false;
};

struct AclrtMemDeleter {
    void operator()(void* ptr) const
    {
        if (ptr != nullptr) {
            aclrtFree(ptr);
        }
    }
};

struct AclblasHandleDeleter {
    void operator()(void* handle) const
    {
        if (handle != nullptr) {
            aclblasDestroy(static_cast<aclblasHandle_t>(handle));
        }
    }
};

// Helper: allocate device memory and copy host data in. Returns nullptr on failure.
static void* MallocAndCopy(const void* hData, size_t bytes)
{
    void* dPtr = nullptr;
    auto ret = aclrtMalloc(&dPtr, bytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(ret == ACL_SUCCESS, return nullptr);
    ret = aclrtMemcpy(dPtr, bytes, hData, bytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(ret == ACL_SUCCESS, aclrtFree(dPtr); return nullptr);
    return dPtr;
}

int aclblasSgemm3mTest(AclContext& ctx)
{
    // C = alpha * (A1*B1 + A2*B2 + A3*B3) + beta * C, column-major, FP32.
    // M=N=K=2, transA=N, transB=N, alpha=1.0, beta=0.0.
    // A is M×(3K) = 2×6 column-major; B is (3K)×N = 6×2 column-major.
    constexpr int m = 2;
    constexpr int n = 2;
    constexpr int k = 2;
    constexpr int lda = m;      // transA=N: lda >= M
    constexpr int ldb = 3 * k;  // transB=N: ldb >= 3K
    constexpr int ldc = m;
    float alpha = 1.0f;
    float beta = 0.0f;

    // Column-major storage. A = [A1 | A2 | A3] concatenated along columns.
    // A1 = A2 = identity (2×2), A3 = zeros (2×2).
    // A layout (2×6, col-major, lda=2): col0-1=A1, col2-3=A2, col4-5=A3.
    std::vector<float> hA(static_cast<size_t>(lda) * 3 * k, 0.0f);
    // A1: identity at columns 0-1
    hA[0] = 1.0f; hA[1] = 0.0f;  // col 0
    hA[2] = 0.0f; hA[3] = 1.0f;  // col 1
    // A2: identity at columns 2-3
    hA[4] = 1.0f; hA[5] = 0.0f;  // col 2
    hA[6] = 0.0f; hA[7] = 1.0f;  // col 3
    // A3: zeros at columns 4-5 (already zero)

    // B = [B1; B2; B3] stacked along rows.
    // B1 = [[1,2],[3,4]], B2 = ones (2×2), B3 = zeros (2×2).
    // B layout (6×2, col-major, ldb=6): rows 0-1=B1, rows 2-3=B2, rows 4-5=B3.
    std::vector<float> hB(static_cast<size_t>(ldb) * n, 0.0f);
    // col 0: [1, 3, 1, 1, 0, 0]
    hB[0] = 1.0f; hB[1] = 3.0f;  // B1 col 0
    hB[2] = 1.0f; hB[3] = 1.0f;  // B2 col 0
    // col 1: [2, 4, 1, 1, 0, 0]
    hB[6] = 2.0f; hB[7] = 4.0f;  // B1 col 1
    hB[8] = 1.0f; hB[9] = 1.0f;  // B2 col 1

    std::vector<float> hC(static_cast<size_t>(ldc) * n, 0.0f);
    const size_t aBytes = static_cast<size_t>(lda) * 3 * k * sizeof(float);
    const size_t bBytes = static_cast<size_t>(ldb) * n * sizeof(float);
    const size_t cBytes = static_cast<size_t>(ldc) * n * sizeof(float);

    // 1. Create ops-blas handle and bind stream.
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);
    std::unique_ptr<void, AclblasHandleDeleter> handle(rawHandle);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handle.get()), ctx.Stream());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    // 2. Allocate Device memory and copy data from Host.
    std::unique_ptr<void, AclrtMemDeleter> dA(MallocAndCopy(hA.data(), aBytes));
    std::unique_ptr<void, AclrtMemDeleter> dB(MallocAndCopy(hB.data(), bBytes));
    std::unique_ptr<void, AclrtMemDeleter> dC(MallocAndCopy(hC.data(), cBytes));
    if (dA == nullptr || dB == nullptr || dC == nullptr) {
        return -1;
    }

    // 3. Call aclblasSgemm3m.
    blasRet = aclblasSgemm3m(
        static_cast<aclblasHandle_t>(handle.get()), ACLBLAS_OP_N, ACLBLAS_OP_N,
        m, n, k, &alpha,
        static_cast<float*>(dA.get()), lda,
        static_cast<float*>(dB.get()), ldb,
        &beta, static_cast<float*>(dC.get()), ldc);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    // 4. Synchronize and copy result back to Host.
    auto aclRet = aclrtSynchronizeStream(ctx.Stream());
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclRet = aclrtMemcpy(hC.data(), cBytes, dC.get(), cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // Expected: C = A1*B1 + A2*B2 + A3*B3 = [[2,3],[4,5]] -> col-major [2,4,3,5].
    for (int col = 0; col < n; ++col) {
        for (int row = 0; row < m; ++row) {
            printf("C[%d][%d] = %f\n", row, col, hC[static_cast<size_t>(col) * ldc + row]);
        }
    }

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSgemm3mTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}
```
