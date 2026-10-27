# Symm/Chemm算子

## 算子概述

Symm（Single-precision Symmetric Matrix Multiplication）算子实现了单精度浮点对称矩阵与普通矩阵的乘法运算；
Chemm（Single-precision Hermitian Matrix Multiplication）算子实现了单精度复数 Hermitian 矩阵与普通矩阵的乘法运算。

数学表达式：

```
LEFT 模式：C := alpha * A * B + beta * C
RIGHT 模式：C := alpha * B * A + beta * C
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSsymm | 单精度浮点对称矩阵乘法 |
| aclblasCsymm | 单精度复数对称矩阵乘法 |
| aclblasChemm | 单精度复数 Hermitian 矩阵乘法 |

## 算子执行接口

### aclblasSsymm

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSsymm(aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int m, int n, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ACL-BLAS 句柄，Host 内存 |
| side | 输入 | aclblasSideMode_t | A 矩阵位置：ACLBLAS_SIDE_LEFT（左侧）或 ACLBLAS_SIDE_RIGHT（右侧），Host 内存 |
| uplo | 输入 | aclblasFillMode_t | A 矩阵存储模式：ACLBLAS_LOWER（下三角）或 ACLBLAS_UPPER（上三角），Host 内存 |
| m | 输入 | int | 矩阵 C 的行数，m >= 0，Host 内存 |
| n | 输入 | int | 矩阵 C 的列数，n >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量 alpha，不可为 nullptr，Host 或 Device 内存（alpha 与 beta 必须同为 Host 或同为 Device） |
| A | 输入 | const float*（FP32） | 对称矩阵，side=LEFT 时 m×m，side=RIGHT 时 n×n，Device 内存 |
| lda | 输入 | int | 矩阵 A 的主维，Host 内存（详见约束说明） |
| B | 输入 | const float*（FP32） | m×n 普通矩阵，Device 内存 |
| ldb | 输入 | int | 矩阵 B 的主维，Host 内存（详见约束说明） |
| beta | 输入 | const float*（FP32） | 标量 beta，不可为 nullptr，Host 或 Device 内存（alpha 与 beta 必须同为 Host 或同为 Device） |
| C | 输入/输出 | float*（FP32） | m×n 矩阵，输入旧值，输出新值，beta!=0 时不可为 nullptr，beta==0 时可为 nullptr，Device 内存 |
| ldc | 输入 | int | 矩阵 C 的主维，Host 内存（详见约束说明） |

#### 约束说明

**通用约束：**

- handle 不可为 nullptr，否则返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR
- side 必须为 ACLBLAS_SIDE_LEFT 或 ACLBLAS_SIDE_RIGHT，uplo 必须为 ACLBLAS_UPPER 或 ACLBLAS_LOWER，非法值返回 ACLBLAS_STATUS_INVALID_ENUM
- m==0 或 n==0 时直接返回 ACLBLAS_STATUS_SUCCESS，不访问任何指针、不校验 ld 参数（BLAS 标准）
- m >= 0, n >= 0，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- side=LEFT 时：lda >= max(1, m)
- side=RIGHT 时：lda >= max(1, n)
- m>0 且 n>0 时，alpha、beta 不可为 nullptr，否则返回 ACLBLAS_STATUS_INVALID_VALUE

**arch35（Ascend 950PR / Ascend 950DT）约束：**

- 矩阵 A、B、C 均按列主序存储（column-major），元素 (row, col) 存储于 col*ld + row 位置
- ldb >= max(1, m)
- ldc >= max(1, m)
- m>0 且 n>0 时，alpha 与 beta 必须同为 Host 指针或同为 Device 指针（统一指针模式，禁止混合），否则返回 ACLBLAS_STATUS_INVALID_VALUE
- m>0 且 n>0 且 alpha!=0 时，A、B 不可为 nullptr；alpha==0 时 A、B 可为 nullptr（BLAS 标准）
- beta==0 时 C 可为 nullptr（BLAS 标准：beta==0 时 C 不需要是有效输入；Device beta 通过 ReadAlphaBetaFromDevice 读回值，Host beta 直接解引用，两种模式下 beta 值均在校验前完成解析，统一判断 beta==0）
- alpha==0 时跳过矩阵乘法，仅执行 C = beta * C（快速路径）

**arch22（Atlas A2 / Atlas A3）约束：**

- 矩阵 A、B、C 均按行主序存储（row-major），元素 (row, col) 存储于 row*ld + col 位置
- ldb >= n
- ldc >= n
- m>0 且 n>0 时，A、B、C 不可为 nullptr，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- alpha、beta 仅支持 Host 指针，不支持 Device 指针

#### 精度验证

采用 `MIXED_TOLERANCE` 混合容差策略（对齐[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) §2.1）：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
|----------|------|------|----------------------|-------------------|
| FLOAT32 | 2^-10 (9.77e-4) | 2^-16 (1.53e-5) | 0.99 | 1e-2 或 32·ULP |

alpha == 0（且 alpha 非空）时使用 `EXACT` 位精确校验（C = beta * C 结果应位精确）。

#### 调用示例

```cpp
// arch35（列主序）调用示例：C = alpha * A * B + beta * C
// side=LEFT, uplo=LOWER, m=4, n=4, alpha=1.0, beta=0.0

#include "acl/acl.h"
#include "cann_ops_blas.h"

// 1. 初始化
aclInit(nullptr);
aclblasHandle_t handle;
aclblasCreate(&handle);

// 2. 准备 Host 数据（列主序存储）
int m = 4, n = 4;
int lda = m, ldb = m, ldc = m;
float alpha = 1.0f;
float beta = 0.0f;

// A (4x4 对称矩阵, 下三角, 列主序, lda=4)
std::vector<float> hA = {
    1.0f, 2.0f, 3.0f, 4.0f,   // col 0
    2.0f, 5.0f, 6.0f, 7.0f,   // col 1
    3.0f, 6.0f, 8.0f, 9.0f,   // col 2
    4.0f, 7.0f, 9.0f, 10.0f   // col 3
};
// B (4x4 普通矩阵, 列主序, ldb=4)
std::vector<float> hB = {
    1.0f, 0.0f, 0.0f, 0.0f,   // col 0
    0.0f, 1.0f, 0.0f, 0.0f,   // col 1
    0.0f, 0.0f, 1.0f, 0.0f,   // col 2
    0.0f, 0.0f, 0.0f, 1.0f    // col 3
};
std::vector<float> hC(static_cast<size_t>(ldc) * n, 0.0f);

// 3. 申请 Device 内存并拷贝数据
size_t aBytes = hA.size() * sizeof(float);
size_t bBytes = hB.size() * sizeof(float);
size_t cBytes = hC.size() * sizeof(float);

void* dA = nullptr;
aclrtMalloc(&dA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMemcpy(dA, aBytes, hA.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE);

void* dB = nullptr;
aclrtMalloc(&dB, bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMemcpy(dB, bBytes, hB.data(), bBytes, ACL_MEMCPY_HOST_TO_DEVICE);

void* dC = nullptr;
aclrtMalloc(&dC, cBytes, ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMemcpy(dC, cBytes, hC.data(), cBytes, ACL_MEMCPY_HOST_TO_DEVICE);

// 4. 执行计算
aclblasSsymm(handle, ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER,
    m, n, &alpha,
    static_cast<const float*>(dA), lda,
    static_cast<const float*>(dB), ldb,
    &beta,
    static_cast<float*>(dC), ldc);

// 5. 同步并拷回结果
aclrtStream stream;
aclblasGetStream(handle, &stream);
aclrtSynchronizeStream(stream);
aclrtMemcpy(hC.data(), cBytes, dC, cBytes, ACL_MEMCPY_DEVICE_TO_HOST);

// 6. 释放资源
aclrtFree(dA);
aclrtFree(dB);
aclrtFree(dC);
aclblasDestroy(handle);
aclFinalize();
```

---

### aclblasCsymm

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

> aclblasCsymm 提供 arch35（`blas/symm/arch35/`，面向 Ascend 950PR / Ascend 950DT）与 arch22（`blas/symm/arch22/`，面向 Atlas A2）两套实现。接口声明位于公共头文件 `include/cann_ops_blas.h`，供各产品线共用，未定义产品线私有平行接口。

#### 函数原型

```cpp
aclblasStatus_t aclblasCsymm(aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo, int m, int n, const aclblasComplex *alpha, const aclblasComplex *A, int lda, const aclblasComplex *B, int ldb, const aclblasComplex *beta, aclblasComplex *C, int ldc)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ACL-BLAS 句柄，Host 内存 |
| side | 输入 | aclblasSideMode_t | A 矩阵位置：ACLBLAS_SIDE_LEFT（左侧）或 ACLBLAS_SIDE_RIGHT（右侧），Host 内存 |
| uplo | 输入 | aclblasFillMode_t | A 矩阵存储模式：ACLBLAS_LOWER（下三角）或 ACLBLAS_UPPER（上三角），Host 内存 |
| m | 输入 | int | 矩阵 C 的行数，m >= 0，Host 内存 |
| n | 输入 | int | 矩阵 C 的列数，n >= 0，Host 内存 |
| alpha | 输入 | const aclblasComplex*（COMPLEX64） | 复数标量 alpha，不可为 nullptr；arch35 为 Host 内存，arch22 为 Device 内存 |
| A | 输入 | const aclblasComplex*（COMPLEX64） | 对称矩阵（A = Aᵀ，不共轭），side=LEFT 时 m×m，side=RIGHT 时 n×n，仅 uplo 指定三角被引用，Device 内存 |
| lda | 输入 | int | 矩阵 A 的主维，Host 内存（详见约束说明） |
| B | 输入 | const aclblasComplex*（COMPLEX64） | m×n 普通矩阵，Device 内存 |
| ldb | 输入 | int | 矩阵 B 的主维，Host 内存（详见约束说明） |
| beta | 输入 | const aclblasComplex*（COMPLEX64） | 复数标量 beta，不可为 nullptr；arch35 为 Host 内存，arch22 为 Device 内存 |
| C | 输入/输出 | aclblasComplex*（COMPLEX64） | m×n 矩阵，输入旧值并原地覆写输出新值，Device 内存 |
| ldc | 输入 | int | 矩阵 C 的主维，Host 内存（详见约束说明） |

#### 约束说明

- handle 不可为 nullptr，否则返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR
- side 必须为 ACLBLAS_SIDE_LEFT 或 ACLBLAS_SIDE_RIGHT，uplo 必须为 ACLBLAS_UPPER 或 ACLBLAS_LOWER，非法值返回 ACLBLAS_STATUS_INVALID_ENUM
- m >= 0，n >= 0，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- m==0 或 n==0 时直接返回 ACLBLAS_STATUS_SUCCESS，不访问任何指针、不校验 ld 参数（BLAS 标准）
- side=LEFT 时 lda >= max(1, m)；side=RIGHT 时 lda >= max(1, n)
- ldb >= max(1, m)，ldc >= max(1, m)，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- m>0 且 n>0 时，alpha、beta 不可为 nullptr，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- m>0 且 n>0 且 alpha!=(0,0) 时，A、B 不可为 nullptr；alpha==(0,0) 时 A、B 可为 nullptr（BLAS 标准）
- beta==(0,0) 时 C 可为 nullptr（BLAS 标准：此时 C 不需要是有效输入，实现也不会读取 C，避免 0×NaN 污染结果）
- alpha==(0,0) 且 beta==(1,0) 时 C 保持不变（Netlib csymm quick return）；alpha==(0,0) 时跳过矩阵乘，仅执行 C = beta * C
- 矩阵 A、B、C 均按列主序（column-major）存储，复数元素 (row, col) 存储于 col*ld + row 位置，实部在前、虚部在后
- A 的对称性（A = Aᵀ）由调用方保证，实现不校验；未被引用的三角内容不参与计算；对角元素虚部不置零、不作修正（区别于 Hermitian 算子 aclblasChemm）
- 不支持超出 lda/ldb/ldc 语义的非连续 Tensor
- 计算为异步执行，依赖 handle 绑定的 stream；读回 Device 结果前需要同步 stream
- uplo 指定的是 **A** 的存储三角（不是 C），另一侧由算子按 A[i][j] = A[j][i] 补齐，对角线元素虚部照常参与运算
- alpha、beta 均为复数，缩放阶段按复数乘法展开
- 输出 C 为一般 M×N 矩阵，不具备任何对称性，全部元素均被写入
- 算子内部会申请库工作区暂存展开后的 A、拆分后的 B 与 GEMM 中间结果；当超出 `ACLBLAS_MAX_WORKSPACE_SIZE`（2 GiB）时返回 `ACLBLAS_STATUS_ALLOC_FAILED` 并在日志中给出所需字节数

#### 实现说明（arch35）

三段式 Kernel 流水（Ascend C kernel 直调）：

1. **Prep（AIV, SIMT）**：按 uplo 将对称三角镜像为完整矩阵，并把 COMPLEX64 的 A、B 拆分为 real / imag 四个 float 平面；镜像只做下标转置，不做共轭
2. **GEMM（AIC, Cube）**：复数乘展开为 4 次实数矩阵乘，在同一次 Kernel 启动内以 (tile × term) 为工作项遍历，结果写入 4 个临时平面
   - term0 = Ar×Br，term1 = Ai×Bi，term2 = Ar×Bi，term3 = Ai×Br
   - prodReal = term0 − term1，prodImag = term2 + term3
   - 列主序适配：Host 侧交换 m↔n 并翻转 side，Kernel 内部按行主序计算 Cᵀ
3. **Scale（AIV, SIMT）**：C = alpha * prod + beta * C，按扁平化的列主序元素空间遍历以获得最好的 AIV 访存合并

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
// 注意：示例代码使用 printf 便于演示，生产代码应使用 dlog（OP_LOGE/OP_LOGI/OP_LOGD）

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

int aclblasCsymmTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-blas 句柄
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasHandle_t>::type, BlasHandleDeleter> handlePtr(rawHandle);

    blasRet = aclblasSetStream(handlePtr.get(), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 2. 准备 Host 数据（side=LEFT，A 为 M×M 对称矩阵，仅存下三角）
    // A 的下三角: A[0][0] = 2+0i, A[1][0] = 3+1i, A[1][1] = 5+0i
    // 算子按对称性补齐上三角，得到完整的 A = [[2+0i, 3+1i], [3+1i, 5+0i]]
    // 列主序存储 A: col0 = [2+0i, 3+1i], col1 = [未引用, 5+0i]
    // B = [[1+0i, 2+0i], [0+1i, 1+0i]]，列主序: col0 = [1+0i, 0+1i], col1 = [2+0i, 1+0i]
    constexpr int m = 2;
    constexpr int n = 2;
    constexpr int lda = m;
    constexpr int ldb = m;
    constexpr int ldc = m;
    std::vector<aclblasComplex> hA = {{2.0f, 0.0f}, {3.0f, 1.0f}, {0.0f, 0.0f}, {5.0f, 0.0f}};
    std::vector<aclblasComplex> hB = {{1.0f, 0.0f}, {0.0f, 1.0f}, {2.0f, 0.0f}, {1.0f, 0.0f}};
    std::vector<aclblasComplex> hC(m * n, {0.0f, 0.0f});
    aclblasComplex alpha = {1.0f, 0.0f};
    aclblasComplex beta = {0.0f, 0.0f};

    // 3. 申请 Device 内存并拷贝数据
    constexpr size_t aSize = m * m * sizeof(aclblasComplex);
    constexpr size_t bSize = m * n * sizeof(aclblasComplex);
    constexpr size_t cSize = m * n * sizeof(aclblasComplex);

    void* rawA = nullptr;
    aclError aclRet = aclrtMalloc(&rawA, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtr(rawA, aclrtFree);

    void* rawB = nullptr;
    aclRet = aclrtMalloc(&rawB, bSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for B failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dBPtr(rawB, aclrtFree);

    void* rawC = nullptr;
    aclRet = aclrtMalloc(&rawC, cSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for C failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dCPtr(rawC, aclrtFree);

    void* rawAlpha = nullptr;
    aclRet = aclrtMalloc(&rawAlpha, sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for alpha failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAlphaPtr(rawAlpha, aclrtFree);

    void* rawBeta = nullptr;
    aclRet = aclrtMalloc(&rawBeta, sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for beta failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dBetaPtr(rawBeta, aclrtFree);

    aclRet = aclrtMemcpy(dAPtr.get(), aSize, hA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(dBPtr.get(), bSize, hB.data(), bSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for B failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(dCPtr.get(), cSize, hC.data(), cSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for C failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(dAlphaPtr.get(), sizeof(aclblasComplex), &alpha, sizeof(aclblasComplex), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for alpha failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(dBetaPtr.get(), sizeof(aclblasComplex), &beta, sizeof(aclblasComplex), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for beta failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasCsymm
    // 计算 C = 1.0 * A * B + 0.0 * C (side=LEFT, uplo=LOWER)
    // 期望结果 C = [[1+3i, 7+1i], [3+6i, 11+2i]]
    blasRet = aclblasCsymm(
        handlePtr.get(), ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER,
        m, n,
        static_cast<const aclblasComplex*>(dAlphaPtr.get()),
        static_cast<const aclblasComplex*>(dAPtr.get()), lda,
        static_cast<const aclblasComplex*>(dBPtr.get()), ldb,
        static_cast<const aclblasComplex*>(dBetaPtr.get()),
        static_cast<aclblasComplex*>(dCPtr.get()), ldc);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCsymm failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印（C 为一般矩阵，全部元素均被更新）
    std::vector<aclblasComplex> cResult(m * n, {0.0f, 0.0f});
    aclRet = aclrtMemcpy(cResult.data(), cSize, dCPtr.get(), cSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet); return aclRet);
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            aclblasComplex val = cResult[static_cast<size_t>(j) * ldc + i];
            LOG_PRINT("C[%d][%d] = %.4f + %.4fi\n", i, j, static_cast<double>(val.real), static_cast<double>(val.imag));
        }
    }

    LOG_PRINT("aclblasCsymm test passed\n");
    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasCsymmTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasCsymmTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

---

### aclblasChemm

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

> 本仓库同时维护 arch35 与 arch22 两套实现：
> - arch35：[`blas/hemm/arch35/chemm_host.cpp`](../hemm/arch35/chemm_host.cpp)
> - arch22：[`blas/hemm/arch22/chemm_host.cpp`](../hemm/arch22/chemm_host.cpp)
>
> 接口声明位于公共头文件 `include/cann_ops_blas.h`，两个产品线共用同一 API。

#### 函数原型

```cpp
aclblasStatus_t aclblasChemm(aclblasHandle_t handle, aclblasSideMode_t side, aclblasFillMode_t uplo,
    int64_t m, int64_t n, const aclblasComplex *alpha, const aclblasComplex *A, int64_t lda,
    const aclblasComplex *B, int64_t ldb, const aclblasComplex *beta, aclblasComplex *C, int64_t ldc)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ACL-BLAS 句柄，Host 内存 |
| side | 输入 | aclblasSideMode_t | Hermitian 矩阵 A 的位置：ACLBLAS_SIDE_LEFT（左侧）或 ACLBLAS_SIDE_RIGHT（右侧），Host 内存 |
| uplo | 输入 | aclblasFillMode_t | A 的存储三角：ACLBLAS_UPPER（上三角）或 ACLBLAS_LOWER（下三角），Host 内存 |
| m | 输入 | int64_t | 矩阵 C 的行数，m >= 0，Host 内存 |
| n | 输入 | int64_t | 矩阵 C 的列数，n >= 0，Host 内存 |
| alpha | 输入 | const aclblasComplex*（COMPLEX64） | 复数标量 alpha，Host 内存 |
| A | 输入 | const aclblasComplex*（COMPLEX64） | Hermitian 矩阵，side=LEFT 时 m×m，side=RIGHT 时 n×n，仅 uplo 指定三角被引用，Device 内存 |
| lda | 输入 | int64_t | 矩阵 A 的主维，Host 内存（详见约束说明） |
| B | 输入 | const aclblasComplex*（COMPLEX64） | m×n 普通矩阵，Device 内存 |
| ldb | 输入 | int64_t | 矩阵 B 的主维，Host 内存（详见约束说明） |
| beta | 输入 | const aclblasComplex*（COMPLEX64） | 复数标量 beta，Host 内存 |
| C | 输入/输出 | aclblasComplex*（COMPLEX64） | m×n 矩阵，输入旧值并原地覆写输出新值，Device 内存 |
| ldc | 输入 | int64_t | 矩阵 C 的主维，Host 内存（详见约束说明） |

#### 约束说明

- handle 不可为 nullptr，否则返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR
- side 必须为 ACLBLAS_SIDE_LEFT 或 ACLBLAS_SIDE_RIGHT，uplo 必须为 ACLBLAS_UPPER 或 ACLBLAS_LOWER，非法值返回 ACLBLAS_STATUS_INVALID_VALUE
- m >= 0，n >= 0，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- m==0 或 n==0 时直接返回 ACLBLAS_STATUS_SUCCESS，不访问任何指针、不校验 ld 参数（BLAS 标准）
- side=LEFT 时：lda >= max(1, m)；side=RIGHT 时：lda >= max(1, n)
- ldb >= max(1, m)，ldc >= max(1, m)，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- m>0 且 n>0 时，alpha、beta 不可为 nullptr，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- m>0 且 n>0 且 alpha!=(0,0) 时，A、B 不可为 nullptr；alpha==(0,0) 时 A、B 可为 nullptr（BLAS 标准）
- beta==(0,0) 时 C 可为 nullptr（BLAS 标准）
- alpha==(0,0) 时跳过矩阵乘法，仅执行 C = beta * C（快速路径）
- 矩阵 A、B、C 均按列主序（column-major）存储，复数元素 (row, col) 存储于 col*ld + row 位置，实部在前、虚部在后
- A 的 Hermitian 性质（A = Aᴴ，对角虚部为 0）由调用方保证，实现不校验；未被引用的三角内容不参与计算，对角虚部按 0 处理
- 不支持超出 lda/ldb/ldc 语义的非连续 Tensor
- 计算为异步执行，依赖 handle 绑定的 stream；读回 Device 结果前需要同步 stream

#### 实现说明（arch35）

Host 按规模、标量、地址范围和 workspace 约束选择 SIMT 或 Cube/AIV 组合路径。Cube 输入使用实数三平面分解，AIV 按原累加顺序合并并执行数值风险检查；风险输出保留有序重算路径。对于 RIGHT/LOWER、深 M 且窄 N 的工作负载，Host 保留通用 checked combine，同时对短收缩维度省略队列初始化和 fallback launch，敏感输出直接 inline 重算，以避免近阈值 case 的额外调度开销。

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

```cpp
// arch35（列主序）调用示例：C = alpha * A * B + beta * C
// side=LEFT, uplo=LOWER, m=2, n=2, alpha=1+0i, beta=0+0i

#include "acl/acl.h"
#include "cann_ops_blas.h"

int main()
{
    aclblasHandle_t handle = nullptr;
    aclblasCreate(&handle);

    // A 为 2x2 Hermitian 矩阵，仅使用下三角：
    // A = [[2+0i, 3-1i],
    //      [3+1i, 5+0i]]
    constexpr int64_t m = 2;
    constexpr int64_t n = 2;
    constexpr int64_t lda = m;
    constexpr int64_t ldb = m;
    constexpr int64_t ldc = m;

    aclblasComplex alpha = {1.0f, 0.0f};
    aclblasComplex beta = {0.0f, 0.0f};
    aclblasComplex A[4] = {{2.0f, 0.0f}, {3.0f, 1.0f}, {0.0f, 0.0f}, {5.0f, 0.0f}};
    aclblasComplex B[4] = {{1.0f, 0.0f}, {2.0f, 0.0f}, {3.0f, 0.0f}, {4.0f, 0.0f}};
    aclblasComplex C[4] = {};

    auto ret = aclblasChemm(
        handle, ACLBLAS_SIDE_LEFT, ACLBLAS_LOWER, m, n, &alpha, A, lda, B, ldb, &beta, C, ldc);

    aclblasDestroy(handle);
    return ret == ACLBLAS_STATUS_SUCCESS ? 0 : 1;
}
```
