# Herkx算子

## 算子概述

Herkx（Single-precision Complex Hermitian Rank-K Update with A≠B）算子实现了单精度复数 Hermitian 矩阵的秩-k扩展更新操作。与标准 HERK（`aclblasCherk`）中 A=B 不同，HERKX 允许 A 和 B 为两个不同的复矩阵，将 `alpha * op(A) * op(B)^H` 加到 Hermitian 矩阵 C 的指定三角区域。其中 alpha 为复数标量，beta 为实数标量（对齐 cuBLAS `cublasCherkx` 的约定），A、B 为复数矩阵，C 为 Hermitian 复矩阵。

数学表达式：

```
trans='N': C = alpha * A * B^H + beta * C
trans='C': C = alpha * A^H * B + beta * C
```

其中 A^H / B^H 为共轭转置（conjugate transpose），即 A^H[i][j] = conj(A[j][i])；A、B 为 N×K（trans='N'）或 K×N（trans='C'）复矩阵，C 为 N×N Hermitian 复矩阵。uplo='U' 时仅更新上三角区域，uplo='L' 时仅更新下三角区域，另一半保持 C 旧值不变。C 满足 Hermitian 性质：C[i][j] = conj(C[j][i])，对角线元素虚部被强制清零。

复数 alpha 展开式（设 alpha = alpha_r + i·alpha_i，乘积 P = op(A)·op(B)^H = P_r + i·P_i）：

```
C_r = alpha_r * P_r - alpha_i * P_i + beta * C_r
C_i = alpha_r * P_i + alpha_i * P_r + beta * C_i    （对角线元素 C_i 强制清零）
```

alpha==0 的判定为复数零：**实部==0 且 虚部==0** 同时成立。纯虚数 alpha（如 (0, 1)）不视为 0，正常参与复数交叉缩放计算。

> **与 HERK 的关键区别**：HERKX 中 A≠B，导致乘积结果 P = A·B^H 一般不具备 Hermitian 对称性（P_r 非对称、P_i 非反对称），对角线虚部理论非零，必须由算子显式强制清零以满足 Hermitian 约束。当 A=B 时，HERKX 退化为 HERK。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasCherkx | 单精度复数 Hermitian 秩-k扩展更新（A≠B） |

## 算子执行接口

### aclblasCherkx

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

> Ascend 950PR/Ascend 950DT 上的 cherkx 依赖 CANN asc-devkit >= 9.1（`ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR >= 1`），版本门控经 `cmake/asc_devkit_version.cmake` 的 `TENSOR_API_OPS` CHERKX 条目生效：低于该版本时 cherkx 源文件不参与编译、对应测试用例跳过。单算子构建口径：`bash build.sh --ops=cherkx --soc=ascend950`。

#### 函数原型

```cpp
aclblasStatus_t aclblasCherkx(
    aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans,
    int n, int k, const aclblasComplex* alpha,
    const aclblasComplex* A, int lda,
    const aclblasComplex* B, int ldb,
    const float* beta, aclblasComplex* C, int ldc);
```

参数顺序与类型映射对齐 cuBLAS 13.4 §2.7.16 `cublasCherkx`（`cuComplex` → `aclblasComplex`、`cublasFillMode_t` → `aclblasFillMode_t` 等，逐位一致）；trans 有效值收窄为 {OP_N, OP_C}。

#### 接口变更说明（Breaking Change）

> **★ API 破坏性变更**：alpha 参数由 `const float*`（实数标量）变更为 `const aclblasComplex*`（复数标量），以对齐 cuBLAS `cublasCherkx`（复数 alpha + 实数 beta）的接口约定。该变更不兼容旧调用代码，调用方须同步迁移；beta 保持 `const float*` 实数标量不变。

**迁移指引**：原实数 alpha 调用方将实数值包装为虚部为 0 的复数标量即可，计算语义不变（alpha 虚部为 0 时，交叉缩放退化为原有的实数缩放）：

```cpp
// 旧接口（实数 alpha，已不再支持）：
float alpha = 1.0f;
aclblasCherkx(handle, uplo, trans, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);

// 新接口（复数 alpha）——原实数值 val 改传 {val, 0.0f}：
aclblasComplex alpha = {1.0f, 0.0f};
aclblasCherkx(handle, uplo, trans, n, k, &alpha, A, lda, B, ldb, &beta, C, ldc);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | C 矩阵存储模式：ACLBLAS_UPPER(121) 仅更新上三角或 ACLBLAS_LOWER(122) 仅更新下三角，Host 内存 |
| trans | 输入 | aclblasOperation_t | A/B 矩阵转置模式：ACLBLAS_OP_N(111) 不转置或 ACLBLAS_OP_C(113) 共轭转置（计算 A^H/B^H）；不支持 ACLBLAS_OP_T(112)，Host 内存 |
| n | 输入 | int | C 矩阵的阶数，n >= 0，Host 内存 |
| k | 输入 | int | A/B 矩阵的第二维度，k >= 0，Host 内存 |
| alpha | 输入 | const aclblasComplex*（复数 FP32） | 复数标量乘数，不可为 nullptr，Host 或 Device 内存（alpha 与 beta 必须同为 Host 或同为 Device 指针） |
| A | 输入 | const aclblasComplex*（复数 FP32） | 输入复矩阵 A，trans='N' 时维度为 N×K，trans='C' 时维度为 K×N，列主序存储，Device 内存；alpha==0（实部虚部均为 0）或 k==0 时可为 nullptr |
| lda | 输入 | int | A 矩阵的主维，trans='N' 时 lda >= max(1, n)，trans='C' 时 lda >= max(1, k)，Host 内存 |
| B | 输入 | const aclblasComplex*（复数 FP32） | 输入复矩阵 B，维度与 A 相同，列主序存储，Device 内存；alpha==0（实部虚部均为 0）或 k==0 时可为 nullptr |
| ldb | 输入 | int | B 矩阵的主维，约束与 lda 相同，Host 内存 |
| beta | 输入 | const float*（FP32） | 实数标量乘数，不可为 nullptr，Host 或 Device 内存（alpha 与 beta 必须同为 Host 或同为 Device 指针） |
| C | 输入/输出 | aclblasComplex*（复数 FP32） | N×N Hermitian 复矩阵，输入旧值，输出新值，仅 uplo 指定三角区域被更新，对角线虚部强制为 0，列主序存储，Device 内存；beta==0 时可为 nullptr（直接成功返回，不更新任何内存） |
| ldc | 输入 | int | C 矩阵的主维，ldc >= max(1, n)，Host 内存 |

#### 支持数据类型

| 数据类型 | 说明 |
|----------|------|
| aclblasComplex（complex\<float\>） | A、B、C 矩阵与 alpha 标量数据类型 |
| float（FP32） | beta 标量数据类型（实数） |

> alpha 为复数标量、beta 为实数标量，这是 cuBLAS `cublasCherkx` 的约定（标准 BLAS 接口集未定义 CHERKX；cuBLAS/MAGMA 的 herkx 均为复数 alpha + 实数 beta，与 CHER2K 的标量约定一致，区别于 CHERK 的实数 alpha）。复数 alpha 按算子概述中的展开式对 C 的实部与虚部做交叉缩放；实数 beta 等比例缩放 C 的实部与虚部。

#### 约束说明

- handle 不可为 nullptr，否则返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR
- uplo 必须为 ACLBLAS_UPPER(121) 或 ACLBLAS_LOWER(122)，非法值返回 ACLBLAS_STATUS_INVALID_ENUM
- trans 必须为 ACLBLAS_OP_N(111) 或 ACLBLAS_OP_C(113)；不支持 ACLBLAS_OP_T(112)（纯转置不含共轭，破坏 Hermitian 语义，cuBLAS 公式对 OP_T 未定义，与 CHERK/CHER2K 行为一致），非法值（含 OP_T）返回 ACLBLAS_STATUS_INVALID_ENUM
- n >= 0, k >= 0，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- n == 0 时直接返回 ACLBLAS_STATUS_SUCCESS，不访问任何指针、不校验 ld 参数（BLAS 标准）
- trans='N' 时：lda >= max(1, n)，ldb >= max(1, n)；trans='C' 时：lda >= max(1, k)，ldb >= max(1, k)，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- ldc >= max(1, n)，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- alpha、beta 不可为 nullptr，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- alpha 与 beta 必须同为 Host 指针或同为 Device 指针（统一指针模式，禁止混合），混合模式返回 ACLBLAS_STATUS_INVALID_VALUE；Host 模式直接解引用（免同步），Device 模式内部通过 D2H 拷贝读回标量真值后再执行校验与计算
- n>0 且 alpha!=0（实部与虚部均为 0 才视为 0，纯虚数不算）且 k>0 时，A、B 不可为 nullptr，否则返回 ACLBLAS_STATUS_INVALID_VALUE；alpha==0 或 k==0 时 A、B 可为 nullptr（BLAS 标准）
- beta!=0 时 C 不可为 nullptr，否则返回 ACLBLAS_STATUS_INVALID_VALUE；beta==0 时 C 可为 nullptr（本算子放宽口径，见下方注），此时直接成功返回、不更新任何内存；Device beta 先读回真值再判断，两种指针模式下 beta==0 判定一致
- alpha==0（复数零）或 k==0 时，跳过矩阵分解与 GEMM 计算，仅执行 C = beta * C 的缩放更新（对角线虚部仍强制清零）
- 当 beta == 0 时，不读取 C 旧值做缩放，uplo 段直接写入计算结果；但 uplo='L' 时对角列的非 uplo 段仍从 C 旧值恢复
- 输出 C 满足 Hermitian 性质：C[i][j] = conj(C[j][i])，对角线元素虚部强制为零
  - 由于 A≠B，乘积对角线虚部理论非零，算子通过 `ZeroDiagonal` 显式强制清零，不依赖数学对称性
- 仅 uplo 指定的三角区域被更新，另一半保持 C 旧值不变
- 矩阵 A、B、C 均按列主序（column-major）存储，复数元素 (row, col) 存储于 col*ld + row 位置，实部在前、虚部在后
- 计算为异步执行，依赖 handle 绑定的 stream；读回 Device 结果前需要同步 stream
- 算子内部会申请库工作区暂存 4 个实矩阵（A/B 的实部/虚部平面）与 4 个 GEMM 临时矩阵：最大 shape（n=k=8192, lda=ldb=8192）时约需 2GB，触及 `ACLBLAS_MAX_WORKSPACE_SIZE`（2 GiB）上限，超出时返回 `ACLBLAS_STATUS_ALLOC_FAILED` 并在日志中给出所需字节数；n=k<=4096 时约 512MB，无风险

> 注：beta==0 时允许 C 为 nullptr 系有意对齐 cher2k arch35 终态的放宽口径，并非 cuBLAS 行为（cuBLAS 对 C 无 nullptr 约定，BLAS 标准中 beta==0 的原义仅是 C 的入口值不参与计算，C 仍是输出矩阵）；本仓 cherk、ssyrkx 对 n>0 时的 C 一律要求非空，调用方从 cherk/ssyrkx 迁移至 cherkx 时请注意此口径差异。

#### 返回值/错误码

| 返回值 | 含义 |
|--------|------|
| ACLBLAS_STATUS_SUCCESS | 执行成功（含 n==0 快速返回、beta==0 且 C 为 nullptr 时的直接成功返回） |
| ACLBLAS_STATUS_HANDLE_IS_NULLPTR | handle 为 nullptr |
| ACLBLAS_STATUS_INVALID_ENUM | uplo 或 trans 为非法枚举值（含 trans=ACLBLAS_OP_T） |
| ACLBLAS_STATUS_INVALID_VALUE | 参数校验失败（n/k 为负、lda/ldb/ldc 不满足约束、alpha/beta 为 nullptr、alpha 与 beta 混合 Host/Device 指针模式、alpha!=0 且 k>0 时 A/B 为 nullptr、beta!=0 时 C 为 nullptr） |
| ACLBLAS_STATUS_ALLOC_FAILED | 工作区需求超出 `ACLBLAS_MAX_WORKSPACE_SIZE`（2 GiB）上限 |
| ACLBLAS_STATUS_INTERNAL_ERROR | 内部错误（核数获取失败、Device 标量 D2H 读回失败等） |

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

int aclblasCherkxTest(AclContext& ctx)
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

    // 2. 准备 Host 数据
    // A 为 N×K 列主序复矩阵 (trans='N'): A = [[1+1i, 2+0i], [3+0i, 4+0i]]
    // 列主序存储: col0 = [1+1i, 3+0i], col1 = [2+0i, 4+0i]
    // B 为 N×K 列主序复矩阵 (trans='N'): B = [[1+0i, 1+1i], [2+0i, 3+0i]]
    // 列主序存储: col0 = [1+0i, 2+0i], col1 = [1+1i, 3+0i]
    constexpr int n = 2;
    constexpr int k = 2;
    constexpr int lda = n;
    constexpr int ldb = n;
    constexpr int ldc = n;
    std::vector<aclblasComplex> hA = {{1.0f, 1.0f}, {3.0f, 0.0f}, {2.0f, 0.0f}, {4.0f, 0.0f}};
    std::vector<aclblasComplex> hB = {{1.0f, 0.0f}, {2.0f, 0.0f}, {1.0f, 1.0f}, {3.0f, 0.0f}};
    std::vector<aclblasComplex> hC(n * n, {0.0f, 0.0f});
    // alpha 为复数标量（对齐 cuBLAS cublasCherkx），beta 为实数标量。
    // Host 指针模式：alpha/beta 直接传 Host 地址（两者必须同为 Host 或同为 Device），
    // 算子自动探测指针位置，Host 模式免 Device→Host 标量回读同步。
    aclblasComplex alpha = {1.0f, 0.0f};
    float beta = 0.0f;

    // 3. 申请 Device 内存并拷贝数据（仅矩阵数据；alpha/beta 为 Host 标量，无需上板）
    constexpr size_t aSize = n * k * sizeof(aclblasComplex);
    constexpr size_t bSize = n * k * sizeof(aclblasComplex);
    constexpr size_t cSize = n * n * sizeof(aclblasComplex);

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

    aclRet = aclrtMemcpy(dAPtr.get(), aSize, hA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dBPtr.get(), bSize, hB.data(), bSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for B failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dCPtr.get(), cSize, hC.data(), cSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for C failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasCherkx
    // 计算 C = (1+0i) * A * B^H + 0.0 * C (trans='N', uplo=LOWER)
    // B^H = [[1+0i, 2+0i], [1-1i, 3+0i]]
    // A * B^H = [[3-1i, 8+2i], [7-4i, 18+0i]]
    // uplo=LOWER: 仅更新下三角 (C[0][0], C[1][0], C[1][1])
    // 对角线虚部强制清零: C[0][0] = 3+0i, C[1][1] = 18+0i
    // 上三角 C[0][1] 保持 C 旧值 = 0+0i
    blasRet = aclblasCherkx(
        handlePtr.get(), ACLBLAS_LOWER, ACLBLAS_OP_N,
        n, k,
        &alpha,
        static_cast<const aclblasComplex*>(dAPtr.get()), lda,
        static_cast<const aclblasComplex*>(dBPtr.get()), ldb,
        &beta,
        static_cast<aclblasComplex*>(dCPtr.get()), ldc);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCherkx failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    std::vector<aclblasComplex> cResult(n * n, {0.0f, 0.0f});
    aclRet = aclrtMemcpy(cResult.data(), cSize, dCPtr.get(), cSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet); return aclRet);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            aclblasComplex val = cResult[static_cast<size_t>(j) * ldc + i];
            LOG_PRINT("C[%d][%d] = %.4f + %.4fi\n", i, j, static_cast<double>(val.real), static_cast<double>(val.imag));
        }
    }
    // 预期输出 (列主序):
    // C[0][0] = 3.0000 + 0.0000i   (下三角, 对角线虚部清零)
    // C[0][1] = 0.0000 + 0.0000i   (上三角, 未更新, 保持 C 旧值)
    // C[1][0] = 7.0000 + -4.0000i  (下三角)
    // C[1][1] = 18.0000 + 0.0000i  (下三角, 对角线虚部清零)

    LOG_PRINT("aclblasCherkx test passed\n");
    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasCherkxTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasCherkxTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

**纯虚数 alpha 示例**（复数 alpha 虚部非 0 时触发交叉缩放；alpha==0 的判定要求实部与虚部同时为 0，纯虚数 alpha 不会被当作 0 处理）：

```cpp
// 计算 C = (0+1i) * A * B^H + 0.0 * C (trans='N', uplo=LOWER)，A/B 取上例数据
aclblasComplex alphaPureImag = {0.0f, 1.0f};
blasRet = aclblasCherkx(handle, ACLBLAS_LOWER, ACLBLAS_OP_N, n, k,
    &alphaPureImag, dA, lda, dB, ldb, &beta, dC, ldc);
// A * B^H = [[3-1i, 8+2i], [7-4i, 18+0i]]
// (0+1i) * (A * B^H) = [[1+3i, -2+8i], [4+7i, 0+18i]]
// uplo=LOWER 且对角线虚部强制清零后:
// C[0][0] = 1.0000 + 0.0000i   (下三角, 对角线虚部清零)
// C[1][0] = 4.0000 + 7.0000i   (下三角)
// C[1][1] = 0.0000 + 0.0000i   (下三角, 对角线虚部清零)
// 上三角 C[0][1] 保持 C 旧值
```

**Device 指针模式示例**（alpha 与 beta 必须同为 Host 或同为 Device，混合模式返回 ACLBLAS_STATUS_INVALID_VALUE；Device 模式下算子内部通过 D2H 拷贝读回标量真值后再执行校验与计算）：

```cpp
aclblasComplex hAlpha = {1.0f, 0.0f};
float hBeta = 0.0f;
aclblasComplex* dAlpha = nullptr;
float* dBeta = nullptr;
aclrtMalloc(reinterpret_cast<void**>(&dAlpha), sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMalloc(reinterpret_cast<void**>(&dBeta), sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
aclrtMemcpy(dAlpha, sizeof(aclblasComplex), &hAlpha, sizeof(aclblasComplex), ACL_MEMCPY_HOST_TO_DEVICE);
aclrtMemcpy(dBeta, sizeof(float), &hBeta, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
blasRet = aclblasCherkx(handle, ACLBLAS_LOWER, ACLBLAS_OP_N, n, k,
    dAlpha, dA, lda, dB, ldb, dBeta, dC, ldc);
```
