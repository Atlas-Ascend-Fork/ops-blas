# Her2k算子

## 算子概述

Her2k（Single-precision Complex Hermitian Rank-2K Update）算子实现了单精度复数 Hermitian 矩阵的秩-2k 更新操作，将 alpha * op(A) * op(B)^H 与 conj(alpha) * op(B) * op(A)^H 之和加到 Hermitian 矩阵 C 的指定三角区域。其中 alpha 为复数标量、beta 为实数标量，A、B 为复数矩阵，C 为 Hermitian 复矩阵。

数学表达式：

```
trans='N': C = alpha * A * B^H + conj(alpha) * B * A^H + beta * C
trans='C': C = alpha * A^H * B + conj(alpha) * B^H * A + beta * C
```

其中 A^H 为 A 的共轭转置（conjugate transpose），即 A^H[i][j] = conj(A[j][i])；A、B 同为 N×K（trans='N'）或 K×N（trans='C'）复矩阵，C 为 N×N Hermitian 复矩阵。uplo='U' 时仅更新上三角区域，uplo='L' 时仅更新下三角区域，另一半保持不变。两项互为共轭转置，故结果恒为 Hermitian：C[i][j] = conj(C[j][i])，对角线元素虚部为零。

beta 之所以限定为实数，是因为复数 beta 会破坏 C 的 Hermitian 性质，这一点与 BLAS 标准一致。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasCher2k | 单精度复数 Hermitian 秩-2k 更新 |

## 算子执行接口

### aclblasCher2k

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持

> Ascend 950PR/Ascend 950DT 上的 cher2k 依赖 CANN asc-devkit >= 9.1（`ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR >= 1`），低于该版本时编译与运行将跳过此算子。

#### 函数原型

```cpp
aclblasStatus_t aclblasCher2k(aclblasHandle_t handle, aclblasFillMode_t uplo, aclblasOperation_t trans, int n, int k, const aclblasComplex* alpha, const aclblasComplex* A, int lda, const aclblasComplex* B, int ldb, const float* beta, aclblasComplex* C, int ldc)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | C 矩阵存储模式：ACLBLAS_UPPER(121) 仅更新上三角或 ACLBLAS_LOWER(122) 仅更新下三角，Host 内存 |
| trans | 输入 | aclblasOperation_t | A、B 的转置模式：ACLBLAS_OP_N(111) 不转置或 ACLBLAS_OP_C(113) 共轭转置，Host 内存 |
| n | 输入 | int | C 矩阵的阶数，n >= 0，Host 内存 |
| k | 输入 | int | 归约维度：trans='N' 时为 A、B 的列数（第二维度），trans='C' 时为 A、B 的行数（第一维度），k >= 0，Host 内存 |
| alpha | 输入 | const aclblasComplex*（复数 FP32） | 复数标量乘数，不可为 nullptr，Host 或 Device 内存 |
| A | 输入 | const aclblasComplex*（复数 FP32） | 输入复矩阵，trans='N' 时维度为 N×K，trans='C' 时维度为 K×N，Device 内存 |
| lda | 输入 | int | A 矩阵的主维，trans='N' 时 lda >= max(1, n)，trans='C' 时 lda >= max(1, k)，Host 内存 |
| B | 输入 | const aclblasComplex*（复数 FP32） | 输入复矩阵，维度与 A 相同，Device 内存 |
| ldb | 输入 | int | B 矩阵的主维，下界与 lda 相同，Host 内存 |
| beta | 输入 | const float*（FP32） | **实数**标量乘数，不可为 nullptr，Host 或 Device 内存 |
| C | 输入/输出 | aclblasComplex*（复数 FP32） | N×N Hermitian 复矩阵，输入旧值，输出新值，仅 uplo 指定三角区域被更新，Device 内存 |
| ldc | 输入 | int | C 矩阵的主维，ldc >= max(1, n)，Host 内存 |

> 注：alpha/beta 支持 Host 或 Device 指针，两代实现均按指针位置自动探测（alpha 与 beta 可独立选择侧别）：Host 指针在 host 侧解引用一次后随 tiling 下发；Device 指针不做 host 侧 D2H 读回——arch22 由 kernel 每次调用从 Device 内存读取真实值，arch35 由 K3/K4 kernel 从 GM 直读并派生标量标志。

#### 约束说明

以下约束对 arch22（Atlas A2）与 arch35（Ascend 950）两代实现均适用：

- n >= 0, k >= 0
- uplo 为 ACLBLAS_UPPER 或 ACLBLAS_LOWER；越界枚举值返回 ACLBLAS_STATUS_INVALID_ENUM
- trans 为 ACLBLAS_OP_N 或 ACLBLAS_OP_C；ACLBLAS_OP_T 为合法枚举但不支持，返回 ACLBLAS_STATUS_INVALID_VALUE；其他越界值返回 ACLBLAS_STATUS_INVALID_ENUM
- trans='N' 时：lda >= max(1, n)、ldb >= max(1, n)
- trans='C' 时：lda >= max(1, k)、ldb >= max(1, k)
- ldc >= max(1, n)
- alpha、beta 不可为 nullptr
- A、B 不可为 nullptr（当 n > 0 且 k > 0 时）
- alpha 为复数、beta 为**实数**：复数 beta 会破坏 C 的 Hermitian 性质，故按 BLAS 标准限定为实数
- 仅更新指定三角，对角线虚部置零。另一三角和 leading-dimension padding 保持原值（值语义不变；实现上 uplo='L' 时对角块的非 uplo（严格上）段按 C 旧值恒等回写，uplo='U' 时非 uplo 段不写出）
- 当 alpha 为 0（或 k 为 0）且 beta 为 1 时按 BLAS 语义直接返回，C 保持原值不变——**包含其对角线虚部**（族内差异注：cherk 同为快返回不写 C，cherkx 无快返回、uplo 三角恒重写且对角虚部恒清零，见各自 README）
- 当 beta = 0 时 C 的取值无需有效（cuBLAS 契约 "If beta == 0 then C does not have to be a valid input"）：uplo 指定三角不读取 C 旧值做缩放、直接写入新值，C 旧值中的 NaN/Inf 不进入 uplo 三角的结果（arch35 的 K3/K4 路径与 arch22 路径行为一致）；C 非 nullptr 时 uplo='L' 对角块的非 uplo（严格上）段仍按 C 旧值恒等回写（语义上不修改，NaN/Inf 原位保留、不扩散），C 为 nullptr 时按 arch35 口径直接成功返回、不发生任何写出
- A、B、C 为列主序 complex64（`aclblasComplex`，即 fp32 实部 + fp32 虚部）存储的 Device 内存
- 工作区大小由对齐后的 n、k 和 kernel 路径共同决定（arch35 通用管线的定量公式见下方 arch35 分组）。申请失败或超过 `ACLBLAS_MAX_WORKSPACE_SIZE` 时返回错误，停止 launch

arch22（Atlas A2）实现：

- C 指针口径：n>0 时 C 必须非空；唯一例外是零乘积（alpha=0 或 k=0）且 beta=0 的无写出 quick return
- kernel 每次调用读取 Device 标量的真实值；同一标量地址更新数值后可复用

arch35（Ascend 950）实现：

- 依赖 CANN asc-devkit >= 9.1，低于该版本编译与运行跳过此算子
- alpha、beta 可为 Host 或 Device 指针（按指针位置自动探测，与 arch22 口径一致）：Host 指针由 host 解引用一次后随 tiling 下发；Device 指针由 K3/K4 kernel 从 GM 直读并派生 isAlphaZero/isBetaZero/skipTemp，无 host 侧 D2H 读回与同步
- C 指针口径：当 n > 0 且 beta != 0 时 C 不可为 nullptr；当 n > 0 且 beta = 0 时 C 允许为 nullptr（返回 SUCCESS 且不发生任何写出）；n = 0 时 C 可为 nullptr
- n <= 8 走 K4 SIMT 融合直算路径（无 workspace）；n > 8 走 Cube+AIV 三阶段通用管线
- 通用管线（n > 8）需约 16·n·k + 16·n² 字节库工作区（A/B 实/虚拆分缓存 4×4·n·k 与 4 份 GEMM 中间结果 4×4·n²；n 按 16 元素向上对齐、总量按 512 字节取整；skipTemp 即 alpha=0 或 k=0 时为 0）；当超出 `ACLBLAS_MAX_WORKSPACE_SIZE`（2 GiB）时返回 ACLBLAS_STATUS_ALLOC_FAILED
- 通用管线的 K2f 融合核在 fallback 形状（baseM=32，即 n<64 或 CHER2K_L1_SHAPE 强制 32×16 档）且 k > tileKChunk 时启用 SEG 分段累加（StreamK 式）：每个 k-chunk 为一段（S=ceil(k/tileKChunk)），段内 L0C 累加、段尾折出至 UB 乒乓槽，AIV 以 fp32 UB 累加器做段间顺序归约后执行既有折叠输出——将 k 深度单链 fp32 顺序舍入链截短为段级短链，收敛大 k 深度累加的舍入噪声（2^-13 精度标准下 TC_EX_0604 类小 n 深 k 用例达标：ratio 98.69% → 100%，maxAbsErr 2.63e-3 → 3.55e-4）；n≥64 主形状（baseM=64）与 k≤tileKChunk（S=1）不启用，n≥64 主路径输出与启用前逐位一致；环境变量 `CHER2K_SEG_K` 默认开，置 0 回退单链旧行为（A/B 对照与现场回退，无需重编）
- 输出 C 满足 Hermitian 性质：C[i][j] = conj(C[j][i])，对角线元素虚部精确为零（代数相消），非 uplo 三角与 ldc padding 字节保持不变
- 异步执行：多个 kernel 全部下发至 handle 绑定的同一 stream，按流语义自动排序，算子内部无需显式同步；调用方读回结果前须自行同步所绑定的 stream（如 `aclrtSynchronizeStream(stream)`，见下方调用示例第 5 步）

测试 CSV 口径：arch22 侧默认 CSV 含前 50 条用例，`CHER2K_FULL_REGRESSION=1` 读取全量 1214 条；arch35 侧为单文件 1203 条（含 200 条 TC_PF 性能用例与 3 条 TC_WB K4 路径锁定用例），性能模式单独读取其中的性能用例。

#### 返回码

| 返回值 | 含义 |
|--------|------|
| ACLBLAS_STATUS_SUCCESS（0） | 成功（含 n=0、(alpha=(0,0) 或 k=0) 且 beta=1 的 quick return，以及 C=nullptr 的无写出 quick return——arch35 为 beta=0，arch22 为零乘积且 beta=0） |
| ACLBLAS_STATUS_HANDLE_IS_NULLPTR（9） | handle 为 nullptr |
| ACLBLAS_STATUS_INVALID_ENUM（10） | uplo 或 trans 为越界枚举值 |
| ACLBLAS_STATUS_INVALID_VALUE（3） | trans=OP_T；n<0 或 k<0；lda/ldb/ldc 违反前导维规则；alpha 或 beta 为 nullptr；n>0 且 k>0 时 A 或 B 为 nullptr；n>0 且 beta!=0 时 C 为 nullptr |
| ACLBLAS_STATUS_INTERNAL_ERROR（6） | 核数查询、alpha/beta 读取、workspace 申请等内部错误 |

> 注：C 指针允许为 nullptr 的条件两代实现不同（见上方约束说明 arch22/arch35 分组），其余返回码口径一致。

#### 与 aclblasCherk 的差异

| 差异点 | aclblasCherk | aclblasCher2k |
|--------|--------------|---------------|
| 输入矩阵 | 单矩阵 A（C = alpha·op(A)·op(A)^H + beta·C） | 双矩阵 A、B（独立装载、同形、无广播） |
| alpha 类型 | 实数（float），同时缩放 C 实部与虚部 | 复数（aclblasComplex），第二项使用 conj(alpha) |
| 数学结构 | A·A^H 天然 Hermitian | M + M^H 对称化构造（M = alpha·A·B^H 或 alpha·A^H·B） |
| 对角虚部 | t3−t4 对角来自两次独立 GEMM，需 epilogue 强制置零 | M[i][i] + conj(M[i][i]) 代数相消，精确为零 |
| 计算量 | 4 次实数 GEMM | 4 次实数 GEMM（4M 分解，第二项按转置读合成） |
| C 指针口径 | n > 0 时一律非空 | 仅 beta != 0 且 n > 0 时非空（beta=0 时可为 nullptr，SUCCESS 无写出） |
| 非法 uplo/trans 返回码 | INVALID_VALUE | 越界枚举返回 INVALID_ENUM；OP_T 返回 INVALID_VALUE |

> 注：表中 cher2k 的 C 指针口径为 arch35 实现；arch22 实现更严格（仅零乘积且 beta=0 时允许 nullptr），见上方约束说明。

#### 精度验证

采用 `MIXED_TOLERANCE` 混合容差策略（对齐[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) §2.1 的 2026-09-17 修订口径），complex64 输出按实部/虚部拆分后分别按 FP32 口径校验：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
|----------|------|------|----------------------|-------------------|
| COMPLEX64（实部/虚部分别按 FP32 口径） | 2^-13 (1.22e-4) | 2^-13 (1.22e-4) | 0.99 | 1e-2 或 32·ULP |

其中，ULP（Unit in the Last Place，末位单位）表示给定浮点数与相邻可表示值之间的间距。
`max_abs_error_limit` 按输出元素逐个计算：第`i`个元素的误差上限为
`max(1e-2, 32 * ULP(abs(golden[i])))`。精度验证需同时满足以下条件：

- 至少 99% 的输出元素满足 `abs(actual[i] - golden[i]) <= atol + rtol * abs(golden[i])`。
- 所有输出元素均满足 `abs(actual[i] - golden[i]) <= max_abs_error_limit[i]`。

附加校验（超出混合容差表的算子专属硬断言）：对角线虚部模长无条件 <= 2^-16（Hermitian 承诺）；非 uplo 三角与 ldc padding 与输入 C 旧值逐位一致（不引用即不污染）。

大抵消场景的 MERE/MARE 复判：her2k 计算 M + M^H 对称叠加，在大 alpha（如 1e4）、大 k 或高均值随机输入下交叉乘抵消会使部分 golden 元素接近 0——此时 float32 在 ~10^3 量级下的固有舍入误差（约 0.01）远超 atol=2^-13，严格混合容差不可达（float cblas 参考链同样无法通过，见测试 CSV 的 *_float_xcheck 列）。因此测试在严格混合容差未通过、且用例 CSV 预置 mere_threshold/mare_multiplier 列（arch35 全量用例预置 2^-13 与 10.0）时，按 MERE/MARE 相对误差口径复判：近零 golden（|golden| < mere_threshold）跳过统计不参与相对误差，其余元素要求平均相对误差 MERE < mere_threshold 且最大相对误差 MARE < mare_multiplier × mere_threshold，判据语义与 test/frame 的 MereMareStrategy 一致，实现见 test/her2k/cher2k/arch35/cher2k_test.cpp 的 EvaluateMereMareDbl。

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

int aclblasCher2kTest(AclContext& ctx)
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
    constexpr int n = 2;
    constexpr int k = 2;
    constexpr int lda = n;
    constexpr int ldb = n;
    constexpr int ldc = n;
    // B = [[1+0i, 0+0i], [1+0i, 1+0i]]，列主序存储: col0 = [1+0i, 1+0i], col1 = [0+0i, 1+0i]
    std::vector<aclblasComplex> hA = {{1.0f, 1.0f}, {3.0f, 0.0f}, {2.0f, 0.0f}, {4.0f, 0.0f}};
    std::vector<aclblasComplex> hB = {{1.0f, 0.0f}, {1.0f, 0.0f}, {0.0f, 0.0f}, {1.0f, 0.0f}};
    std::vector<aclblasComplex> hC(n * n, {0.0f, 0.0f});
    aclblasComplex alpha = {1.0f, 0.0f};  // alpha 为复数
    float beta = 0.0f;                    // beta 为实数

    // 3. 申请 Device 内存并拷贝数据
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

    void* rawAlpha = nullptr;
    aclRet = aclrtMalloc(&rawAlpha, sizeof(aclblasComplex), ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for alpha failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAlphaPtr(rawAlpha, aclrtFree);

    void* rawBeta = nullptr;
    aclRet = aclrtMalloc(&rawBeta, sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST);
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

    aclRet = aclrtMemcpy(dBetaPtr.get(), sizeof(float), &beta, sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for beta failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasCher2k
    // 计算 C = alpha * A * B^H + conj(alpha) * B * A^H + beta * C (trans='N', uplo=LOWER)
    // B^H = [[1+0i, 1+0i], [0+0i, 1+0i]]
    // A * B^H = [[1+1i, 3+1i], [3+0i, 7+0i]]
    // B * A^H = (A * B^H)^H = [[1-1i, 3+0i], [3-1i, 7+0i]]
    // 两项之和 = [[2+0i, 6+1i], [6-1i, 14+0i]]
    // 结果为 Hermitian: C[0][1] = conj(C[1][0])，对角线虚部为零
    // uplo=LOWER: 仅更新下三角 (C[0][0], C[1][0], C[1][1])
    blasRet = aclblasCher2k(
        handlePtr.get(), ACLBLAS_LOWER, ACLBLAS_OP_N,
        n, k,
        static_cast<const aclblasComplex*>(dAlphaPtr.get()),
        static_cast<const aclblasComplex*>(dAPtr.get()), lda,
        static_cast<const aclblasComplex*>(dBPtr.get()), ldb,
        static_cast<const float*>(dBetaPtr.get()),
        static_cast<aclblasComplex*>(dCPtr.get()), ldc);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCher2k failed. ERROR: %d\n", blasRet);
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

    LOG_PRINT("aclblasCher2k test passed\n");
    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasCher2kTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasCher2kTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```
