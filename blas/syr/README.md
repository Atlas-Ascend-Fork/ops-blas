# Syr算子

## 算子概述

syr (Symmetric Rank-1 Update) 实现对称矩阵的秩-1更新操作。该算子将 `alpha * x * x^T` 加到对称矩阵 A 的指定三角区域，仅上三角或下三角区域被引用和更新。

数学表达式：

```
A := alpha * x * x^T + A
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSsyr | 单精度对称秩-1更新 |
| aclblasCsyr | 单精度复数对称秩-1更新（不共轭） |

## 算子执行接口

### aclblasSsyr

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSsyr(aclblasHandle_t handle, aclblasFillMode uplo, const int n, const float* alpha, const float* x, const int incx, float* A, const int lda)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode | ACLBLAS_UPPER(121) 或 ACLBLAS_LOWER(122)，指定更新的三角区域，Host 内存 |
| n | 输入 | int | 矩阵阶数，n >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量乘数指针，Host/Device 内存 |
| x | 输入 | const float*（FP32） | 输入向量指针，长度至少 1 + (n-1) * abs(incx)，Device 内存 |
| incx | 输入 | int | x 的元素间步长，incx != 0，Host 内存 |
| A | 输入/输出 | float*（FP32） | 对称矩阵指针，维度 (lda, n)，Device 内存 |
| lda | 输入 | int | A 的主维度，lda >= max(1, n)，Host 内存 |

#### 约束说明

- n >= 0
- incx != 0
- lda >= max(1, n)

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
int aclblasSsyrTest(AclContext& ctx)
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
    constexpr int incx = 1;
    constexpr int lda = n;
    constexpr size_t xSize = n * sizeof(float);
    constexpr size_t aSize = n * n * sizeof(float);

    std::vector<float> hX = {1.0f, 2.0f};
    std::vector<float> hA(n * n, 0.0f);
    float alpha = 1.0f;

    void* rawX = nullptr;
    aclError aclRet;
    aclRet = aclrtMalloc(&rawX, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for x failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dXPtr(rawX, aclrtFree);

    void* rawA = nullptr;
    aclRet = aclrtMalloc(&rawA, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtr(rawA, aclrtFree);

    aclRet = aclrtMemcpy(dXPtr.get(), xSize, hX.data(), xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for x failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(dAPtr.get(), aSize, hA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A failed. ERROR: %d\n", aclRet); return aclRet);

    blasRet = aclblasSsyr(
        static_cast<aclblasHandle_t>(handlePtr.get()), ACLBLAS_LOWER, n, &alpha,
        static_cast<const float*>(dXPtr.get()), incx, static_cast<float*>(dAPtr.get()), lda);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSsyr failed. ERROR: %d\n", blasRet);
              return blasRet);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    std::vector<float> aResult(n * n, 0.0f);
    aclRet = aclrtMemcpy(aResult.data(), aSize, dAPtr.get(), aSize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet); return aclRet);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            LOG_PRINT("A[%d][%d] = %f\n", i, j, aResult[j * lda + i]);
        }
    }

    LOG_PRINT("aclblasSsyr test passed\n");
    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSsyrTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSsyrTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

### aclblasCsyr

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持（arch35 实现）
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持（arch22 实现）
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持（arch22 实现）

#### 函数原型

```cpp
aclblasStatus_t aclblasCsyr(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n,
    const aclblasComplex* alpha, const aclblasComplex* x, const int incx,
    aclblasComplex* A, const int lda);
```

对标 cuBLAS `cublasCsyr`，计算单精度复数（COMPLEX64）对称秩-1更新：

```
A := alpha * x * x^T + A
```

其中 `x^T` 为普通转置（**不共轭**，区别于厄米特秩-1更新 `aclblasCher` 的 `x^H`）；alpha 为复数标量。A 为对称（非厄米特）复数矩阵，列主序存储，仅 `uplo` 指定的三角区域被引用和更新，另一三角不读不写；对角元素为普通复数元素，虚部不作任何假定或置零处理。

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | ACLBLAS_UPPER(121) 或 ACLBLAS_LOWER(122)，指定引用/更新的三角区域，Host 内存 |
| n | 输入 | int | 矩阵阶数，n >= 0，Host 内存；n = 0 或 alpha = (0,0) 为合法 quick return |
| alpha | 输入 | const aclblasComplex*（COMPLEX64） | 复数标量乘数指针，Host/Device 内存 |
| x | 输入 | const aclblasComplex*（COMPLEX64） | 输入复数向量指针，逻辑长度 n，物理长度至少 1 + (n-1) * abs(incx)，Device 内存 |
| incx | 输入 | int | x 的元素间步长（复数元素单位），incx != 0，可正可负（负步长按 Netlib ssyr 语义反向遍历），Host 内存 |
| A | 输入/输出 | aclblasComplex*（COMPLEX64） | 对称复数矩阵指针，维度 (lda, n)，列主序，原地更新，Device 内存 |
| lda | 输入 | int | A 的主维度，lda >= max(1, n)，Host 内存 |

#### 约束说明

接口级约束（arch22 与 arch35 实现一致）：

- 执行 `A := alpha * x * x^T + A`，其中 `x^T` 为普通转置，不进行共轭；仅 `uplo` 指定的三角区域（含对角线）被引用和更新，另一三角区域及 padding 行保持不变；对角元素按普通复数计算，虚部不会被强制置零
- n >= 0，incx != 0，lda >= max(1, n)
- handle 有效且 n = 0 时直接返回成功；n > 0 时先检查 uplo、lda、incx 和 alpha 指针，检查通过后若 alpha = (0, 0) 则返回成功且不更新 A；n > 0 且 alpha != (0, 0) 时 x 和 A 不得为空指针
- alpha 支持 Host/Device 指针传入：Device 指针内部 D2H 读回，Host 指针直接读取

实现差异：

- arch22（Atlas A2/A3）：单核 UB 容量约束下支持 n <= 4864；lda 与 |incx| 须满足元素级跨度约束：(n-1)·lda + n（A）与 (n-1)·|incx| + 1（x）均 ≤ 2^31，超界返回 `ACLBLAS_STATUS_INVALID_VALUE`（n = 0 quick return 不受影响）
- arch35（Ascend 950PR/DT）：无固定 n 上限（4096 为调优参数，不是接口尺寸限制）；`aclblasComplex` 自然对齐 4 字节，x/A 满足 8 字节对齐时走 float2 快路径，否则走标量 GM 路径（含独立 x/A 偏移回归）；复数乘法以独立舍入的 `__fma` 写法逐乘积舍入，保证大数溢出时 Inf/NaN 分类与 CPU golden 一致（单位 alpha 的零乘项保留，避免省略 `0*Inf` 后改变 NaN 传播）

#### 执行模型说明（arch22 同步接口）

**arch22 实现为同步接口**：`aclblasCsyr` 返回前在内部完成 `aclrtSynchronizeStream`，返回时本次更新已全部写入 A。这与 cuBLAS/aclBLAS 常规算子「仅入队、由调用方自行同步」的异步契约不同——调用方无需也无法依赖本接口在 arch22 上的异步排队行为；需要流水线并行的调用方应在更上层组织多流。arch35 实现为常规异步入队语义，由调用方按需 `aclrtSynchronizeStream`。

成因：arch22 实现采用 host 侧预构建 x 交错镜像、按调用上传的 staging 方案，且 kernel 以原子累加写（SetAtomicAdd）完成原地更新；在当前驱动代际上，流同步不能可靠等待在飞原子写，因此实现以按调用的设备内存分配/释放作为排空点，并在返回前显式同步流，以保证 staging 缓冲生命周期与更新可见性的严格正确。

中期方向：per-stream staging 缓冲池复用；kernel 去原子化后去除强制同步，使接口回到仅入队的异步契约。

#### 返回值

- `ACLBLAS_STATUS_SUCCESS`：成功
- `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`：handle 为空指针
- `ACLBLAS_STATUS_INVALID_ENUM`：uplo 取值非法
- `ACLBLAS_STATUS_INVALID_VALUE`：n < 0、incx = 0、lda < max(1, n)、alpha/x/A 为空指针（n > 0 且 alpha ≠ (0,0) 时）、n 超出实现支持上限（arch22：n > 4864）或元素级跨度超界（arch22）

#### 调度配置说明（arch35）

Host 使用 `GetAivCoreCount()` 查询当前设备的 AIV（向量核）数量，实际启动的
block 数为 `min(n, AIV 核数)`。`CSYR_TUNED_AIV_BLOCKS = 56` 表示既有
Ascend 950PR 优化配置所针对的 56 个 AIV block，并非所有支持设备的固定核数。
只有实际启动配置为 56 个 block、每个 block 1024 个 SIMT 线程时，才选择固定步长
特化；其他配置使用运行时的 block/thread 数进行遍历。

64 × 64 三角分块将 1024 个线程划分为 64 个行 lane 和 16 个列 lane，每个列 lane
按 16 列的步长遍历。该路径仅在非零 Host alpha、`incx = 1`、8 字节对齐、
`1024 < n <= 4096`、`lda <= 4096` 且满足上述启动配置时启用。
其他情形根据步长、对齐及矩阵尺寸选择循环列或标量 GM 路径。512/1024 线程策略及
4096 阶快路径上限均为调优参数，不是接口尺寸限制；数据直接在 GM 中访问，
不存在对应大小的 UB 缓冲区分配。

#### 调用示例

```cpp
#include <cstdio>
#include <memory>
#include <vector>
#include "acl/acl.h"
#include "cann_ops_blas.h"

// 前置：已完成 aclInit / aclrtSetDevice / aclblasCreate / aclblasSetStream
aclblasStatus_t CsyrExample(aclblasHandle_t handle, aclrtStream stream)
{
    constexpr int n = 4;
    constexpr int lda = n;
    std::vector<aclblasComplex> hX(n, {1.0f, 1.0f});
    std::vector<aclblasComplex> hA(n * lda, {0.0f, 0.0f});
    aclblasComplex alpha = {1.0f, 0.0f};
    const size_t xBytes = hX.size() * sizeof(aclblasComplex);
    const size_t aBytes = hA.size() * sizeof(aclblasComplex);

    // 设备内存经 unique_ptr RAII 托管（deleter 为 aclrtFree）：析构时释放，
    // 任意错误返回路径都零泄漏，与逐处手动 aclrtFree 等价。
    void* rawX = nullptr;
    aclError aclRet = aclrtMalloc(&rawX, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        fprintf(stderr, "aclrtMalloc for x failed. ERROR: %d\n", static_cast<int>(aclRet));
        return ACLBLAS_STATUS_ALLOC_FAILED;
    }
    std::unique_ptr<void, aclError (*)(void*)> dXPtr(rawX, aclrtFree);
    auto* dX = static_cast<aclblasComplex*>(rawX);

    void* rawA = nullptr;
    aclRet = aclrtMalloc(&rawA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_SUCCESS) {
        fprintf(stderr, "aclrtMalloc for A failed. ERROR: %d\n", static_cast<int>(aclRet));
        return ACLBLAS_STATUS_ALLOC_FAILED;  // dX 由 dXPtr 析构释放
    }
    std::unique_ptr<void, aclError (*)(void*)> dAPtr(rawA, aclrtFree);
    auto* dA = static_cast<aclblasComplex*>(rawA);

    aclRet = aclrtMemcpy(dX, xBytes, hX.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        fprintf(stderr, "aclrtMemcpy H2D for x failed. ERROR: %d\n", static_cast<int>(aclRet));
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    aclRet = aclrtMemcpy(dA, aBytes, hA.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_SUCCESS) {
        fprintf(stderr, "aclrtMemcpy H2D for A failed. ERROR: %d\n", static_cast<int>(aclRet));
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    const aclblasStatus_t blasRet = aclblasCsyr(handle, ACLBLAS_UPPER, n, &alpha, dX, 1, dA, lda);
    if (blasRet != ACLBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "aclblasCsyr failed. STATUS: %d\n", static_cast<int>(blasRet));
        return blasRet;
    }
    aclRet = aclrtSynchronizeStream(stream);
    if (aclRet != ACL_SUCCESS) {
        fprintf(stderr, "aclrtSynchronizeStream failed. ERROR: %d\n", static_cast<int>(aclRet));
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }
    aclRet = aclrtMemcpy(hA.data(), aBytes, dA, aBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    if (aclRet != ACL_SUCCESS) {
        fprintf(stderr, "aclrtMemcpy D2H failed. ERROR: %d\n", static_cast<int>(aclRet));
        return ACLBLAS_STATUS_INTERNAL_ERROR;
    }

    return ACLBLAS_STATUS_SUCCESS;  // dX/dA 由 dXPtr/dAPtr 析构释放
}
```

示例中 `std::unique_ptr`（deleter 为 `aclrtFree`）的 RAII 托管与逐处手动 `aclrtFree` 完全等价：每个错误返回路径都自动释放已申请的设备内存，成功路径在函数返回时统一释放。

#### 测试说明

测试遵循[ST 开发指南](../../docs/zh/develop/st_develop_guide.md)，目录为
`test/syr/csyr/`。参数解析和 CPU golden 位于接口目录（arch22 与 arch35 ST 共用），
NPU wrapper、GTest 用例及同名 CSV 分别位于 `arch22/` 与 `arch35/`，
使用公共 `BlasTest`、数据生成和精度校验框架。

在已配置 CANN 环境的设备上，从仓库根目录运行（以 Atlas A2/A3 为例）：

```bash
bash build.sh --ops=csyr --soc=ascend910B3 --run
```

arch22 测试覆盖上下三角、正负步长、参数校验、Host/Device alpha、特殊浮点值、
快速返回、G3 未引用区域快照断言与性能用例；arch35 测试覆盖上下三角、正负步长、
参数校验、Host/Device alpha、特殊浮点值、4 字节地址对齐，以及输入向量、另一
三角区域和 padding 不被修改的检查。
