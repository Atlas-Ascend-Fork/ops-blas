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

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

#### 函数原型

```cpp
aclblasStatus_t aclblasCsyr(
    aclblasHandle_t handle, aclblasFillMode_t uplo, const int n,
    const aclblasComplex* alpha, const aclblasComplex* x, const int incx,
    aclblasComplex* A, const int lda);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | ACLBLAS_UPPER(121) 或 ACLBLAS_LOWER(122)，指定更新的三角区域，Host 内存 |
| n | 输入 | int | 矩阵阶数，n >= 0，Host 内存 |
| alpha | 输入 | const aclblasComplex*（COMPLEX64） | 复数标量乘数指针，Host/Device 内存 |
| x | 输入 | const aclblasComplex*（COMPLEX64） | 输入复向量，物理长度至少为 1 + (n - 1) * abs(incx)，Device 内存 |
| incx | 输入 | int | x 的复数元素间步长，支持正、负步长且 incx != 0，Host 内存 |
| A | 输入/输出 | aclblasComplex*（COMPLEX64） | 列主序复对称矩阵，维度 (lda, n)，Device 内存 |
| lda | 输入 | int | A 的主维度，lda >= max(1, n)，Host 内存 |

#### 约束说明

- 执行 `A := alpha * x * x^T + A`，其中 `x^T` 为普通转置，不进行共轭。
- 仅 `uplo` 指定的三角区域（含对角线）被引用和更新；另一三角区域及 padding 行保持不变。
- 对角元素按普通复数计算，虚部不会被强制置零。
- n >= 0，incx != 0，lda >= max(1, n)。
- handle 有效且 n = 0 时直接返回成功；n > 0 时先检查 uplo、lda、incx 和 alpha 指针，
  检查通过后若 alpha = (0, 0)，则返回成功且不更新 A。
- n > 0 且 alpha != (0, 0) 时，x 和 A 不得为空指针。

arch35 的 complex64 快路径保留单位 alpha 的零乘项，避免省略
`0*Inf` 后改变 NaN 传播。复数乘法使用直接写在表达式中的 `__fma(a,b,0)`
独立舍入每个 FP32 乘积，再进行加减；这避免有限大数乘积溢出时，被收缩的
乘加表达式与 CPU golden 产生不同的 Inf/NaN 分类。此实现使用仓库统一的编译配置，
不设置算子专用编译选项。
`aclblasComplex` 的自然对齐为 4 字节；x 或 A 未满足 8 字节对齐时走标量 GM
路径，满足条件时继续采用 float2 快路径。测试包含独立 x/A 偏移回归。

#### 调度配置说明

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

初始化 ACL、创建 handle、绑定 stream 及申请 Device 内存的步骤与
`aclblasSsyr` 示例相同。以下代码展示复数接口的核心调用：

```cpp
constexpr int n = 2;
constexpr int incx = 1;
constexpr int lda = n;
aclblasComplex alpha{1.0f, 2.0f};

// dX 和 dA 分别指向已初始化的 Device 复数向量和列主序复数矩阵。
aclblasStatus_t status = aclblasCsyr(
    handle, ACLBLAS_LOWER, n, &alpha,
    static_cast<const aclblasComplex*>(dX), incx,
    static_cast<aclblasComplex*>(dA), lda);
CHECK_RET(status == ACLBLAS_STATUS_SUCCESS, return status);
CHECK_RET(aclrtSynchronizeStream(stream) == ACL_SUCCESS, return ACLBLAS_STATUS_EXECUTION_FAILED);
```

#### 测试说明

测试遵循[ST 开发指南](../../docs/zh/develop/st_develop_guide.md)，目录为
`test/syr/csyr/`。参数解析和 CPU golden 位于接口目录，NPU wrapper、GTest 用例及
同名 CSV 位于 `arch35/`，使用公共 `BlasTest`、数据生成和精度校验框架。

在已配置 CANN 环境的 Ascend 950 设备上，从仓库根目录运行：

```bash
bash build.sh --ops=csyr --soc=ascend950 --run
```

测试覆盖上下三角、正负步长、参数校验、Host/Device alpha、特殊浮点值、4 字节地址对齐，
以及输入向量、另一三角区域和 padding 不被修改的检查。
