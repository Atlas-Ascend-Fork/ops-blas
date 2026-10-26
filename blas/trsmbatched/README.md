# Trsmbatched算子

## 算子概述

批量三角矩阵线性方程组求解（Batched Triangular Solve），对多个独立的三角线性方程组进行批量求解，解矩阵 X[i] 原地覆盖 B[i]。

数学表达式：

```
side='L': op(A[i]) * X[i] = alpha * B[i]
side='R': X[i] * op(A[i]) = alpha * B[i]
```

其中 op(A) 由 trans 参数决定：trans='N' 时 op(A)=A，trans='T' 时 op(A)=A^T，trans='C' 时 op(A)=A^H（实数下等价于 A^T）。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasStrsmBatched | 单精度浮点批量三角求解 |
| [aclblasCtrsmBatched](#aclblasctrsmbatchedcomplex64) | 单精度复数批量三角求解（当前实现：Atlas 950PR，arch35） |

## 算子执行接口

### aclblasStrsmBatched

#### 产品支持情况

| 产品 | 实现支持 | 验证状态 |
|---|---|---|
| Ascend 950PR / Ascend 950DT | 支持（arch35） | 沿用原实现 |
| Atlas A3 训练/推理系列（含 Atlas 800I A3） | 支持（arch22） | 1200 项测试全部通过 |
| Atlas A2 训练/推理系列（含 Atlas 800I A2） | 支持（arch22） | 1200 项测试全部通过 |

> Ascend 950PR/Ascend 950DT 上的 strsmbatched 依赖 CANN asc-devkit >= 9.1（`ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR >= 1`），低于该版本时编译与运行将跳过此算子。

#### 函数原型

```cpp
aclblasStatus_t aclblasStrsmBatched(
    aclblasHandle_t handle,
    aclblasSideMode_t side,
    aclblasFillMode_t uplo,
    aclblasOperation_t trans,
    aclblasDiagType_t diag,
    int m,
    int n,
    const float* alpha,
    const float* const A[],
    int lda,
    float* const B[],
    int ldb,
    int batchCount);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| side | 输入 | aclblasSideMode_t | 三角矩阵位置：ACLBLAS_SIDE_LEFT（A 在左侧）/ ACLBLAS_SIDE_RIGHT（A 在右侧），Host 内存 |
| uplo | 输入 | aclblasFillMode_t | 三角存储模式：ACLBLAS_UPPER（上三角）/ ACLBLAS_LOWER（下三角），Host 内存 |
| trans | 输入 | aclblasOperation_t | 转置模式：ACLBLAS_OP_N（不转置）/ ACLBLAS_OP_T（转置）/ ACLBLAS_OP_C（共轭转置，实数下等价于 T），Host 内存 |
| diag | 输入 | aclblasDiagType_t | 对角线类型：ACLBLAS_NON_UNIT（非单位对角线）/ ACLBLAS_UNIT（单位对角线，假定为 1 不访问），Host 内存 |
| m | 输入 | int | B 的行数，>= 0，Host 内存 |
| n | 输入 | int | B 的列数，>= 0，Host 内存 |
| alpha | 输入 | const float* | 缩放因子指针，alpha=0 时 B 置零（不访问 A），Host 内存 |
| A | 输入 | const float* const[] | 批量三角矩阵指针数组，A[i] 指向第 i 个三角矩阵 A[i]；side=LEFT 时 A[i] 为 (M,M) 矩阵，side=RIGHT 时 A[i] 为 (N,N) 矩阵，Device 内存 |
| lda | 输入 | int | A[i] 的 leading dimension，side=LEFT 时 lda >= max(1,M)，side=RIGHT 时 lda >= max(1,N)，Host 内存 |
| B | 输入/输出 | float* const[] | 批量右端/解矩阵指针数组，B[i] 指向第 i 个 (M,N) 矩阵，输出时原地覆盖为解矩阵 X[i]，Device 内存 |
| ldb | 输入 | int | B[i] 的 leading dimension，ldb >= max(1,M)，Host 内存 |
| batchCount | 输入 | int | 批量大小，>= 0，Host 内存。0 表示无批量，为合法 no-op，直接返回成功且不访问 A/B 指针数组（允许为 nullptr）；< 0 返回 ACLBLAS_STATUS_INVALID_VALUE |

#### 约束说明

- m >= 0, n >= 0
- batchCount >= 0；batchCount == 0 为合法 no-op（对齐 cuBLAS 批量族约定）：标量参数（side/uplo/trans/diag/m/n/alpha/batchCount/lda/ldb）仍完整校验、非法组合照常报错，仅 A/B 指针数组不被校验与访问（允许为 nullptr），校验通过后直接返回 ACLBLAS_STATUS_SUCCESS；batchCount < 0 返回 ACLBLAS_STATUS_INVALID_VALUE
- side 必须为 ACLBLAS_SIDE_LEFT 或 ACLBLAS_SIDE_RIGHT
- uplo 必须为 ACLBLAS_UPPER 或 ACLBLAS_LOWER
- trans 必须为 ACLBLAS_OP_N、ACLBLAS_OP_T 或 ACLBLAS_OP_C
- diag 必须为 ACLBLAS_NON_UNIT 或 ACLBLAS_UNIT
- alpha 不为 nullptr
- B 不为 nullptr（当 batchCount > 0 时）
- A 不为 nullptr（当 alpha != 0 且 batchCount > 0 时）
- lda >= max(1, K)，其中 K = (side==LEFT ? m : n)
- ldb >= max(1, m)
- 计算路径仅按 m/n 阈值选择（SIMT / Blocked），对 lda/ldb 无额外对齐要求
- 仅支持 FP32 数据类型
- 单次调用临时工作区（Workspace）需求为 batchCount * (tempBatchStride + 8192) * sizeof(float) 字节，其中 tempBatchStride = align16(m) * align16(n)（非对齐跨度场景另含对齐重排区）
- 受底层 aclblasHandle_t 最大工作区限制（ACLBLAS_MAX_WORKSPACE_SIZE = 2 GiB），调用方需保证单次调用的总工作区需求不超过 2 GiB，超出上限将返回 ACLBLAS_STATUS_ALLOC_FAILED。对于大尺寸叠加超大 Batch（例如 m=n>=1024 且 batchCount>=1024 等超 2 GiB 的组合），建议调用方按 Batch 分批调用

边界情况处理：

- m = 0 或 n = 0 时为合法 no-op，直接返回 ACLBLAS_STATUS_SUCCESS（不访问矩阵数据）
- alpha = 0 时 A 不被引用，B 的逻辑区域直接写零（不读旧值，NaN/Inf 不传播）
- 各 batch 相互独立，B[i] 不可重叠；UNIT 对角假定为 1 且不访问

#### 精度验证

采用 `MIXED_TOLERANCE` 混合容差策略（对齐[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) §2.1）：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
|----------|------|------|----------------------|-------------------|
| FLOAT32 | 2^-10 (9.77e-4) | 2^-16 (1.53e-5) | 0.99 | 1e-2 或 32·ULP |

其中，ULP（Unit in the Last Place，末位单位）表示给定浮点数与相邻可表示值之间的间距。
`max_abs_error_limit` 按输出元素逐个计算：第`i`个元素的误差上限为
`max(1e-2, 32 * ULP(abs(golden[i])))`。精度验证需同时满足以下条件：

- 至少 99% 的输出元素满足 `abs(actual[i] - golden[i]) <= atol + rtol * abs(golden[i])`。
- 所有输出元素均满足 `abs(actual[i] - golden[i]) <= max_abs_error_limit[i]`。

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

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

class AclContext {
public:
    explicit AclContext(int deviceId) : deviceId_(deviceId) {}

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
    int deviceId_;
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
    void operator()(aclblasHandle_t handle) const
    {
        if (handle != nullptr) {
            aclblasDestroy(handle);
        }
    }
};

int aclblasStrsmBatchedTest(AclContext& ctx)
{
    // 2 batches, each A is 3x3 lower triangular, B is 3x2
    constexpr int batchCount = 2;
    constexpr int m = 3;
    constexpr int n = 2;
    constexpr int lda = m;
    constexpr int ldb = m;
    float alpha = 1.0f;

    // Host data for batch 0: A0 (lower triangular, column-major)
    // A0 = [2  0  0]    B0 = [8  10]
    //      [1  3  0]        [5   7]
    //      [4  2  1]        [6   3]
    float hA0[lda * m] = {2.0f, 1.0f, 4.0f, 0.0f, 3.0f, 2.0f, 0.0f, 0.0f, 1.0f};
    float hB0[ldb * n] = {8.0f, 5.0f, 6.0f, 10.0f, 7.0f, 3.0f};

    // Host data for batch 1: A1 (lower triangular, column-major)
    // A1 = [1  0  0]    B1 = [1  2]
    //      [2  1  0]        [3  4]
    //      [3  2  1]        [5  6]
    float hA1[lda * m] = {1.0f, 2.0f, 3.0f, 0.0f, 1.0f, 2.0f, 0.0f, 0.0f, 1.0f};
    float hB1[ldb * n] = {1.0f, 3.0f, 5.0f, 2.0f, 4.0f, 6.0f};

    size_t aBytes = static_cast<size_t>(lda) * m * sizeof(float);
    size_t bBytes = static_cast<size_t>(ldb) * n * sizeof(float);
    size_t ptrArrayBytes = static_cast<size_t>(batchCount) * sizeof(float*);

    // Allocate Device memory for A0, A1, B0, B1
    void* rawA0 = nullptr;
    auto aclRet = aclrtMalloc(&rawA0, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dA0(rawA0);

    void* rawA1 = nullptr;
    aclRet = aclrtMalloc(&rawA1, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dA1(rawA1);

    void* rawB0 = nullptr;
    aclRet = aclrtMalloc(&rawB0, bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dB0(rawB0);

    void* rawB1 = nullptr;
    aclRet = aclrtMalloc(&rawB1, bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dB1(rawB1);

    // Copy host data to device
    aclRet = aclrtMemcpy(dA0.get(), aBytes, hA0, aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dA1.get(), aBytes, hA1, aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dB0.get(), bBytes, hB0, bBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dB1.get(), bBytes, hB1, bBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // Build pointer arrays on host, then copy to device
    float* hAPtrs[batchCount] = {
        static_cast<float*>(dA0.get()),
        static_cast<float*>(dA1.get())
    };
    float* hBPtrs[batchCount] = {
        static_cast<float*>(dB0.get()),
        static_cast<float*>(dB1.get())
    };

    void* rawAPtrArray = nullptr;
    aclRet = aclrtMalloc(&rawAPtrArray, ptrArrayBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dAPtrArray(rawAPtrArray);

    void* rawBPtrArray = nullptr;
    aclRet = aclrtMalloc(&rawBPtrArray, ptrArrayBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dBPtrArray(rawBPtrArray);

    aclRet = aclrtMemcpy(dAPtrArray.get(), ptrArrayBytes, hAPtrs, ptrArrayBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dBPtrArray.get(), ptrArrayBytes, hBPtrs, ptrArrayBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // Create ops-blas handle
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasHandle_t>::type, AclblasHandleDeleter> handle(rawHandle);

    blasRet = aclblasSetStream(handle.get(), ctx.Stream());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    // Call aclblasStrsmBatched: solve op(A[i]) * X[i] = alpha * B[i]
    blasRet = aclblasStrsmBatched(
        handle.get(),
        ACLBLAS_SIDE_LEFT,
        ACLBLAS_LOWER,
        ACLBLAS_OP_N,
        ACLBLAS_NON_UNIT,
        m, n, &alpha,
        static_cast<const float* const*>(dAPtrArray.get()), lda,
        static_cast<float* const*>(dBPtrArray.get()), ldb,
        batchCount);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    aclRet = aclrtSynchronizeStream(ctx.Stream());
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // Copy results back to host
    float hResult0[ldb * n] = {};
    float hResult1[ldb * n] = {};
    aclRet = aclrtMemcpy(hResult0, bBytes, dB0.get(), bBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(hResult1, bBytes, dB1.get(), bBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    // Print results
    printf("Batch 0 result:\n");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("  X[%d,%d] = %f\n", i, j, hResult0[j * ldb + i]);
        }
    }
    printf("Batch 1 result:\n");
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            printf("  X[%d,%d] = %f\n", i, j, hResult1[j * ldb + i]);
        }
    }

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasStrsmBatchedTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}
```

## aclblasCtrsmBatched（complex64）

#### 产品支持情况

| 产品 | 支持情况 |
| --- | --- |
| Atlas 950PR（Ascend 950PR） | 支持 |

已在 Atlas 950PR / CANN 9.1.0 上进行实机验证。

公共声明位于 `include/cann_ops_blas.h`。参数顺序与实数同族一致，alpha/A/B 类型改为
`aclblasComplex`（float32 real、float32 imag）。N/T/C 分别表示原矩阵、转置、共轭转置。
A/B 是设备侧指针数组，alpha 是 Host 标量；统一 m/n/lda/ldb/batchCount，列主序。
LEFT 求解 op(A)X=alpha B；RIGHT 求解 X op(A)=alpha B，B 原地覆写。

仅访问指定三角，UNIT 不读对角；alpha=0 不读 A 和 B 的旧值，只清零逻辑 B 区域。
B 的 padding 保留，各 batch 的 B 不可重叠。无奇异性检测，不支持设备 alpha。
非法枚举、负维度、batchCount<1、错误前导维、空 alpha/B 返回 INVALID_VALUE；
空 handle 返回 HANDLE_IS_NULLPTR。先验证元数据和 alpha/B，再处理 m/n=0 no-op；
空矩阵允许未使用的 lda/ldb 为 0，非空矩阵才要求满足相应尺寸下界。
非空维度且 alpha 非零要求非空 A 指针数组。B[i] 及实际需要读取的 A 元素须位于有效设备内存；alpha=0 和 UNIT 隐含对角对应的数据不被使用。

实现位于 `arch35/ctrsmbatched_*`，依赖 asc-devkit >= 9.1，与实数文件共存。
实现按职责维护六个文件：Host（`ctrsmbatched_host.cpp`）、小矩阵与有序求解（`ctrsmbatched_kernel.cpp`）、UNIT（`ctrsmbatched_unit_kernel.cpp`）、Cube（`ctrsmbatched_cube_kernel.cpp`）、方阵流水（`ctrsmbatched_square_kernel.cpp`），以及共享参数和基础复数运算（`ctrsmbatched_kernel.h`）。
设备入口源文件由仓库现有规则自动收集，不生成包装源码、不排除单个源文件，也不设置算子专用编译选项。UNIT 与方阵入口通过 `CTRSM_BATCHED_SHARED_ONLY` 复用主入口源文件中的私有模板，避免重复实例化主入口，并保持各入口的 UB 分配独立。源文件有效行数均低于 2000，共享头文件低于 500。

小矩阵采用 SIMT 搬运、分组 SIMD 求解和 DMA 写回；8/16/32 阶方阵在
LEFT / N / NON_UNIT / alpha=1 时使用流水搬运。较大 NON_UNIT 使用对角块预求逆与
Cube 更新，workspace 不足时使用 AIV 原地求解。NON_UNIT 对 A/B 同时施加行缩放，
减少极端对角引起的倒数溢出和下溢。

UNIT 使用与 OpenBLAS 对齐的向量分块顺序；较大三角矩阵先求解至多 96 阶的前缀，
仅在实部和虚部均已传播为 NaN 时填充后续结果，否则从未改写的输入执行完整有序求解。
初始 alpha 缩放保留四次独立乘法舍入，分块更新按参考顺序使用 FMA。
短矩阵 LEFT UNIT 的 8 行微块由相邻两行共享已求解 B 的加载，并将宽 RHS 分成
128 列一组计算；每行保留独立累加器和原有 FMA 顺序。
面板、微块和行调度在标量层执行，每个向量计算范围仅处理一行或一对相邻行，避免长跳转编译选项；UNIT 的一阶矩阵走不读取 A 元素的有序路径。
UNIT 不进入 Cube 管线，不分配逆矩阵增长标志。

输入和累加结果保存为 FLOAT32。k<128 的 Cube 对角求解关闭 HF32；其余 Cube
求解和更新仅在 abs(alpha.real)+abs(alpha.imag)≤1 时启用 HF32。
Cube workspace 为 `8*batchCount*kp*(2*kp+rp)` 字节，kp 将三角阶数补齐到 8，
rp 将 RHS 数补齐到 16。复用 handle workspace；首次扩容可能同步 stream，足够时异步排队。
用户 workspace 不足、超过限制或分配失败时，在修改 B 前回退到向量路径。
用户 workspace 保持原有地址与所有权，其他运行时错误返回调用方。
Cube 输入按矩阵存储方向选择 DMA 打包分块，转置时采用四行交错的线程排列和
32 字节输入填充。独立输出 tile 使用两个 L1 缓冲区重叠下一 tile 的加载；
L0 和 Fixpipe 保持串行，事件保证缓冲区读取完成后才可复用。

当前参考环境为 OpenBLAS 0.3.25 / Cooperlake / SINGLE_THREADED；构建和参考库配置见下方测试说明。
精度阈值、原始 1000 条精度与 200 条性能用例均保持不变。
异步执行错误由调用方在 stream 同步时检查。

[精度与性能测试说明](../../test/trsmbatched/ctrsmbatched/README.md)。

## Atlas A2/A3（arch22）实现与自测

实现位于 `arch22/`，使用 Ascend C kernel 直调，共用上述公开接口。
所有 kernel 提交至 handle 绑定的 stream；Host 不读取矩阵内容，除 alpha 标量与 A/B 指针数组在 Device 内存时需同步读取校验外，不在正常调用中同步设备。

在安装了 Netlib BLAS、LAPACK、GTest 的 Ascend Linux 环境，从仓库根目录执行：

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --soc=ascend910b --ops=strsmbatched
# 默认冒烟：前 50 条 CSV 用例及 3 项独立接口检查
build/test/trsmbatched/strsmbatched/strsmbatched_test
# 完整验收：显式启用全量 CSV 及补充回归，再按精度/性能筛选
GSB_FULL_TEST=1 build/test/trsmbatched/strsmbatched/strsmbatched_test --gtest_filter='*-*TC_PF*'
GSB_FULL_TEST=1 build/test/trsmbatched/strsmbatched/strsmbatched_test --gtest_filter='*TC_PF*'
```

A3 使用与实际设备匹配的 `--soc=ascend910_9382` 等参数重新编译并复验。测试支持通过
`ASCEND_DEVICE_ID` 环境变量覆盖设备编号。

`test/trsmbatched/strsmbatched/arch22/` 保留全量 `strsmbatched_test.csv`（1000 精度、200 性能）。
默认仅加载 `strsmbatched_test_smoke.csv`，其内容为精选的 51 条用例，供流水线快速冒烟使用；
设置 `GSB_FULL_TEST=1` 时加载全量任务 CSV。两种模式均保留独立接口检查；冒烟通过不能替代全量精度和性能验收。
工作区按 `batchCount * (align16(m) * align16(n) + 8192) * sizeof(float)` 计算，受 handle 2 GiB 上限约束，超限返回 `ACLBLAS_STATUS_ALLOC_FAILED`，超大规格组合建议按 batch 分批调用。

