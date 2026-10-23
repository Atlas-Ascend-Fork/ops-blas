# Sgbmv算子（aclblasSgbmv）

## 算子概述

sgbmv (General Banded Matrix-Vector Multiplication) 计算带状矩阵与向量的乘加（BLAS-2，带状存储 Banded Storage）。A 为 m×n 带状实数矩阵（kl 条下对角线 + ku 条上对角线），列主序带状存储于 lda×n 一维数组：`A(i,j)`（0 基，max(0,j-ku) ≤ i ≤ min(m-1,j+kl)）位于 `A[j*lda + (ku+i-j)]`（1 基即 `A(ku+1+i-j, j)`），带外区域不引用（由带状结构隐含为 0）。向量 y 入口时为旧值，出口时按 `y = alpha*op(A)*x + beta*y` 原地覆写。支持转置与共轭转置（实数下 OP_C 与 OP_T 语义等价），支持负步长（incx/incy < 0 时向量反向存储）。

数学表达式：

```
y = alpha * op(A) * x + beta * y
```

其中 op(A) 由 trans 决定：ACLBLAS_OP_N 时 op(A) = A；ACLBLAS_OP_T 时 op(A) = A^T；ACLBLAS_OP_C 时 op(A) = A^H（实数矩阵共轭为自身，OP_C 与 OP_T 语义等价，host 侧折叠为 T 路径）。x 逻辑长度 trans=N 时为 n、T/C 时为 m；y 逻辑长度 trans=N 时为 m、T/C 时为 n。

本接口与 cuBLAS `cublasSgbmv` 参数序列一一对应（handle 及参数顺序逐参数对应），语义参考 Netlib `sgbmv.f`。**带状存储行数为 kl+ku+1（lda ≥ kl+ku+1），与 LAPACK 部分带状例程的 2*kl+ku+1 工作数组布局不同，勿混淆。**

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSgbmv | 单精度实数带状矩阵-向量乘加（原地覆写 y） |
| aclblasCgbmv | 单精度复数带状矩阵-向量乘法 |

## 算子执行接口

### aclblasSgbmv

#### 产品支持情况

| 产品 | 是否支持 |
|------|:---:|
| Ascend 950PR / Ascend 950DT | 支持（arch35 实现） |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | 支持（arch22 实现） |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | 支持（arch22 实现） |
| Atlas 200I/500 A2 推理产品 | 不支持 |
| Atlas 推理系列产品 | 不支持 |
| Atlas 训练系列产品 | 不支持 |

- Atlas A2/A3 系列产品由 `blas/gbmv/arch22/` 实现支持，精度/性能验证环境为 Atlas 800I A2（Ascend 910B3）+ CANN 9.1.0。
- 接口声明位于 `include/cann_ops_blas.h`，供其他产品线共用。

#### 函数原型

```cpp
aclblasStatus_t aclblasSgbmv(
    aclblasHandle_t    handle,
    aclblasOperation_t trans,
    int                m,
    int                n,
    int                kl,
    int                ku,
    const float        *alpha,
    const float        *A,
    int                lda,
    const float        *x,
    int                incx,
    const float        *beta,
    float              *y,
    int                incy)
```

> 本接口为单段式（ops-blas handle 式 BLAS 直调，kernel 直调工程模式，非 aclnn 两段式）：入参校验与 kernel 下发在同一次调用内完成；执行 stream 通过 `aclblasSetStream` 绑定到 handle。

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带执行 stream（经 `aclblasSetStream` 绑定），Host 内存；须为 `aclblasCreate` 创建的有效句柄，为 nullptr 时返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR |
| trans | 输入 | aclblasOperation_t | 矩阵 A 的操作类型：ACLBLAS_OP_N（不转置）、ACLBLAS_OP_T（转置）、ACLBLAS_OP_C（共轭转置，实数下同 T），Host 内存；取值不在枚举集合内返回 ACLBLAS_STATUS_INVALID_ENUM |
| m | 输入 | int | 矩阵 A 的行数，m ≥ 0；m=0 时为合法 no-op，Host 内存 |
| n | 输入 | int | 矩阵 A 的列数，n ≥ 0；n=0 时为合法 no-op，Host 内存 |
| kl | 输入 | int | A 的下带宽（主对角线以下的非零对角线数），kl ≥ 0；实现加严：m>0 时 kl < m（见约束说明，与 cuBLAS 的已知行为差异），Host 内存 |
| ku | 输入 | int | A 的上带宽（主对角线以上的非零对角线数），ku ≥ 0；实现加严：n>0 时 ku < n，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量乘数，Host 内存；不可为 nullptr（即使 alpha=0）；alpha=0 时不引用 A 与 x |
| A | 输入 | const float*（FP32） | 带状矩阵，维度为 lda×n（存带状部分），列主序，Device 内存，只读；带外槽位不引用 |
| lda | 输入 | int | 带状矩阵 A 的存储行数（前导维度），lda ≥ kl+ku+1，Host 内存 |
| x | 输入 | const float*（FP32） | 输入向量，Device 内存，只读；逻辑长度 trans=N 时为 n、trans=T/C 时为 m，物理长度 ≥ 1+(len-1)\|incx\| |
| incx | 输入 | int | x 中连续元素之间的步长，incx ≠ 0（可正可负，负值时 x 反向存储），Host 内存 |
| beta | 输入 | const float*（FP32） | 标量乘数，Host 内存；不可为 nullptr；beta=0 时 y 不必是有效输入（结果 y 被覆写） |
| y | 输入/输出 | float*（FP32） | 输出向量（读入旧值、原地覆写新值），Device 内存；逻辑长度 trans=N 时为 m、trans=T/C 时为 n，物理长度 ≥ 1+(len-1)\|incy\| |
| incy | 输入 | int | y 中连续元素之间的步长，incy ≠ 0（可正可负，负值时 y 反向存储），Host 内存 |

#### 约束说明

- handle 必须为 `aclblasCreate` 创建的有效句柄，且经 `aclblasSetStream` 绑定 stream；读回 Device 结果前须同步 stream（`aclrtSynchronizeStream`，异步执行）。
- m ≥ 0、n ≥ 0、kl ≥ 0、ku ≥ 0；**lda ≥ kl+ku+1**（Netlib `sgbmv` / cuBLAS `cublasSgbmv` 一致口径）。
- **与 cuBLAS 的已知行为差异（保守兼容披露）**：m>0 时要求 kl<m、n>0 时要求 ku<n，违反返回 `ACLBLAS_STATUS_INVALID_VALUE`；cuBLAS/Netlib 对 kl≥m/ku≥n 的退化带状输入照常计算。该校验为本实现的保守加严，与 cuBLAS/Netlib 存在已知行为差异。
- incx ≠ 0、incy ≠ 0，支持负步长（x/y 可为反向向量，Netlib 语义反向遍历）。
- alpha、beta 不可为 nullptr（即使取值为 0 也须传有效 Host 指针）。
- m=0 或 n=0 为合法 no-op 直接返回成功（不引用 A、x、y，此时三者允许为 nullptr——A/x/y 判空在该分支不生效）。
- **alpha=0 特例（不引用 A 与 x，A/x 允许为 nullptr）**：beta=1 时 y 逐位不变（host 侧 quick return，位精确）；beta=0 时 y 各元素置 +0.0（位精确）；其余 beta 值 y 逐元素缩放（y = beta·y，单次 fp32 乘位精确）。
- beta=0 时 y 不必是有效输入（可用 NaN 初始化 y）；alpha=0 时 A、x 不被引用。
- y 为原地覆写，不返回视图；不支持超出 incx/incy/lda 步长语义的非连续内存访问；无 broadcast；确定性计算不做要求。
- 数据类型仅支持 FLOAT32（单精度实数）；非零形状能力口径：m/n/kl/ku 为 int 运行时入参（接口域 m,n ≤ 2^31-1），A 元素偏移全程 uint64 核算；测试工程覆盖口径 m,n ≤ 65536、lda·n ≤ 2^30（1 073 741 824，建模软约束，非算子能力上限）。
- 精度：FLOAT32 按生态算子开源精度标准——rtol = 2^-10 (9.77e-4)、atol = 2^-16 (1.53e-5)、required_matched_ratio = 0.99、逐元素 max_abs_error_limit = 1e-2 或 32×ULP；alpha=0 计算路径与 quick return 位精确（EXACT）；golden 由 cblas（Netlib `sgbmv`）生成，输出 y 全向量逐元素验证。全量自测 **1201/1201 通过**（仓内 1200 条用例全量 + handle-null 负向；最低 matchedRatio 0.99805、最大 maxAbsErr 3.91e-03）。
- **性能口径为 kernel-only**（msprof plain 模式 op_summary Task Duration，warmup 10 + 有效采样 60、每 case 3 轮取中位）；测量环境 Atlas 800I A2（910B3）+ CANN 9.1.0，四标杆（1024²/16、2048²/32、T 1024²/16、4096²/64）实测 **4.900 / 7.150 / 4.559 / 12.350 µs**（标杆 12.53 / 15.54 / 10.58 / 21.98 µs，余量 1.78–2.56×）。

#### 错误码

| 返回值 | 说明 |
|--------|------|
| ACLBLAS_STATUS_SUCCESS (0) | 接口调用成功（含 m=0 / n=0 合法 no-op，及 alpha=0 且 beta=1 时 y 位精确不变的 quick return） |
| ACLBLAS_STATUS_INVALID_VALUE (3) | m/n/kl/ku < 0；m>0 时 kl ≥ m 或 n>0 时 ku ≥ n（加严校验）；lda < kl+ku+1；incx = 0 或 incy = 0；alpha/beta 为 nullptr；m>0 且 n>0 时 y 为 nullptr；m>0 且 n>0 且 alpha≠0 时 A 或 x 为 nullptr |
| ACLBLAS_STATUS_HANDLE_IS_NULLPTR (9) | handle 为空指针 |
| ACLBLAS_STATUS_INVALID_ENUM (10) | trans 取值不在 {ACLBLAS_OP_N, ACLBLAS_OP_T, ACLBLAS_OP_C} 内 |

- 校验顺序：handle → trans（枚举）→ dims（m/n/kl/ku/lda，含加严）→ scalars（incx/incy/alpha/beta）→ arrays（A/x/y 条件判空）→ quick return 判定 → kernel 下发。
- 资源类异常（如库内部内存分配失败）可能返回 ACLBLAS_STATUS_ALLOC_FAILED 等其他状态码，语义见 `include/cann_ops_blas_common.h`。

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
#include "acl/acl.h"
#include "cann_ops_blas.h"

// ... aclInit / aclrtSetDevice / aclrtCreateStream 之后：
aclblasHandle_t handle = nullptr;
aclblasCreate(&handle);                 // 创建 ops-blas 句柄
aclblasSetStream(handle, stream);       // 绑定执行 stream

// A（m=n=3、kl=ku=1 三对角）带状存储：lda=kl+ku+1=3，A(i,j) 位于 A[j*lda+(ku+i-j)]（0 基）
//   A = | 1 2 . |        hA（列主序逐列堆叠，· 为带外槽位置 0）：
//       | 3 1 2 |          {0,1,3,  2,1,3,  2,1,0}
//       | . 3 1 |
float hA[9] = {0.f, 1.f, 3.f,  2.f, 1.f, 3.f,  2.f, 1.f, 0.f};
float hX[3] = {1.f, 1.f, 1.f};
float alpha = 1.0f;
float beta = 0.0f;                      // alpha/beta 为 Host 指针，不可为 nullptr
// ... aclrtMalloc + aclrtMemcpy 上板（dA/dX/dY）...
aclblasSgbmv(handle, ACLBLAS_OP_N, 3, 3, 1, 1, &alpha, dA, 3, dX, 1, &beta, dY, 1);
aclrtSynchronizeStream(stream);         // 读回结果前同步 stream
// ... aclrtMemcpy 回读 dY，期望 y = A·x = {3, 6, 4} ...
aclblasDestroy(handle);
```
### aclblasCgbmv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持（仅 arch35 实现）
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持（当前无 arch22 实现）
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持（当前无 arch22 实现）

- `aclblasCgbmv` 当前仅提供 arch35（Ascend 950PR / Ascend 950DT）实现，Atlas A2/A3 系列产品对应的算子库中不包含该接口。

#### 函数原型

```cpp
aclblasStatus_t aclblasCgbmv(aclblasHandle_t handle, aclblasOperation_t trans, int m, int n, int kl, int ku, const aclblasComplex *alpha, const aclblasComplex *A, int lda, const aclblasComplex *x, int incx, const aclblasComplex *beta, aclblasComplex *y, int incy)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| trans | 输入 | aclblasOperation_t | 矩阵操作类型：ACLBLAS_OP_N（op(A)=A）、ACLBLAS_OP_T（op(A)=A^T）、ACLBLAS_OP_C（op(A)=A^H，共轭转置），Host 内存 |
| m | 输入 | int | 矩阵 A 的行数，Host 内存 |
| n | 输入 | int | 矩阵 A 的列数，Host 内存 |
| kl | 输入 | int | A 的下带宽（主对角线以下的次对角线条数），Host 内存 |
| ku | 输入 | int | A 的上带宽（主对角线以上的次对角线条数），Host 内存 |
| alpha | 输入 | const aclblasComplex*（COMPLEX64） | 复数标量乘数，Host/Device 内存 |
| A | 输入 | const aclblasComplex*（COMPLEX64） | 列主序带状存储的矩阵，数组维度 lda x n，元素 A(i,j)（1-based）存于 A(ku+1+i-j, j)，Device 内存 |
| lda | 输入 | int | 带状矩阵 A 存储的主维长度，Host 内存 |
| x | 输入 | const aclblasComplex*（COMPLEX64） | 向量，trans=ACLBLAS_OP_N 时包含 n 个元素，否则包含 m 个元素，Device 内存 |
| incx | 输入 | int | x 中连续元素之间的步长，可为负，Host 内存 |
| beta | 输入 | const aclblasComplex*（COMPLEX64） | 复数标量乘数。beta 为 (0,0) 时不读取 y 的旧值，Host/Device 内存 |
| y | 输入/输出 | aclblasComplex*（COMPLEX64） | 向量，trans=ACLBLAS_OP_N 时包含 m 个元素，否则包含 n 个元素，原地覆写，Device 内存 |
| incy | 输入 | int | y 中连续元素之间的步长，可为负，Host 内存 |

#### 约束说明

- m >= 0，n >= 0；m == 0 或 n == 0 时为合法 no-op：trans、kl/ku 下界、lda、incx/incy、alpha/beta 非空等检查仍先执行，通过后返回 ACLBLAS_STATUS_SUCCESS，不访问 A、x、y。
- kl >= 0，ku >= 0，lda >= kl + ku + 1；m、n 均大于 0 时还要求 kl <= m - 1，ku <= n - 1。
- incx != 0，incy != 0；负步长按 BLAS 语义从物理存储的高地址端开始遍历，调用方传入物理存储的起始地址。
- alpha、beta 不可为 nullptr，可各自位于 Host 或 Device 内存，由算子内部通过 aclrtPointerGetAttributes 自动判断；位于 Device 时由 kernel 直接读取，不引入同步。
- m、n 均大于 0 时 A、x、y 不可为 nullptr（y 为输出，beta 为 (0,0) 时同样要求非空）。
- 非法参数返回 ACLBLAS_STATUS_INVALID_VALUE（含 alpha/beta 指针位置查询失败），trans 取值非法同样返回 ACLBLAS_STATUS_INVALID_VALUE（与 aclblasSgbmv 一致），handle 为 nullptr 返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR，获取 AI Vector 核数失败返回 ACLBLAS_STATUS_EXECUTION_FAILED。
- alpha == (0,0) 且 beta == (1,0) 时直接返回；alpha == (0,0) 时仅计算 y = beta * y，不读取 A、x。
- 只访问带内元素：带外三角区与 lda 填充行不被读取，y 的步长间隙保持不变。
- x、y 不得与 A 重叠。函数在 handle 绑定的 stream 上异步执行，读取 y 前需同步该 stream。

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
#include <cstdio>
#include <memory>
#include <type_traits>
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
struct BlasHandleDeleter {
    void operator()(aclblasHandle_t h) const { aclblasDestroy(h); }
};

int aclblasCgbmvTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasHandle_t>::type, BlasHandleDeleter> handlePtr(rawHandle);

    blasRet = aclblasSetStream(handlePtr.get(), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 3 x 3 tridiagonal matrix (kl = ku = 1) in band storage, column by column:
    // column 0: [unused, A(0,0), A(1,0)], column 1: [A(0,1), A(1,1), A(2,1)], column 2: [A(1,2), A(2,2), unused]
    constexpr int m = 3;
    constexpr int n = 3;
    constexpr int kl = 1;
    constexpr int ku = 1;
    constexpr int lda = kl + ku + 1;
    constexpr int incx = 1;
    constexpr int incy = 1;
    constexpr size_t aSize = lda * n * sizeof(aclblasComplex);
    constexpr size_t xSize = n * sizeof(aclblasComplex);
    constexpr size_t ySize = m * sizeof(aclblasComplex);

    aclblasComplex alpha = {1.0f, 0.0f};
    aclblasComplex beta = {0.0f, 0.0f};
    std::vector<aclblasComplex> hA = {{0.0f, 0.0f}, {1.0f, 1.0f}, {2.0f, 0.0f},
                                      {3.0f, 0.0f}, {4.0f, -1.0f}, {5.0f, 0.0f},
                                      {6.0f, 0.0f}, {7.0f, 2.0f}, {0.0f, 0.0f}};
    std::vector<aclblasComplex> hX = {{1.0f, 0.0f}, {0.0f, 1.0f}, {1.0f, 1.0f}};
    std::vector<aclblasComplex> hY(m, {0.0f, 0.0f});

    void* rawA = nullptr;
    aclError aclRet = aclrtMalloc(&rawA, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, AclMemDeleter> dAPtr(rawA);

    void* rawX = nullptr;
    aclRet = aclrtMalloc(&rawX, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for x failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, AclMemDeleter> dXPtr(rawX);

    void* rawY = nullptr;
    aclRet = aclrtMalloc(&rawY, ySize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for y failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, AclMemDeleter> dYPtr(rawY);

    aclRet = aclrtMemcpy(dAPtr.get(), aSize, hA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dXPtr.get(), xSize, hX.data(), xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for x failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(dYPtr.get(), ySize, hY.data(), ySize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for y failed. ERROR: %d\n", aclRet); return aclRet);

    blasRet = aclblasCgbmv(
        handlePtr.get(), ACLBLAS_OP_N, m, n, kl, ku, &alpha,
        static_cast<const aclblasComplex*>(dAPtr.get()), lda, static_cast<const aclblasComplex*>(dXPtr.get()), incx,
        &beta, static_cast<aclblasComplex*>(dYPtr.get()), incy);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCgbmv failed. ERROR: %d\n", blasRet);
              return blasRet);

    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    std::vector<aclblasComplex> yResult(m);
    aclRet = aclrtMemcpy(yResult.data(), ySize, dYPtr.get(), ySize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy result failed. ERROR: %d\n", aclRet); return aclRet);
    // expected: y = (1, 4), (9, 10), (5, 14)
    for (int i = 0; i < m; i++) {
        LOG_PRINT("y[%d] = (%f, %f)\n", i, yResult[i].real, yResult[i].imag);
    }

    LOG_PRINT("aclblasCgbmv test passed\n");
    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasCgbmvTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasCgbmvTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```
