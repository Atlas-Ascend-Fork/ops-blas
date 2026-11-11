# Spmv算子

## 算子概述

Spmv（Symmetric Packed Matrix-Vector Multiplication）算子实现了对称压缩矩阵与向量的乘法运算。该算子针对对称矩阵的存储特性进行了优化，采用压缩存储格式以节省内存空间，并高效完成矩阵与向量的乘加运算。

数学表达式：

```
y = alpha * A * x + beta * y
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSspmv | 单精度浮点对称压缩矩阵-向量乘法 |
| aclblasChpmv | 单精度复数 Hermitian 压缩矩阵-向量乘法 |

## 算子执行接口

### aclblasSspmv

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSspmv(aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float* alpha, const float* AP, const float* x, int incx, const float* beta, float* y, int incy)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | ACLBLAS_UPPER(121) 或 ACLBLAS_LOWER(122)，Host 内存 |
| n | 输入 | int | 方阵阶数，n >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量 alpha 的指针，Host 内存 |
| AP | 输入 | const float*（FP32） | 对称压缩矩阵，共 n(n+1)/2 个元素，Device 内存 |
| x | 输入 | const float*（FP32） | 输入向量，n 个元素，Device 内存 |
| incx | 输入 | int | x 的步长，incx != 0（可正可负），Host 内存 |
| beta | 输入 | const float*（FP32） | 标量 beta 的指针，Host 内存 |
| y | 输入/输出 | float*（FP32） | 输入/输出向量，n 个元素，Device 内存 |
| incy | 输入 | int | y 的步长，incy != 0（可正可负），Host 内存 |

#### 约束说明

- n >= 0
- incx != 0, incy != 0

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](https://gitcode.com/cann/ops-blas/blob/master/docs/zh/develop/compile_and_run_example.md)。

```cpp
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

int aclblasSspmvTest(AclContext& ctx)
{
    constexpr int n = 3;
    constexpr int incx = 1;
    constexpr int incy = 1;
    constexpr size_t apSize = n * (n + 1) / 2 * sizeof(float);
    constexpr size_t xSize = n * sizeof(float);
    constexpr size_t ySize = n * sizeof(float);
    float alpha = 1.0f;
    float beta = 1.0f;

    float hAP[n * (n + 1) / 2] = {1.0f, 2.0f, 5.0f, 3.0f, 6.0f, 9.0f};
    float hX[n] = {1.0f, 2.0f, 3.0f};
    float hY[n] = {10.0f, 20.0f, 30.0f};

    void *rawAP = nullptr;
    auto aclRet = aclrtMalloc(&rawAP, apSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dAP(rawAP);

    void *rawX = nullptr;
    aclRet = aclrtMalloc(&rawX, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dX(rawX);

    void *rawY = nullptr;
    aclRet = aclrtMalloc(&rawY, ySize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dY(rawY);

    aclRet = aclrtMemcpy(dAP.get(), apSize, hAP, apSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dX.get(), xSize, hX, xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dY.get(), ySize, hY, ySize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);
    std::unique_ptr<void, AclblasHandleDeleter> handle(rawHandle);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handle.get()), ctx.Stream());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    blasRet = aclblasSspmv(
        static_cast<aclblasHandle_t>(handle.get()), ACLBLAS_UPPER, n, &alpha,
        static_cast<float*>(dAP.get()), static_cast<float*>(dX.get()), incx, &beta,
        static_cast<float*>(dY.get()), incy);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    aclRet = aclrtSynchronizeStream(ctx.Stream());
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclRet = aclrtMemcpy(hY, ySize, dY.get(), ySize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    return 0;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSspmvTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}
```

### aclblasChpmv

计算单精度复数 Hermitian 压缩矩阵与向量的乘加：`y = alpha * A * x + beta * y`。

#### 产品支持情况

| 产品 | 支持情况 |
|------|----------|
| Ascend 950PR | 支持 |
| Ascend 950DT | 支持 |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | 不支持 |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | 不支持 |

#### 函数原型

```cpp
aclblasStatus_t aclblasChpmv(aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const aclblasComplex* alpha, const aclblasComplex* AP, const aclblasComplex* x, int incx, const aclblasComplex* beta, aclblasComplex* y, int incy)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | ACLBLAS_UPPER 或 ACLBLAS_LOWER，指定压缩存储的三角，Host 内存 |
| n | 输入 | int | 方阵阶数，n >= 0，Host 内存 |
| alpha | 输入 | const aclblasComplex*（complex64） | 复数标量 alpha 的指针，Host 内存 |
| AP | 输入 | const aclblasComplex*（complex64） | 按列压缩存储的三角矩阵，共 n(n+1)/2 个元素，Device 内存，只读 |
| x | 输入 | const aclblasComplex*（complex64） | 输入向量，逻辑长度为 n，Device 内存，只读 |
| incx | 输入 | int | x 的非零步长，可正可负，Host 内存 |
| beta | 输入 | const aclblasComplex*（complex64） | 复数标量 beta 的指针，Host 内存 |
| y | 输入/输出 | aclblasComplex*（complex64） | 逻辑长度为 n 的向量，原地更新，Device 内存 |
| incy | 输入 | int | y 的非零步长，可正可负，Host 内存 |

#### 约束说明

- AP 按列连续存储 UPPER 或 LOWER 三角，没有前导维度参数；另一半矩阵由共轭对称确定，对角元素只使用实部。
- 向量存储长度至少为 `1+(n-1)*abs(inc)`（n > 0）；负步长时传入地址仍是分配起点。计算容量时应先将步长提升到 64 位再取绝对值。
- 参数检查依次为：空 handle、负 n、n=0、非法 uplo、零步长及空指针。空 handle 优先返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`；句柄有效时，负 n 返回 `ACLBLAS_STATUS_INVALID_VALUE`，n=0 直接返回成功。
- n > 0 时，非法 uplo 返回 `ACLBLAS_STATUS_INVALID_ENUM`；零 incx/incy 或 alpha/beta/AP/x/y 为空返回 `ACLBLAS_STATUS_INVALID_VALUE`。
- alpha=0 时仍要求上述指针非空；计算路径不读取 AP/x。beta=0 时不读取旧 y。
- 调用在 handle 绑定的 stream 上异步执行；alpha/beta 按值传入，Device 缓冲区须保持有效直到 stream 完成。AP、x 和 y 应使用互不重叠的存储区。

#### 自测

在 Ascend 950PR、CANN 9.1.0 环境下，使用 Netlib BLAS/CBLAS 3.10.0 作为数值参考。在仓库根目录执行，参考库路径替换为本机实际路径：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export CHPMV_NETLIB_CBLAS=/absolute/path/netlib-build/lib/libcblas.so.3.10.0
export CHPMV_NETLIB_BLAS=/absolute/path/netlib-build/lib/libblas.so.3.10.0
export BLAS_DEVICE_ID=0
export EAGER_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64"
cmake -S . -B build -DSOC_VERSION=ascend950 -DBUILD_TEST=ON -DTEST_NAMES=chpmv \
  -DREFBLAS_INCLUDE_DIR=/absolute/path/netlib-build/include \
  -DREFBLAS_LIB="$CHPMV_NETLIB_CBLAS" -DREFLAPACK_LIB:FILEPATH=
cmake --build build --target chpmv_test -j8
python3 test/spmv/chpmv/run_self_test.py --binary build/test/spmv/chpmv/chpmv_test --output ../chpmv_self_test
```

输出目录须尚不存在。`run_self_test.py` 检查原 CSV 的 1200 项（含 200 项性能形状）全部注册，并执行全部 2504 项固定、原始及补充测试；逐项核对 XML、实虚部完整比较记录、失败与跳过情况，输出 `summary.json`、`precision.jsonl` 和原始日志。性能形状在此仅验证精度，不代替正式性能计时。

内存检查使用独立的 Debug 构建，显式指定 `-O0`。当前 CANN 9.1.0 / mssanitizer 26.1.0 环境下，优化构建与插桩结合时会出现运行异常，不能将普通构建直接用于本项检查。检查产物使用同一份源码和参考库，仅供 mssanitizer 使用，不用于性能计时：

```bash
cmake -S . -B ../chpmv_memcheck_build -DSOC_VERSION=ascend950 -DBUILD_TEST=ON -DTEST_NAMES=chpmv \
  -DCMAKE_BUILD_TYPE=Debug -DCMAKE_ASC_FLAGS_DEBUG="-O0 -g" \
  -DCANN_3RD_LIB_PATH="$PWD/build/third_party" -DCMAKE_ASC_FLAGS="-g --cce-enable-sanitizer" \
  -DREFBLAS_INCLUDE_DIR=/absolute/path/netlib-build/include \
  -DREFBLAS_LIB="$CHPMV_NETLIB_CBLAS" -DREFLAPACK_LIB:FILEPATH=
cmake --build ../chpmv_memcheck_build --target chpmv_test -j8
python3 test/spmv/chpmv/run_memcheck.py --binary ../chpmv_memcheck_build/test/spmv/chpmv/chpmv_test --output ../chpmv_memcheck
```

`run_memcheck.py` 使用 `mssanitizer --tool=memcheck --leak-check=yes` 分进程检查契约与成功路径，覆盖 UPPER/LOWER、连续与负步长、SIMT/SIMD 分派边界和 UB 暂存复用。每组核对实际执行名单；成功路径还必须有预期数量的 `chpmv_kernel` 启动及对应的 `No error detected.` 记录。工具缺失、超时、诊断错误、漏测或缺少内核记录均返回非零状态，并记录在 `summary.json`，不能作为通过。

#### 调用示例

示例使用 RAII 管理资源。具体编译和运行方法见[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

```cpp
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

int aclblasChpmvTest(AclContext& ctx)
{
    constexpr int n = 2;
    constexpr int incx = 1;
    constexpr int incy = 1;
    constexpr size_t apSize = n * (n + 1) / 2 * sizeof(aclblasComplex);
    constexpr size_t xSize = n * sizeof(aclblasComplex);
    constexpr size_t ySize = n * sizeof(aclblasComplex);
    aclblasComplex alpha{1.0f, 0.0f};
    aclblasComplex beta{0.0f, 0.0f};

    aclblasComplex hAP[n * (n + 1) / 2] = {{2.0f, 0.0f}, {1.0f, 2.0f}, {3.0f, 0.0f}};
    aclblasComplex hX[n] = {{1.0f, 1.0f}, {2.0f, -1.0f}};
    aclblasComplex hY[n] = {};

    void *rawAP = nullptr;
    auto aclRet = aclrtMalloc(&rawAP, apSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dAP(rawAP);

    void *rawX = nullptr;
    aclRet = aclrtMalloc(&rawX, xSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dX(rawX);

    void *rawY = nullptr;
    aclRet = aclrtMalloc(&rawY, ySize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    std::unique_ptr<void, AclrtMemDeleter> dY(rawY);

    aclRet = aclrtMemcpy(dAP.get(), apSize, hAP, apSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dX.get(), xSize, hX, xSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);
    aclRet = aclrtMemcpy(dY.get(), ySize, hY, ySize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);
    std::unique_ptr<std::remove_pointer_t<aclblasHandle_t>, AclblasHandleDeleter> handle(rawHandle);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handle.get()), ctx.Stream());
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    blasRet = aclblasChpmv(
        static_cast<aclblasHandle_t>(handle.get()), ACLBLAS_UPPER, n, &alpha,
        static_cast<aclblasComplex*>(dAP.get()), static_cast<aclblasComplex*>(dX.get()), incx, &beta,
        static_cast<aclblasComplex*>(dY.get()), incy);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, return blasRet);

    aclRet = aclrtSynchronizeStream(ctx.Stream());
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    aclRet = aclrtMemcpy(hY, ySize, dY.get(), ySize, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, return aclRet);

    CHECK_RET(hY[0].real == 6.0f && hY[0].imag == 5.0f &&
              hY[1].real == 9.0f && hY[1].imag == -4.0f, return 1);
    return 0;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasChpmvTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, return ret);
    return 0;
}
```
