# Syr2算子

## 算子概述

syr2 (Symmetric Rank-2 Update) 实现对称秩-2更新操作。该算子将两个向量的外积组合加到对称矩阵的指定三角区域。

数学表达式：

```
A = alpha * x * y^T + alpha * y * x^T + A
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSsyr2 | 单精度对称秩-2更新 |
| aclblasCsyr2 | 单精度复数对称秩-2更新，不共轭 |

## 算子执行接口

### aclblasSsyr2

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSsyr2(aclblasHandle_t handle, aclblasFillMode_t uplo, int n, const float *alpha, const float *x, int incx, const float *y, int incy, float *A, int lda)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| uplo | 输入 | aclblasFillMode_t | 指定矩阵 A 的存储格式。ACLBLAS_LOWER(122): 下三角，ACLBLAS_UPPER(121): 上三角，Host 内存 |
| n | 输入 | int | 向量 x 和 y 中的元素个数，矩阵 A 的行列数。n >= 0，Host 内存 |
| alpha | 输入 | const float*（FP32） | 标量 alpha 指针，向量乘积缩放因子，Host 内存 |
| x | 输入 | const float*（FP32） | 输入向量，对应公式中的 x。数据类型支持 FLOAT32，数据格式支持 ND，shape 为 [n]，Device 内存 |
| incx | 输入 | int | x 相邻元素间的内存地址偏移量，incx != 0，Host 内存 |
| y | 输入 | const float*（FP32） | 输入向量，对应公式中的 y。数据类型支持 FLOAT32，数据格式支持 ND，shape 为 [n]，Device 内存 |
| incy | 输入 | int | y 相邻元素间的内存地址偏移量，incy != 0，Host 内存 |
| A | 输入/输出 | float*（FP32） | 输入/输出矩阵，对应公式中的 A。数据类型支持 FLOAT32，数据格式支持 ND，shape 为 [n, n]，Device 内存 |
| lda | 输入 | int | 矩阵 A 的每列元素的存储步长，lda >= max(1, n)，Host 内存 |

#### 约束说明

- n >= 0，n==0 时直接返回成功
- incx != 0，incy != 0
- lda >= max(1, n)
- 算子输入 shape 为 [n]、[n]、[n, n]，输出 shape 为 [n, n]
- 算子实际计算时，不支持 ND 高维度运算（不支持维度 >= 3 的运算）
- Host 侧不做流同步，调用方需自行管理同步

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
int aclblasSsyr2Test(AclContext& ctx)
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
    constexpr int incy = 1;
    constexpr int lda = n;
    constexpr size_t vSize = n * sizeof(float);
    constexpr size_t aSize = n * n * sizeof(float);

    std::vector<float> hX = {1.0f, 2.0f};
    std::vector<float> hY = {3.0f, 4.0f};
    std::vector<float> hA(n * n, 0.0f);
    float alpha = 1.0f;

    void* rawX = nullptr;
    aclError aclRet;
    aclRet = aclrtMalloc(&rawX, vSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for x failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dXPtr(rawX, aclrtFree);

    void* rawY = nullptr;
    aclRet = aclrtMalloc(&rawY, vSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for y failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dYPtr(rawY, aclrtFree);

    void* rawA = nullptr;
    aclRet = aclrtMalloc(&rawA, aSize, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> dAPtr(rawA, aclrtFree);

    aclRet = aclrtMemcpy(dXPtr.get(), vSize, hX.data(), vSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for x failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(dYPtr.get(), vSize, hY.data(), vSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for y failed. ERROR: %d\n", aclRet); return aclRet);

    aclRet = aclrtMemcpy(dAPtr.get(), aSize, hA.data(), aSize, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A failed. ERROR: %d\n", aclRet); return aclRet);

    blasRet = aclblasSsyr2(
        static_cast<aclblasHandle_t>(handlePtr.get()), ACLBLAS_LOWER, n, &alpha,
        static_cast<const float*>(dXPtr.get()), incx, static_cast<const float*>(dYPtr.get()), incy,
        static_cast<float*>(dAPtr.get()), lda);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSsyr2 failed. ERROR: %d\n", blasRet);
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

    LOG_PRINT("aclblasSsyr2 test passed\n");
    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSsyr2Test(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSsyr2Test failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

### aclblasCsyr2

#### 产品支持情况

| 产品 | 支持情况 |
| --- | --- |
| Ascend 950PR | 支持 |
| Atlas A3 训练系列产品 / Atlas A3 推理系列产品 | 支持 |
| Atlas A2 训练系列产品 / Atlas A2 推理系列产品 | 支持 |

#### 函数原型

```cpp
aclblasStatus_t aclblasCsyr2(
    aclblasHandle_t handle, aclblasFillMode_t uplo, int n,
    const aclblasComplex* alpha, const aclblasComplex* x, int incx,
    const aclblasComplex* y, int incy, aclblasComplex* A, int lda);
```

接口声明位于 `include/cann_ops_blas.h`。Atlas A2/A3 实现位于 `blas/syr2/arch22/`，Ascend 950PR 实现位于 `blas/syr2/arch35/`。

#### 参数与约束

| 参数 | 说明 |
| --- | --- |
| handle | 有效库句柄，使用 `aclblasSetStream` 绑定执行 stream。 |
| uplo | `ACLBLAS_UPPER` 或 `ACLBLAS_LOWER`，仅更新对应三角区域。 |
| n | 矩阵阶数，`n >= 0`。 |
| alpha | Host 侧 complex64 标量指针。 |
| x / incx | Device 只读向量，`incx != 0`。 |
| y / incy | Device 只读向量，`incy != 0`。 |
| A / lda | 列主序 Device 输入输出矩阵，`lda >= max(1, n)`。 |

计算公式为 `A += alpha*x*y^T + alpha*y*x^T`。本接口为普通转置，不对 x 或 y 做共轭；只读写 `uplo` 指定的三角区域，另一三角及 `lda` padding 保持不变。`n=0` 或 `alpha=(0,0)` 时返回成功且不访问 x、y、A。

Host 侧不对 stream 同步。Ascend 950PR 上，连续布局 `incx=incy=1` 且 `1 <= n <= 4096` 使用 AIV 优化路径，其他合法布局使用通用 SIMT 路径。

#### 调用示例

下面示例在 Ascend 950PR 上调用 `aclblasCsyr2`，更新列主序复数矩阵的上三角。`alpha` 位于 Host，x、y 和 A 位于 Device；调用后同步绑定的 stream 再读回结果。完整源码见 [example_csyr2.cpp](../../test/syr2/csyr2/examples/example_csyr2.cpp)。

```cpp
#include <cstdio>
#include <cstring>
#include <initializer_list>

#include "acl/acl.h"
#include "cann_ops_blas.h"

namespace {
struct Resources {
    bool initialized = false;
    bool deviceSet = false;
    aclblasHandle_t handle = nullptr;
    aclrtStream stream = nullptr;
    void* x = nullptr;
    void* y = nullptr;
    void* a = nullptr;
    ~Resources()
    {
        if (stream != nullptr) {
            aclrtSynchronizeStream(stream);
        }
        for (void* allocation : {x, y, a}) {
            if (allocation != nullptr) {
                aclrtFree(allocation);
            }
        }
        if (handle != nullptr) {
            aclblasDestroy(handle);
        }
        if (stream != nullptr) {
            aclrtDestroyStream(stream);
        }
        if (deviceSet) {
            aclrtResetDevice(0);
        }
        if (initialized) {
            aclFinalize();
        }
    }
};
} // namespace

#define CHECK(call, success) do { \
    const auto status = (call); \
    if (status != (success)) { \
        std::fprintf(stderr, "%s failed: %d\n", #call, static_cast<int>(status)); \
        return 1; \
    } \
} while (false)

int main()
{
    Resources r;
    CHECK(aclInit(nullptr), ACL_SUCCESS);
    r.initialized = true;
    CHECK(aclrtSetDevice(0), ACL_SUCCESS);
    r.deviceSet = true;
    CHECK(aclblasCreate(&r.handle), ACLBLAS_STATUS_SUCCESS);
    CHECK(aclrtCreateStream(&r.stream), ACL_SUCCESS);
    CHECK(aclblasSetStream(r.handle, r.stream), ACLBLAS_STATUS_SUCCESS);

    const aclblasComplex alpha{1.0f, 1.0f};
    const aclblasComplex x[]{{1.0f, 2.0f}, {-1.0f, 1.0f}};
    const aclblasComplex y[]{{2.0f, -1.0f}, {3.0f, 2.0f}};
    // Column-major storage, n=lda=2. Only UPPER is updated.
    aclblasComplex a[]{{1.0f, 1.0f}, {2.0f, 3.0f}, {2.0f, 3.0f}, {-1.0f, 4.0f}};
    const aclblasComplex expected[]{{3.0f, 15.0f}, {2.0f, 3.0f}, {-11.0f, 12.0f}, {-13.0f, -4.0f}};
    CHECK(aclrtMalloc(&r.x, sizeof(x), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    CHECK(aclrtMalloc(&r.y, sizeof(y), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    CHECK(aclrtMalloc(&r.a, sizeof(a), ACL_MEM_MALLOC_HUGE_FIRST), ACL_SUCCESS);
    CHECK(aclrtMemcpy(r.x, sizeof(x), x, sizeof(x), ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    CHECK(aclrtMemcpy(r.y, sizeof(y), y, sizeof(y), ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    CHECK(aclrtMemcpy(r.a, sizeof(a), a, sizeof(a), ACL_MEMCPY_HOST_TO_DEVICE), ACL_SUCCESS);
    CHECK(aclblasCsyr2(r.handle, ACLBLAS_UPPER, 2, &alpha,
        static_cast<const aclblasComplex*>(r.x), 1, static_cast<const aclblasComplex*>(r.y), 1,
        static_cast<aclblasComplex*>(r.a), 2), ACLBLAS_STATUS_SUCCESS);
    CHECK(aclrtSynchronizeStream(r.stream), ACL_SUCCESS);
    CHECK(aclrtMemcpy(a, sizeof(a), r.a, sizeof(a), ACL_MEMCPY_DEVICE_TO_HOST), ACL_SUCCESS);
    for (int index = 0; index < 4; ++index) {
        std::printf("A[%d,%d] = (%g,%g)\n", index % 2, index / 2, a[index].real, a[index].imag);
    }
    CHECK(std::memcmp(a, expected, sizeof(a)), 0);
    std::puts("PASS: complex symmetric UPPER update; lower element preserved");
    return 0;
}
```

在已配置 CANN 环境的 ops-blas 根目录编译并运行：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cmake -S . -B build-release -DSOC_VERSION=ascend950 \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$ASCEND_HOME_PATH/lib64/cmake"
cmake --build build-release --target ops_blas -j8
g++ -std=c++17 test/syr2/csyr2/examples/example_csyr2.cpp \
  -Iinclude -I"$ASCEND_HOME_PATH/include" -Lbuild-release \
  -L"$ASCEND_HOME_PATH/lib64" -Wl,-rpath,"$PWD/build-release" \
  -lops_blas -lascendcl -o build-release/example_csyr2
./build-release/example_csyr2
```

CANN 安装路径按实际环境调整。示例输出如下，未选中的下三角元素 `A[1,0]` 保持不变：

```text
A[0,0] = (3,15)
A[1,0] = (2,3)
A[0,1] = (-11,12)
A[1,1] = (-13,-4)
PASS: complex symmetric UPPER update; lower element preserved
```
