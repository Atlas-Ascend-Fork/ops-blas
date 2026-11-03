# Dgmm算子

## 算子概述

Dgmm（Diagonal Matrix-Matrix Multiplication）算子实现了对角矩阵与普通矩阵的乘法运算，使用一个向量构造对角矩阵，按行（LEFT）或按列（RIGHT）对输入矩阵进行缩放，核心运算为逐元素乘法。

数学表达式：

```
LEFT 模式：  C = diag(x) * A，  C[i,j] = x[i] * A[i,j]    （x 长度为 m，按行缩放）
RIGHT 模式： C = A * diag(x)，  C[i,j] = A[i,j] * x[j]    （x 长度为 n，按列缩放）
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSdgmm | 单精度实数对角矩阵乘法 |
| aclblasCdgmm | 单精度复数对角矩阵乘法，当前仅支持 LEFT 模式，RIGHT 模式暂未实现 |

### sdgmm 与 cdgmm 差异

两个接口均为对角矩阵乘法，但存在以下行为差异，使用时需注意：

| 差异项 | aclblasSdgmm | aclblasCdgmm |
|--------|--------------|--------------|
| 数据类型 | FP32 实数 | Complex32 复数 |
| 支持模式 | LEFT / RIGHT | 仅 LEFT（RIGHT 返回 ACLBLAS_STATUS_NOT_SUPPORTED） |
| incx == 0 | 允许，等价于标量乘法 C = x[0] * A | 拒绝，返回 ACLBLAS_STATUS_INVALID_VALUE |
| 存储布局 | 列主序（column-major） | 行主序（row-major），与 cuBLAS cublasCdgmm 的列主序语义存在差异 |
| 产品支持 | Ascend 950PR / Ascend 950DT | Atlas A2 / Atlas A3 系列 |

## 算子执行接口

### aclblasSdgmm

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

> Ascend 950PR/Ascend 950DT 上的 sdgmm 依赖 CANN asc-devkit >= 9.1（`ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR >= 1`），低于该版本时编译与运行将跳过此算子。

#### 函数原型

```cpp
aclblasStatus_t aclblasSdgmm(aclblasHandle_t handle, aclblasSideMode_t mode, int m, int n, const float *A, int lda, const float *x, int incx, float *C, int ldc)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| mode | 输入 | aclblasSideMode_t | 缩放模式：ACLBLAS_SIDE_LEFT（按行缩放，x 长度为 m）或 ACLBLAS_SIDE_RIGHT（按列缩放，x 长度为 n），Host 内存 |
| m | 输入 | int | 矩阵 A/C 的行数，m >= 0，Host 内存 |
| n | 输入 | int | 矩阵 A/C 的列数，n >= 0，Host 内存 |
| A | 输入 | const float*（FP32） | 输入矩阵，列主序存储，维度 m×n，Device 内存 |
| lda | 输入 | int | 矩阵 A 的主维（leading dimension），lda >= max(1, m)，Host 内存 |
| x | 输入 | const float*（FP32） | 对角向量，mode=LEFT 时长度为 m，mode=RIGHT 时长度为 n，Device 内存 |
| incx | 输入 | int | x 中相邻元素的步长，可为任意整数（包括 0，incx==0 时等价于标量乘法 C = x[0] * A），可为负数表示反向访问 x，Host 内存 |
| C | 输出 | float*（FP32） | 输出矩阵，列主序存储，维度 m×n，Device 内存 |
| ldc | 输入 | int | 矩阵 C 的主维（leading dimension），ldc >= max(1, m)，Host 内存 |

#### 约束说明

- handle 不能为 nullptr，否则返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR
- mode 必须为 ACLBLAS_SIDE_LEFT 或 ACLBLAS_SIDE_RIGHT，否则返回 ACLBLAS_STATUS_INVALID_ENUM
- m >= 0, n >= 0，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- m == 0 或 n == 0 时为 no-op，直接返回 ACLBLAS_STATUS_SUCCESS（跳过以下所有校验）
- 以下约束仅在 m > 0 且 n > 0 时生效：
- incx 可为任意整数（包括 0，incx==0 时等价于标量乘法 C = x[0] * A），可为负数表示反向访问 x
- lda >= max(1, m)，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- ldc >= max(1, m)，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- A、x、C 不能为 nullptr，否则返回 ACLBLAS_STATUS_INVALID_VALUE
- 原地执行（A==C）要求 lda == ldc，否则返回 ACLBLAS_STATUS_INVALID_VALUE

#### 精度验证

采用 `MIXED_TOLERANCE` 混合容差策略（对齐[生态算子开源精度标准](https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/experimental_standard.md) §2.1）：

| 数据类型 | rtol | atol | required_matched_ratio | max_abs_error_limit |
|----------|------|------|----------------------|-------------------|
| FLOAT32 | 2^-10 (9.77e-4) | 2^-16 (1.53e-5) | 0.99 | 1e-2 或 32·ULP |

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。

以下示例演示 LEFT 模式（C = diag(x) * A），取 m = 3, n = 3, x = [2, 3, 4]：

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

struct BlasHandleDeleter {
    void operator()(aclblasHandle_t h) const { aclblasDestroy(h); }
};

int aclblasSdgmmTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-blas 句柄
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasHandle_t>::type, BlasHandleDeleter> handlePtr(rawHandle);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 2. 准备 Host 数据
    // LEFT 模式：C = diag(x) * A，x 长度为 m
    int m = 3;
    int n = 3;
    int incx = 1;
    int lda = m;
    int ldc = m;
    // x = [2, 3, 4]
    std::vector<float> xHostData = {2.0f, 3.0f, 4.0f};
    // A (列主序，3x3):
    //   列0=[1,2,3], 列1=[4,5,6], 列2=[7,8,9]
    std::vector<float> aHostData = {1.0f, 2.0f, 3.0f,
                                    4.0f, 5.0f, 6.0f,
                                    7.0f, 8.0f, 9.0f};
    size_t xBytes = static_cast<size_t>(m) * sizeof(float);
    size_t aBytes = static_cast<size_t>(lda) * static_cast<size_t>(n) * sizeof(float);
    size_t cBytes = static_cast<size_t>(ldc) * static_cast<size_t>(n) * sizeof(float);

    // 3. 申请 Device 内存并拷贝数据
    void* xRawMem = nullptr;
    auto aclRet = aclrtMalloc(&xRawMem, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for x failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> xDevicePtr(xRawMem, aclrtFree);
    aclRet = aclrtMemcpy(xDevicePtr.get(), xBytes, xHostData.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for x failed. ERROR: %d\n", aclRet); return aclRet);

    void* aRawMem = nullptr;
    aclRet = aclrtMalloc(&aRawMem, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> aDevicePtr(aRawMem, aclrtFree);
    aclRet = aclrtMemcpy(aDevicePtr.get(), aBytes, aHostData.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A failed. ERROR: %d\n", aclRet); return aclRet);

    void* bRawMem = nullptr;
    aclRet = aclrtMalloc(&bRawMem, cBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for C failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<void, aclError (*)(void*)> bDevicePtr(bRawMem, aclrtFree);

    // 4. 调用 aclblasSdgmm（LEFT 模式）
    blasRet = aclblasSdgmm(
        static_cast<aclblasHandle_t>(handlePtr.get()), ACLBLAS_SIDE_LEFT, m, n,
        static_cast<const float*>(aDevicePtr.get()), lda,
        static_cast<const float*>(xDevicePtr.get()), incx,
        static_cast<float*>(bDevicePtr.get()), ldc);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSdgmm failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    // 预期 C = diag(x) * A（列主序）:
    //   列0=[2,6,12], 列1=[8,15,24], 列2=[14,24,36]
    std::vector<float> resultData(ldc * n, 0.0f);
    aclRet = aclrtMemcpy(resultData.data(), cBytes, bDevicePtr.get(), cBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", aclRet); return aclRet);
    for (int col = 0; col < n; col++) {
        for (int row = 0; row < m; row++) {
            LOG_PRINT("C[%d,%d] = %f\n", row, col, resultData[col * ldc + row]);
        }
    }

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSdgmmTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSdgmmTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

预期输出（C = diag(x) * A，列主序）：

```
C[0,0] = 2.000000
C[1,0] = 6.000000
C[2,0] = 12.000000
C[0,1] = 8.000000
C[1,1] = 15.000000
C[2,1] = 24.000000
C[0,2] = 14.000000
C[1,2] = 24.000000
C[2,2] = 36.000000
```

### aclblasCdgmm

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：不支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasCdgmm(aclblasHandle_t handle, aclblasSideMode_t mode, int m, int n, const aclblasComplex *A, int lda, const aclblasComplex *x, int incx, aclblasComplex *C, int ldc)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| mode | 输入 | aclblasSideMode_t | 缩放模式：当前仅支持 ACLBLAS_SIDE_LEFT（C = diag(x) * A，x 长度为 m）；ACLBLAS_SIDE_RIGHT 返回 ACLBLAS_STATUS_NOT_SUPPORTED，Host 内存 |
| m | 输入 | int | 矩阵 A/C 的行数，m >= 0，Host 内存 |
| n | 输入 | int | 矩阵 A/C 的列数，n >= 0，Host 内存 |
| A | 输入 | const aclblasComplex* | 输入矩阵，行主序存储，维度 m×n，Device 内存 |
| lda | 输入 | int | 矩阵 A 的行跨度（row-major），lda >= max(1, n)，Host 内存 |
| x | 输入 | const aclblasComplex* | 对角向量，LEFT 模式下长度为 m，Device 内存 |
| incx | 输入 | int | x 中相邻元素的步长，incx != 0，可为负数，Host 内存 |
| C | 输出 | aclblasComplex* | 输出矩阵，行主序存储，维度 m×n，Device 内存 |
| ldc | 输入 | int | 矩阵 C 的行跨度（row-major），ldc >= max(1, n)，Host 内存 |

#### 约束说明

- 当前仅支持 `ACLBLAS_SIDE_LEFT` 模式（C = diag(x) * A，C[i,j] = x[i] * A[i,j]）
- `ACLBLAS_SIDE_RIGHT` 是合法枚举，但当前返回 `ACLBLAS_STATUS_NOT_SUPPORTED`
- A 和 C 使用行主序（row-major）存储，与 cuBLAS `cublasCdgmm` 的列主序语义存在差异，不能把 cuBLAS 的列主序输入未经转换直接传入
- `lda` 和 `ldc` 表示行主序的行跨度（相邻两行起始位置之间的复数元素数），约束为不小于 n
- handle 不能为 nullptr，否则返回 `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
- mode 必须为 `ACLBLAS_SIDE_LEFT` 或 `ACLBLAS_SIDE_RIGHT`，否则返回 `ACLBLAS_STATUS_INVALID_ENUM`
- m >= 0, n >= 0，否则返回 `ACLBLAS_STATUS_INVALID_VALUE`
- incx != 0（可为负数，表示反向访问 x），否则返回 `ACLBLAS_STATUS_INVALID_VALUE`
- lda >= max(1, n)，否则返回 `ACLBLAS_STATUS_INVALID_VALUE`
- ldc >= max(1, n)，否则返回 `ACLBLAS_STATUS_INVALID_VALUE`
- 当 m > 0 且 n > 0 时，A、x、C 不能为 nullptr，否则返回 `ACLBLAS_STATUS_INVALID_VALUE`
- m == 0 或 n == 0 时为 no-op，直接返回 `ACLBLAS_STATUS_SUCCESS`
- 支持正负 `incx`
- 支持独立 `lda`、`ldc`（lda != ldc）
- C 的 padding 区域（C[i*ldc+n ... i*ldc+ldc-1]）不被修改
- 支持 `A == C && lda == ldc` 的原地执行；`A == C && lda != ldc` 返回 `ACLBLAS_STATUS_INVALID_VALUE`
- arch22 支持；arch35 暂不支持 Cdgmm
