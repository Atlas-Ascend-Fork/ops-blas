# LtMatmul算子

## 算子概述

LtMatmul 算子实现了 BLAS Lt 通用矩阵乘法，核心运算为 D = alpha * op(A) * op(B) + beta * C。其中 A、B 为输入矩阵，C 为累加矩阵，D 为输出矩阵，alpha 和 beta 为标量，op(A)/op(B) 支持不转置（N）和转置（T）。当前实现支持 FP32、MXFP8（E4M3FN）、MXFP4（E2M1）三种输入类型组合，输出支持 FP32 和 BF16。

数学表达式：

```
D = alpha * op(A) * op(B) + beta * C
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasLtMatmul | 通用矩阵乘法（支持 FP32 / MXFP8 / MXFP4 路径） |

## 算子执行接口

### aclblasLtMatmul

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：不支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：不支持

> MXFP8/MXFP4 量化路径依赖 CANN asc-devkit >= 9.1（`ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR >= 1`）。

#### 函数原型

```cpp
aclblasStatus_t aclblasLtMatmul(aclblasLtHandle_t lightHandle, aclblasLtMatmulDesc_t computeDesc, const void* alpha, const void* A, aclblasLtMatrixLayout_t Adesc, const void* B, aclblasLtMatrixLayout_t Bdesc, const void* beta, const void* C, aclblasLtMatrixLayout_t Cdesc, void* D, aclblasLtMatrixLayout_t Ddesc, const aclblasLtMatmulAlgo_t* algo, void* workspace, size_t workspaceSizeInBytes, aclrtStream stream)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| lightHandle | 输入 | aclblasLtHandle_t | aclBLASLt 库上下文句柄，由 `aclblasLtCreate` 创建，不可为 NULL，否则返回 `ACLBLAS_STATUS_NOT_INITIALIZED`，Host 内存 |
| computeDesc | 输入 | aclblasLtMatmulDesc_t | 矩阵乘法描述符，设置 transA/transB、epilogue、scale 指针等属性，不可为 NULL，Host 内存 |
| alpha | 输入 | const void*（FP32） | 用于乘法的 float 标量指针，不可为 NULL，Host 内存 |
| A | 输入 | const void* | 输入矩阵 A，数据类型由 Adesc 指定，m>0 且 n>0 时不可为 NULL，Device 内存 |
| Adesc | 输入 | aclblasLtMatrixLayout_t | 矩阵 A 的 layout 描述符（rows/cols/ld/order/dtype），Host 内存 |
| B | 输入 | const void* | 输入矩阵 B，数据类型由 Bdesc 指定，m>0 且 n>0 时不可为 NULL，Device 内存 |
| Bdesc | 输入 | aclblasLtMatrixLayout_t | 矩阵 B 的 layout 描述符，Host 内存 |
| beta | 输入 | const void*（FP32） | 用于累加的 float 标量指针，不可为 NULL，beta=0 时 C 可不参与计算，Host 内存 |
| C | 输入 | const void* | 累加矩阵 C，beta=0 时可为 NULL，Device 内存 |
| Cdesc | 输入 | aclblasLtMatrixLayout_t | 矩阵 C 的 layout 描述符，Host 内存 |
| D | 输出 | void* | 输出矩阵 D，维度 m x n，m>0 且 n>0 时不可为 NULL，Device 内存 |
| Ddesc | 输入 | aclblasLtMatrixLayout_t | 矩阵 D 的 layout 描述符，指定输出数据类型（FP32 或 BF16），Host 内存 |
| algo | 输入 | const aclblasLtMatmulAlgo_t* | 算法描述符，可为 NULL（使用默认算法），Host 内存 |
| workspace | 输入 | void* | 工作空间内存，可为 NULL，非 NULL 时需 16B 对齐，Device 内存 |
| workspaceSizeInBytes | 输入 | size_t | 工作空间大小（字节），Host 内存 |
| stream | 输入 | aclrtStream | AscendCL 执行流，Host 内存 |

#### 约束说明

- M、N、K >= 0；M=0 或 N=0 时为空操作，直接返回 `ACLBLAS_STATUS_SUCCESS`
- dtypeA / dtypeB 须为同类型组合：FP32×FP32、MXFP8×MXFP8、MXFP4×MXFP4；其他组合返回 `ACLBLAS_STATUS_NOT_SUPPORTED`
- dtypeC 当前固定为 FP32
- dtypeD 支持 FP32 或 BF16；MXFP8/MXFP4 路径支持 FP32 或 BF16 输出，FP32 路径输出 FP32
- computeType 当前仅支持 `ACLBLAS_COMPUTE_32F`
- transA / transB 支持 N、T，对应 `ACLBLAS_OP_N`（不转置）、`ACLBLAS_OP_T`（转置）
- MXFP8/MXFP4 路径要求 K 为 32 的整数倍，否则返回 `ACLBLAS_STATUS_INVALID_VALUE`
- order 必须显式设为 `ACLBLASLT_ORDER_ROW`（layout 描述符的默认值为 `ACLBLASLT_ORDER_COL`，A/B/C/D 任一未设置为 ROW 均返回 `ACLBLAS_STATUS_NOT_SUPPORTED`，不再静默按行主序解释）；lda / ldb / ldc / ldd 须 >= 矩阵物理列数；MXFP4 的 ld 为逻辑元素 leading dim（2 个 FP4 元素打包为 1 字节）
- epilogue 当前仅支持 `ACLBLASLT_EPILOGUE_DEFAULT`
- MXFP8/MXFP4 路径须通过 computeDesc 设置 scaleA / scaleB（`ACLBLASLT_MATMUL_DESC_A/B_SCALE_POINTER`），E8M0 格式，按 K 方向每 32 元素一组，不可为 NULL
- algo 可为 NULL，使用默认算法
- workspace 非 NULL 时需 16B 对齐；algo 非 NULL 时 workspaceSizeInBytes 须 >= algo->max_workspace_bytes

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](../../docs/zh/develop/compile_and_run_example.md)。使用 BLASLt 接口时，需在 CMakeLists.txt 中额外链接 `libops_blasLt.so`。

```cpp
#include <cstdio>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "cann_ops_blasLt.h"

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
struct LtHandleDeleter {
    void operator()(aclblasLtHandle_t h) const { aclblasLtDestroy(h); }
};
struct LtDescDeleter {
    void operator()(aclblasLtMatmulDesc_t d) const { aclblasLtMatmulDescDestroy(d); }
};
struct LtLayoutDeleter {
    void operator()(aclblasLtMatrixLayout_t d) const { aclblasLtMatrixLayoutDestroy(d); }
};

int aclblasLtMatmulTest(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 aclBLASLt 句柄与描述符
    aclblasLtHandle_t rawHandle = nullptr;
    auto blasRet = aclblasLtCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasLtCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasLtHandle_t>::type, LtHandleDeleter> handlePtr(rawHandle);

    aclblasLtMatmulDesc_t rawDesc = nullptr;
    blasRet = aclblasLtMatmulDescCreate(&rawDesc, ACLBLAS_COMPUTE_32F, ACL_FLOAT);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasLtMatmulDescCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasLtMatmulDesc_t>::type, LtDescDeleter> descPtr(rawDesc);

    const uint64_t m = 16, n = 16, k = 16;
    aclblasLtMatrixLayout_t rawAdesc = nullptr;
    aclblasLtMatrixLayout_t rawBdesc = nullptr;
    aclblasLtMatrixLayout_t rawCdesc = nullptr;
    aclblasLtMatrixLayout_t rawDdesc = nullptr;
    int32_t order = ACLBLASLT_ORDER_ROW;

    blasRet = aclblasLtMatrixLayoutCreate(&rawAdesc, ACL_FLOAT, m, k, k);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("MatrixLayoutCreate A failed. ERROR: %d\n", blasRet);
              return blasRet);
    blasRet = aclblasLtMatrixLayoutCreate(&rawBdesc, ACL_FLOAT, k, n, n);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("MatrixLayoutCreate B failed. ERROR: %d\n", blasRet);
              return blasRet);
    blasRet = aclblasLtMatrixLayoutCreate(&rawCdesc, ACL_FLOAT, m, n, n);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("MatrixLayoutCreate C failed. ERROR: %d\n", blasRet);
              return blasRet);
    blasRet = aclblasLtMatrixLayoutCreate(&rawDdesc, ACL_FLOAT, m, n, n);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("MatrixLayoutCreate D failed. ERROR: %d\n", blasRet);
              return blasRet);

    std::unique_ptr<std::remove_pointer<aclblasLtMatrixLayout_t>::type, LtLayoutDeleter> aDescPtr(rawAdesc);
    std::unique_ptr<std::remove_pointer<aclblasLtMatrixLayout_t>::type, LtLayoutDeleter> bDescPtr(rawBdesc);
    std::unique_ptr<std::remove_pointer<aclblasLtMatrixLayout_t>::type, LtLayoutDeleter> cDescPtr(rawCdesc);
    std::unique_ptr<std::remove_pointer<aclblasLtMatrixLayout_t>::type, LtLayoutDeleter> dDescPtr(rawDdesc);

    aclblasLtMatrixLayoutSetAttribute(aDescPtr.get(), ACLBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    aclblasLtMatrixLayoutSetAttribute(bDescPtr.get(), ACLBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    aclblasLtMatrixLayoutSetAttribute(cDescPtr.get(), ACLBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));
    aclblasLtMatrixLayoutSetAttribute(dDescPtr.get(), ACLBLASLT_MATRIX_LAYOUT_ORDER, &order, sizeof(order));

    // 2. 准备 Host 数据
    float alpha = 1.0f;
    float beta = 0.0f;
    std::vector<float> hA(m * k, 1.0f);
    std::vector<float> hB(k * n, 1.0f);

    // 3. 申请 Device 内存并拷贝数据
    void* rawA = nullptr;
    void* rawB = nullptr;
    void* rawD = nullptr;
    size_t aBytes = hA.size() * sizeof(float);
    size_t bBytes = hB.size() * sizeof(float);
    size_t dBytes = m * n * sizeof(float);

    auto aclRet = aclrtMalloc(&rawA, aBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for A failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc(&rawB, bBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for B failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMalloc(&rawD, dBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for D failed. ERROR: %d\n", aclRet); return aclRet);

    std::unique_ptr<void, AclMemDeleter> aDevicePtr(rawA);
    std::unique_ptr<void, AclMemDeleter> bDevicePtr(rawB);
    std::unique_ptr<void, AclMemDeleter> dDevicePtr(rawD);

    aclRet = aclrtMemcpy(aDevicePtr.get(), aBytes, hA.data(), aBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for A failed. ERROR: %d\n", aclRet); return aclRet);
    aclRet = aclrtMemcpy(bDevicePtr.get(), bBytes, hB.data(), bBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for B failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasLtMatmul：D = alpha * A * B + beta * C（beta=0，C 可为 nullptr）
    blasRet = aclblasLtMatmul(
        handlePtr.get(), descPtr.get(), &alpha,
        aDevicePtr.get(), aDescPtr.get(), bDevicePtr.get(), bDescPtr.get(),
        &beta, nullptr, cDescPtr.get(),
        dDevicePtr.get(), dDescPtr.get(),
        nullptr, nullptr, 0, stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasLtMatmul failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    std::vector<float> hD(m * n, 0.0f);
    aclRet = aclrtMemcpy(hD.data(), dBytes, dDevicePtr.get(), dBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", aclRet);
              return aclRet);
    LOG_PRINT("D[0] = %f (expected %f)\n", hD[0], static_cast<float>(k));

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasLtMatmulTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasLtMatmulTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

