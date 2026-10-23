# Scal算子

## 算子概述

向量缩放算子，实现向量乘以标量的运算。包含实数向量缩放（Sscal）和复数向量缩放（Cscal）。

数学表达式：

```
x[i] = alpha * x[i]  (i = 0 .. n-1，步长为 incx)
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSscal | 实数向量乘以标量 |
| aclblasCscal | 复数向量乘以复数标量 |
| aclblasCsscal | 复数向量乘以实数标量 |

## 算子执行接口

### aclblasSscal

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSscal(aclblasHandle_t handle, int n, const float* alpha, float* x, int incx);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 向量 x 中的元素个数，Host 内存 |
| alpha | 输入 | const float*（FP32） | 指向标量乘数的指针，Host 内存 |
| x | 输入/输出 | float*（FP32） | float 向量，包含 n 个元素，Device 内存 |
| incx | 输入 | int | x 中连续元素之间的步长，Host 内存 |

#### 约束说明

**Ascend 950PR / Ascend 950DT（arch35）：**

- n 为整数；n <= 0 时为 no-op（直接返回 ACLBLAS_STATUS_SUCCESS，不修改 x）
- incx 为整数；incx <= 0 时为 no-op（不修改 x，对齐参考 BLAS cblas_sscal 的 IF (INCX.LE.0) RETURN 语义）；incx > 0 时支持任意步长
- handle 不能为 nullptr，否则返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR
- n > 0 时 alpha、x 不能为 nullptr，否则返回 ACLBLAS_STATUS_INVALID_VALUE

**Atlas A2 / Atlas A3 系列产品（arch22）：**

- incx 参数当前实现未启用（固定按连续向量 incx=1 处理，传入的 incx 取值不生效）
- n < 0 时返回 ACLBLAS_STATUS_INVALID_VALUE；n == 0 时为 no-op（直接返回 ACLBLAS_STATUS_SUCCESS，不访问 alpha / x）
- handle 不能为 nullptr，否则返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR
- n > 0 时 alpha、x 不能为 nullptr，否则返回 ACLBLAS_STATUS_INVALID_VALUE


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

int aclblasSscalTest(AclContext& ctx)
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
    int n = 5;
    int incx = 1;
    float alpha = 2.0f;
    std::vector<float> xHostData = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};  // 缩放后期望 {2,4,6,8,10}
    size_t xBytes = n * sizeof(float);

    // 3. 申请 Device 内存并拷贝数据
    void* rawMem = nullptr;
    auto aclRet = aclrtMalloc(&rawMem, xBytes, ACL_MEM_MALLOC_HUGE_FIRST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMalloc for x failed. ERROR: %d\n", aclRet); return aclRet);
    std::unique_ptr<float, AclMemDeleter> xDevicePtr(static_cast<float*>(rawMem));

    aclRet = aclrtMemcpy(xDevicePtr.get(), xBytes, xHostData.data(), xBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtMemcpy for x failed. ERROR: %d\n", aclRet); return aclRet);

    // 4. 调用 aclblasSscal（alpha 为 Host 指针，原地缩放）
    blasRet = aclblasSscal(handlePtr.get(), n, &alpha, xDevicePtr.get(), incx);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSscal failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 5. 同步等待任务执行结束
    aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    std::vector<float> resultData(n, 0);
    aclRet = aclrtMemcpy(resultData.data(), xBytes, xDevicePtr.get(), xBytes, ACL_MEMCPY_DEVICE_TO_HOST);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("copy result from device to host failed. ERROR: %d\n", aclRet);
              return aclRet);
    for (int i = 0; i < n; i++) {
        LOG_PRINT("result[%d] is: %f\n", i, resultData[i]);
    }

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasSscalTest(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasSscalTest failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

### aclblasCscal

#### 产品支持情况

- Ascend 950PR：支持（CANN 9.1.0，arch35）
- Ascend 950DT：不支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasCscal(
    aclblasHandle_t handle, int n, const aclblasComplex* alpha,
    aclblasComplex* x, int incx);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 向量 x 中的逻辑复数元素个数，Host 标量；n≤0 为 no-op |
| alpha | 输入 | const aclblasComplex* | 指向 Host complex64 标量；不是按值参数或 Device 指针 |
| x | 输入/输出 | aclblasComplex* | Device complex64 向量；按 incx 选择元素并原地写回 |
| incx | 输入 | int | 复数元素步长；incx>0 支持正步长，incx≤0 为 no-op |

#### 约束说明

- 本节新增的行为及测试说明针对 Ascend 950PR / arch35；其他架构支持状态沿用上游，未在本次重新测试。
- `aclblasComplex` 定义以公共 `include/cann_ops_blas_common.h` 为准，实部、虚部各 float32，按 real／imag 交织，占 8 字节。此接口不是 `aclblasCsscal`。
- 对 `i=0,...,n-1`，执行 `x[i*incx] = alpha * x[i*incx]`；必须先读两个原始分量，再写回 `(a*c-b*d, a*d+b*c)`。不改变 stride 空隙。
- 正参数时，x 的有效物理存储至少为 `1+(n-1)*incx` 个复数；支持运行时 n，无额外 leading dimension、broadcast 或视图输出。内部 64 位跨度检查不能替代调用者保证指针及分配有效。
- 参数处理优先级：空 handle → n／incx 非正 no-op → alpha／x 空指针检查 → alpha=(1,0) no-op → 地址跨度检查 → 启动计算。非正 incx 不是反向遍历。
- alpha=(0,0) 不是 no-op；按本任务文字将选中元素的两个分量置为正零，原地存储的其他位置不变。此约定与原始 Netlib 对部分 Inf／NaN／零符号输入的结果不同，测试单列 raw CBLAS 差异，设计评审需确认该约定。
- alpha 实虚分量支持 FLOAT32 值域，包括 Inf／NaN；比较 NaN 类别，不保证 NaN payload。alpha=(1,0) 保持输入原字节。
- handle 中绑定的 stream 必须有效；计算异步执行，接口内不做 stream 同步。读回、释放、复用 x 或跨 stream 消费前，调用者负责同步和依赖关系。
- 不需要额外 GM workspace 参数；不在该接口内部申请 Device 辅助缓冲。

#### 返回值

| 返回码 | 950PR 行为 |
|---|---|
| ACLBLAS_STATUS_SUCCESS | 合法 no-op，或 Host 已提交异步计算；不表示设备同步执行已经成功 |
| ACLBLAS_STATUS_HANDLE_IS_NULLPTR | handle 为空，优先于其他检查 |
| ACLBLAS_STATUS_INVALID_VALUE | 活跃调用时 alpha／x 为空，或字节跨度／地址算术溢出 |
| ACLBLAS_STATUS_EXECUTION_FAILED | 无法产生合法执行规划，例如获取的 AIV 核数为 0 |

异步设备错误需检查 `aclrtSynchronizeStream` 等 ACL 调用的返回值。

#### 调用示例

下面展示 950PR 上的最小完整生命周期。预期 `(2-3i)*(1+2i)=8+i`，`(2-3i)*(-1+i)=1+5i`。

```cpp
#include <cstdio>
#include <stdexcept>
#include "acl/acl.h"
#include "cann_ops_blas.h"

int main() {
    aclrtStream stream = nullptr;
    aclblasHandle_t handle = nullptr;
    void* device = nullptr;
    bool initialized = false, selected = false;
    auto aclCheck = [](aclError e) {
        if (e != ACL_SUCCESS) throw std::runtime_error("ACL call failed");
    };
    auto blasCheck = [](aclblasStatus_t e) {
        if (e != ACLBLAS_STATUS_SUCCESS) throw std::runtime_error("BLAS call failed");
    };
    auto cleanup = [&] {
        if (stream) aclrtSynchronizeStream(stream);
        if (device) aclrtFree(device);
        if (handle) aclblasDestroy(handle);
        if (stream) aclrtDestroyStream(stream);
        if (selected) aclrtResetDevice(0);
        if (initialized) aclFinalize();
    };
    try {
        aclCheck(aclInit(nullptr)); initialized = true;
        aclCheck(aclrtSetDevice(0)); selected = true;
        aclCheck(aclrtCreateStream(&stream));
        blasCheck(aclblasCreate(&handle));
        blasCheck(aclblasSetStream(handle, stream));
        aclblasComplex alpha{2.0f, -3.0f}; // Host
        aclblasComplex x[2]{{1.0f, 2.0f}, {-1.0f, 1.0f}};
        aclCheck(aclrtMalloc(&device, sizeof(x), ACL_MEM_MALLOC_HUGE_FIRST));
        aclCheck(aclrtMemcpy(device, sizeof(x), x, sizeof(x), ACL_MEMCPY_HOST_TO_DEVICE));
        blasCheck(aclblasCscal(handle, 2, &alpha,
                              static_cast<aclblasComplex*>(device), 1));
        aclCheck(aclrtSynchronizeStream(stream));
        aclCheck(aclrtMemcpy(x, sizeof(x), device, sizeof(x), ACL_MEMCPY_DEVICE_TO_HOST));
        std::printf("(%g,%g) (%g,%g)\n", x[0].real, x[0].imag, x[1].real, x[1].imag);
        cleanup(); return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what()); cleanup(); return 1;
    }
}
```

将示例保存为 `example.cpp`。在已运行 `bash build.sh --ops=cscal --soc=ascend950` 的仓库根目录执行（库目录按实际构建调整）：

```bash
source /usr/local/Ascend/cann/set_env.sh
g++ -std=c++17 example.cpp -Iinclude -I"$ASCEND_HOME_PATH/include" \
  -Lbuild -Wl,-rpath,"$PWD/build" \
  -L"$ASCEND_HOME_PATH/lib64" -Wl,-rpath,"$ASCEND_HOME_PATH/lib64" \
  -lops_blas -lascendcl -o example
./example
```

#### 测试与已验证配置

见 [Cscal 测试说明](../../test/scal/cscal/README.md)。原交付分支在 Ascend 950PR、CANN 9.1.0 上记录了原题精度 1000/1000、补充精度 181/181、性能 200/200 通过；计时采用 ACL timeline 单次 stream 区间，10 次预热、64 次有效样本，并在每次计时前恢复输入。同条件复测仍有波动，且当前上游基线未重新做设备复测。完整原始结果和构建哈希保留在[验收交付分支](https://gitcode.com/zee_920/ops-blas/tree/delivery/cscal950-acceptance/docs/cscal950/acceptance/20260915)，不将支持声明扩大到 950DT 或未测工具链。

### aclblasCsscal

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：待支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：待支持

#### 函数原型

```cpp
aclblasStatus_t aclblasCsscal(
    aclblasHandle_t handle,
    int n,
    const float* alpha,
    aclblasComplex* x,
    int incx)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | 向量 x 的复数元素个数，Host 内存；n ≤ 0 时为 no-op |
| alpha | 输入 | const float*（FP32） | 指向实数标量乘数的指针，Host 内存 |
| x | 输入/输出 | aclblasComplex*（FP32 complex） | 复数向量，Device 内存，原地更新 |
| incx | 输入 | int | x 中连续复数元素之间的步长，Host 内存；incx ≤ 0 时为 no-op |

#### 约束说明

- n ≤ 0 或 incx ≤ 0 时为 no-op（返回 ACLBLAS_STATUS_SUCCESS，不修改 x）
- alpha=0.0 时为置零操作（非 no-op）
- alpha=1.0 时允许提前返回
- n > 0 时 alpha、x 不得为 nullptr
