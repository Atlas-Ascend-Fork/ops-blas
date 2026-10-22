# {算子名称}算子

<!--
模板使用说明：
1. 将所有 {占位符} 替换为实际内容，占位符命名规则见下表。
2. 按实际包含的接口增删子章节（### 节）。如仅有单精度接口则只保留一个 ### 节；如有更多接口（如 aclblasH{op}、aclblasC{op}）则继续添加。
3. 参数说明列必须写清楚参数含义和内存位置（Host 内存/Device 内存）。
4. 约束说明如无约束则写"无"，不允许留空。
5. 函数原型必须写在同一行内，不允许换行，且末尾不加 `;`。
6. 调用示例必须使用 RAII 模式（AclContext 类 + std::unique_ptr）管理 ACL 资源和 Device 内存，与 compile_and_run_example.md 一致；须由开发者在本地跑通后再上库。
7. 完成文档后删除本使用说明注释块。

占位符约定：

| 占位符 | 含义 | 示例 |
|--------|------|------|
| {算子名称} | 算子中文名 | 向量缩放 |
| {op} | 算子英文缩写（小写） | scal |
| {功能描述} | 算子功能的一句话描述 | 对向量进行标量缩放 |
| {运算描述} | 核心数学运算描述 | y = alpha * x |
| {参数含义} | 各参数的具体含义 | 向量元素个数 |
-->

## 算子概述

{一段话描述算子的功能定位和核心运算。例如："{op} 算子实现了{功能描述}，核心运算为{运算描述}。"}

数学表达式：

```
{数学公式，使用纯文本。如需要可用 LaTeX 格式（$$...$$）}
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasS{op} | {单精度功能描述} |
| aclblasD{op} | {双精度功能描述} |

## 算子执行接口

### aclblasS{op}

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：{支持/不支持}
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：{支持/不支持}
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：{支持/不支持}

#### 函数原型

```cpp
aclblasStatus_t aclblasS{op}(aclblasHandle_t handle, int n, const float *x, float *y, ...)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | {参数含义}，Host 内存 |
| x | 输入 | const float*（FP32） | {参数含义}，Device 内存 |
| y | 输入/输出 | float*（FP32） | {参数含义}，Device 内存 |

#### 约束说明

- n >= 0
- incx != 0

{如无约束则写"无"，不允许留空。}

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](compile_and_run_example.md)。

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

int aclblasS{op}Test(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-blas 句柄
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasHandle_t>::type, decltype(&aclblasDestroy)>
        handlePtr(rawHandle, aclblasDestroy);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 2. 准备 Host 数据
    // {按算子需求定义参数并初始化 Host 数据，如 std::vector<float> hX(n, 1.0f); }

    // 3. 申请 Device 内存并拷贝数据
    // {按算子需求使用 aclrtMalloc + std::unique_ptr + aclrtFree 管理 Device 内存}
    // {按算子需求使用 aclrtMemcpy 拷贝数据到 Device}

    // 4. 调用 aclblasS{op}
    // blasRet = aclblasS{op}(static_cast<aclblasHandle_t>(handlePtr.get()), ...);
    // CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasS{op} failed. ERROR: %d\n", blasRet);
    //           return blasRet);

    // 5. 同步等待任务执行结束
    auto aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    // {按算子需求使用 aclrtMemcpy 拷贝结果回 Host 并打印验证}

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasS{op}Test(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasS{op}Test failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```

### aclblasD{op}

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：{支持/不支持}
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：{支持/不支持}
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：{支持/不支持}

#### 函数原型

```cpp
aclblasStatus_t aclblasD{op}(aclblasHandle_t handle, int n, const double *x, double *y, ...)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| n | 输入 | int | {参数含义}，Host 内存 |
| x | 输入 | const double*（FP64） | {参数含义}，Device 内存 |
| y | 输入/输出 | double*（FP64） | {参数含义}，Device 内存 |

#### 约束说明

- n >= 0

{如无约束则写"无"，不允许留空。}

#### 调用示例

示例代码如下，仅供参考，具体编译和执行过程请参考[编译与运行样例](compile_and_run_example.md)。

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

int aclblasD{op}Test(AclContext& ctx)
{
    aclrtStream stream = ctx.Stream();

    // 1. 创建 ops-blas 句柄
    aclblasHandle_t rawHandle = nullptr;
    auto blasRet = aclblasCreate(&rawHandle);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasCreate failed. ERROR: %d\n", blasRet);
              return blasRet);
    std::unique_ptr<std::remove_pointer<aclblasHandle_t>::type, decltype(&aclblasDestroy)>
        handlePtr(rawHandle, aclblasDestroy);

    blasRet = aclblasSetStream(static_cast<aclblasHandle_t>(handlePtr.get()), stream);
    CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasSetStream failed. ERROR: %d\n", blasRet);
              return blasRet);

    // 2. 准备 Host 数据
    // {按算子需求定义参数并初始化 Host 数据，如 std::vector<double> hX(n, 1.0); }

    // 3. 申请 Device 内存并拷贝数据
    // {按算子需求使用 aclrtMalloc + std::unique_ptr + aclrtFree 管理 Device 内存}
    // {按算子需求使用 aclrtMemcpy 拷贝数据到 Device}

    // 4. 调用 aclblasD{op}
    // blasRet = aclblasD{op}(static_cast<aclblasHandle_t>(handlePtr.get()), ...);
    // CHECK_RET(blasRet == ACLBLAS_STATUS_SUCCESS, LOG_PRINT("aclblasD{op} failed. ERROR: %d\n", blasRet);
    //           return blasRet);

    // 5. 同步等待任务执行结束
    auto aclRet = aclrtSynchronizeStream(stream);
    CHECK_RET(aclRet == ACL_SUCCESS, LOG_PRINT("aclrtSynchronizeStream failed. ERROR: %d\n", aclRet); return aclRet);

    // 6. 将结果从 Device 拷贝回 Host 并打印
    // {按算子需求使用 aclrtMemcpy 拷贝结果回 Host 并打印验证}

    return ACL_SUCCESS;
}

int main()
{
    AclContext ctx(0);
    auto ret = ctx.Init();
    CHECK_RET(ret == ACL_SUCCESS, return ret);

    ret = aclblasD{op}Test(ctx);
    CHECK_RET(ret == ACL_SUCCESS, LOG_PRINT("aclblasD{op}Test failed. ERROR: %d\n", ret); return ret);
    return 0;
}
```
