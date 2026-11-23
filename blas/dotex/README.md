# DotEx 算子

## 算子概述

扩展精度向量点积算子，实现两个向量的点积运算，支持混合精度计算。内部统一以 FP32 精度累加，最终结果转换为输入类型。

数学表达式：

```
result = Σ(x[i * incx] * y[i * incy]),  i = 0 ... n-1
```

包含以下接口：

| 接口名       | 功能简述              |
|-------------|----------------------|
| aclblasDotEx | 扩展精度向量点积 |

## 算子执行接口

### aclblasDotEx

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 接口说明

```cpp
aclblasStatus_t aclblasDotEx(
    aclblasHandle_t handle, int n,
    const void *x, aclDataType xType, int incx,
    const void *y, aclDataType yType, int incy,
    void *result, aclDataType resultType,
    aclDataType executionType)
```

| 参数          | 输入/输出 | 说明              |
|--------------|---------|-------------------|
| handle       | 输入     | 算子句柄            |
| n            | 输入     | 向量元素个数         |
| x            | 输入     | 向量 x 的设备地址     |
| xType        | 输入     | x 的数据类型         |
| incx         | 输入     | x 的存储步长         |
| y            | 输入     | 向量 y 的设备地址     |
| yType        | 输入     | y 的数据类型         |
| incy         | 输入     | y 的存储步长         |
| result       | 输出     | 点积结果的设备地址     |
| resultType   | 输入     | result 的数据类型     |
| executionType | 输入    | 中间计算精度，固定为 ACL_FLOAT |

#### 返回值

| 值                              | 说明           |
|---------------------------------|---------------|
| ACLBLAS_STATUS_SUCCESS          | 成功           |
| ACLBLAS_STATUS_HANDLE_IS_NULLPTR | handle 为空     |
| ACLBLAS_STATUS_INVALID_VALUE    | 参数非法        |
| ACLBLAS_STATUS_NOT_SUPPORTED    | 不支持的类型组合  |
| ACLBLAS_STATUS_ALLOC_FAILED     | 内存分配失败     |
| ACLBLAS_STATUS_EXECUTION_FAILED | 执行失败        |

#### 支持的数据类型

| x/y 类型 | result 类型 | execution 类型 |
|----------|------------|---------------|
| FP32     | FP32       | FP32          |
| FP16     | FP16       | FP32          |
| BF16     | BF16       | FP32          |

#### 注意事项

- x 和 y 的数据类型必须相同
- result 类型必须与 x/y 类型一致
- executionType 必须为 ACL_FLOAT

## 编译与运行

```bash
bash build.sh --run --ops=dotex --soc=ascend950
```
