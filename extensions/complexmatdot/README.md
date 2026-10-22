# ComplexMatDot算子

## 算子概述

ComplexMatDot（复数矩阵点乘）算子实现了两个复数矩阵的逐元素乘法运算，是 BLAS 基础线性代数库中的扩展算子之一。该算子针对复数运算特性进行了优化，使用 GatherMask 操作高效完成复数矩阵的逐元素乘法。

数学表达式：

```
result[i, j] = matx[i, j] * maty[i, j]
```

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasComplexMatDot | 复数矩阵逐元素点乘 |

## 算子执行接口

### aclblasComplexMatDot

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：不支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasComplexMatDot(
    aclblasHandle_t handle, int m, int n, const aclblasComplex* matx, const aclblasComplex* maty,
    aclblasComplex* result);
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | 执行上下文句柄，Host 内存 |
| m | 输入 | int | 矩阵的行数，Host 内存 |
| n | 输入 | int | 矩阵的列数，Host 内存 |
| matx | 输入 | const aclblasComplex* | 复数矩阵，维度为 m x n，Device 内存 |
| maty | 输入 | const aclblasComplex* | 复数矩阵，维度为 m x n，Device 内存 |
| result | 输出 | aclblasComplex* | 复数矩阵，维度为 m x n，Device 内存 |

#### 约束说明

无
