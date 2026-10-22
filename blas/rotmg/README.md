# Rotmg算子

## 算子概述

构造修正 Givens 旋转参数（Construct Modified Givens Rotation），根据输入的标量值构造 modified Givens 旋转矩阵 H 的参数，使得 H 满足特定的对角化条件。rotmg 是纯标量计算，仅操作 4 个标量值（d1, d2, x1, y1），构造旋转参数，其输出通常直接作为下游 rotm 算子的输入。

数学表达式：

```
H^T * diag(d1_new, d2_new) * H = diag(d1, d2)
H * [x1, y1]^T = [x1_new, 0]^T    (sflag != -2)
```

其中第一式适用于非负权重 `d1`、`d2`；第二式描述实际旋转分支。`sflag == -2` 时 H 为单位矩阵，输入保持不变，例如 `d2 == 0` 且 `y1 != 0` 时不保证第二式。H 为 2x2 变换矩阵，由 `param` 数组按 BLAS 标准编码：

| param[0] (sflag) | H 矩阵 | 存储的元素 |
|------------------|--------|-----------|
| -1.0 | [[h11, h12], [h21, h22]] | param[1..4] 全部存储 |
| 0.0 | [[1, h12], [h21, 1]] | param[2]=h21, param[3]=h12 |
| 1.0 | [[h11, 1], [-1, h22]] | param[1]=h11, param[4]=h22 |
| -2.0 | [[1, 0], [0, 1]] | 无（恒等变换，仅设 sflag） |

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasSrotmg | 单精度浮点构造修正 Givens 旋转参数 |

## 算子执行接口

### aclblasSrotmg

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持

#### 函数原型

```cpp
aclblasStatus_t aclblasSrotmg(aclblasHandle_t handle, float *d1, float *d2, float *x1, const float *y1, float *param)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | ops-blas 库上下文句柄，携带 stream，Host 内存 |
| d1 | 输入/输出 | float*（FP32） | x 的缩放因子，Host 或 Device 内存 |
| d2 | 输入/输出 | float*（FP32） | y 的缩放因子，Host 或 Device 内存 |
| x1 | 输入/输出 | float*（FP32） | 向量的第一个分量，Host 或 Device 内存 |
| y1 | 输入 | const float*（FP32） | 向量的第二个分量，Host 或 Device 内存 |
| param | 输出 | float*（FP32） | 5 个元素的旋转参数数组，Host 或 Device 内存 |

#### 约束说明

- d1, d2, x1, y1, param 必须全部位于 Host 侧或全部位于 Device 侧；混合 Host/Device 指针将返回 ACLBLAS_STATUS_INVALID_VALUE
- Host 路径：直接 CPU 计算，无 kernel 开销，无数据搬运
- Device 路径：启动 device kernel（1 block SIMT），计算和结果都留在 device 上
