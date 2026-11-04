# GemmGroupedBatchedEx算子

## 算子概述

通用矩阵乘法分组批量扩展接口（GEMM Grouped Batched Ex），在一次调用中执行多组列主序 GEMM。每个组共享矩阵形状、转置方式、leading dimension、缩放系数和数据类型；同组内可以包含多个独立矩阵实例。

数学表达式：

```
C[index] = alpha[g] * op(A[index]) * op(B[index]) + beta[g] * C[index]
```

其中 `index` 按组顺序连续编号，实例总数为 `sum(groupSize[g])`。

包含以下接口：

| 接口名 | 功能简述 |
|--------|---------|
| aclblasGemmGroupedBatchedEx | 通用矩阵乘法分组批量扩展接口 |

## 算子执行接口

### aclblasGemmGroupedBatchedEx

#### 产品支持情况

- Ascend 950PR / Ascend 950DT：支持
- Atlas A3 训练系列产品 / Atlas A3 推理系列产品：支持（arch22，FLOAT32主路径）
- Atlas A2 训练系列产品 / Atlas A2 推理系列产品：支持（arch22，FLOAT32主路径）

#### 函数原型

```cpp
aclblasStatus_t aclblasGemmGroupedBatchedEx(aclblasHandle_t handle, const aclblasOperation_t transaArray[], const aclblasOperation_t transbArray[], const int mArray[], const int nArray[], const int kArray[], const void* alphaArray, const void* const Aarray[], aclDataType Atype, const int ldaArray[], const void* const Barray[], aclDataType Btype, const int ldbArray[], const void* betaArray, void* const Carray[], aclDataType Ctype, const int ldcArray[], int groupCount, const int groupSize[], aclblasComputeType_t computeType)
```

#### 参数说明

| 参数名 | 输入/输出 | 参数类型 | 说明 |
|--------|----------|---------|------|
| handle | 输入 | aclblasHandle_t | BLAS handle，其中的 stream 用于异步下发设备任务，不可为空，Host 内存 |
| transaArray | 输入 | const aclblasOperation_t[] | 长度为 groupCount，每组 A 的操作，可为 ACLBLAS_OP_N、ACLBLAS_OP_T 或 ACLBLAS_OP_C；实数类型下 C 等同于 T，Host 内存 |
| transbArray | 输入 | const aclblasOperation_t[] | 长度为 groupCount，每组 B 的操作，取值同 transaArray，Host 内存 |
| mArray | 输入 | const int[] | 长度为 groupCount，每组输出矩阵 C 的行数，必须 >= 0，Host 内存 |
| nArray | 输入 | const int[] | 长度为 groupCount，每组输出矩阵 C 的列数，必须 >= 0，Host 内存 |
| kArray | 输入 | const int[] | 长度为 groupCount，每组乘法的规约维度，必须 >= 0，Host 内存 |
| alphaArray | 输入 | const void* | 长度为 groupCount，computeType 为 ACLBLAS_COMPUTE_16F 时元素按 FP16 存储；为 ACLBLAS_COMPUTE_32F 时元素按 FP32 存储，Host 内存 |
| Aarray | 输入 | const void* const [] | 长度为 sum(groupSize) 的设备指针数组，每个元素指向一个列主序 A 矩阵，Device 内存 |
| Atype | 输入 | aclDataType | 所有 A 矩阵的元素类型，Host 内存 |
| ldaArray | 输入 | const int[] | 长度为 groupCount，A 的 leading dimension；非转置时至少为 max(1, m[g])，转置时至少为 max(1, k[g])，Host 内存 |
| Barray | 输入 | const void* const [] | 长度为 sum(groupSize) 的设备指针数组，每个元素指向一个列主序 B 矩阵，Device 内存 |
| Btype | 输入 | aclDataType | 所有 B 矩阵的元素类型，FP8 输入允许 E4M3 和 E5M2 混合使用，Host 内存 |
| ldbArray | 输入 | const int[] | 长度为 groupCount，B 的 leading dimension；非转置时至少为 max(1, k[g])，转置时至少为 max(1, n[g])，Host 内存 |
| betaArray | 输入 | const void* | 长度为 groupCount，元素存储类型与 alphaArray 相同，Host 内存 |
| Carray | 输入/输出 | void* const [] | 长度为 sum(groupSize) 的设备指针数组，每个元素指向一个列主序输入/输出 C 矩阵，Device 内存 |
| Ctype | 输入 | aclDataType | 所有 C 矩阵的元素类型，Host 内存 |
| ldcArray | 输入 | const int[] | 长度为 groupCount，C 的 leading dimension，至少为 max(1, m[g])，Host 内存 |
| groupCount | 输入 | int | 分组数量，必须 >= 0；为 0 时接口直接返回成功，Host 内存 |
| groupSize | 输入 | const int[] | 长度为 groupCount，每组包含的矩阵实例数，每个值必须 >= 0，Host 内存 |
| computeType | 输入 | aclblasComputeType_t | 缩放系数和计算精度类型，支持 ACLBLAS_COMPUTE_16F 或 ACLBLAS_COMPUTE_32F，Host 内存 |

#### 约束说明

- groupCount >= 0
- m, n, k >= 0（每组）
- groupSize >= 0（每组）
- lda >= max(1, 非转置时为 m[g]，转置时为 k[g])
- ldb >= max(1, 非转置时为 k[g]，转置时为 n[g])
- ldc >= max(1, m[g])
- Aarray、Barray、Carray 本身以及它们指向的矩阵均位于 Device；其余数组位于 Host
- 当实例总数大于 0 时，三个设备指针数组均不可为空

支持的数据类型：

| Atype | Btype | Ctype | computeType |
|-------|-------|-------|-------------|
| ACL_FLOAT16 | ACL_FLOAT16 | ACL_FLOAT16 | ACLBLAS_COMPUTE_16F、ACLBLAS_COMPUTE_32F |
| ACL_BF16 | ACL_BF16 | ACL_BF16 | ACLBLAS_COMPUTE_32F |
| ACL_FLOAT8_E4M3FN | ACL_FLOAT8_E4M3FN | ACL_FLOAT16 | ACLBLAS_COMPUTE_32F |
| ACL_FLOAT8_E5M2 | ACL_FLOAT8_E5M2 | ACL_FLOAT16 | ACLBLAS_COMPUTE_32F |
| ACL_FLOAT8_E4M3FN | ACL_FLOAT8_E5M2 | ACL_FLOAT16 | ACLBLAS_COMPUTE_32F |
| ACL_FLOAT8_E5M2 | ACL_FLOAT8_E4M3FN | ACL_FLOAT16 | ACLBLAS_COMPUTE_32F |

## Atlas A2/A3（arch22）支持范围

arch22 支持 Atype/Btype/Ctype 均为 `ACL_FLOAT` 且 computeType 为
`ACLBLAS_COMPUTE_32F`。其他组合返回 `ACLBLAS_STATUS_NOT_SUPPORTED`；上方
FP16/BF16/FP8 类型表为 arch35 的支持范围，不表示 arch22 同样支持。
公共 API 仍复用 `include/cann_ops_blas.h` 的已有声明。

Aarray、Barray、Carray 本身及矩阵位于 Device；组内实例按组顺序展开。
Host 组数组长度须为 groupCount，不能广播。alphaArray、betaArray 为连续
FLOAT32 标量数组，arch22 支持 Host 或 Device 存储；Device 标量在句柄
stream 上执行时读取。arch35 标量布局仍按其已有实现的 Host 约束。

所有矩阵为列主序，支持 N/T/C（实数 C 等同 T）和合法 leading dimension
padding。m/n/k、groupCount、groupSize 非负。dtype 校验后，groupCount=0
直接成功；groupCount>0 则校验 Host 数组与所有组参数，然后在实例总数>0
时要求三个设备指针数组非空。空 m/n 或 groupSize 的组不执行计算。

k=0 或 alpha=0 时不读取 A/B 矩阵与对应单个指针；beta=1 保持 C，beta=0
直接写零，其余值缩放 C。正常 GEMM 的 beta=0 也不读取旧 C。结果仅覆盖
C 的逻辑 m×n 区域，padding 保持不变。异步调用使用 handle 绑定的 stream；
读回结果或释放输入前须同步该 stream。工作区由 handle 管理；用户提供的
工作区不足时返回 `ACLBLAS_STATUS_ALLOC_FAILED`。

两种架构的非法 transa/transb 均返回 `ACLBLAS_STATUS_INVALID_ENUM`，
与公共 CPU 参考实现及 A2/A3 任务书 §2.4/§2.5 一致。

返回值包含 SUCCESS、HANDLE_IS_NULLPTR、NOT_SUPPORTED、INVALID_ENUM、
INVALID_VALUE、ALLOC_FAILED；异步设备错误还须通过 stream 同步结果检查。
裸数组 API 不携带数组长度，CSV 长度错误由测试加载器单独检测。

arch22 的稠密小矩阵在 Host α=1、β=0、m/ldc 满足直接写出约束时使用 Cube，
并在同 stream 的 AIV 检查 A/B 指数位；存在极大值、Inf 或 NaN 时按 Netlib
逐K顺序重算该矩阵。检测与重算都属于算子执行，并在同一 stream 中顺序完成。
其他小矩阵保留 AIV 路径；K 以1024分段，尾块仅写有效逻辑元素。
