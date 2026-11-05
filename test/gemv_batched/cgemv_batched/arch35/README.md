# aclblasCgemvBatched 测试说明 (arch35 / Ascend 950PR)

> **算子**: `aclblasCgemvBatched` — 单精度复数批量矩阵-向量乘
> **适配硬件**: Ascend 950PR (arch35)
> **CANN 版本**: 9.1.0
> **数据类型**: complex64 (单精度复数，实部/虚部交错 float32)
> **接口形态**: Aarray/xarray/yarray 均为指针数组 (与 cuBLAS cublasCgemvBatched 一致)

## 1. 公式

对每个 batch b (0 ≤ b < batchCount):

```
y[b] = alpha * op(A[b]) * x[b] + beta * y[b]
```

- `trans=N`: op(A)=A, A 为 m×n, x 长 n, y 长 m
- `trans=T`: op(A)=A^T, x 长 m, y 长 n
- `trans=C`: op(A)=A^H (共轭转置), x 长 m, y 长 n

## 2. 文件结构

| 文件 | 作用 |
|------|------|
| `cgemv_batched_test.cpp` | CSV 驱动 GTest 主程序 |
| `cgemv_batched_test.csv` | 1200 条测试用例 (1000 精度 + 200 性能) |
| `gpu_baseline.csv` | 200 条 GPU H100 性能基线 |
| `cgemv_batched_param.h` | CSV 列解析为 CgemvBatchedParam 结构体 |
| `cgemv_batched_golden.h` | CPU golden 参考 (复数 cgemv 逐批计算) |
| `cgemv_batched_npu_wrapper.h` | NPU wrapper: 连续缓冲→指针数组→调用 API |
| `CMakeLists.txt` | 构建配置 |

## 3. 测试用例分类

| 类别 | 前缀 | 条数 | 说明 |
|------|------|------|------|
| L0 基础 | TC_L0 | 6 | trans 3 × 尺寸 (4,8) × batchCount=2 |
| L1 方阵 | TC_SQ | 81 | 27 种尺寸 × trans 3 × batchCount=2 |
| L1 矩形 | TC_MN | 24 | 8 组 m≠n × trans 3 |
| L2 标量 | TC_AB | 24 | 8 组 alpha/beta × trans 3 |
| L4 前导维 | TC_LD | 8 | lda=m+4 padding × trans(N/T) × 4 尺寸 |
| L4 步长 | TC_INC | 72 | incx×incy ∈ ±1/±2/±3 × trans(N/T) |
| L5 填充 | TC_FL | 18 | 均匀/全零/交替/极端/Inf/NaN |
| L5b 覆盖 | TC_CV | 35 | batchCount 扫描 + 中等尺寸 |
| L6 边界 | TC_ED | 28 | 非正维度/非法 lda/零步长/空指针/非法枚举 |
| EX 扩展 | TC_EX | 704 | 确定性采样组合 |
| PF 性能 | TC_PF | 200 | 4 条典型 case + 小尺寸大 batch + 随机组合 |

## 4. 精度阈值

COMPLEX64 按 FLOAT32 分量判定 (实部/虚部独立):

- atol = 2⁻¹³ (1.22e-4), rtol = 2⁻¹³ (1.22e-4)
- matched_ratio ≥ 0.99 且 max_abs_error ≤ 1e-2 或 32×ULP

## 5. 测试步骤

### 步骤一: 编译算子与测试

```bash
cd /path/to/ops-blas
bash build.sh --soc=ascend950 --ops=cgemv_batched
```

### 步骤二: 精度测试

```bash
# 直接运行 GTest (排除性能用例，仅测精度)
./build/test/gemv_batched/cgemv_batched/cgemv_batched_test \
  --gtest_filter=-*TC_PF*
```

### 步骤三: 性能测试

测试代码已内置 aclrtEvent 事件计时 (5 warmup + 10 samples)，直接运行 GTest 性能用例即可输出逐 case 的 NPU 耗时、GPU 标杆耗时、speedup 与 PASS/SLOWER 判定，末尾汇总 total/PASS/SLOWER/mean speedup。

```bash
./build/test/gemv_batched/cgemv_batched/cgemv_batched_test \
  --gtest_filter=*TC_PF*
```

> 注: 任务书 §3.3 要求用 msprof op 采集 task duration，但本环境 (Ascend 950PR / CANN 9.1.0) msprof op 不可用 (Profiling channel start failed，硬件/驱动限制)。aclrtEvent 事件计时直接测量 kernel 执行耗时，与 msprof op 采集的 task duration 等价，已作为替代方案内置于测试代码。

### 步骤四: 性能判定

- NPU 平均单次耗时 ≤ GPU H100 标杆耗时 / 0.4 (即不高于标杆 2.5 倍)
- warmup 5 次后有效采样 10 次取平均
- `gpu_baseline.csv` 的 `gpu_ms` 列为 H100 实测基线

## 6. 接口修正说明

`include/cann_ops_blas.h` 中 `aclblasCgemvBatched` 声明已从单指针形态修正为指针数组形态:

```cpp
// 修正前 (单指针):
aclblasStatus_t aclblasCgemvBatched(..., const aclblasComplex* A, int lda,
    const aclblasComplex* x, int incx, ..., aclblasComplex* y, int incy, int batchCount);

// 修正后 (指针数组, 与 aclblasSgemvBatched / cuBLAS 一致):
aclblasStatus_t aclblasCgemvBatched(..., const aclblasComplex* const Aarray[], int lda,
    const aclblasComplex* const xarray[], int incx, ..., aclblasComplex* const yarray[], int incy, int batchCount);
```

同步修正了 `blas/gemv_batched/arch22/cgemv_batched_host.cpp` 的入参与批量循环。
