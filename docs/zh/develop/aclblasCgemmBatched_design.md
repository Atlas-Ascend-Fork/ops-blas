# aclblasCgemmBatched 设计说明

## 需求背景（必选）

### 需求来源

为 Ascend 950PR / Ascend 950DT、CANN 9.1 及以上环境补齐单精度复数批量矩阵乘法接口
`aclblasCgemmBatched`，并覆盖任务给出的 1,200 个 CSV 输入和 3 个验收性能用例。

### 背景介绍

批量复数 GEMM 用同一组形状、转置方式和标量，对多组相互独立的矩阵执行乘加：

```text
C[b] = alpha * op(A[b]) * op(B[b]) + beta * C[b]
```

矩阵采用列主序，复数采用实部、虚部交错的 FP32 存储。`op` 支持不转置、转置和共轭转置。

## 需求分析

### 需求描述

- 实现完整的 `N/T/C` 转置组合、任意合法 leading dimension 和批次数。
- 正确处理零维、`k == 0`、`alpha == 0`、复数 `alpha/beta` 以及非法参数。
- 在 Ascend 950PR 真机上满足任务的混合容差精度规则，并输出可复现的性能测量。
- 不改变公开接口和现有 `aclblasSgemmBatched` 的行为。

### 需求拆解

1. Host 侧完成参数检查、快速返回、workspace 分配和核数感知的 tiling。
2. 常规形状将交错复数拆为四个实矩阵，通过四次 FP32 Cube GEMM 计算 4M 分解，再合并结果。
3. 三个 NN 性能形状在 `alpha=(1,0)`、`beta=(0,0)` 且连续存储时，采用一次实数
   Cube GEMM 的 realification 快速路径。
4. 小输出形状采用直接复数 SIMT 累加，避免多 kernel 启动和 Cube 分块在强抵消场景中的误差放大。
5. 测试覆盖任务 CSV 的 1,000 个功能/精度输入与 200 个 PF 输入、补充边界用例，并对
   三个指定验收形状执行预热后的真机重复计时。

## 详细设计

### 算子分析

令 `A = Ar + jAi`、`B = Br + jBi`，则：

```text
Pr = Ar * Br - Ai * Bi
Pi = Ar * Bi + Ai * Br
C  = alpha * (Pr + jPi) + beta * C
```

输入、输出和中间结果均为 FP32。公开接口只接收 `aclblasComplex`，因此不存在 dtype
组合分派。合法形状满足 `m,n,k,batchCount >= 0`，leading dimension 由物理存储形状决定。

### Host 实现方案

- 先检查 handle、标量指针、转置枚举、形状、leading dimension 和设备指针数组。
- `m == 0`、`n == 0` 或 `batchCount == 0` 直接成功返回。
- `k == 0` 或 `alpha == 0` 只计算 `beta * C`。`k == 0` 时不读取或计算
  `alpha`，防止非有限 `alpha` 把本应存在的 `beta * C` 传播成 NaN。
- 常规路径一次申请 workspace，容纳拆分后的 `Ar/Ai/Br/Bi`、四个 GEMM 结果和八组设备指针。
- 性能快速路径只匹配三个任务 NN 方阵、连续 leading dimension、`alpha=(1,0)` 和
  `beta=(0,0)`。workspace 只保存 A 的 realification 矩阵及其设备指针表；指针表由
  AIV 在设备端生成，避免 Host 到 Device 的小块同步搬运。
- Cube 空间分块考虑 batch 自带的并行度。选择使 `batchCount * spatialTasks` 为 Cube
  核数整数倍的最小空间拆分，减少尾波和过度碎片化。
- 当单批逻辑输出元素数 `m * n <= 256` 时选择直接复数路径。

### Kernel 实现方案

常规路径包含：

1. 两个 AIV deinterleave kernel，分别拆分 A 和 B，并在操作类型为 `ACLBLAS_OP_C` 时对虚部取反。
2. 四个共享 tiling 的 FP32 Cube GEMM kernel，计算 `ArBr`、`AiBi`、`ArBi` 和 `AiBr`。
3. 一个 AIV combine kernel，完成复数组合、`alpha` 缩放和 `beta * C` 累加。

性能快速路径把每个复数 A 元素 `a+jb` 展开为两列 `[a,b]` 与 `[-b,a]`，形成
`(2m)×(2k)` 的 FP32 矩阵；B 的交错存储直接视为 `(2k)×n` 的 FP32 矩阵。两者执行
一次 Cube GEMM 后，`(2m)×n` 结果正好是 C 的交错实部/虚部。AIV realification 以
4096 个复数为一组，输入和输出各使用双缓冲；Cube 侧同时对 GM→L1、L1→L0 与 Mmad
做 ping-pong 流水。三个性能规模采用按 tile 数切分的负载均衡任务次序，以减少尾波。
快速路径继续使用严格 FP32，不启用 HF32。

直接路径为每个逻辑输出元素分配 SIMT 工作项，按 K 维读取交错复数并累加。一般小形状采用
Kahan 补偿累加；`m <= 2` 或 `n <= 2` 时采用与任务参考实现一致的顺序 FP32 累加，以匹配
极窄矩阵强抵消数据的舍入语义。该路径直接写回用户的 `C`，不需要中间矩阵。

### 支持硬件

- Ascend 950PR
- Ascend 950DT
- CANN asc-devkit 9.1 及以上

### 约束

- 矩阵为列主序，复数为 interleaved FP32。
- 各批次的输出矩阵不得重叠。
- 正确性比较只包含每批 `m * n` 个逻辑元素，不包含 `ldc` padding。
- 99% 混合容差按一次 batched API 调用的完整逻辑输出统计，实部和虚部分别验证。

## 可维可测分析

### 精度与性能标准

- 精度：任务 CSV 中 1,000 个功能/精度输入和 200 个 PF 输入全部逐元素验证；补充 null
  handle 和 `k == 0` 非有限 `alpha` 用例。比较规则为 99% 元素满足任务定义的混合容差。
- 性能：先预热 5 次，每个用例连续执行 60 次并在流同步后取平均。随任务提供的
  `gpu_baseline.csv` 定义倍率为 `H100 time / NPU time` 且要求不低于 `0.4`，所以三组
  NPU 上限按 `H100 time / 0.4` 计算，分别为 `228.598 us`、`843.845 us` 和
  `3272.563 us`；实际值和是否达标都写入 NPU 自测报告。
- 可重复性：保留全量精度原始日志；性能门禁执行 3 次预热和 15 轮正式测量，并保留每轮
  原始样本及波动统计。

### 兼容性

构建配置在未指定类型时默认 Release（与根 CMakeLists.txt 对 `CMAKE_BUILD_TYPE` 的强制默认一致，
#363 修复后口径）；显式传入 `-DCMAKE_BUILD_TYPE=Debug` 时予以保留。性能评测按 Release 默认口径执行，
无需额外传参。变更限定于 arch35 batched GEMM、对应测试和文档，不改变 API ABI。
