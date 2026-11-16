# scnrm2 测试用例及测试指导参考

> **任务书**: 社区任务提供的外部材料，未随本仓库分发
> **适配硬件**: Ascend 950PR
> **CANN 版本**: 9.1.0
> **数据类型**: complex64（单精度复数）输入，FLOAT32 实数标量输出
> **内存预算**: 单用例 host 侧 ≤ 4GB（设计选择，非硬件限制）

## 算子说明

**接口**: `aclblasScnrm2`
**公式**: `result = ||x||_2 = sqrt(Σ (|real(x[k])|² + |imag(x[k])|²))`（i = 1..n，k = 1+(i-1)*incx）
**输出**: 单个 float 标量（L2 范数），精度验证为**标量比对**（单值相对/绝对误差），非逐元素矩阵比对
**参数**: n ≥ 0（n <= 0 为合法 quick return，result = 0.0f）；incx ≠ 0（可正可负）
**维度**: x 逻辑一维 [n]，物理长度 1+(n-1)*|incx|

## 用例规模

仓内 [`arch35/scnrm2_test.csv`](arch35/scnrm2_test.csv) 保存 **1200 条** CSV 用例：

- 精度：1000 条，覆盖基础组合 × 尺寸扫描 × 步长组合 × 填充模式 × 边界负向
- 性能/内存：200 条（TC_PF 前缀），前 3 条与任务书 §3.3 精确匹配的典型 case（n=1048576/2097152/4194304，incx=1）

尺寸范围 [0, 1048576]（精度）/ [1, 4194304]（性能），含 0、1、小质数、2 的幂及 2 的幂 ±1、非对齐值。性能用例一律连续访存（incx=1，生成脚本末尾带断言自检）；步长等非连续场景属功能维度，由精度用例 TC_INC/TC_EX 覆盖。

原始任务的 `gen_csv.py` 和 `gpu_baseline.csv` 未随本目录分发。仓内测试直接读取上述
CSV；性能验收需要另行取得与该 CSV 对应的 200 条权威 GPU 基线，并通过 `--baseline`
指定文件路径。

## 用例分类

| 类别 | 前缀 | 条数 | 说明 |
|------|------|------|------|
| L0 基础 | TC_L0 | 8 | 小尺寸（n=1/8）× 基础步长（±1/±2） |
| L1 尺寸 | TC_SQ | 38 | 38 种尺寸（1→1048576，含质数/边界/非对齐），incx=1 |
| L2 步长 | TC_INC | 18 | incx ∈ ±1/±2/±3 × n ∈ {16,64,256} |
| L5 填充 | TC_FL | 12 | 均匀随机/全零/交替/极端值/Inf/NaN × 2 尺寸 |
| L6 边界 | TC_ED | 10 | n=0 quick return（×4 步长）、负 n quick return（×2）、空指针 x/result、incx=0（负向期望 INVALID_VALUE）、n=0+空指针组合 |
| EX 扩展 | TC_EX | 914 | 尺寸 × 步长 × 填充的确定性采样（扩展精度条数的主力类别） |
| PF 性能 | TC_PF | 200 | 3 条任务书典型 case + 小尺寸 + 规模扫描 + 特殊填充 + 混合（均为连续访存，incx=1） |

> 仓内性能验收脚本要求恰好 200 条性能用例及 200 条基线；本目录不提供扩展条数的生成命令。

> 边界语义说明：n <= 0（含负 n）为合法 quick return（result = 0.0f，期望 `ACLBLAS_STATUS_SUCCESS`），对齐 cuBLAS 文档与仓内同族接口 aclblasSnrm2/Snrm2Ex 口径；incx = 0、n > 0 时 x/result 空指针为负向用例（期望 `ACLBLAS_STATUS_INVALID_VALUE`）。handle 空指针（`ACLBLAS_STATUS_HANDLE_IS_NULLPTR`）由测试工程层面覆盖，不在 CSV 中表达。

## 精度验收

仓内直接编译与运行（先加载 CANN 环境）：

```bash
bash build.sh --ops=scnrm2 --soc=ascend950 --run
```

`blas/CMakeLists.txt` 自动收集 `arch35` 下的 Host 和 Kernel 源文件；
`test/nrm2/scnrm2/CMakeLists.txt` 注册 `scnrm2_test` GTest 目标，无需重复添加源码列表。
根 `CMakeLists.txt` 在包含 Scnrm2 arch35 源码时链接 `opapi` 和 `nnopbase`，提供 Norm 和描述符 API。
测试包装器通过 `aclblasGetStream` 获取当前流，仅同步该流，失败时不读回无效输出。
额外回归用例 `SmallWorkspaceFallback` 检查小工作区下确实完成计算，
`InsufficientWorkspacePreservesHostResult` 检查工作区不足的错误码和 Host 结果保护。
`NumericalBoundaries` 覆盖连续/负步长下的大有限值、小有限值、Inf 和 NaN。
`OffsetInputAndOutput` 覆盖输入偏移一个复数、输出偏移一个 float 的非 32 字节对齐地址。
`LargeUnalignedFallback` 在 n=1048576、小工作区触发自定义 kernel 时复现核间非对齐起始地址，
覆盖整块 tile 的 `DataCopyPad` 路径。
`NormalDistributionPrecision` 使用固定 seed 20260918 与确定性 Box-Muller 变换生成标准正态
实部/虚部，覆盖 n=1/64/4096/65536 与 incx=1/-2，补足权威 CSV uniform 随机分布之外的正态输入。
权威 `scnrm2_test.csv` 是不可修改的评测输入，其中 `RANDOM_NORM_5_5` 随机填充实际由权威
`gen_csv.py` 按 uniform 分布生成，因此仓内不直接覆盖该 CSV。`MixedDistributionPrecision`
在保留该权威 uniform CSV 的基础上，用固定 seed 20260919 确定性生成 exactly 500 个
uniform[-5,5] 向量 case 和 500 个 normal 向量 case；normal 的 μ∈[-5,5]、σ∈[0.1,2]，
实部/虚部使用独立随机流，两组各覆盖 n=1/17/64/257/1024/4096 与 incx=1/-2/3/-4，
并在测试内断言 500/500 数量、分布参数边界和正负步长覆盖。该专项用于补足任务书字面
50% uniform + 50% normal 要求，不改变权威 CSV 的验收语义。

Golden 使用 Netlib CBLAS 复数范数：将原始 FLOAT32 分量无损提升为 double，调用
`cblas_dznrm2`，再将结果转回 FLOAT32。这样保留 CBLAS 的缩放归约与特殊值语义，
同时避免 `cblas_scnrm2` 在长向量上串行 FP32 累加误差超过验收的 32 ULP 限制。
精度阈值不变，负步长按照任务书允许的绝对步长等价计算。

仅运行精度用例时，在仓库根目录编译并直接执行 GTest 二进制，使用过滤器排除
`TC_PF_` 前缀的性能用例。本目录未提供 `verify_accuracy.py`：

```bash
bash build.sh --ops=scnrm2 --soc=ascend950
./build/test/nrm2/scnrm2/scnrm2_test --gtest_filter='-*TC_PF*'
# 使用已编译的二进制在设备 1 上运行精度用例
BLAS_DEVICE_ID=1 ./build/test/nrm2/scnrm2/scnrm2_test --gtest_filter='-*TC_PF*'
# 仅运行 L0 基础用例
./build/test/nrm2/scnrm2/scnrm2_test --gtest_filter='*TC_L0*'
```

### 精度阈值

测试通过 `applyMixedTolerance(cfg, ACL_FLOAT, golden)` 设置 FLOAT32 混合容差，
默认值见 [`test/frame/mixed_tolerance.h`](../../frame/mixed_tolerance.h)：

- atol = rtol = 2⁻¹³ = 0.0001220703125
- matched_ratio ≥ 0.99，且每个元素的绝对误差 ≤ max(1e-2, 32×ULP(golden))。
  本算子输出只有一个标量，因此该标量必须满足混合容差判定及绝对误差上限。

参考: https://gitcode.com/cann/opbase/blob/master/docs/zh/ops_precision_standard/mixed_tolerance_standard.md

## 性能验收

仓内 `verify_performance.py` 以权威 wrapper 的调用界面为基础，修正结果解析：
使用严格模式的 `[STRICT_PERF] average_us` 事件耗时，而不是 GTest 整 case 耗时；
baseline 按权威 CSV 行序与 `TC_PF_1001`～`TC_PF_1200` 关联，避免重复 `(n, incx)` 错配。
执行 TC_PF 用例采集 NPU 耗时，与 `gpu_baseline.csv` 关联比对：

```bash
python test/nrm2/scnrm2/verify_performance.py --repo /path/to/ops-blas \
  --soc ascend950 --strict --baseline /path/to/gpu_baseline.csv --timeout 3600
python test/nrm2/scnrm2/verify_performance.py --repo /path/to/ops-blas \
  --soc ascend950 --strict --skip-build --device 1 \
  --baseline /path/to/gpu_baseline.csv --output /path/to/results.csv
```

- **判定口径**：NPU 平均单次耗时 ≤ 标杆耗时 / 倍率（标杆耗时为`gpu_baseline.csv` 的 `gpu_ms` 列，比较时换算为 us）；仅当 200 条全部有基线、warmup=20、samples=100 且 ratio≥0.4 时退出 0。
- 该 wrapper 会设置 `SCNRM2_STRICT_PERF=1`；若未显式传 `--strict`，也支持用同名环境变量开启。

## 补充说明
- 代码上库时需要同步提交测试工程代码，可能需要新增、复用或者改造当前库上test目录下测试工程，详细要求以仓库最新代码规范和贡献指南要求为准

- 部分特殊场景下自动生成的case可能导致标杆出现异常行为使测试行为无意义，此时可根据实际场景过滤掉或者修改这些case并给出相应的说明
