# aclblasCgemmBatched 真机测试

本目录包含 `aclblasCgemmBatched`（complex64 批量矩阵乘，列主序）的 CSV 驱动精度测试、
边界回归和固定性能契约，按架构分目录：

- `arch35/`：Ascend 950PR / 950DT
- `arch22/`：Atlas A2 训练/推理系列（Atlas 800T A2、Atlas 800I A2/A3，910B3 实测）。
  完整复现说明（环境/构建/精度/性能/结果核对/FAQ）见 MR 描述中的复现与验收说明

两个架构共用 `cgemm_batched_param.h` / `cgemm_batched_golden.h` 与相同的 CSV 格式，
各 1,200 个用例：前 1,000 个功能/精度用例覆盖基础转置组合、尺寸扫描、标量、batch、
leading dimension、输入分布（含 Inf/NaN）、边界负向和扩展组合；后 200 个 `TC_PF`
行保留任务性能输入。构建时 CMake 按 `--soc` 对应的 `SOC_ARCH_DIRS` 自动选择
arch 目录中的测试源（`ascend910b3` → `arch22`，`ascend950` → `arch35`）。

| 精度类别 | 数量 |
|---|---:|
| `TC_L0` 基础转置组合 | 18 |
| `TC_SQ` 尺寸扫描 | 23 |
| `TC_AB` alpha/beta | 72 |
| `TC_BC` batch 扫描 | 13 |
| `TC_LD` leading dimension/padding | 12 |
| `TC_FL` 输入分布与 Inf/NaN | 6 |
| `TC_CV` 中等形状覆盖 | 72 |
| `TC_ED` 边界与负向 | 29 |
| `TC_EX` 扩展组合 | 755 |
| 合计 | 1,000 |

## 构建

在 CANN 9.1 环境中使用仓库根目录的构建脚本（推荐，自动设置工具链并编译算子与测试）：

```bash
# Atlas A2/A3（arch22）
bash build.sh --soc=ascend910b3 --ops=cgemm_batched
# Ascend 950PR（arch35）
bash build.sh --soc=ascend950 --ops=cgemm_batched
```

测试程序位于：

```text
build/test/gemm_batched/cgemm_batched/cgemm_batched_test
```

## CI 冒烟与全量验收（arch22）

参考 `csyr2k` 的默认冒烟、显式全量机制，A2/A3 默认加载 245 条基础 CSV 用例及全部
669 条专项回归，保留转置、标量、步长、负向、Inf/NaN 和历史缺陷覆盖。
755 条 `TC_EX` 扩展组合及 200 条 `TC_PF` 输入仅在全量模式加载，减少 CI 的 CPU golden
计算与内存开销。原始 CSV、golden 和精度阈值不变；冒烟通过不能代替完整验收。

设置 `CGEMM_BATCHED_FULL_CSV=1` 恢复全部 1,869 条 CSV 输入。
`CGEMM_BATCHED_RUN_ALL_PERF=1` 也会加载全量，避免性能测试被截断。
该调整不影响 arch35。

## 正确性与复测（arch22，910B3）

```bash
# 原始非 PF 输入、669 条补充回归和 5 项专项测试（不执行 CsvPerformance）
CGEMM_BATCHED_FULL_CSV=1 ./build/test/gemm_batched/cgemm_batched/cgemm_batched_test \
  --gtest_filter='*CsvDriven/*:CgemmBatchedTest.NullHandle:CgemmBatchedTest.KZeroIgnoresNonFiniteAlpha:CgemmBatchedTest.OffsetPointersAndStreamOrdering:CgemmBatchedTest.FusedMultiplyAddReference:CgemmBatchedTest.ConcurrentStreamsIndependentWorkspaces-*TC_PF*'

# 后 200 条 TC_PF 输入的逐元素正确性（CPU golden，耗时较长）
CGEMM_BATCHED_FULL_CSV=1 ./build/test/gemm_batched/cgemm_batched/cgemm_batched_test \
  --gtest_filter='CgemmBatched/CgemmBatchedTest.CsvDriven/*TC_PF_*'
```

两条命令合计覆盖原始 1,200 个 CSV 输入、669 项补充回归和 5 项专项，共 1,874 项。golden 由 cblas（Netlib BLAS 复数实现
`cblas_cgemm`）逐批生成；验证将一次 batched 调用的全部逻辑输出聚合，实部和虚部分别
应用 99% 混合容差（atol=2⁻¹⁶、rtol=2⁻¹⁰ 生态标准口径），`ldc` padding 单独按位检查；
`alpha = (0,0)` 用例走 `C[i] = beta*C[i]` 精确路径位级比对。`TC_PF` 的 GTest 用时包含
输入生成、CPU golden 和逐元素比对，只用于正确性，不作为 NPU kernel 性能证据。

### 特殊值参考语义

非零乘积路径采用完整复数乘法的无穷恢复规则，与 `ctrsv` / `cgbmv` 一致。
`FusedMultiplyAddReference` 同时对照编译器 `__mulsc3` 检查 28,561 组边界组合。
不同 BLAS 构建对 Inf/NaN 的处理可能不同：部分 limited-range 构建将本应恢复的无穷保留为
NaN；OpenBLAS 的累加顺序也可能不同。复现 CI 时应使用与 CI 一致的参考 BLAS，记录其
路径及版本，不能仅凭本机不同参考库的结果声明 CI 通过。测试仍调用原始 `cblas_cgemm`，
不放宽特殊值检查或精度阈值。

## NPU 性能（arch22，910B3）

```bash
CGEMM_BATCHED_FULL_CSV=1 ./build/test/gemm_batched/cgemm_batched/cgemm_batched_test \
  --gtest_filter='CgemmBatchedTest.TC_PF_PerformanceContract'
```

每个形状预热 5 次、计时 60 次，设备流同步后输出 `average_us`。验收上限为任务书 §3.3
的标杆耗时（COMPLEX64，Avg time）：

| 用例 | 标杆耗时（上限） |
|---|---:|
| 256x256x256, batch=32, NN | 342.44 us |
| 512x512x512, batch=16, NN | 1462.32 us |
| 1024x1024x1024, batch=8, NN | 5018.64 us |
| 2048x2048x2048, batch=4, NN | 18032.75 us |

`TC_PF_*` 的 CSV 性能扫描（更多形状与转置组合）默认跳过，需要时置
`CGEMM_BATCHED_RUN_ALL_PERF=1` 开启。性能用例在超过标杆时会失败，这是预期的契约
行为；CPU、模拟器或 mock 结果不能作为性能证据。


## arch22 严格验收与补充回归

构建和复验步骤见 MR 描述中的复现与验收说明。
任务原始 CSV 的 1,200 行保持原样，另有 `arch22/cgemm_batched_regression.csv` 的
669 条回归，覆盖九种转置、不同平面大小、跨 K=8192 分块、非有限 alpha 的 k=0、
beta=0 与 NaN 旧 C、beta=1 保持原值、非规格化数的位精确缩放，以及各 90 条均匀/正态分布及对应复数标量。
新增的 235 条回归覆盖 4864 元素合并分段边界、长行、多分段、九种转置的 Inf/NaN、有限数溢出、非零 beta 及 K=8193，并覆盖非单位alpha与极端有限A/B导致的参考中间量溢出。
`OffsetPointersAndStreamOrdering` 检查 8/16/24 字节偏移矩阵地址、乱序批地址、
两次连续 stream 提交、输入不变、C 前后保护区和 padding、用户工作区强制 2+1 分轮。

所有批次的实部、虚部都遍历比较，按 case 汇总 99% 混合容差；每批的最小匹配率
另行输出供排查，绝对误差上限逐元素检查。NaN 只匹配 NaN，Inf 必须符号相同；
alpha=0/k=0 时有限结果按位比较（包含正负零），NaN payload 不作跨平台约束。
C padding 全部按位检查。CPU golden 使用 Netlib CBLAS，按不重叠输出列并行执行
以减少大例时间，每个元素的归约次序不变。

精度与性能直接使用本目录的 GTest 程序执行，命令见上文及
MR 描述中的复现与验收说明。自测采集与汇总工具保存在仓外，不随源码提交。
附带 `arch22/gpu_baseline.csv` 保留任务提供的 200 条 GPU 基线，按
transA/transB/m/n/k/batchCount 关联，以 `gpu_ms*1000/0.8` 为参考耗时上限逐项判定。
四个任务书门槛与全部 200 条扩展参考比较分别报告，不能用 GTest 执行成功代替性能达标。

## msprof kernel 性能补充统计

任务包测试指导允许用msprof采集精确kernel耗时。前述`average_us`是连续提交和最终同步的墙钟平均值，不能与kernel执行时间混称。两种口径分别保留，不改GPU参考值。

使用仓外工具对完整200项采集核验XML、host提交/同步分组与全部device任务，剔除5次预热后对60次调用取算术平均。单kernel用Task Duration；固定多kernel序列用首kernel开始到末kernel结束的跨度，包含调用内间隔。多stream或无法归属的数据报错，不遗漏计算阶段。

另统计第 6–15 次与第 11–60 次调用的均值，对应评审主窗口和稳态窗口。
每轮三个窗口都须通过；中位数不能覆盖某一轮的失败。
GTest 预热后有一次同步，关键形状仍需连续提交的独立探针复核。
完整口径和环境要求见 MR 描述中的复现与验收说明。


## 多种子随机输入复验（arch22）

性能缓冲现在按CSV的`a_fill/b_fill/c_fill`生成（200项均为实部/虚部独立均匀分布[-5,5]）。
历史P100及更早的`CsvPerformance`使用全零缓冲，其结果只代表零输入对照；
`CsvDriven`精度测试一直按CSV生成输入。构造随机数、输入指纹和H2D均在预热及计时之前。

`CGEMM_BATCHED_TEST_SEED_OFFSET`为十进制uint32，默认0；每个基础种子加偏移后按uint32回绕，
再沿用各批次A/B/C的+0/+1/+2及虚部+7919规则。固定值、Inf/NaN和负向用例的填充语义不变。
种子偏移也作用于调用共用生成器的指针/并发回归；固定手工构造数据的专项不受影响。
每个性能用例记录整个A/B/C输入的FNV-1a 64位指纹，精度XML记录基础种子及偏移。

多种子复验时设置 `CGEMM_BATCHED_TEST_SEED_OFFSET`，按复现说明直接运行 GTest。
性能采集使用 `--task-time=on --ai-core=off`，不同时采集会扰动小 kernel 的流水线计数。
逐轮核对输入指纹、二进制及全部预热/正式调用，不去慢样本、不减启动常数、不取最快轮。
保持设备独占，性能阶段不与精度、编译或其他 NPU 任务并行。

运行结果、日志、XML 和 profiler 原始数据保存于仓外目录，不提交到代码分支。
A2 与 A3 的验收结果分别保存，不能以某一设备的通过结果替代另一设备复验。


### 完整性与内存检查门禁

`arch22` 的主 CSV 固定 1200 项（245 基础、755 扩展、200 性能形状），
回归 CSV 固定 669 项，两者名称全局唯一。`CsvCaseCount` 在过滤前检查完整输入。
下列脚本要求新输出目录；任何失败、跳过、执行名称/数量不匹配均返回非零。

```bash
python test/gemm_batched/cgemm_batched/run_self_test.py --binary build/test/gemm_batched/cgemm_batched/cgemm_batched_test --output /tmp/cgemm-full
# 加 --smoke 运行基础与回归队列；默认 full 同时检查全部扩展与性能形状的精度。
```

内存检查须使用独立构建目录，并在正常 CANN 构建参数之外传入
`-DCMAKE_ASC_FLAGS="-g --cce-enable-sanitizer"`，然后执行：

```bash
python test/gemm_batched/cgemm_batched/run_memcheck.py --binary build-sanitize/test/gemm_batched/cgemm_batched/cgemm_batched_test --output /tmp/cgemm-memcheck
```

队列包括无效参数、零 K、大 alpha、偏移/非连续指针与受限 workspace 多轮复用，
以及 8193 长 K 的分块累加。检查日志必须存在正数且配对的 kernel 检查记录；
缺失插桩、工具告警、精度失败或执行不完整都不能通过。
结果分别写入 `self_test_summary.json` 和 `memcheck_summary.json`。
