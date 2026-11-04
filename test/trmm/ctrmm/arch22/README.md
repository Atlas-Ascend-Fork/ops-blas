# aclblasCtrmm 测试（Atlas A2/A3，arch22）

环境：Atlas A2 910B3 或 A3，CANN 9.1.0，GTest，Netlib BLAS 的 CBLAS 接口。
请明确链接 Netlib，避免自动选中另一个 BLAS 实现。

Inf/NaN 回归须使用与验收一致的参考库构建。已确认的验收库来源为
`lapack-devel-3.10.0-5.oe2203sp4.aarch64`，封装库 `libnetlib_ctrmm_accept.so.1`
的 SHA256 为 `f7a40cc860792a6ef512ab3b75fa6a45802736d4baaea84084a459ba99626c40`。
仅同为 Netlib 3.10.0 不足以保证复数非有限值传播一致；Ubuntu 的构建可能不同。
若使用验收封装库，应按验收方式保留 `LD_PRELOAD`，并核对运行日志中的接管横幅。

```bash
source /usr/local/Ascend/cann/bin/setenv.bash
export LINUX_INCLUDE_PATH="$ASCEND_HOME_PATH/$(uname -m)-linux/include"
export EAGER_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64"
cmake -S . -B build-ctrmm \
  -DSOC_VERSION=ascend910b3 -DBUILD_TEST=ON -DTEST_NAMES=ctrmm \
  -DCANN_3RD_LIB_PATH=/path/to/parent-of-ops-tensor \
  -DREFBLAS_INCLUDE_DIR=/path/to/netlib/include \
  -DREFBLAS_LIB=/path/to/netlib/libblas.so
cmake --build build-ctrmm --target ctrmm_test -j8
build-ctrmm/test/trmm/ctrmm/ctrmm_test \
  --gtest_filter='*-*TC_PF*' --gtest_output=xml:ctrmm-accuracy.xml
```

以下文件路径相对于 `test/trmm/ctrmm/`。A2/A3 与 950 复用公共参数头文件
`ctrmm_param.h` 和 golden 头文件 `ctrmm_golden.h`。A2/A3 使用其中的
`CtrmmGolden` 并行接口，950 保留原有 `aclblasCtrmm_cpu` 接口；两者的输出
padding 处理和负向用例调用约定保持不变。
测试入口和 GTest 注册位于 `arch22/ctrmm_test.cpp`；fixture、执行与精度校验位于
`arch22/ctrmm_arch22_test.h`，补充用例的数据生成与模式组合位于
`arch22/ctrmm_test_inputs.h`。拆分不改变测试名称、CSV 路径和精度判据。

较大输入的 CPU golden 复用仓内 `blas_test::ForEachChunk` 并行调用同一个
`cblas_ctrmm`：LEFT 按输出列分块，RIGHT 按输出行分块，保留完整三角矩阵与
每个输出元素的归约顺序。小规模计算仍串行执行。线程数由公共工具根据 CPU
亲和性选择，最多 32 个；可用 `BLAS_GOLDEN_THREADS=4` 限制并行度，或设置
`BLAS_GOLDEN_THREADS=1` 复核串行结果。该设置不改变用例选择、比较器或 NPU 计算。
并行优化不会统一不同参考库构建的 Inf/NaN 传播；仍需按上述要求核对参考库。

当前 PR 的 `arch22/ctrmm_test.csv` 包含 1199 行用例：任务精度用例 999 行、
TC_PF 性能用例 200 行；另有 24 项补充 GTest，全量执行共 1223 项。
因验收与 CI 参考库构建的 Inf/NaN 传播口径尚未统一，当前测试集移除了
`TC_FL_119`、`NonFiniteReferencePropagation` 和 `SensitiveBranchCombinations`。
这是测试范围调整，不代表对应特殊值差异已修复；历史完整测试报告仍按其对应版本解读。
TC_PF 的 200 行可单独执行
精度对照（不设置 CTRMM_PERF_ONLY），或者执行性能采样：

```bash
CTRMM_PERF_ONLY=1 CTRMM_EVENT_TIMING=1 \
  build-ctrmm/test/trmm/ctrmm/ctrmm_test --gtest_filter='*TC_PF_1001'
```

性能模式预热 5 次，采样 11 次；`CTRMM_PERF` 行记录全部 ACL Event 样本与平均值。
性能模式不做 CPU golden，也不应代替精度测试。msprof 采集时去掉
CTRMM_EVENT_TIMING，用 `--aic-mode=task-based --aic-metrics=PipeUtilization
--task-time=on`。每次正常Cube调用包含pack、cube、finish；q>256还包含repack，
剔除前 5 组后对后 11 组求和，另保留端到端跨度作为参考。

输出 `CTRMM_ACCURACY` 包含每个用例的实/虚部匹配比例、最大绝对误差和特殊值
不匹配数。按任务书使用atol=rtol=2^-13、匹配比例≥0.99，且最大绝对误差≤max(0.01,32*ULP(g_low))。g_low为最大误差位置的FP32 golden，同误差点取较小ULP。日志同时输出固定0.01超限数；标准冻结于opbase提交dcdd10634671a961b6aee09776d1aa20091e6386。
测试还检查输出 padding 和前后 guard，并用 NaN 填充禁止读取的 A 区域。
特殊值失败时自动输出前 32 个不匹配分量的行列坐标和 actual/golden；全部分量仍
参与原有判定，特殊值不匹配数仍须为 0、实虚部匹配比例仍须均达标。

额外 GTest 覆盖原地及不同 ldb/ldc、8 字节偏移、alpha=0 且 A/B=null、
小矩阵块边界与偏移（同时检查 alpha=1）、
q=64/65/100/128 与 r=513/641 的长条矩阵全部 24 种模式及原地/padding、部分重叠拒绝，以及
共享 workspace 的大 alpha / alpha=1 连续调用（中间不插入 Host 同步；分别检查敏感与普通调用）。
设计说明位于独立的cann-competitions仓库：`04_tasks/01_community-task-2026/tasklist/09-71-aclblasCtrmm-A2A3/Xztxzt/docs/design.md`。

内存采集独立运行，不与 profiler 同时执行：

```bash
CTRMM_MEMORY_ONLY=1 build-ctrmm/test/trmm/ctrmm/ctrmm_test --gtest_filter='*TC_PF*'
```

该模式每例执行一次，输出 `CTRMM_MEMORY`：矩阵字节数、guard 字节数、handle
实际 workspace 容量，以及分配前/同步后的设备空闲量。handle 跨用例复用，因此
后续小用例可能仍持有之前大用例扩容后的 workspace。设备空闲量为全局读数，
不应将其变化直接当成本算子的独占峰值。

精度复核新增 `NumericalRoutingBoundaries`（32及alpha实/虚分量8两侧/等值）、
`CancellationAndReductionLength`（16/64/128/257/1024/4096项相消和独立随机输入）、
`CertifiedAndOrderedBranches`（24模式、原地、低幅及极小分量）以及
`EmulatedFmaProducts`（普通和跨指数范围的对角复数乘法）。
完整4096性能用例的精度必须单独运行，不能用性能模式的执行成功代替：

```bash
unset CTRMM_PERF_ONLY CTRMM_MEMORY_ONLY CTRMM_EVENT_TIMING
build-ctrmm/test/trmm/ctrmm/ctrmm_test --gtest_filter='*TC_PF_1005'
```

当前实现用复数项耦合的FP32 Cube计算；q≤256采用分组规约，q>256逐复数项累加。分流同时考虑alpha的L1范数、有效分量幅度以及A实/虚系数的近共线程度；这些是风险指标，不是覆盖全部FP32输入的误差证明。敏感输入保留Netlib分支顺序，性能会明显下降。

补充24项GTest还覆盖独立幅度/规约长度/seed、重复舍入、128阶Small、576次短Cube分支组合，以及非恒定相消（180次）、小矩阵不同相位比例相消（72次）、短规约长条矩阵相消（72次）、幅度16的邻近值与规约长度边界（180次）。全部使用任务容差；`CTRMM_CHECK_EXACT=1`可附加零误差诊断，不属于默认验收标准。

最新精度、性能和历史失败证据必须按对应源文件和二进制版本解读。报告生成与原始profiler证据随社区任务材料提交，流程见[贡献指南](../../../../CONTRIBUTING.md)。

`PairedUniformGaussianDistributions`补充真正的正态分布：使用`std::normal_distribution`，与理论方差一致的均匀分布成对测试，各196次。覆盖8个短/中规约长度的24模式及1024/4096长规约；补充输入由测试直接构造，框架填充器保持不变，避免把`RANDOM_NORM`的名称误当作正态覆盖。

新增`ShortCancellationAlphaScale`覆盖60次短规约变体；`ShortCancellationSmallShapes`覆盖144次相邻短尺寸；`BatchedFmaDomainBoundaries`覆盖216次实际域界及其相邻值。`ApplicationLlmTriangularProducts`包含36个GDN形式和6个SSD形式的实数输入，`ApplicationComplexFactorProducts`包含36个复数因子输入。它们是可复现合成数据，不是真实模型张量或模型性能验收。输入范围与用途依据见设计文档1.2.1及5.1。

## CI 冒烟与完整回归

CI 按 PR 改动选择算子后，默认通过 `--smoke` 执行现有 CSV 抽样。
显式 CSV 增量名单优先保留，避免被抽样覆盖；`RUN_FLAGS=--no-smoke` 可关闭抽样。
`build.sh` 在家族展开和自动发现后按首次出现顺序去重；同时选择 `trmm` 与
`ctrmm` 时，CTRMM 测试程序只执行一次。
本算子的独立冒烟与完整回归可使用下列命令。

在仓库根目录执行以下命令。冒烟仍生成 CPU golden 并检查精度，
不要设置 `CTRMM_PERF_ONLY` 或 `CTRMM_MEMORY_ONLY` 来代替精度验证。

```bash
unset CTRMM_PERF_ONLY CTRMM_MEMORY_ONLY CTRMM_EVENT_TIMING
bash build.sh --ops=ctrmm --run --smoke  # 现有 CSV 抽样
bash build.sh --ops=ctrmm --run          # 完整 CSV 与补充测试
```

手动执行全仓完整回归可使用
`RUN_FLAGS=--no-smoke bash scripts/ci/run_example.sh all`。
