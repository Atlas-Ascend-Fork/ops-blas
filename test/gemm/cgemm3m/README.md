# aclblasCgemm3m（Atlas 950PR）测试

实现入口：`blas/gemm/arch35/cgemm3m_host.cpp`；公共声明：`include/cann_ops_blas.h`。
要求 CANN 9.1.0、Ascend950PR，使用 arch35 直接调用 Ascend C kernel。
共用既有 Cgemm kernel，并强制所有非退化标量场景执行三次实数矩阵乘法。
融合路径与非融合路径分别采用 P0/P1/P2 融合公式及标准 Ar×Br、Ai×Bi、
(Ar+Ai)×(Br+Bi) 公式；既有 `aclblasCgemm` 保持原 Gauss 公式。
具体表达式见 `blas/gemm/README.md`。

三乘法非融合路径联合选择输出 tile 与 1～4 个 K 分区，估算最忙核心的工作量和部分结果搬运成本；
小输出保留原有占用率策略。融合路径将长 K 点积限制为每段最多两个 L1 面板，
按固定顺序在片上合并，减少 FP32 长链累加误差。三乘法仅分配三份中间乘积空间。
这两项策略仅由输入尺寸和硬件容量决定，不按测试编号或随机种子分派。
细长矩阵在输出 tile 覆盖至少 3/4 Cube 核且 K 在缓冲容量约束内时，
直接分面板在片上准备复数输入并完成三乘法，省去全局三平面工作区和两次内核启动。
并行度不足时保留独立乘积/K 分区路径。

2048～4096 阶连续方阵、alpha=1/beta=0 使用较短的 128 项 K 分段，同时记录部分和轨迹。
结合输入行/列能量筛选 FP32 累加风险较高的输出，对选中位置并行生成三乘法项。
当前纠正重算只在 2048～4096 阶启用；每个向量 lane 先计算一段 K 项的部分和，
以前序段的前缀为起点重放该段，再扣除前缀，以保留重放时的 FP32 舍入行为。
同时保留不扣前缀的直接分段归约；若最大前缀幅度达到直接结果幅度的 32 倍，
采用直接结果以避免严重抵消时的前缀扣除误差。
分段归约是近似重算，精度由完整用例实测判据约束，
不承诺与 Netlib 逐位等价。另有定向用例检查正负 K 段抵消为零及保留非零残差。
筛选仅依赖当前输入和部分和，不读取参考结果、测试编号、随机种子或预存坐标。
额外存储由 handle 工作区统一管理，不扩写用户 C。所有新增准备、筛选、纠正内核均计入性能。
风险估计用于选择计算路径，并非任意输入的严格误差上界；精度仍由下述完整实测判据验证。

## 自验工具

`run_self_test.py`、`run_memcheck.py` 及两者共用的 `test_gate.py` 随测试工程提供，性能基线位于
`arch35/gpu_baseline.csv`。执行下文的 Netlib 准备和 msprof 汇总命令前，先将 PR #735
附件 `pr735_selftest_evidence_20261115.zip` 中的两个脚本解压到仓库根目录，保留原有相对路径。
从仓库根目录执行下文命令即可复现；无需下载或解压额外附件。

## 构建

从 ops-blas 根目录执行，先安装编译器、GTest、OpenBLAS/LAPACK 开发包：

```bash
# 将 CANN_HOME 改为本机 set_env.sh 所在的目录。
CANN_HOME=/usr/local/Ascend/cann-9.1.0
source "$CANN_HOME/set_env.sh"
CMAKE_INCLUDE_PATH=/usr/include/openblas CMAKE_LIBRARY_PATH=/usr/lib64 \
  bash build.sh --ops=cgemm3m,cgemm --soc=ascend950
```

`build.sh` 会重新建立 build 目录。后续只改源码时可使用：

```bash
export LINUX_INCLUDE_PATH="$ASCEND_HOME_PATH/$(uname -m)-linux/include"
export EAGER_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64"
cmake --build build --target cgemm3m_test cgemm_test -j8
```

## 功能与精度

`arch35/cgemm3m_test.csv` 是任务原始 1200 行用例：1000 条功能精度、200 条性能场景。
默认全部进行精度检查，每条非空成功用例从相同输入调用两次并逐字节检查确定性。
另含空句柄、空输出哨兵、k=0/alpha=0 且 A/B 为空、分段抵消的 4 个补充测试，
以及 3 个判据边界测试。
按本任务书 §3.2 的 FLOAT32 混合容差标准，实部和虚部分开检查。有限分量误差不超过
`(1+abs(golden))/8192` 才计入 matched；每个分量须同时满足 matched_ratio≥99% 和
max_abs_error≤max(0.01,32 ULP)，后者在最大误差点的 golden 计算。非有限值按任务书规则检查，
padding 必须逐字节不变。先前附加的 100% 逐点门禁高于任务标准，不能代替官方判据。
旧版精度和性能报告不能作为当前混合算法及最终源码的通过证据。
不使用 CSV 中旧框架的 mere/mare 字段替代任务精度要求。

验收参考库固定为 Netlib BLAS/CBLAS 3.10.0，来自 Reference-LAPACK 官方 v3.10.0 源包。
需 `gfortran`、CMake 和 C 编译器；脚本核对源包 SHA256，在指定目录独立构建，不替换系统库。
源码许可证位于下载源码根目录 LICENSE。构建采用 Netlib 官方 make.inc.example 的 Fortran 参数 `-O2 -frecursive`，不启用 fast-math。

```bash
bash test/gemm/cgemm3m/scripts/prepare_netlib.sh "$PWD/../reference"
CGEMM3M_REFERENCE=$(cd ../reference && pwd)
LD_PRELOAD="$CGEMM3M_REFERENCE/build/lib/libcblas.so $CGEMM3M_REFERENCE/build/lib/libblas.so" \
  ./build/test/gemm/cgemm3m/cgemm3m_test > netlib_accuracy.log 2>&1
```

两库的预加载次序确保 cblas_cgemm 及其 Fortran cgemm_ 都来自指定版本。
可用 `LD_DEBUG=bindings` 对 `--gtest_filter=*TC_L0_001` 运行一次核实实际绑定。
未设置预加载时使用工程默认 OpenBLAS，仅作开发交叉检查，不声称等同任务参考库。

只跑功能集：`--gtest_filter=-*TC_PF*`；只跑性能场景精度：`--gtest_filter=*TC_PF*`。
旧接口回归：`./build/test/gemm/cgemm/cgemm_test`。
输入生成复用上游 `makeBlasComplexMatrix`，固定 CSV seed，虚部分量使用独立 seed。

## msprof 性能自测

先通过对应精度测试。性能入口只接受原始 TC_PF 用例（alpha=1、beta=0），设备内存分配一次，
反复执行同一输入；每次同步 stream 并检查返回值。该入口跳过 CPU golden，不生成主机耗时指标。
不要将 GTest 的毫秒数用作设备耗时。

默认预热 5 次、采样 10 次，全部有效样本取算术平均，不剔除慢样本：

```bash
CGEMM3M_PROFILE=1 msprof --output="$PWD/prof_cgemm3m" --task-time=on --ai-core=off \
  --application="./build/test/gemm/cgemm3m/cgemm3m_test --gtest_filter=*TC_PF*" > profile.log 2>&1
python3 test/gemm/cgemm3m/scripts/summarize_msprof.py \
  prof_cgemm3m profile.log performance.csv
```

脚本对每次 API 调用的所有 kernel task duration 求和，检查 kernel 顺序、调用数量、
GTest 成功和基线参数一致性；缺记录直接失败。输出汇总 CSV 和含全部预热/采样的 `.raw.csv`，
同时保留从首 kernel 开始到末 kernel 结束的 span，便于识别启动间隙。
GPU 基线按原始用例编号匹配并验证 8 个参数，避免相同形状的重复基线误配。
默认必须覆盖 CSV 全部 200 个 TC_PF 用例，缺项、重复或未知用例均报错。
局部开发探索需显式传入 `--allow-partial`，日志会标记 PARTIAL EXPLORATION，不能用于全量性能结论。
倍率为 GPU_us/NPU_us；各条 >=0.4，全部 200 条的倍率算术平均 >=0.5。
`npu_us` 和 `npu_us_all` 始终是全部有效样本均值，`ratio`、`pass_0_4` 及日志统计均使用此口径。
可显式传 `--drop-largest N` 生成开发对照列 `npu_us_trimmed`，剔除数记录为 `trimmed_discarded`；
此选项不会改变倍率或通过判定，正式统计的 `discarded` 始终为 0。

任务指定 **msprof op** 模式，预热 5 次、采样 10 次全部平均。本次环境原始 msopcom b4ffdb3 的 op 模式出现
`StartFFTSTask Profiling channel start failed`，即使进程返回 0，也未产生成功记录。
底层原因为逻辑设备0被转换为物理设备4后传给仅接受逻辑设备号的接口。
官方 msopcom d1564487 的 DeviceContext.cpp 已改用逻辑设备查询；使用包含该修复的工具后复测。
上面的 application 模式结果只作为开发探索，不能标注正式验收通过。
默认采样数量虽与任务口径一致，application 模式仍不能代替任务指定的 msprof op 模式。
保留 msprof 原始目录和日志；不要仅保留筛选后的平均值。

## PR 检视补充：用例完整性与内存安全检查

`Cgemm3mCsvIntegrity.CsvCaseCount` 断言 CSV 共 1200 条、TC_PF 共 200 条、
其余功能精度共 1000 条，且用例名非空并唯一。当前完整 GTest 为 1208 项。
`run_self_test.py` 先核对 CSV，再逐项比较 GTest 注册和实际 XML 执行结果；
缺行、重复、漏执行、跳过、失败或非零退出均失败，并输出 `self_test_summary.json`。
脚本清除过滤、分片和 profiling 环境变量，保证 TC_PF 也执行参考精度检查。

完成上面的构建和 Netlib 准备后，从仓库根目录执行完整功能/精度自检：

```bash
python3 test/gemm/cgemm3m/run_self_test.py \
  --binary build/test/gemm/cgemm3m/cgemm3m_test \
  --reference ../reference/build/lib --device 0 --output self_test_cgemm3m
```

华为 [CANN 9.1 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/devaids/optool/docs/en/user_guide/compile_option_config.md)
说明 Ascend 950 当前不支持 `--cce-enable-sanitizer`、`--sanitizer` 和 `-O0`。
内存检查使用独立 O2/-g 构建，以 mssanitizer 的快速定位模式运行；该模式不能检查全部指令，
也不替换上面的精度和性能构建：

```bash
export LINUX_INCLUDE_PATH="$ASCEND_HOME_PATH/$(uname -m)-linux/include"
export EAGER_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64"
CMAKE_INCLUDE_PATH=/usr/include/openblas CMAKE_LIBRARY_PATH=/usr/lib64 \
cmake -S . -B build_memcheck -DBUILD_TEST=ON -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DSOC_VERSION=ascend950 '-DTEST_NAMES=cgemm3m;cgemm' \
  '-DCMAKE_ASC_FLAGS_RELWITHDEBINFO=-O2 -g'
cmake --build build_memcheck --target cgemm3m_test -j8
python3 test/gemm/cgemm3m/run_memcheck.py \
  --binary build_memcheck/test/gemm/cgemm3m/cgemm3m_test \
  --reference ../reference/build/lib --device 0 --output memcheck_cgemm3m
```

输出目录必须不存在，以免旧 XML 或日志混入新结果。每次内存检查默认限时 400 秒，
可通过 `--timeout` 调整；超时或未完成记录均判失败。
脚本不会安装参考库、修改设备配置或跳过失败；使用指定两份 Netlib 动态库并记录其 SHA256。

内存安全检查响应 PR #735 的具体检视意见，使用环境中真实可用的
`mssanitizer --tool=memcheck`，可用 `--sanitizer` 指定工具路径。
TC_L0 18 项、TC_ED 32 项分别在独立进程检查全部契约；TC_SQ_039、TC_SQ_045、
TC_PF_1001、TC_PF_1052、TC_PF_1200 是覆盖常规、混合形状及大方阵路径的代表用例。
对 TC_SQ_045 检查 path kernel 和 select kernel 的全部 block；
对 2077 阶 TC_PF_1052 检查 pack、permute、prefix 全部 block，单独检查 pack 尾块 24，
并检查 select 全部 block。select 不传 `--block-id`，采用工具默认的全部 block 检测，
每个用例仅启动一次检查；`--cache-size=128` 避免全核模式超过工具的总缓存限制。
各次运行均要求 GTest 完整通过，且 sanitizer 日志中 `Start memcheck` 与
`No error detected` 数量为正且一致，无错误或记录丢失诊断。
生成 `memcheck_summary.json`，明确记录代表用例范围，保留每次 GTest XML、运行日志和 sanitizer 日志。
summary 预登记全部检查组为 `NOT_RUN`，实际完成快速检测的组标为 `PASS_UNINSTRUMENTED`，失败组保留 `FAIL` 和错误信息。
select 每个用例保留一次全部 block 检查结果；失败后停止，不继续使用异常设备。
超时或检查记录不完整不自动判定为工具缺陷，`tool_limitation` 明确说明根因尚未确证。
SQ/PF 内存检查不宣称覆盖 108+200 条队列的每一条；完整精度由 `run_self_test.py` 覆盖 1208 项。
工具缺失、无有效检查记录或失败时必须标为未通过，不以退出码 0 或旧精度结果代替。
2026-10-31 在 950PR/CANN 9.1.0 上，修复 select 的 UB 分配及计数访问后，
使用相同执行和校验函数的快速驱动完成上述 12 组检查，94/94 条记录配对，未报告错误或记录丢失。
本脚本已同步采用该全 block 参数；原始结果、实际运行入口和源码版本对应关系保存在评审附件中。
这项是越界等内存安全检查，不是任务书 §3.4 不涉及的内存占用指标，也不用于性能计时。

本轮自动评审后续修订：950快速检测成功状态为 `PASS_UNINSTRUMENTED`，同时记录
`compiler_instrumented=false`、`memory_safety_certified=false`；退出0表示快速检测成功，
不表示完整编译插桩内存安全认证。下游应读取明确状态及检测范围。
以下为 2026-10-31 的阶段记录：此前 12 组 94/94 记录属于提交 46286de；
当时事件与短 K 分流修改尚待目标设备验证。后续验证结果见本节末尾，不再以该历史说明表示当前状态。

后续评审候选版本还调整了通用三乘积预处理及epilogue，使其共同使用
Ar*Br、Ai*Bi、(Ar+Ai)*(Br+Bi)；融合路径保留自身配套公式。
rounding的packed矩阵子缓冲独立按32字节对齐，分配大小与启动偏移使用同一函数。
TrackPath需求只由显式张量布局推导；平台可用UB不足时，若通用192KiB预算仍满足，
则回退到非融合三乘积路径；未知容量或连通用预算也不足时返回错误。
非有限值校验按任务书 §3.2：NaN/Inf 混配等效，异号 Inf 不等效。
截至 2026-10-31 记录时，这些后续修改尚未做目标 CANN 编译或 NPU 复测；
该历史状态已由下述 2026-11-02 验证记录补充。

后续候选恢复standard三乘积连续输入的SIMD反交织；仅非连续输入走SIMT预处理。
correctRounding必须同时满足fusedTile非零，避免不执行纠正时仍申请其工作区。
select使用__vector__混合编程启动约定：第二参数dyn_ub_size为动态UB字节数；
分配、启动及Host容量检查统一使用CGEMM_SELECT_UB_ELEMENTS/CGEMM_SELECT_UB_BYTES。
不能套用__aicore__入口的l2ctrl指针约定将该参数改为nullptr。
以上为 2026-10-31 候选说明；当时仅完成本地检查。后续目标设备验证结果如下。

### 2026-11-02 验证结果（2026-11-04 补记）

在 Ascend 950PR / CANN 9.1.0 上完成混合算法候选构建及自验；参考库固定为
Netlib BLAS/CBLAS 3.10.0，精度按本 README 的任务书 §3.2 判据执行。

| 检查 | 结果 | 证据 |
|---|---|---|
| 目标构建 | 通过 | `hybrid_build.log` |
| 全量精度及同输入双跑确定性 | 1207/1207，0 失败、0 跳过 | `hybrid_full_accuracy/full.xml`、`self_test_summary.json` |
| 正式 msprof op 性能 | 200/200 达标，最低比值 0.415309，平均比值 0.645595 | `performance_hybrid_final/performance_checks.json` |
| PR CI | 页面显示 `ci-pipeline-passed`，2026-11-02 20:35:43（北京时间）关联 | PR #735 企业标签 |

精度和性能使用相同库及测试二进制，SHA256 分别为：

- `libops_blas.so`：`93f33210a532053aaa3f3f4883b84eec41ea93e2e4886161dbdf561c32e8ece3`。
- `cgemm3m_test`：`0fce01e13dd37dc8bde50dcec2858640ae66c28ee6584bfd78ad4761acce91c3`。

当轮验证报告记载待推送的 10 个改动文件与服务器测试源码 SHA256 一致。
2026-11-04 核对 PR head `1f16e4b781670c4df15f037e87590d4f4fb09eca` 的公开差异，
算子 Host、数值内核、测试 C++ 及构建入口与保存的源码快照一致；测试 CSV 原始 SHA256
也与全量自测摘要一致。快照另有后续内存检查脚本改动，不能将其视为已测脚本的原样副本。

内存工具口径沿用已有评审认可：2026-10-31 08:14（北京时间），MaskHunter2008
在回复 197212264 中认可提交 46286de2 的 12 组、94/94 记录及 950PR 快速检测模式的限制。
11 月 2 日混合候选的 epilogue 定点检查中，ED309、ED336 各有 2 条开始和 2 条无错误记录，
工具退出码为 0；对应插桩运行存在 GTest 数值断言差异，原始日志及审计记录均保留。
此处分别记录工具输出和插桩数值现象，不将其认定为已证实的 kernel 内存缺陷，
也不把此前评审认可扩大为对该轮全部插桩对照的重新认可。

原始证据保存于跨电脑交接仓的 `docs/pr735_review7_evidence_20261102/`；
复现命令及范围见 `docs/pr735_review7_validation_20261102.md`。
代码 PR：https://gitcode.com/cann/ops-blas/pull/735 。

### 2026-11-09 合并前验证结果与上游同步

前缀相消保护及旧 Cgemm 兼容性修复后，在健康 Ascend 950PR / CANN 9.1.0 上完成新一轮验证。
参考库为预加载的 Netlib BLAS/CBLAS 3.10.0，实测于北京时间 11 月 9 日 01:36～02:30 完成。
原始证据见 PR #735 证据附件中的 `docs/validation/cgemm3m/2026-11-09/`。

| 检查 | 实测结果 | 证据位置 |
|---|---|---|
| 目标构建 | cgemm3m_test、cgemm_test 构建通过 | `build_record/build.log` |
| 完整精度及适用 CSV 用例双跑确定性 | 1208/1208，0 失败、0 跳过，含 SegmentCancellation | `accuracy/self_test_summary.json`、`accuracy/full.xml` |
| 旧 Cgemm 回归 | 51/51，0 失败、0 跳过 | `legacy/prefix_legacy_cgemm.xml` |
| 正式 msprof op 性能 | 200/200 达标；最低倍率 0.4149422841，平均倍率 0.6406365460 | `performance/performance_checks.json`、全部 2000 样本及原始 kernel CSV |
| 098419e6 官方 CI | `ci-pipeline-passed`，11 月 9 日核实 | PR #735 |
| 合并前内存检查记录 | 定界材料已上传 | 本轮验证目录“内存检查待确认项” |

表中证据路径相对于本轮验证目录。性能为 msprof op 每次接口所属全部 kernel 的设备
task duration 之和，工具预热 5 次、有效采样 10 次，不剔除样本。

设备实测后仅拆分两条变量声明生成 d652c34；再同步上游 f37a077 并解决声明、文档冲突，
生成 `098419e6309db0b852668133893353cb5ad1de32`，本 PR 的 24 个实现与测试文件与 d652c34 逐字节一致。
精确声明替换、实测/提交文件清单与产物哈希均保存在本轮验证目录。

本次补记录时，上游又推进至 `182f7b44260392a4c91ad955603a1ec7ce24c758`，
新增批处理三乘法并修改共用内核。合并时分别保留其 Gauss 路径与本 PR 单矩阵标准 3M 路径；
rounding 内核和单矩阵测试 C++、CSV 未修改，共用 Host、compute、epilogue、on-chip 和 tiling 已变化。
### 2026-11-10 合并后设备验证

合并后版本在 Ascend 950PR / CANN 9.1.0 上完成验证。实测源码为
`0f8c125e22e0cad1d7cda3e95d2fddb042b8554c` 加两文件 SDK 兼容补丁，使用 Netlib 3.10.0。
补丁用于检测 SDK 的 `ACL_HIFLOAT4` 并保护测试 CSV 类型别名，补丁随 PR 附件提供。

| 检查 | 合并后设备实测 |
|---|---|
| 目标构建 | 通过，使用上述 SDK 补丁 |
| 全量精度与适用用例双跑确定性 | 1208/1208，0 失败、0 跳过 |
| 旧 Cgemm 回归 | 51/51，0 失败、0 跳过 |
| 正式 msprof op 性能 | 200/200 达标，最低 GPU/NPU 0.414649、平均 0.635419 |
| 内存快检 | 12 组已执行，11 组 PASS_UNINSTRUMENTED，ED 组失败 |

ED309、ED313、ED317、ED336、ED337 出现数值断言失败，另有一条 8 字节 GM 写越界告警。
告警和数值差异的根因仍待定。源码与产物哈希、构建日志、原始 XML、2000 个性能样本和各组内存日志
见同一附件中的 `docs/validation/cgemm3m/2026-11-10/`。
