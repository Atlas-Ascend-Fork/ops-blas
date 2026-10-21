# aclblasCtrsmBatched 精度与性能测试

测试采用仓库 GTest + CSV 框架，共享 `test/frame/test_main.cpp` 入口。
`arch35/ctrsmbatched_test.cpp` 包含设备调用、CBLAS golden、精度比较和 ACL 事件计时，
`ctrsmbatched_param.h` 解析用例参数，`CMakeLists.txt` 注册测试目标。

算子实现维护在 `blas/trsmbatched/arch35` 的六个文件中：Host、主设备入口、UNIT、
Cube、方阵流水五个 `.cpp` 和一个共享头文件。入口源码由仓库现有 CMake 规则自动收集，
不生成包装文件、不排除单个源文件、不设置专用编译选项。UNIT 和方阵入口复用
主入口中的私有模板，入口间保持独立的 UB 分配。
测试入口按输入准备、逐批 CBLAS 参考、实部/虚部比较和事件计时拆分为独立函数。

## 用例与验收要求

| 文件 | 用途 |
| --- | --- |
| `arch35/ctrsmbatched_test.csv` | 任务附件原始用例：1000 条精度及边界用例、200 条性能用例 |
| `arch35/gpu_baseline.csv` | 全部 200 条性能用例的任务附件 GPU 标杆 |
| `arch35/ctrsmbatched_contract.csv` | 独立的 165 条参数、路径和内存契约用例，不替代官方原始用例 |

官方 CSV 保留全部原始参数、随机种子和期望返回码，包括 TC_ED_180、TC_ED_182
的零维矩阵及 lda/ldb=0；不修改用例以适配实现。性能用例编号为 TC_PF_1001～1200，
标杆按原始行号匹配并核对七项矩阵参数，不能只按 shape 匹配。

实部、虚部分别比较，atol=2^-16、rtol=2^-10、匹配率≥0.99；每个分量同时限制
绝对误差≤max(0.01,32×ULP(golden))。NaN/Inf 的位置、类别及 Inf 符号必须一致。
参考解自身溢出的用例输出 `CTRSM_REFERENCE_OVERFLOW`，报告应单列这些记录；
不缩小原始随机分布或用其他输入替代此类用例。

测试同时检查 A 保持只读、B padding 未修改、UNIT 对角及未引用三角不被读取；
alpha=0 时不分配 A 矩阵。A/B 设备指针数组包含各批次独立分配的矩阵地址。
每批 B 的输出仍位于原地址，A/B 指针数组逐项保持不变。`NOACCESS_UNIT` 用例将
一阶 UNIT 的 A 元素地址设为不可访问地址 `0x1`，结果正确且同步成功才通过；
一般 UNIT 用例对对角和未使用三角填 NaN，分别验证这些值不参与运算。

## 评审契约用例

`ctrsmbatched_contract.csv` 使用独立、确定性的种子，包含 24 种
LEFT/RIGHT × UPPER/LOWER × N/T/C × UNIT/NON_UNIT 组合。每种组合含 8 阶方阵、
17 阶矩形、193 阶矩形和宽 RHS 矩阵，共 96 条有限值精度用例，覆盖方阵、小矩阵、
UNIT 与 Cube/向量路径。另含 24 条 alpha=0 + null A + NaN B、24 条一阶 UNIT
不可访问 A、6 条零维以及 15 条非法参数用例。每个有效矩阵用例有 3 个独立 batch、
非紧密 lda/ldb，实部/虚部分别与 CBLAS 比较，并检查 A、B padding 和指针数组。

先使用正常 workspace 运行，再使用仅 16 字节的用户 workspace 重复运行。
后一轮同时断言 workspace 地址、容量与 16 字节哨兵不变，证明空间不足时成功回退
且没有越界使用用户缓冲区。Cube 候选矩阵在前一轮具备足够 workspace。
两轮均应完整执行 165/165；不得以补充用例的通过替代官方 1000+200 条验收。

官方和契约用例均直接运行 GTest 二进制，命令见下方完整测试步骤。

## 环境与构建

设备为 Atlas 950PR，CANN / asc-devkit ≥9.1。按仓库根目录说明初始化子模块、
安装 GTest 与 CBLAS/LAPACK 依赖并加载 CANN 环境。`CANN_3RD_LIB_PATH` 指向
已准备的仓库第三方依赖目录。以下命令在仓库根目录执行，路径按实际安装位置替换。

验收参考配置为 OpenBLAS 0.3.25 / Cooperlake / SINGLE_THREADED。
测试逐 batch 调用 `cblas_ctrsm` 生成 golden；不同内核的舍入顺序可能不同，
复现时应核对 `openblas_get_config()`、`openblas_get_corename()` 与线程数。
开发侧参考库 `libopenblas-r0.3.25.so` 的 SHA-256 为
`bee7c8d09de00a3e52998a65578804a0e4c933e2d9501a8a2a07b4466c5439a8`。
使用 Cooperlake 内核的 CPU 需支持 AVX-512；测试程序使用 Release，不启用 fast-math。

```bash
export OPENBLAS_CORETYPE=Cooperlake
export OPENBLAS_NUM_THREADS=1
cmake -S . -B build -DSOC_VERSION=ascend950 -DBUILD_TEST=ON \
  -DTEST_NAMES=ctrsmbatched -DCMAKE_BUILD_TYPE=Release \
  -DCANN_3RD_LIB_PATH=/absolute/path/to/deps \
  -DREFBLAS_INCLUDE_DIR=/absolute/path/to/include \
  -DREFBLAS_LIB=/absolute/path/to/libopenblas-r0.3.25.so \
  -DREFLAPACK_LIB=/absolute/path/to/libopenblas-r0.3.25.so
cmake --build build --target ctrsmbatched_test -j8
export LD_LIBRARY_PATH="$PWD/build:/absolute/path/to/reference:$LD_LIBRARY_PATH"
ldd build/test/trsmbatched/ctrsmbatched/ctrsmbatched_test
```

更换参考依赖后重新配置 CMake 并检查 `CMakeCache.txt` 的 `REFBLAS_LIB`。
运行前确认 `ldd` 实际加载的算子及参考库路径，记录二者的 SHA-256。
若参考库需要较新的 glibc，可单独解包兼容的 glibc，以如下脚本包装测试程序，
在下方测试步骤中将 `CTRSM_TEST_BINARY` 设置为该包装程序的绝对路径：

```bash
#!/bin/sh
exec /absolute/path/to/glibc/ld-linux-x86-64.so.2 \
  --library-path /absolute/path/to/glibc:/absolute/path/to/build:$LD_LIBRARY_PATH \
  --preload /absolute/path/to/libopenblas-r0.3.25.so \
  /absolute/path/to/ctrsmbatched_test "$@"
```

该加载器配置仅用于测试进程，不替换系统 libc。

性能测试还应记录 CPU 亲和性与 NPU 的 NUMA 节点，减少微秒级调用入队时的调度波动。
本次 x86_64 服务器的 NPU 位于 NUMA 1，复测将测试进程固定在该节点的物理 CPU
`64-127`。其他服务器应根据 `lscpu`、NPU PCI 地址对应的
`/sys/bus/pci/devices/<PCI地址>/numa_node` 和 `local_cpulist` 选择本机 CPU，
不能直接套用该编号。可在下述测试程序命令前加 `taskset -c 64-127`；
仍须完整执行全部用例并保留每条 60 次事件采样，不能删除失败样本。

## 完整测试步骤

精度和性能必须使用同一构建的算子动态库，以下 Bash 命令在仓库根目录执行。
先清除遗留的用例过滤、分片、采样数和 workspace 设置，明确指定官方输入；
输出目录必须是新目录，已有同名目录时请另取名称。

```bash
set -euo pipefail
unset BLAS_TEST_CASES GTEST_FILTER GTEST_SHARD_INDEX GTEST_TOTAL_SHARDS GTEST_REPEAT
unset CTRSM_BATCHED_BENCH_SAMPLES CTRSM_BATCHED_WORKSPACE_BYTES
export CTRSM_BATCHED_CSV="$PWD/test/trsmbatched/ctrsmbatched/arch35/ctrsmbatched_test.csv"
export CTRSM_BATCHED_GPU_BASELINE="$PWD/test/trsmbatched/ctrsmbatched/arch35/gpu_baseline.csv"
CTRSM_TEST_BINARY="$PWD/build/test/trsmbatched/ctrsmbatched/ctrsmbatched_test"
mkdir -p results
mkdir results/ctrsm_accuracy results/ctrsm_performance
"$CTRSM_TEST_BINARY" --gtest_color=no --gtest_filter='*-*TC_PF_*' \
  --gtest_output="xml:$PWD/results/ctrsm_accuracy/gtest.xml" \
  2>&1 | tee results/ctrsm_accuracy/run.log
"$CTRSM_TEST_BINARY" --gtest_color=no --gtest_filter='*TC_PF_*' \
  --gtest_output="xml:$PWD/results/ctrsm_performance/gtest.xml" \
  2>&1 | tee results/ctrsm_performance/run.log
```

核对 GTest 日志与 XML：精度应执行并通过 1000/1000，性能应执行并通过 200/200；
用例名称应与官方 CSV 一一对应。缺失、跳过、失败或重复均不能作为通过的验收结果。
性能用例先检查精度，再预热 5 次、采样 60 次；每次恢复原始 B，
恢复和 CPU 参考计算位于事件计时范围之外。ACL 事件在同一 stream 上记录单次
调用并转换为 us。每条用例均须满足 `gpu_ms*1000/avg_us > 0.4`，等于 0.4 也失败。
不能使用 GTest 进程总运行时间代替算子耗时。测试 CPP 直接读取全部 GPU 标杆、
核对编号及矩阵参数，并逐条断言门槛；不依赖额外 Python 脚本完成性能判定。

随后运行两轮契约用例，沿用上方环境、同一测试二进制和动态库：

```bash
mkdir results/ctrsm_contract results/ctrsm_contract_workspace16B
CTRSM_BATCHED_CSV="$PWD/test/trsmbatched/ctrsmbatched/arch35/ctrsmbatched_contract.csv" \
"$CTRSM_TEST_BINARY" --gtest_color=no --gtest_filter='*' \
  --gtest_output="xml:$PWD/results/ctrsm_contract/gtest.xml" \
  2>&1 | tee results/ctrsm_contract/run.log
CTRSM_BATCHED_CSV="$PWD/test/trsmbatched/ctrsmbatched/arch35/ctrsmbatched_contract.csv" \
CTRSM_BATCHED_WORKSPACE_BYTES=16 \
"$CTRSM_TEST_BINARY" --gtest_color=no --gtest_filter='*' \
  --gtest_output="xml:$PWD/results/ctrsm_contract_workspace16B/gtest.xml" \
  2>&1 | tee results/ctrsm_contract_workspace16B/run.log
```

每轮契约用例应执行并通过 165/165，同样检查是否缺失、跳过或重复。
`set -o pipefail` 保证使用 `tee` 保存日志时仍保留测试失败的退出状态。

`gtest.xml` 和 `run.log` 保存逐条结果及原始数据。日志中 `CTRSM_ACCURACY` 行包含
逐 batch 的实部/虚部匹配率、最大误差及非有限分量数；`CTRSM_PERF` 行依次记录
用例名称、事件均值（us）、采样数、矩阵与指针数组申请字节数、活动 handle workspace
容量。采样数必须为 60，内存字段表示显式申请量与活动容量，不是设备实测峰值。
验收报告可从这些原始记录整理，但不能替代完整执行结果和上述性能断言。
