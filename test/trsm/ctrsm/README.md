# aclblasCtrsm 测试指导

## 测试范围

`arch35/ctrsm_test.csv` 包含 1200 条用例：

- 1000 条功能、精度和参数校验用例，覆盖 side / uplo / trans / diag 的 24 种组合、方形与矩形 shape、lda / ldb padding、UNIT / NON_UNIT、特殊 alpha、Inf / NaN 和负向输入。
- 200 条 `TC_PF_` 性能用例，全部逐条验证 `GPU/NPU > 0.4`，其中 `TC_PF_1001`、`TC_PF_1002`、`TC_PF_1003` 对应任务书表中的三个示例。
- `CtrsmArch35Test.NullHandle` 额外验证空句柄返回值。
- `CtrsmRegression` 额外包含 12 条回归：将 `TC_FL_110/116/117` 分别替换为 7、42、20260911、20260912 四个种子，覆盖极值 A、极值 B 和 Inf B。官方 CSV 保持原样。
- `CtrsmTailRegression` 额外包含 408 条回归：覆盖 65～71 的全部 32B 余数、255/1025、257/255、255/511，以及分派/分块边界 63/32、64/31、64/32、65/33、127/63、128/64、129/65。每种形状覆盖全部 24 种 side/uplo/trans/diag 组合、正负复数 alpha、lda+3/ldb+5 padding；检查实部、虚部及 padding 保持不变。
- `RejectUnrepresentableBlockedLayout` 检查左右两种模式的 8 组超大维度，在分配工作区或启动 kernel 前拒绝无法表示的 Matmul 行距/矩阵字节大小。
- `NumericalPathBoundaries` 在 5 种形状下验证大数除法、零右端与 Inf 系数、alpha=1 与 Inf 右端，共 15 组输入。
- `ScaledDiagonalAllOperations` 在 14 种形状、24 种组合下分别验证 1e20 大数和 1e-20 小右端，共 672 组输入，检查实虚部与 padding；NON_UNIT 对角至少为 order+5，遵循任务书远离零的要求，UNIT 对角及未引用三角使用 NaN 毒化。
- `NonFiniteAllOperations` 在 3 种分派边界、24 种组合和两种 alpha 下验证正/负 Inf 右端、NaN 右端、零右端配 Inf 系数，共 576 组输入。非有限结果按 Netlib 分量比较，不将 NaN 自动当作通过。
- `OffsetBuffersAndUserWorkspace` 验证 A/B 偏移一个复数元素、B 首尾哨兵、空 workspace 拒绝、不足空间返回 ALLOC_FAILED、足量用户 workspace 计算和恢复默认 workspace。

分块准备阶段仅检查有效输入，不读取 UNIT 对角和未引用三角。检测到非有限数、大幅值或过小对角时，设备端使用补偿 SIMT 求解；其余输入保留 Cube 路径。该分派不依据用例编号，不在 Host 读取矩阵或增加流同步。

精度 golden 使用任务书指定的 Netlib BLAS `cblas_ctrsm` 生成。COMPLEX64 的实部和虚部分别按 FLOAT32 标准验证：`rtol=2^-10`、`atol=2^-16`、`matched_ratio>=0.99`，同时满足最大绝对误差限制。性能用例的精度检查在计时区间之外执行。

OpenBLAS 虽然提供同名 CBLAS 接口，但极值和 Inf / NaN 的结果可能与 Netlib 不同，不能替代本任务的验收 Golden。请同时确认链接路径和运行时实际加载路径；系统默认 `libblas.so.3` 可能通过 alternatives 指向 OpenBLAS。

## 环境

- 硬件：Ascend 950PR
- 目标 CANN：9.1.0
- 当前真机实测 CANN：9.1.0 正式版
- ASC 编译优化：`-O3`

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export EAGER_LIBRARY_PATH=/usr/local/Ascend/cann-9.1.0/x86_64-linux/lib64
export LINUX_INCLUDE_PATH=/usr/local/Ascend/cann-9.1.0/x86_64-linux/include
```

`EAGER_LIBRARY_PATH` 和 `LINUX_INCLUDE_PATH` 请根据实际 CANN 安装目录调整。上述路径已在
Ascend 950PR / CANN 9.1.0 正式版环境验证。

## 编译

在 ops-blas 根目录执行：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DREFBLAS_LIB=/usr/lib/x86_64-linux-gnu/blas/libblas.so.3 \
  -DREFBLAS_INCLUDE_DIR=/usr/include/x86_64-linux-gnu \
  -DSOC_VERSION=ascend950 \
  -DBUILD_TEST=ON \
  -DTEST_NAMES=ctrsm
cmake --build build --target ctrsm_test -j2
```

示例 Netlib 路径对应本次 x86_64 Ubuntu 环境，其他环境应指定实际安装的 Netlib 库及 `cblas.h` 目录。

源文件由仓库的架构目录发现机制自动收集，优化等级沿用构建模式。Release 下
可在详细构建日志中确认 `ctrsm_*kernel.cpp` 的编译命令包含 `-O3`，不额外覆盖 Debug 的优化等级。
统一内核源文件 `ctrsm_kernel.cpp` 使用源码级 `#pragma clang fp contract(off)`，保留误差补偿中普通乘加的
分步舍入；显式 `fmaf` 仍用于计算融合残差。无需为本算子修改顶层 CMake 或统一编译选项。

验收使用完整 `Release` 构建。配置后检查 `build/CMakeCache.txt` 中
`CMAKE_BUILD_TYPE:STRING=Release`，并确认实际加载的是当前构建目录的 `libops_blas.so`。
仅为设备内核添加 `-O3` 不等价于完整 Release 验证。需要对比 Debug 时，使用独立的
`build_debug` 目录并显式传入 `-DCMAKE_BUILD_TYPE=Debug`，避免混用二进制和日志。

## 运行

```bash
export LD_LIBRARY_PATH="$PWD/build:/usr/lib/x86_64-linux-gnu/blas:/usr/local/Ascend/cann-9.1.0/x86_64-linux/lib64:$LD_LIBRARY_PATH"
export ASCEND_RT_VISIBLE_DEVICES=0

# 确认 libblas.so.3 指向 Netlib 目录，libops_blas.so 指向本次 build 目录
ldd ./build/test/trsm/ctrsm/ctrsm_test

# 1000 条 CSV 功能/精度用例 + 6 个独立测试 + 12 条特殊值回归 + 408 条尾块回归
./build/test/trsm/ctrsm/ctrsm_test --gtest_filter=-\*TC_PF\*

# 单独复现官方非 PF 集（1001 项）与新增回归（12 项）
./build/test/trsm/ctrsm/ctrsm_test --gtest_filter='Ctrsm/CtrsmArch35Test.*:CtrsmArch35Test.NullHandle-*TC_PF*'
./build/test/trsm/ctrsm/ctrsm_test --gtest_filter='*TC_REG*'

# 覆盖所有 32B 尾块余数及 255×1025 等尺寸，全部 side/uplo/trans/diag 组合
./build/test/trsm/ctrsm/ctrsm_test --gtest_filter='*TC_TAIL*:*RejectUnrepresentableBlockedLayout*'

# 数值边界、所有操作组合的极值与 Inf、偏移和用户工作区
./build/test/trsm/ctrsm/ctrsm_test --gtest_filter='CtrsmArch35Test.*'

# 三个示例用例的快速排查（不能代替全量性能验收）
./build/test/trsm/ctrsm/ctrsm_test \
  --gtest_filter='*TC_PF_1001:*TC_PF_1002:*TC_PF_1003'

# 完整 200 条性能用例，包含计时区间外的 CBLAS 精度检查
./build/test/trsm/ctrsm/ctrsm_test --gtest_filter='*TC_PF*' > ctrsm_performance.log 2>&1

# 一次完成所有 1626 项 GTest 测试并核查性能（独立测试内部另含上述输入组合）
./build/test/trsm/ctrsm/ctrsm_test > ctrsm_full.log 2>&1
```

全部 200 条性能用例均先执行 10 次 warmup，再通过 ACL Runtime Event 采样 51 次。
计时覆盖一次完整 `aclblasCtrsm` 调用的设备执行，包含准备、求解和回写内核，
不包含输入 B 的 Device-to-Device 恢复或 CPU Golden。不能使用 GTest 打印的整条测试墙钟时间作为 NPU 耗时。

GPU 基线来自随任务提供的 `arch35/gpu_baseline.csv`，原始单位为 ms，比较前转换为 us。
按 `(m,n,side,uplo,trans,diag)` 完整参数匹配；重复参数必须具有相同基线。
每条用例必须满足 `GPU_us / average_us > 0.4`，等于 0.4 也不通过。
基线缺失、缺少预热、采样不足、精度失败、结果缺失或比例不达标均不能判定验收通过。
以下仅列出任务书表中的三个示例限值，其余 197 条同样执行全量比例门槛：

| 用例 | m, n | side / uplo / trans / diag | 限值 (us) |
|---|---|---|---:|
| TC_PF_1001 | 256, 256 | LEFT / UPPER / N / NON_UNIT | 266.06 |
| TC_PF_1002 | 512, 512 | LEFT / LOWER / N / NON_UNIT | 558.6 |
| TC_PF_1003 | 1024, 1024 | RIGHT / UPPER / T / NON_UNIT | 1488.72 |

运行结果须同时满足：功能/精度及回归集 `1426/1426 PASSED`（官方非 PF 1000 项、特殊值回归 12 项、尾块回归 408 项、独立测试 6 项），
完整 PF 集 `200/200 PASSED`，测试程序退出码为 0。
逐条检查日志中的 `[CTRSM_PERF]` 记录，确认 200 个官方 PF 用例无缺失、
`warmup=10`、`samples=51` 且 `gpu_npu_ratio>0.4`。
一次执行完整集时应为 `1626/1626 PASSED`。用例的实部、虚部统计也必须各自通过。
若设备不是逻辑 0 卡，请将 `ASCEND_RT_VISIBLE_DEVICES` 改为验收环境分配的设备号。
