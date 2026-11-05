# aclblasSsbmv arch22（Atlas A2/A3，910B3）测试说明

本文说明如何复现 `aclblasSsbmv` 在 arch22 上的精度与性能自测结果。

## 1. 目录内容

| 文件 | 说明 |
|---|---|
| `ssbmv_test.cpp` | gtest 用例：CSV 驱动 + 边界/负向/特殊值/大步长用例 |
| `ssbmv_test.csv` | 随任务提供的 1200 条用例（1000 条精度 + 200 条性能 `TC_PF_*`） |
| `ssbmv_perf.cpp` | 性能门禁：事件计时暖平均，对照 `ssbmv_perf_baseline.csv` 的 GPU 基线 |
| `ssbmv_perf_baseline.csv` | 200 条 `TC_PF_*` 的基线（`gpu_ms`，由任务包 `gpu_baseline.csv` 与用例逐条配对） |
| `run_self_test.py` | 自测门禁：CSV 用例数断言 + 精度全量执行 |
| `run_memcheck.py` | mssanitizer memcheck 门禁：代表用例的 kernel 越界检查 |
| `ssbmv_npu_wrapper.h` | 把 host 侧指针搬到 device、调用算子、取回结果的薄封装 |
| `README.md` | 本文件 |

Golden 由 `test/sbmv/ssbmv/ssbmv_golden.h` 调用 **Netlib `cblas_ssbmv`** 生成（生态标准实现）。

## 2. 构建与运行

前置：CANN 9.1.0 环境，且已 `apt-get install -y libblas-dev liblapack-dev`（`cmake/test.cmake` 硬依赖）。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 两个变量 set_env.sh 不设置；为空时测试目标会去链 /libascendcl.so 并报 "acl/acl.h: No such file"
export LINUX_INCLUDE_PATH=/usr/local/Ascend/ascend-toolkit/latest/include
export EAGER_LIBRARY_PATH=/usr/local/Ascend/ascend-toolkit/latest/lib64
cd <ops-blas>
# 注意：测试目标由 BUILD_TEST + TEST_NAMES 控制，缺一不可
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=ON -DTEST_NAMES=ssbmv && cmake --build build -j32
```

精度用例（排除 200 条性能用例）：

```bash
cd build/test/sbmv/ssbmv
./ssbmv_test --gtest_filter='-*TC_PF*'          # 期望: [  PASSED  ] 1012 tests.
./ssbmv_test                                    # 期望: 1212 tests（TC_PF 只做一次下发 smoke）
```

性能门禁（200 条 `TC_PF_*` 的事件计时暖平均；任务书 §3.3 的 4 组标杆为硬门禁）：

```bash
./ssbmv_perf                                    # 期望: verdict=PASS（4 条门禁用例）
SSBMV_PERF_CASE=TC_PF_1001 ./ssbmv_perf         # 只测一条
```

自测 / memcheck 门禁：

```bash
python3 run_self_test.py                        # CSV 1200 条断言 + 精度全量（0 失败）
python3 run_memcheck.py                         # mssanitizer memcheck（约 90s，12 个 kernel）
python3 run_memcheck.py --filter '*TC_L0_*'     # 换一组用例
```

单条复现：

```bash
./ssbmv_test --gtest_filter='*TC_L0_003'
SSBMV_DEBUG=1 ./ssbmv_test --gtest_filter='*TC_L0_003'   # n<=8 时额外打印 A/x/y 输入
```

## 3. 精度判定口径

- 逐元素混合容差（`test/frame/verify.h`）：`|actual-golden| <= atol + rtol*|golden|`，
  FLOAT32 取 `rtol = 2^-10`、`atol = 2^-16`；用例通过要求 `matchedRatio >= 0.99`。
- **`alpha == 0`**：走 `y = beta*y` 轻量路径，单次 float 乘，**位精确**。
- **就地（`x == y`）**：算子先把 x 整段备份进 handle 的 workspace 再下发（`CopyInPlaceX`）；workspace
  不够时库自有的原地扩容，用户注入的扩容不了则返回 `ACLBLAS_STATUS_ALLOC_FAILED`。就地路径不做
  「退回单核」的降级：kernel 按 chunk 分块写回，单核一样会踩掉后续 chunk 还要当 x 读的行（实测
  n=4096/k=8/incy=2 时后一半全错，见 PR 讨论）。
- **步长上限**：kernel 的物理下标（`XPhysicalPos`/`YPhysicalPos`）在 uint32 域内算
  `logical * |inc|`，因此要求 `(n-1)*|incx|` 与 `(n-1)*|incy|` 都不超过 `UINT32_MAX`；超出时
  host 返回 `ACLBLAS_STATUS_INVALID_VALUE`（否则乘积回绕成缓冲区内的错误下标，结果静默错误）。
  该约束与 csrot 的 `(n-1)*stride` 显式拒绝同口径。
- **带宽上限**：合法输入要求 `lda > k` 且 `lda` 为 `int`，因此 `k` 的有效域是 `0 <= k <= INT_MAX-1`；
  `k = INT_MAX` 不存在合法 `lda`，host 返回 `ACLBLAS_STATUS_INVALID_VALUE`。校验按 `lda > k`
  比较，不在 int 域计算 `k + 1`（该加法在 `k = INT_MAX` 时是溢出 UB）。`k >= n` 的宽带仍按
  列形式支持。
- **负向用例**：非法 uplo 按任务书 §2.4 返回 `ACLBLAS_STATUS_INVALID_ENUM`（仓内 arch35 用
  `INVALID_VALUE`，本算子以任务书为准，分歧见 `/workspace/ASC/REVIEW.md`）。
- **特殊值组（`RANDOM_EXTREME` / `VALUE_INF` / `VALUE_NAN` 填充）**：混合容差无法表达非有限值，
  且 band 值接近 `FLT_MAX` 时乘积溢出为 Inf，**溢出后相消的结果依赖求和顺序**，与参考实现不可能
  逐元素一致（`TC_FL_124` 第 13 行：double 精确和为 +3.32e38，参考实现给 -Inf、本算子给 +Inf）。
  因此该组按「边界不越界」判定，见 `SsbmvVerifyNonFinite`：
  1. golden 有限的元素 → 本算子必须有限且落在同一混合容差内；
  2. golden 饱和的元素 → 本算子必须保持饱和（不锁符号）。

  该判据仍能抓住这类用例真正的用途：越界读、脏缓冲、未初始化累加器都会表现为「参考有限而我们非有限」。
- **手写边界用例**（CSV 的 incx/incy 仅覆盖 {-3..3, 0}，以下由 `ssbmv_test.cpp` 补充）：
  - `LargeStridesMatchReference`：incx=1000、incy=-1000；
  - `ExtremeStridesMatchReference`：±2^31 步长（x/y 跨度 8GiB，HBM 不足时自动跳过）；
  - `RejectsStrideSpanBeyondUint32`：`(n-1)*|inc| > UINT32_MAX` 时 x、y 各判一次 `INVALID_VALUE`；
  - `RejectsBandWidthWithoutValidLda`：`k = INT_MAX` 时 int 域不存在合法 `lda`，`lda` 取 1、-1、
    `INT_MAX` 三档均返回 `ACLBLAS_STATUS_INVALID_VALUE`（校验不在 int 域计算 `k + 1`）；
  - `BetaZeroIgnoresStridedY`：`beta=0、incy=2` 时旧 y 填 NaN/Inf，结果仍必须是 `alpha*A*x`
    （BLAS 契约要求 beta==0 时不引用 y，不能出现 `0*Inf=NaN`）；
  - `WideBandMatchesReference`：`k >= n` 的宽带（n=256、k=30000、lda=30001），UPPER/LOWER ×
    `incx ∈ {1,2}` × `incy ∈ {1,-1}` 与 cblas 逐元素比对；宽带列形式不建 2k 的 x 窗口、
    A/x 段按 `SSBMV_MIRROR_TILE` 分块归约，UB 占用由 host 逐项精确预算，装不下时在下发前
    返回 `ACLBLAS_STATUS_NOT_SUPPORTED`（不会把越界的 UB 分配带进核里）；
  - 4 条就地用例（`AlphaZeroInPlaceIsExact`、`InPlaceAlphaNonZeroMatchesReference`、
    `InPlaceXOverDefaultWorkspaceGrowsWorkspace`、`InPlaceWithoutWorkspaceReportsAllocFailed`）。

## 4. 性能验收

任务书 §3.3 的 4 组标杆为 13.57 / 24.67 / 14.41 / 47.74 us（= `gpu_baseline.csv` 的
`gpu_ms / 0.8`），采样口径为「warmup 后有效采样 >50 次取平均」。

### 4.1 仓内门禁（`ssbmv_perf`）

`ssbmv_perf` 按任务书口径测量：同一流上 **warmup 10 次后连续下发 64 次**，用一对 event 卡住
整段再除以次数（流水线暖平均；含每次下发的 host 入队开销，与 msprof 的 kernel Task Duration
口径不同）。基线在 `ssbmv_perf_baseline.csv`，限值 = `gpu_ms * 1000 / 0.8`。

- 任务书 §3.3 的 4 组标杆（`TC_PF_1001~1004`）是硬门禁，超限即测试失败；
- 其余 196 条 `TC_PF_*` 只打印 `avg_us/limit_us/ratio`（`verdict=INFO`）。原因：本口径含每次
  下发的 host 入队开销，几微秒的小 shape 会高于按 kernel Task Duration 计的 msprof 口径，
  两者不能套同一个限值；它们的验收以 §4.2 的 msprof 口径为准。

**实测（910B3 / CANN 9.1.0，2026-10-29，本目录 `./ssbmv_perf`）**：

| case | n | k | uplo | 限值(us) | avg_us（多次运行） | 倍率 | 判定 |
|---|---|---|---|---|---|---|---|
| TC_PF_1001 | 1024 | 16 | UPPER | 13.571 | 6.0 ~ 6.5 | ≥ 2.1 | PASS |
| TC_PF_1002 | 2048 | 32 | UPPER | 24.672 | 6.2 ~ 6.4 | ≥ 3.9 | PASS |
| TC_PF_1003 | 1024 | 16 | LOWER | 14.415 | 6.0 ~ 6.3 | ≥ 2.3 | PASS |
| TC_PF_1004 | 4096 | 64 | UPPER | 47.738 | 11.7 ~ 12.1 | ≥ 3.9 | PASS |

### 4.2 msprof 单发 / 冷启动口径（历史记录）

> 本节是交付期在开发机上用仓外脚本（`/workspace/ASC/tmp/accept_repro/run_case_avg.sh`）与
> `msprof` 采集的历史记录，脚本与扫描报告未随仓提供；仓内可复现的门禁见 §4.1。

任务书 §3.3 验收口径的复现：同一进程内 warmup 12 次后连续 64 次下发取后 52 次平均，从
`msprof` 的 `op_summary_*.csv` 取每发的 `Task Duration(us)`。

```bash
# 历史脚本（不在仓内）：
# BIN=$(pwd)/ssbmv_test bash /workspace/ASC/tmp/accept_repro/run_case_avg.sh TC_PF_1001 64 12
```

**期望值（910B3 实测，2026-09-23，5 轮中位）**：

| case | n | k | uplo | 预算(us) | 期望平均(us) | 倍率 |
|---|---|---|---|---|---|---|
| TC_PF_1001 | 1024 | 16 | UPPER | 13.571 | 4.1 ~ 4.2 | ≥ 3.2 |
| TC_PF_1002 | 2048 | 32 | UPPER | 24.672 | 6.2 ~ 6.5 | ≥ 3.8 |
| TC_PF_1003 | 1024 | 16 | LOWER | 14.415 | 4.1 ~ 4.2 | ≥ 3.4 |
| TC_PF_1004 | 4096 | 64 | UPPER | 47.738 | 12.2 ~ 12.8 | ≥ 3.7 |

该口径下 **200 条 TC_PF 全部达标**（单进程扫描，最紧的 10 条仍有 ≥1.45 倍余量）。

### 4.3 单发冷启动口径（历史记录，保守下界）

每个用例起**全新进程**、只调用一次，取 msprof 的 `kernel Task Duration`；判据
`kernel_us <= gpu_ms / 0.8`。该口径每个用例都重付一次性的冷启动开销（实测 2.2~5.3us），
比 §4.2 的暖平均高 60%~130%，只作为稳健性下界：

| case | n | k | uplo | 预算(us) | 期望 kernel_us | 倍率 |
|---|---|---|---|---|---|---|
| TC_PF_1001 | 1024 | 16 | UPPER | 13.571 | 7.6 ~ 8.0 | ≥ 1.68 |
| TC_PF_1002 | 2048 | 32 | UPPER | 24.672 | 11.1 ~ 11.8 | ≥ 2.09 |
| TC_PF_1003 | 1024 | 16 | LOWER | 14.415 | 7.7 ~ 8.2 | ≥ 1.76 |
| TC_PF_1004 | 4096 | 64 | UPPER | 47.738 | 17.5 ~ 18.1 | ≥ 2.64 |

## 5. 自测门禁（`run_self_test.py` / `run_memcheck.py`）

- `run_self_test.py`：断言 `ssbmv_test.csv` 恰好 **1200 条 = 1000 精度 + 200 性能**，跑
  `-*TC_PF*` 的精度套件并要求 0 失败，且执行数 == gtest 列出总数 - 200（当前 1012）。
- `run_memcheck.py`：用 `mssanitizer --tool=memcheck` 跑一组代表用例（就地/workspace/大步长/
  宽带 k>=n/beta==0 strided-y 手写用例 + CSV 的窄带、相位副本、宽带列形式、strided 各一条），
  要求每个 kernel 都报
  `No error detected` 且 gtest 无失败。每个 pattern 独立进程执行：`InPlaceWithoutWorkspace...`
  在 mssanitizer 下会让同进程后续 launch 报 107017（工具与用例的交互，与被检算子无关）。
  默认集合约 2 分钟（宽带用例的 8 个 kernel 各带 30MB A）；`--filter` 可指定任意 gtest filter。

## 6. 已实测的环境语义（复现时可能踩到）

`/workspace/ASC/tmp/probe_pad/` 是可一键复现的最小探针工程（`bash run.sh`），实测结论：

1. `DataCopyPad` GM→UB 多 block 搬运，**UB 侧目的地按 32B datablock 落位**，4 字节块无法紧凑打包。
2. `DataCopyPad` 的 `rightPadding` 超过一个 32B 块时**整条搬运静默失效**（不报错、不写入）。
3. 标量写 GM 走每核私有 cache，**两个核写同一条 64B cache line 时后刷的核会用旧值覆盖对方**，
   跨核 strided y 写会静默丢数据（单核不复现）。
4. `Gather`（Level-2）在 dav-2201/float 可用，偏移单位为字节、**允许 4 字节非对齐**。

算子实现如何规避这四点见 `blas/sbmv/arch22/ssbmv_kernel.cpp` 的注释与
`/workspace/ASC/DECISIONS.md` D-012。

## 7. 精度复现（官方脚本）

官方脚本在任务包 `<test_cases>` 目录（不在本仓）：

```bash
python3 <test_cases>/verify_accuracy.py --repo <ops-blas> --soc ascend910b3
# 期望末尾：汇总: PASS=1000  FAIL=0  (共1000)  结论: ALL PASS
```
