# aclblasCrot 测试工程（arch35）

CSV 驱动的 GTest 测试工程，用于验证 `aclblasCrot`（complex64 向量 Givens 平面旋转）的
功能、边界、精度与性能。用例与算子实现分离，新增用例只需改 CSV。

## 目录结构

```
test/rot/crot/
├── CMakeLists.txt              # ops_blas_add_gtest_tests(${OPS_BLAS} crot_test)
├── crot_param.h                # CSV 行 -> 参数结构体
├── crot_golden.h               # CPU golden（按任务书 §2.1.1 公式实现）
├── run_self_test.py            # 自检门禁：CSV 构成校验 + 全量执行 + 汇总 JSON
├── run_memcheck.py             # 内存安全：mssanitizer / valgrind / 降级运行
└── arch35/
    ├── crot_test.cpp           # GTest 用例主体（CSV 驱动 + 7 个固定用例）
    ├── crot_npu_wrapper.h      # H2D/D2H 与 c/s 标量指针位置控制
    └── crot_test.csv           # 用例数据（1206 条：精度 1006 + 性能 200）
```

## 编译

```bash
cd <repo-root>
source /root/ascend_env.sh          # 或 source <CANN>/set_env.sh
bash build.sh --soc=ascend950 --ops=crot
```

产物：`build/test/rot/crot/crot_test`

## 运行

```bash
# 全部用例（精度 + 性能）
./build/test/rot/crot/crot_test

# 仅精度用例（排除性能用例）
./build/test/rot/crot/crot_test --gtest_filter='-*TC_PF*'

# 按类别跑
./build/test/rot/crot/crot_test --gtest_filter='*TC_L0*'      # 基础
./build/test/rot/crot/crot_test --gtest_filter='*TC_ED*'      # 边界与负向
./build/test/rot/crot/crot_test --gtest_filter='*TC_PF_1001*' # 单条性能用例

# 指定设备
ASCEND_DEVICE_ID=1 ./build/test/rot/crot/crot_test --gtest_filter='-*TC_PF*'
```

或用自检门禁脚本（会自行编译并解析结果，详见下文「自检门禁」）：

```bash
python3 test/rot/crot/run_self_test.py --repo <repo-root> --device 0
```

## CSV 列说明

| 列 | 说明 |
|---|---|
| `case_name` | 用例名，前缀决定类别（`TC_L0/SQ/INC/CS/FL/PM/ED/EX/PF`） |
| `description` | 用途描述 |
| `n` | 元素个数；`n <= 0` 为 no-op 用例 |
| `x` / `y` | 填充模式串：`RANDOM_NORM_5_5`、`RANDOM_ALTER`、`RANDOM_EXTREME`、`VALUE_NORM_0/INF/NAN`、`NULLPTR` |
| `incx` / `incy` | 步长，可为正/负/零 |
| `c` | 旋转余弦（实数） |
| `s_real` / `s_imag` | 旋转正弦（复数）的实部/虚部 |
| `expect_result` | 期望返回码（`ACLBLAS_STATUS_SUCCESS` / `ACLBLAS_STATUS_INVALID_VALUE`） |
| `mere_threshold` / `mare_multiplier` | 模板保留列（与同族工程 CSV 格式对齐），当前判定实现未消费该列（见下文「判定逻辑」） |
| `random_seed` | 填充用随机种子（虚部使用 `seed + 1000`） |
| `cs_ptr_mode` | c/s 标量指针位置：`host` / `device` / `mixed` |

## 判定逻辑

- **返回码**：与 `expect_result` 严格相等；非 SUCCESS 的用例只比对返回码。
- **精度**：golden 由 `crot_golden.h` 在 CPU 上按任务书 §2.1.1 公式重算，实部/虚部分别经
  `applyMixedTolerance(ACL_FLOAT)` 取容差（`atol = rtol = 2^-13`，即
  `test/frame/mixed_tolerance.h` 的 ACL_FLOAT 默认值），经 `Verifier::verifyVector` 按
  `|actual - golden| <= atol + rtol*|golden|` 判定，要求 `matched_ratio >= 0.99` 且
  `max_abs_error <= max(1e-2, 32*ULP)`。CSV 的 `mere_threshold` / `mare_multiplier`
  列为模板保留字段，当前不参与判定。
- **特殊值**：输入含 NaN 时要求输出实部/虚部全为 NaN；单位旋转（c=1,s=0）与普通用例
  一致走混合容差判定，无逐位相等/EXACT 比对分支。
- **n <= 0**：只比对返回码（应为 SUCCESS），并确认 x/y 未被修改。

## 用例构成

`arch35/crot_test.csv` 共 **1206 条**，按 `case_name` 前缀分布如下：

| 前缀 | 条数 | 类别 |
|---|---|---|
| `TC_L0` | 8 | 基础功能 |
| `TC_SQ` | 38 | 长度扫描 |
| `TC_INC` | 36 | 步长组合（正/负/零/混合） |
| `TC_CS` | 20 | c/s 来源（host/device/mixed）与空标量 |
| `TC_FL` | 12 | 特殊填充（NaN/Inf/极值） |
| `TC_PM` | 6 | 特殊参数（空 handle、`n<=0`） |
| `TC_ED` | 10 | 边界与负向 |
| `TC_EX` | 876 | 扩展覆盖 |
| `TC_PF` | 200 | 性能形状（另由性能门禁计时） |
| 合计 | 1206 | 其中精度用例 1006 条 |

装载期由 `LoadAndValidateCrotCases()` 断言总数 1206、`TC_PF` 条数 200、case_name 唯一；
任一不符直接抛 `std::runtime_error`，避免 CSV 被截断、拷贝失败或行尾变化后套件静默变绿。

## 预期结果

全部 1213 个用例（1206 条 CSV + 7 个固定用例）PASS：

```
[==========] 1213 tests from 2 test suites ran.
[  PASSED  ] 1213 tests.
```

## 性能门禁

`CrotTest.PerformanceGate` 对 `blas/rot/README.md` 声明的三档形状做**算子级计时**：

- 计时口径：每轮 1 对 ACL event 包住 100 次连续下发（事件开销摊薄），预热 10 次，共 3 轮取最优值；
- 数据常驻显存，不含 H2D/D2H 与 golden 比对，与验收口径（kernel 平均单次耗时）一致；
- 三档阈值：`n=1048576 ≤ 21.96us`、`n=2097152 ≤ 43.70us`、`n=4194304 ≤ 119.52us`，超限即 FAIL；
- 每次运行打印 `[PERF] n=... kernel_best_us=... kernel_avg_us=... budget_us=...`。

实测（Ascend 950PR / CANN 9.1.0）：**17.15 / 31.10 / 65.50 us**。

## 内存安全检查

```bash
python3 test/rot/crot/run_memcheck.py --repo <repo-root> --device 0
```

按 `msmemcheck`(mssanitizer) → `valgrind` → 降级（`ASCEND_LAUNCH_BLOCKING=1` 普通运行）的顺序选择工具；
默认过滤器 `*TC_L0_*:*TC_INC_*:*TC_CS_*:*TC_EX_STBIG_*` **包含真正下发 kernel 的成功路径用例**
（非仅负向用例），要求至少 60 条被执行、0 失败，mssanitizer 下还要求出现 `No error detected`。

## 自检门禁

```bash
python3 test/rot/crot/run_self_test.py --repo <repo-root> --device 0
```

校验 CSV 总数/`TC_PF` 条数/用例名唯一性，执行全量套件，比对"CSV 条数 == 实际执行条数"，
并要求 `[  PASSED  ]` 计数 ≥ 1213，最后输出 `self_test_summary.json`（含精度与 `[PERF]` 记录）。

## 别名输入用例

固定用例 `CrotTest.AliasedBuffersMatchGolden` 覆盖 x、y 指向同一块缓冲的两种情形：
等步长（`incx == incy == 2`，各元素独立）与不等步长（`incx=1, incy=-1`，两个逻辑槽位共享
同一物理元素）。后者由 host 侧检测基址相同后强制走单线程串行路径（tiling 的 `serialOnly`），
结果按逻辑下标顺序链式求值，与 golden 一致。

该用例经过变异验证：把 host 的 `serialOnly` 强制置 0 后，用例立即报 real/imag 全量不符；
恢复后通过。CSV 通道无法覆盖该场景，因为 wrapper 总是为 x、y 分别分配缓冲。

## 差距元素不被写的断言

固定用例 `CrotTest.GapElementsUntouched` 用 `inc=4`、`n=9` 的整跨度缓冲填哨兵值，旋转后再逐元素检查
非步长槽位（gap 元素）保持原值；精度比对只看步长槽位，靠这条用例才能发现索引换算错位导致的静默写坏。

## 新增用例

直接在 `arch35/crot_test.csv` 追加一行即可，无需改代码。注意：

- `incx`/`incy` 为负时，缓冲区按 `(n-1)*|inc|+1` 分配，第 i 个逻辑元素放在 `(n-1-i)*|inc|` 位置；
- `incx`/`incy` 为 0 时按参考实现语义（同一元素串行旋转 N 次），缓冲区按 n 分配；
- 空指针负向用例把对应填充列写成 `NULLPTR`，并把 `expect_result` 设为 `ACLBLAS_STATUS_INVALID_VALUE`。

## 溢出拒绝用例的测法说明

`crot_host.cpp` 中 `(n-1)*|stride| > INT32_MAX` 的 `ACLBLAS_STATUS_INVALID_VALUE` 分支**无法通过 CSV 覆盖**：
`aclblasCrot_npu` 在调用算子前会先按 `(n-1)*|inc|+1` 分配显存，`incx=INT32_MIN` 需要约 17 GB，会先分配失败，
根本走不到校验分支。

因此该分支由固定用例 `CrotTest.StridedOffsetOverflowRejected` 覆盖：它直接用少量显存调用公共接口
`aclblasCrot`（绕过 wrapper 的预分配），验证：

- `n=3, incx=INT32_MIN` → `INVALID_VALUE`
- `n=3, incy=INT32_MIN` → `INVALID_VALUE`
- `n=3, incx=INT32_MAX`（`2*INT32_MAX > INT32_MAX`）→ `INVALID_VALUE`
- `n=1, incx=INT32_MAX`（跨度仍为 1 个元素）→ `SUCCESS`，边界不误杀

边界成功用例必须让 kernel 只访问真实分配的显存：`n=2` 配 `INT32_MAX` 时跨度是 2^31 个元素（16 GiB），
用小块显存跑会构成故意的越界访问，因此该形状由 wrapper 按完整跨度分配的 `TC_EX_STBIG` 用例覆盖。

大步长的**成功**路径由 CSV 用例 `TC_EX_STBIG_001 ~ 006` 覆盖（`|inc|` 取 1000 / 4096，含负步长与零步长混合）。
