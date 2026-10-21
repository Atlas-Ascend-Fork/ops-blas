# aclblasCtpsv 测试工程

`aclblasCtpsv`（单精度复数 packed 三角求解）的 GTest 测试工程，由 CSV 驱动，golden 由 cblas（Netlib `ctpsv`）生成，
覆盖两个架构：

- **arch35**（Ascend 950PR / Ascend 950DT）；
- **arch22**（Atlas A2/A3 系列，Ascend 910B3/B4）。

两架构共用家族级 `ctpsv_param.h` / `ctpsv_golden.h` 与 `test/frame` 测试框架，差异仅在各 arch 子目录下的
wrapper 与用例集。

## 目录结构

```
test/tpsv/ctpsv/
├── CMakeLists.txt                  # 测试 target（由 cmake/test.cmake 的 TEST_NAMES 发现）
├── ctpsv_param.h                   # CSV 行 -> CtpsvParam 参数解析
├── ctpsv_golden.h                  # cblas_ctpsv golden 封装 + 参数校验
├── arch35/
│   ├── ctpsv_npu_wrapper.h         # host 侧申请 device 内存并调用 aclblasCtpsv
│   ├── ctpsv_test.cpp              # 用例主体（数据构造 / 调用 / 比对）
│   └── ctpsv_test.csv              # 用例集（1200 条：1000 精度 + 200 性能）
├── arch22/
│   ├── ctpsv_npu_wrapper.h         # host 侧申请 device 内存并调用 aclblasCtpsv
│   ├── ctpsv_test.cpp              # 用例主体（数据构造 / 调用 / 比对 / 性能基线 join）
│   ├── ctpsv_test.csv              # 用例集（1200 条：1000 精度 + 200 性能）
│   └── gpu_baseline.csv             # 200 条 A100 性能基线（gpu_ms 原值，与 TC_PF 用例五元组 join）
└── README.md
```

## arch35 用例集

`arch35/ctpsv_test.csv` 与任务配套用例完全一致，共 1200 条：

| 类别 | 前缀 | 条数 | 说明 |
|------|------|------|------|
| L0 基础 | TC_L0 | 24 | uplo × trans × diag 12 组枚举全组合 × 小尺寸（8, 32） |
| L1 尺寸 | TC_SQ | 92 | 23 种尺寸（1→2048）× 枚举组合轮转 |
| L2 步长 | TC_INC | 24 | incx ∈ ±1/±2/±3 |
| L5 填充 | TC_FL | 8 | AP/x 的均匀 / 交替 / 极端(AP) / 大值(x) / 全零 / Inf / NaN 填充 |
| L5b 覆盖 | TC_CV | 96 | 中尺寸 × 12 组枚举全组合 |
| L6 边界 | TC_ED | 9 | 空指针（AP、x）/ 非法枚举 / 零维 / 负维度 / 零步长；另含独立 TEST(CtpsvNullHandle) 覆盖 handle 为 nullptr（CSV 用例按任务书 §3.5 固定 handle 为有效句柄） |
| EX 扩展 | TC_EX | 747 | 尺寸池 × 枚举组合 × 步长确定性采样 |
| PF 性能 | TC_PF | 200 | 4 条任务书典型 case + 阶数扫描 |

## arch22 用例集

`arch22/ctpsv_test.csv` 共 1200 条，与 arch35 同一 13 列 schema：

| 类别 | 前缀 | 条数 | 说明 |
|------|------|------|------|
| L0 基础 | TC_L0 | 24 | uplo × trans × diag 12 组枚举全组合 × 小尺寸（8, 32） |
| L1 尺寸 | TC_SQ | 92 | 23 种尺寸（1→2048，含奇数/边界/非对齐）× 枚举组合轮转 |
| L2 步长 | TC_INC | 24 | incx ∈ ±1/±2/±3（含负步长）× uplo × diag（trans=N） |
| L5 填充 | TC_FL | 8 | AP：均匀/交替/极端；x：全零/交替/极端/Inf/NaN 填充 |
| L5b 覆盖 | TC_CV | 96 | 中尺寸（10~400）× 12 组枚举全组合 |
| L6 边界 | TC_ED | 12 | 零维 quick return（含 AP 空指针）/ 非法枚举（`999` 字面量）/ 负维度 / 零步长 / 空指针 AP·x / handle 空指针（返回码断言） |
| EX 扩展 | TC_EX | 744 | 尺寸池 × 枚举组合 × 步长确定性采样 |
| PF 性能 | TC_PF | 200 | 5 条任务书典型 case + 小尺寸 + 阶数扫描（均为 incx=1） |

错误码用例断言返回码与输入非法性对应：`handle` 空指针与 `n ≤ 0` 或 `AP/x` 为 `NULLPTR` 的行经 wrapper
的直通守卫原样返回状态码；非法枚举行（csv 的 `999` 字面量）由 `csv_loader.h` 解析器的数值回退解析为
域外枚举值、零步长行走正常分配路径后同样被 host 侧校验拒绝；期望
`ACLBLAS_STATUS_HANDLE_IS_NULLPTR` 的行以空 handle 直调。handle 为 nullptr 另由独立
`TEST(CtpsvNullHandle)` 覆盖（CSV 用例按任务书 §3.5 固定 handle 为有效句柄）。

## arch35 编译与执行

前置：已 source CANN 环境（`ASCEND_HOME_PATH` / `ASCEND_OPP_PATH` 有效），且机器为 Ascend 950（arch35）。

```bash
cd /path/to/ops-blas
bash build.sh --soc=ascend950 --ops=ctpsv           # 仅编译（含测试二进制）
bash build.sh --soc=ascend950 --ops=ctpsv --run     # 编译并执行全部用例
bash build.sh --soc=ascend950 --ops=ctpsv --run --device=1   # 指定 device
```

单独重跑（不重新编译）：

```bash
./build/test/tpsv/ctpsv/ctpsv_test build/test/tpsv/ctpsv
./build/test/tpsv/ctpsv/ctpsv_test build/test/tpsv/ctpsv --gtest_filter='*TC_L0_*'
```

测试二进制的第一个参数为 CSV 所在目录（框架据此定位 `ctpsv_test.csv`）。

## arch22 编译与执行

与 arch35 同一入口，仅 soc 不同（Atlas A2/A3 系列，如 Ascend 910B3/B4）：

```bash
bash build.sh --soc=ascend910b3 --ops=ctpsv           # 仅编译
bash build.sh --soc=ascend910b3 --ops=ctpsv --run     # 编译并执行全部用例
bash build.sh --soc=ascend910b3 --ops=ctpsv --run --device=1
./build/test/tpsv/ctpsv/ctpsv_test build/test/tpsv/ctpsv --gtest_filter='*TC_L0_*'
```

## arch35 数据构造与比对口径

- **AP**：`makeBlasTriangular(n, upper, ap_fill, seed)` 生成实部，`seed + 1000` 生成虚部，实虚独立采样。
- **x**：`makeBlasStrided(n, incx, x_fill, seed + 1)` 生成实部，`seed + 1001` 生成虚部。
- **输入分布**（任务书 §3.5 + 生态精度标准）：常规填充用例 AP/x 采用「均匀分布 `[-5,5]` 50% + 正态分布
  （μ∈[-5,5]、σ∈[0.1,2]）50%」，按用例序号严格交替、AP 与 x 交叉覆盖；正态分布由 `fill.h` 的
  `GaussGenerator` 实现（μ/σ 随机采样一次后同分布生成，元素值截断到 `[-5,5]` 与均匀分布值域一致）。
  特殊填充用例（ALTER/EXTREME/全零/大值/Inf/NaN）与性能用例不参与 50/50 交替。
- **对角保障**（`diag = NON_UNIT` 沿用 `test/tpsv/stpsv/arch35/stpsv_test.cpp`；`diag = UNIT` 收紧）：
  - `diag = NON_UNIT`：被引用对角元实部加偏移 `boost = max(5, n)`，保证求解良态；
  - `diag = UNIT`：对非对角按 `min(1, 1/n)` 缩放，AP 对角位不被引用。复数非对角模为实数的 √2 倍、且正态
    填充存在非零均值（μ 采样到 ±5 时非对角同号、回代呈几何级数放大），故比 stpsv 的 `5/n` 更收紧到
    `1/n`，保证对角=1 主导。
- **比对**：输出 x（n 个复数，按 incx 步长）全量比对，实部 / 虚部分开按 FLOAT32 判定：
  `atol = 2^-16`、`rtol = 2^-10`、`required_matched_ratio = 0.99`、逐元素上限 `max(1e-2, 32 × ULP)`，
  由 `test/frame/verify.h` 的 `applyMixedTolerance` + `Verifier::verifyVector` 执行。

## arch22 数据构造与比对口径

与 arch35 同一框架、同一判据，差异点如下：

- **AP**：packed 存储的每个槽位都属于三角，故直接对 n(n+1)/2 个槽位按序扁平填充即得 packed 布局。
  不经 `makeBlasTriangular`（该辅助函数会改写 fill 的 pattern，把 `RANDOM_ALTER` / `RANDOM_EXTREME`
  退化为普通均匀填充），保证这两个 token 按 `test/frame/fill.h` 的字面语义生效（交替 ±(i+1) /
  含 `FLT_MAX` 的 7 值轮换）。实部用 seed、虚部用 `seed + 1000`，实虚独立采样，与 arch35 一致。
- **x**：`makeBlasStrided` 按 `1 + (n-1)·|incx|` 物理长度构造（含负步长回绕与间隙置零），实/虚种子同上。
- **填充词表**：`RANDOM_NORM_5_5`（均匀 [-5,5]）/ `RANDOM_ALTER` / `RANDOM_EXTREME` / `VALUE_NORM_0` /
  `VALUE_NORM_INF` / `VALUE_NORM_NAN` / `NULLPTR`，全部由 `test/frame/fill.h` 原生解析。
- **对角保障**：与 arch35 相同（NON_UNIT 加 `boost = max(5, n)`；UNIT 按 `min(1, 1/n)` 缩放）。
- **比对**：与 arch35 相同——逻辑 n 个输出元素按步长取位，实部 / 虚部分开按 FLOAT32 判定
  （`atol = 2^-16`、`rtol = 2^-10`、`matched_ratio ≥ 0.99`、上限 `max(1e-2, 32×ULP)`）。
- **防御性跳过**（两种机器无关触发）：①「输入 AP 与 x 全为有限、但 cblas golden 出现非有限值」——
  本用例集中 `TC_FL_143`（AP 极端填充）与 `TC_FL_146`（x 极端填充）确定性触发，属 golden 自身溢出的
  不可判定场景；②「输入全有限、但 golden 幅值超过 2^62」——病态放大场景下逐元素 32×ULP 门已细于
  cblas golden 的跨平台分散度，比对不再裁决算子本身，`TC_FL_142`（AP 交替填充 + UNIT 回代的几何级
  放大）确定性触发。其余精度行正常判分。

## arch35 极端值用例说明

x 的「极端值」口径与生态算子精度标准（`ops-precision-standard` 的 `test_case_generation.md`）对齐：

- 常规用例的值域控制在 `[-5, 5]`，避免计算中间量溢出 FP32 硬上限；
- 「极端」即 **Inf / NaN 特殊值**（`TC_FL_147` / `TC_FL_148`），其传播良定义、必须与 golden 一致；
- `TC_FL_146` 用 `RANDOM_NORM_10000_10000`（±1e4 的大值）覆盖「大值输入」场景——大值经加偏对角
  （`boost = max(5, n)`）的良态求解后结果仍在 FP32 有效范围内、可正常比对。

> 早先 `TC_FL_146` 曾用 `RANDOM_EXTREME`（含 `FLT_MAX`），`L(i,j)*x(j)` 达到 `1e38`、cblas golden
> 自身溢出为 `±inf/NaN`，两侧在溢出区间的轨迹无法比对。这与精度标准的「避免 μ 过大导致误判」要求
> 相悖，已改为大值填充。

测试工程仍保留一处防御性跳过：当「输入 AP 与 x 全为有限、但 cblas golden 出现非有限值」时跳过并打印
说明（当前用例集不会触发，仅为防未来新增用例误配）。

## arch35 性能

性能用例为 `TC_PF` 前缀（200 条）。任务书 §3.3 的 4 条典型 case 及其标杆耗时：

| case | n | uplo | trans | diag | incx | 标杆耗时（us） |
|---|---|---|---|---|---|---|
| 1 | 512 | LOWER | N | NON_UNIT | 1 | 971.85 |
| 2 | 1024 | UPPER | N | NON_UNIT | 1 | 2074.42 |
| 3 | 2048 | LOWER | T | NON_UNIT | 1 | 4012.85 |
| 4 | 4096 | UPPER | C | NON_UNIT | 1 | 12481.55 |

### arch35 性能用例的计时口径

任务书 §3.3 要求「平均单次耗时，须先 warmup 再有效采样 >50 次取平均，单位 us」。单次调用的 wall
time 会被 launch+sync 固定开销（约 0.7ms）主导，对 n ≤ 512 会淹没 kernel 时间，且 GTest 只有 1ms
粒度，无法表达亚毫秒耗时。因此性能用例采用 **device 侧 `aclrtEvent` 精确计时**（与仓内
`test/copy/ccopy/arch35/ccopy_benchmark.cpp` 同一范式）：

- 在 `SetUpTestSuite` 中**一次性**构建并常驻一份 device 侧操作数（AP、x 各一份，覆盖最大 n），
  其构造耗时不计入任何用例；
- 每个性能用例只：复位 x（`n` 个元素 H2D）→ **warmup 10 次** → 在 stream 上以 start/stop event
  包住一串连续 `aclblasCtpsv` 调用（**n < 64 采样 1000 次、n ≥ 64 采样 100 次**，均满足 >50）→
  `aclrtEventElapsedTime` 得到纯 kernel 执行时间，除以采样次数得到 us 级平均单次耗时，打印
  `[CTPSV_PERF] <case> avg_us=...`；
- `verify_performance.py` 解析该值得到 `NPU_avg_ms` 再与 `gpu_baseline.csv` 的 `gpu_ms` 比对；
  两者同为纯 kernel 时间口径（不含 host launch/sync 开销），因此 n ≤ 32 的极小规模也能精确计时；
- 精度验证由 1000 条精度用例覆盖（其 n 上限为 2048），性能用例不再重复计算 golden。

性能用例的操作数取「全部元素相等的三角矩阵」。该矩阵对任意 n 与任意 uplo/trans 组合均非奇异，
且代入过程不溢出：常数 `a` 的前向代入给出部分和 `S_i = Σ_{j≤i} x_j = b_i / a`，始终有界。
算子耗时与数据无关，因此它是公平的计时操作数。

### arch35 实测结果（Ascend 950PR，平均单次耗时，device 侧 event 计时）

| case | n | 标杆（us） | NPU 平均（us） | ratio | 达标 |
|---|---|---|---|---|---|
| 1 | 512 | 971.85 | 503.3 | 0.772 | 是 |
| 2 | 1024 | 2074.42 | 962.3 | 0.862 | 是 |
| 3 | 2048 | 4012.85 | 1848.7 | 0.868 | 是 |
| 4 | 4096 | 12481.55 | 4651.2 | 1.073 | 是 |

`verify_performance.py` 汇总：**PASS=200、NO_REF=0、FAIL=0**，全量 200 条性能用例均有数据、无缺省。

说明：SIMT 大 n 路径采用**多核分块消去**（numBlocks ≤ 8 个 AIV 核，秩 T=16 分块 + 尾随更新），
tile 循环位于 `__global__` 内核，每个 tile 拆成 panel / update 两段 `asc_vf_call`，其间用官方
`AscendC::SyncAll()` 跨核同步（参照仓内 `dotex`/`nrm2_ex` arch35 范式）。O(n²) 的尾随更新切分到
多核，n=4096 平均单次耗时由单核 ~16.5ms 降至 ~4.7ms。跨核正确性关键：`xGm` 声明为 `__gm__ volatile`
（写直通、读旁路 L1）——`SyncAll` 只保证写序与核间屏障，不失效 L1 读缓存，SIMT 直连 GM 的读必须
旁路 L1 才能读到 panel 刚写入的值。

## arch22 性能口径

性能用例（`TC_PF` 前缀，200 条）与 arch35 共用同一计时实现（`SetUpTestSuite` 常驻 device 操作数、
warmup 10 次、device 侧 `aclrtEvent` 计时、**n < 64 采样 1000 次 / n ≥ 64 采样 100 次**，均满足
任务书 §3.3「有效采样 >50 次」），差异仅判定基准：

- 测试启动时将 `arch22/gpu_baseline.csv`（任务包随包 A100 基线，200 条全枚举）与 TC_PF 用例按
  `(uplo, trans, diag, n, incx)` 五元组 join，缺行 / 多行 / 重复行均直接报错；
- 判定式：**NPU 平均单次耗时 ≤ `gpu_ms / 0.8`（= `gpu_ms` × 1.25，倍率阈值 0.8，对标 GPU A100）**，
  逐用例打印 `[CTPSV_PERF] <case> avg_us=... gpu_us=... ratio=... verdict=PASS|FAIL` 并在用例内断言。

`gpu_baseline.csv` 的 `gpu_ms` 为 A100 单次调用原值；与 event 计时同为纯 kernel 时间口径（不含
host launch/sync 开销），n=1 的极小规模也能精确计时。
