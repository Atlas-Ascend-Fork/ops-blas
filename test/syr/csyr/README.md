# csyr ST 测试工程（test/syr/csyr）

`aclblasCsyr`（复数对称秩-1 更新，kernel 直调）的 CSV 驱动 GTest 测试工程。
测试设计真值源：本 README（17 列 CSV 结构、填充 token 表、G1-G4 落位、执行前注记）。

## 1. 工程结构

```text
test/syr/csyr/
├── CMakeLists.txt              # ops_blas_add_gtest_tests(csyr_test csyr_ut)
├── csyr_param.h                # 17 列 CSV 参数解析
├── csyr_golden.h               # 自实现 golden（不共轭、对角普通复数、仅 uplo 三角；
│                               #   按列流式 API 供大 n 用例现算现比）
├── csyr_fill_ext.h             # 填充扩展（RANDOM_GAUSS_5_2 / VALUE_NEG_ZERO /
│                               #   INJECT_NAN_ONE_ELEMENT / SINGLE_POS_INF），其余委托 test/frame/fill.h
├── csyr_verify_ext.h           # 精度双门累加器（spec 门 + CSV MERE/MARE 门，实/虚分判，NaN==NaN 判等）
│                               #   + G3 快照/整存 bitwise 比较器（与 gtest 解耦，可独立复用）
├── README.md                   # 本文件（G4 口径说明见 §4）
└── arch22/
    ├── csyr_test.cpp           # 4 个 CSV 驱动 suite（统一 INSTANTIATE 前缀 Csyr）：CsyrL0Test(18)/CsyrL1Test(68)/CsyrL2Test(40) + 回归 CsyrFullTest(csyr_test.csv smoke 前 50)
    ├── csyr_ut.cpp             # host/tiling 单元测试（独立 gtest 二进制）
    ├── csyr_npu_wrapper.h      # H2D/D2H 封装（host 指针直通规则与 x_len 短缓冲支持）
    ├── csyr_l0_test_cases.csv  # L0 用例表
    ├── csyr_l1_test_cases.csv
    ├── csyr_l2_test_cases.csv
    └── csyr_test.csv           # 回归表（smoke 版：任务用例池前 50 条，共享 PreSmoke 池口径）
```

## 2. 复现步骤

环境：一台已装 CANN 9.1.0 的 NPU 机器（宿主或容器均可），NPU 卡号由 `build.sh --device=N` 指定
（会把 `-DTEST_DEVICE_ID=N` 编入测试二进制；默认 0 号卡）。

```bash
# 编译（ops-blas 仓根目录）
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh && \
  bash build.sh --ops=csyr --soc=ascend910b3 --device=0
```

运行 gtest（分级执行；`--device` 同时透传给 build.sh）：

```bash
./build/test/syr/csyr/csyr_test --gtest_filter='*/CsyrL0Test.*'            # L0 18 条
./build/test/syr/csyr/csyr_test --gtest_filter='*/CsyrL1Test.*:*/CsyrL2Test.*'      # L1+L2 108 条
./build/test/syr/csyr/csyr_test --gtest_filter='*/CsyrFullTest.*'          # 回归：csyr_test.csv 前 50 条 smoke
./build/test/syr/csyr/csyr_test --gtest_filter='*/CsyrFullTest.*-*TC_PF*'  # 回归非性能部分

# 4 标杆 case 的 kernel 纯耗时以 msprof 采集（warmup 后有效采样 >50 次取平均）：
#   msprof --application ./build/test/syr/csyr/csyr_test \
#          --gtest_filter=**/L2_034:...
```

## 3. CSV 列结构与填充 token

17 列（csyr_test.csv 为 14 列子集，缺省三列走解析默认值）：`case_name, description, uplo, n, alpha_real, alpha_imag, x, incx, a, lda,
x_len, handle, untouched_check, expect_result, mere_threshold, mare_multiplier, random_seed`。

| 列 | 语义 |
|---|---|
| x_len | 空=默认 n；数字=x 逻辑长度覆写（物理缓冲 1+(x_len-1)·\|incx\|，K 维不匹配场景） |
| handle | VALID（默认）/ NULLPTR（以 nullptr 句柄调用，预期 9） |
| untouched_check | TRUE：执行前快照未引用三角（含 padding 行），执行后逐元素 **bitwise** 比对一致 |
| expect_result | ACLBLAS_STATUS_{SUCCESS(0), INVALID_VALUE(3), HANDLE_IS_NULLPTR(9), INVALID_ENUM(10)} |
| mere_threshold / mare_multiplier | 2⁻¹³ / 10.0，工程 MERE/MARE 双门（与 spec 门同时判定） |

填充 token：`RANDOM_NORM_5_5`（均匀 [-5,5]）、`RANDOM_GAUSS_5_2`（正态，μ~U(-5,5)、
σ~U(0.1,2)，实/虚独立）、`VALUE_NORM_0/INF/NAN`、`VALUE_NEG_ZERO`（位级 -0.0，
G3 掩码符号翻转暴露）、`RANDOM_ALTER`、`RANDOM_EXTREME`、`INJECT_NAN_ONE_ELEMENT`、
`SINGLE_POS_INF`、`NULLPTR`。

A 侧填充说明：A 全存储按 token 填充（未引用三角持有非平凡值，强化 G3 检测力）；
精度比对仅覆盖 uplo 三角（含对角），未引用三角/padding 由快照断言锁定"不读不写"。
`alpha_real=null` 表示 alpha 空指针负向用例；n/lda/incx 负向值按字面透传（不钳位）。

## 4. G4：n 上限统一口径（必读）

**本算子统一支持上限 n ≤ 4864**。当前构建（S=3 三重缓冲）B=1 起决定作用的布局为交织
瘦槽域：`4·nPad + S·(2·nPad+64) + 32 ≤ 49152 floats`（即 `10·nPad + 224`，S=3；
nPad ≤ 4892 均可容纳，配额沿用 4864）。S=3 下 SoA 回退布局仅容纳 nPad ≤ 4032，
派发动态降级（csyr_host.cpp batchCols 预算）。

- `n = 4864`（L2_018）与 `n = 4863`（L2_020，非 64 倍数尾块）为合法输入，返回 SUCCESS；
- **`n > 4864`（L2_019，n=4865）返回 `ACLBLAS_STATUS_INVALID_VALUE`——超出支持上限
  的输入按参数非法拒绝，口径以上述 UB 预算推导为准，特此说明**；
- 代码落点：`blas/syr/arch22/csyr_host.cpp` 的 `CSYR_MAX_N = 4864` 校验（校验序第 3 步）。
- **lda/|incx| 上界**：元素级 GM 偏移跨度
  `(n-1)·lda + n`（A）与 `(n-1)·|incx| + 1`（x）须 ≤ 2^31，超界返回
  `ACLBLAS_STATUS_INVALID_VALUE`（n==0 quick return 不受影响；契约见
  blas/syr/README.md 的 aclblasCsyr API 说明）。

## 5. 实现注记

1. **大 n 内存（L2_018/020）**：n=4864 时 A≈189MB/条。工程采用**流式 golden**
   （`ComputeGoldenCsyrColumn` 逐列现算现比，不物化整幅 golden 矩阵），host 峰值
   ≈ 输入快照 + actual 两份 ≈ 380MB + 设备侧 A 一份，单用例预算内。
2. **golden 结合序对齐**：golden 与 NPU kernel 同样把 α 折叠进列标量
   `t_j = α·x(j)`，再按 kernel 的 muls+融合 axpy（vmla 单舍入）粒度对齐：
   `Δre = fmaf(x(i).im, −t_j.im, x(i).re·t_j.re)`、
   `Δim = fmaf(x(i).im, t_j.re, x(i).re·t_j.im)`（乘加结合顺序对齐；fmaf 为显式库调用，`-ffp-contract=off` 仅约束其余表达式）。
   两段舍入（先积后加各舍一次）的参考实现会在 α=1e6 近抵消元素上把大乘积项
   （~1.8e8）的半 ULP（8.0）泄漏为绝对误差，触碰逐元素门 max(1e-2, 32·ULP)；
   fma 对齐后 kernel 与 golden 全 α 量级 bitwise 一致。
3. **K 维不匹配（L2_005/006）**：CSV 登记 expect=INVALID_VALUE（spec 建模
   raises_error）。C 接口无从感知 x 物理长度（host 校验序不含该项，实现设计裁定），
   执行前须知：若实现不可检测，该两条降级为
   "进程不崩溃 + 未下发越界写（G3 快照佐证）"并在报告记录。工程已按 x_len 列
   分配短缓冲并以声明的 n 调用，两种结论都能承载。
4. **quick return（n=0 / α=(0,0)）**：除返回 SUCCESS 外，断言 A 全存储 bitwise 不变；n=0 时存储为空自然满足。
5. **NaN 判等**：actual 与 golden 同为 NaN 判 matched（±Inf 位相等亦判 matched）；
   NaN payload / ±0 符号差异仅在 G3 快照（bitwise）中报告。
6. **确定性（L1_085/086）**：同参连跑 3 次，每次对 golden 按容差双门判定，run_i 对
   run_1 再按同门比对（不要求 bit-exact，spec determinism.required=false）。
7. **性能用例**：`pf_smoke*`（L0，warmup 5 + 采样 10）只验链路不判达标；
   `pf_*`/`TC_PF*`（L2，warmup 10 + 采样 60）输出
   `[PERF] <case> n=<n> uplo=<U|L> incx=<i> avg_time=<x> us (...)`；正式达标判定以
   msprof kernel 纯耗时为准。
8. **已知敏感性**：α=1e6 类大值用例在 float32 下存在抵消点（α·x(i)·x(j) 中间量
   ~1e7、结果可近零），任何 ULP 级结合序差异都可能触碰 max_abs_error 门
   （max(1e-2, 32·ULP)）。golden 已按 kernel 结合序对齐以最小化该差异；若此类
   用例仍失败，属真实精度发现，按规格口径记录，不放宽用例。
