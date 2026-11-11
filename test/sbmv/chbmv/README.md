# aclblasChbmv 测试说明

## Ascend 950PR（arch35）

`aclblasChbmv`（单精度复数 Hermitian 带状矩阵-向量乘，`y = alpha * A * x + beta * y`）在 Ascend 950PR
（arch35 / DAV-3510）上的精度与性能测试。

### 目录结构

| 文件 | 说明 |
| --- | --- |
| `CMakeLists.txt` | 注册 `chbmv_test` 测试目标 |
| `chbmv_param.h` | CSV 行到用例参数的解析 |
| `chbmv_golden.h` | Netlib `cblas_chbmv` 参考实现封装 |
| `arch35/chbmv_npu_wrapper.h` | Host/Device 搬运与算子调用 |
| `arch35/chbmv_test.cpp` | CSV 驱动的 GTest 用例与专项 `TEST_F` |
| `arch35/chbmv_test.csv` | 任务配套用例：1000 条精度 + 200 条性能（`TC_PF_` 前缀） |
| `arch35/chbmv_test_supplement.csv` | 补充用例 18 条：`lda > k + 1` 的带宽前导维 padding（主 CSV 全为紧凑 `lda == k + 1`，无法暴露列步长误用 `k + 1`） |

### 编译

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cd <ops-blas 仓库根目录>
bash build.sh --soc=ascend950 --ops=chbmv
```

产物：

- 算子库 `build/libops_blas.so`
- 测试程序 `build/test/sbmv/chbmv/chbmv_test`

### 精度测试

```bash
BLAS_DEVICE_ID=0 ./build/test/sbmv/chbmv/chbmv_test --gtest_filter='-*TC_PF*'
```

`chbmv_test_supplement.csv` 与主 CSV 同目录时会被自动追加到参数化用例集；主 CSV 保持与任务包
逐字节一致，补充用例单独成表，覆盖 `lda = k + 2 / k + 3 / k + 4`、正负步长与两种三角。
用例集完整性由 GTest 启动时的 CSV 契约校验保证（列名、1200/200 条数量、`case_name` 唯一），
契约不满足时直接以可读信息失败，不会静默跳过。

判定口径：golden 由 Netlib BLAS 3.10.0 `cblas_chbmv(CblasColMajor, ...)` 生成，实部、虚部分别按 FLOAT32
标准比较（`rtol = atol = 2^-13`、`matched_ratio >= 0.99`、`max_abs_error <= max(1e-2, 32*ULP)`）。
NaN/Inf 按任务书 §3.2 的等效规则判定：同为 NaN、同为同号 Inf、或一方 NaN 一方 ±Inf 视为等效，
有限与非有限不等效，非有限值不计入匹配率统计。

除 CSV 用例外，`chbmv_test.cpp` 另含专项回归：空句柄、对角虚部 Inf/NaN 毒值、正态分布与随机复数
标量、大值防溢出反例（`A = 1e38` 配 `alpha = 0.25`）、以及 `lda > k + 1` 的 padding 覆盖。

### 性能测试

性能以 `msprof op` 采集的 Kernel `Task Duration(us)` 为准。GTest 直接运行性能用例时使用
ACL Event 计时（先 5 次 warmup，再 55 次采样取平均），输出 `[PERF_EVENT] <case> <avg_us>`：

```bash
CHBMV_EVENT_TIMING=1 BLAS_DEVICE_ID=0   ./build/test/sbmv/chbmv/chbmv_test --gtest_filter='*TC_PF*'
```

需要提交 profiler 原始数据时，可对同一用例使用：

```bash
msprof op --application="./build/test/sbmv/chbmv/chbmv_test --gtest_filter=*TC_PF_1001*" \
  --aic-metrics=BasicInfo --warm-up=5 --launch-count=10 --output=./chbmv_prof
```

判定口径：任务书 §3.3 要求 `GPU 平均耗时 / NPU 平均耗时 >= 0.4`（即 NPU 不高于 GPU 的 2.5 倍）。
GTest 逐例打印 NPU 平均耗时，与提交包中的 GPU 基线表按 `case_name` 对齐后即可核算比值。

## Atlas A2 / A3（arch22）

This test suite calls the public `aclblasChbmv` API on an A2/A3 device with CANN 9.1.0. Self-validation needs one supported hardware model; this README does not claim device results. Build, CPU/static checks, device accuracy, event timing, and formal profiler timing are separate evidence.

### Cases and original artifacts

- `arch22/chbmv_test.csv`: the task's unchanged 1,000 accuracy cases and 200 `TC_PF_` performance cases.
- `arch22/gpu_baseline.csv`: the task's unchanged 200 A100 references.
- `arch22/chbmv_supplement_test.csv`: separate generated coverage, automatically appended by `Cases()` in `arch22/chbmv_test.cpp`.
- `gen_supplement.py`: deterministic generator and static contract checker. It never overwrites either original artifact and checks their SHA256 before and after its work.
- `arch22/chbmv_gaussian_mirror_test.csv`: optional, separate equal-budget Gaussian partners; never named `TC_PF_` and never assigned a GPU baseline.
- `gen_gaussian_mirror.py`: deterministic generator/checker, preserving the original 1,200 rows, the existing 131 rows and GPU baseline byte-for-byte.
- An optional `--mapping-report` export records complete source-to-partner parameters, component budgets and real/imaginary seed mapping as separate validation evidence, outside the product-source diff.
- `arch22/chbmv_case_registry.json`: expected registry and distribution accounting; generated coverage is not a device pass.

Original SHA256 identities, verified against the downloaded task ZIP:

```text
chbmv_test.csv    190f8cd07fa4691051280ae421ecb8c721e63dd4abecc519e0beaf35613b658f
gpu_baseline.csv  dd2cb87a50589bde139bd42014e911eb4c8c05f31aac0fbdfb0633359bad4369
```

Generate or check only the supplementary data from the repository root:

```bash
python3 test/sbmv/chbmv/gen_supplement.py
python3 test/sbmv/chbmv/gen_supplement.py --check
python3 test/sbmv/chbmv/gen_gaussian_mirror.py
python3 test/sbmv/chbmv/gen_gaussian_mirror.py --check
```

By default the mirror generator writes or checks only the Gaussian CSV and compact registry. To retain the complete per-case mapping as separate validation evidence, pass an explicit output path (relative to the current working directory):

```bash
python3 test/sbmv/chbmv/gen_gaussian_mirror.py \
  --mapping-report ../chbmv_validation/chbmv_gaussian_mirror_mapping.json
python3 test/sbmv/chbmv/gen_gaussian_mirror.py --check \
  --mapping-report ../chbmv_validation/chbmv_gaussian_mirror_mapping.json
```

The scoped `.gitignore` excludes the legacy `arch22/chbmv_gaussian_mirror_mapping.json` and local `GAUSSIAN_MIRROR_DELIVERY.json`; existing files are preserved. Neither file is required by the NPU test executable. No mapping report is written or checked unless `--mapping-report` is supplied, and `--check` never writes any output.

The printed qualification `STATIC_CSV_CHECK_ONLY_NOT_NPU_PASS` confirms only CSV structure, original identities, deterministic regeneration, paired coverage and parameter contracts. It is not a device result.

### Supplementary coverage and distribution accounting

The supplement has 131 cases: 96 paired random-distribution cases, 18 special-value/degenerate cases, and 17 parameter-priority/overflow cases. None is a new performance case and none receives a fabricated GPU reference.

The random stratum contains 48 pairs: 24 `(n, declared k, lda padding, incx, incy)` configurations, each tested for UPPER and LOWER. Each pair has one `RANDOM_NORM_5_5` case and one `GAUSS_NORM_5_2` case. Members have identical physical matrix and vector element counts, so both row counts and generated element budgets are exactly 50%/50% within this stratum. Alpha and beta each have an equal number of independently sampled float32 components in the two distributions. Their seeds and sampled values are fixed in the CSV.

`GAUSS_NORM_5_2` uses the repository's existing `GaussGenerator`: a mean sampled in `[-5,5]`, a standard deviation in `[0.1,2]`, and samples clipped to `[-5,5]`. The scalar generator uses the same parameter ranges and clipping policy. Real/imaginary tensor fills use separate seeds in `MakeComplex`; real/imaginary scalar components are sampled independently.

The original 96-case random stratum alone was insufficient to balance the much larger original uniform tensor data set. The optional mirror now adds **1,194 accuracy cases**: partners for 1,176 positive-n SUCCESS rows in the original CSV and 18 unpaired special-value rows in the supplement. The existing 48 uniform/Gaussian pairs are retained without another mirror. Every partner keeps geometry, strides, fixed scalar values and special tensor tokens; only `RANDOM_NORM_5_5` tensor fields become `GAUSS_NORM_5_2`. Its seed is the source seed plus 1,000,000,000, with separate recorded real/imaginary offsets. `TC_GS_FROM_PF_1001` and the other 199 original-PF partners are accuracy-only cases and do not match `TC_PF_` filters.

With all three CSVs, genuinely random tensor components have equal generated budgets per real/imaginary component: A **287,263,711 uniform + 287,263,711 Gaussian**, x **686,350 + 686,350**, and y **686,198 + 686,198**. `MakeVector` draws n logical values per component; strided sentinel gaps are not random samples. The ledger separately reports mathematical references, excluding padding/diagonal-imaginary components and alpha/beta-zero terms. Those budgets also match per distribution. Repeated PF launches reuse an input and do not count as additional random draws.

The official `gen_csv.py` contains no `random.uniform` or Gaussian scalar generation: all original alpha/beta values are fixed `SCALARS`, literals or null markers. Their fixed values therefore remain unchanged. Only the existing 96-case supplementary random stratum supplies random alpha/beta: each parameter has **96 uniform + 96 Gaussian real/imaginary components**, with the required independently sampled normal mean/standard-deviation and clipping policy. There are also **4,776 fixed components per scalar parameter** across numeric CSV cases. These fixed/special values are a separate stratum; this README does **not** claim that all scalar values including constants are 50% Gaussian.

Expected full registry is **2,525 CSV cases + 4 hand-written GTests = 2,529 tests**, including 2,484 positive-n SUCCESS CSV numerical comparisons and exactly 200 original performance cases. Positive alpha=0/beta=1 comparisons may be host quick returns, so numerical comparisons must not be relabelled as kernel launches. Without the optional Gaussian mirror the prior registry remains 1,335. `Cases()` prints the actual loaded CSV registry, and the JSON lists all four hand-written names. Input generation is test data; the operator itself is not random.

Coverage includes:

- `n=1` through 2,048, power-of-two boundaries and nonaligned lengths; `k=0/1`, small and broad bands, full bands, and declared `k>n-1`.
- UPPER/LOWER for each paired shape; compact and padded lda; strides `+/-1`, `+/-2`, `+/-3` including opposite signs.
- Declared `k>n-1` retains the original physical band row: the UPPER diagonal is at row `k`, not row `min(k,n-1)`.
- `beta=0` with NaN/Inf in old y, which should not contaminate a finite result; NaN/Inf inputs with nonzero alpha; alpha-zero scaling with finite A/x; beta-one propagation.
- `n=0` combined separately and jointly with invalid enum/k/lda/strides/null pointers, expected SUCCESS after a valid handle. Negative n remains invalid. Existing code also has a null-handle-before-quick-return test.
- `k=INT32_MAX` with inadequate lda, `INT32_MIN` strides even at n=1, and vector span `INT32_MAX+1`; alpha-zero/null-A and beta-zero/null-y still remain invalid at positive n.

Negative and zero-length cases use the runner's tiny dummy buffers. They must not allocate the enormous physical spans implied by invalid inputs.

### Build and real-device accuracy

On the assigned device, load its CANN 9.1.0 environment first, then from the repository root:

```bash
bash build.sh --soc=ascend910b --ops=chbmv
## A3 uses --soc=ascend910_93 instead.
BLAS_DEVICE_ID=0 ./build/test/sbmv/chbmv/chbmv_test --gtest_list_tests
BLAS_DEVICE_ID=0 ./build/test/sbmv/chbmv/chbmv_test --gtest_filter='*TC_SU_*' --gtest_output=xml:chbmv_supplement.xml
BLAS_DEVICE_ID=0 ./build/test/sbmv/chbmv/chbmv_test --gtest_filter='-*TC_PF_*' --gtest_output=xml:chbmv_accuracy.xml
```

For portable packages set `CHBMV_TEST_CASE_DIR=/absolute/case_directory`; this directory contains the original CSV and the optional supplementary/mirror CSVs. If that variable is unset, the binary falls back to its compiled `__FILE__` source directory. Moving a binary therefore does not require retaining its old source path when the environment variable is supplied. For example, use `CHBMV_TEST_CASE_DIR="$PWD/test/sbmv/chbmv/arch22"` before the test command. Capture full stdout/stderr, XML, device model, CANN/driver versions and exact source/library/CSV hashes. Check the printed actual registry and expected JSON; a zero-case run is not a pass. New Gaussian cases have not received a device result merely by generating or listing them.

The golden path promotes original complex64 values to double precision, uses `cblas_zhbmv` in column-major Hermitian-band mode, and rounds the final result to complex64. Record the linked CBLAS implementation/version. Logical y elements are checked in full; real/imaginary components have separate statistics. Finite comparisons use `atol=rtol=2^-13`, matched ratio at least 0.99, and maximum absolute error at most `max(1e-2, 32*ULP)` with ULP anchored at the maximum-error golden value. Non-finite comparisons follow taskbook §3.2 and are counted separately.

The runner checks output guards and unused strided-y slots for positive normal cases. Its current nonnormal CSV branch also downloads y and checks all sentinel/guard bytes after n=0 or expected-error calls. The dedicated `ZeroDimensionDoesNotWrite` test combines n=0 with invalid arguments and verifies unchanged y. `DiagonalImaginaryAndUnusedCellsNaN` explicitly poisons only ignored diagonal-imaginary/unused band locations; `BetaOneMixedInfinityKeepsFiniteComponent` checks mixed finite/infinite components. These current assertions supersede the older README's return-code-only n=0 description. They remain distinct from all-NaN fill CSV cases.

### Performance

Only original `TC_PF_` cases have the provided GPU references. The runner prints `method=aclrtEvent`, with five warmups and ten timed calls. These event averages are distinct from taskbook §3.3's formal `msprof op` Task Duration requirement. Keep event output for cross-checking; do not relabel it as profiler data.

Collect one selected PF case at a time to avoid mixing unrelated kernel samples. For example, after obtaining its exact name from `--gtest_list_tests`:

```bash
BLAS_DEVICE_ID=0 msprof op --application='./build/test/sbmv/chbmv/chbmv_test --gtest_filter=*TC_PF_1001*' --output=./prof_chbmv_case1001
```

Use the actual generated profiler table and documented kernel identity to select at least ten valid post-warmup task durations. Check the matched reference and units (`gpu_ms` versus microseconds); the task criterion is mean NPU duration no greater than GPU duration divided by 0.8. Preserve individual samples and raw tables. Missing Task Duration is unmeasured, not a pass. The taskbook says memory validation is not applicable; retain the required template files with that explanation instead of inventing a memory benchmark.
