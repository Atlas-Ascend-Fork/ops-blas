# aclblasCgbmv 测试工程

`aclblasCgbmv`（单精度复数带状矩阵-向量乘，`y = alpha * op(A) * x + beta * y`）在 Ascend 950PR（arch35）上的精度与性能测试。

## 文件

| 文件 | 说明 |
| --- | --- |
| `CMakeLists.txt` | ascend950 下注册 `cgbmv_test`（精度）与 `cgbmv_benchmark`（性能） |
| `cgbmv_param.h` | CSV 参数解析：trans 支持枚举名与整数；`alpha_real`/`beta_real` 为 `null` 时传空指针；填充串支持 `GAUSS_<μ>_<σ>` 正态分布；可选列见下文 |
| `cgbmv_golden.h` | Netlib 参考 BLAS `cblas_cgbmv` golden，参数检查与算子一致 |
| `arch35/cgbmv_npu_wrapper.h` | Host↔Device 搬运（可选 alpha/beta 放 Device、A/x/y 地址偏移）、调用、同步与回读 |
| `arch35/cgbmv_test.cpp` | 17 条 `TEST_F`（接口/标量/异常结果回归 16 条、T/C 最终加法溢出 1 条）、2 条主机判定器 `TEST` 和 CSV 驱动的 `TEST_P` |
| `arch35/cgbmv_test.csv` | 任务配套用例：1000 条精度 + 200 条性能（`TC_PF_` 前缀）。除 `TC_ED_241` 的 `expect_result` 一格外保持原样，见「对任务配套 CSV 的改动」 |
| `arch35/cgbmv_test_supplement.csv` | 补充用例 1065 条：正态分布数据 36、alpha/beta 纯虚数与大值 30、A/x/y 对齐偏移 18、任务用例的正态分布镜像 981 |
| `arch35/cgbmv_benchmark.cpp` | 性能程序：读取 `TC_PF_` 用例，warmup 后以 event 计时取平均 |

## 编译与运行

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --soc=ascend950 --ops=cgbmv

# 精度：任务 1000 条 + 补充 1065 条 + 17 条 TEST_F + 2 条判定器 TEST（共 2084 条），预期全部通过、退出码 0
./build/test/gbmv/cgbmv/cgbmv_test '--gtest_filter=-*TC_PF_*'
# 只跑补充用例
./build/test/gbmv/cgbmv/cgbmv_test --gtest_filter="CgbmvSup/*"
# 200 条性能用例带 golden 的正确性校验
./build/test/gbmv/cgbmv/cgbmv_test --gtest_filter=*TC_PF_*

# 性能：输出 [CGBMV_PERF] case=... avg_us=...
./build/test/gbmv/cgbmv/cgbmv_benchmark
CGBMV_BENCHMARK_CASE=TC_PF_1001 ./build/test/gbmv/cgbmv/cgbmv_benchmark   # 按用例名子串过滤
```

`CGBMV_BENCH_WARMUP`（≥0，默认 10）与 `CGBMV_BENCH_REPEATS`（>0，默认 100）可覆盖 warmup 与计时次数。

任务配套的 `verify_performance.py` 以 GTest 单条用例的毫秒级整体耗时计时（含造数、H2D/D2H、CPU golden 与比对），在本算子上会输出 FAIL/NO_REF，不代表算子性能；性能以 `cgbmv_benchmark` 的 event 计时为准（可用 msprof 交叉验证）。

## 精度判定

实部、虚部分别按 FLOAT32 判定（复用仓内 `MIXED_TOLERANCE`）：rtol = 2⁻¹⁰，atol = 2⁻¹⁶，matched_ratio ≥ 0.99，逐元素绝对误差上限 max(1e-2, 32·ULP)。golden 为 NaN 的分量要求输出为 NaN，golden 为 Inf 的分量要求输出为同符号 Inf；CSV 中的 `mere_threshold` / `mare_multiplier` 列不参与判定。

输入中不应被读取的位置（带外三角区、lda 填充行、x 步长空洞）填 NaN，y 的步长空洞填哨兵值并逐位校验不变，用于检测越界访问。

用例判为失败时，测试会额外用双精度 `cblas_zgbmv` 在同一份输入上计算高精度参考，打印 NPU 结果与 golden 各自偏离参考的程度（按容差 max(1e-2, 32·ULP) 折算的倍数，以及 |参考| ≥ 1 的元素上的 ULP 数），判定结果不变；设置环境变量 `CGBMV_FP64_STATS=1` 时对每条 CSV 用例都打印这组对比。诊断另列有限比较数、参考非有限数、参考超出 FP32 范围数、非有限不匹配数，以及 matched 数和可表示参考数；零误差不能脱离这些计数解释为通过。

`TCFinalAdditionOverflow` 覆盖 T/C × beta=1/一般复数 × incx=1/-2 共八个子场景，带宽 65、incy=-3，点积和 beta·y 各自有限、仅最终相加溢出，同时校验 y 的步长空洞未被改写；八个子场景均与 golden 一致。

```bash
CGBMV_FP64_STATS=1 ./build/test/gbmv/cgbmv/cgbmv_test '--gtest_filter=-*TC_PF_*' | grep "FP64 reference"
```

## 对任务配套 CSV 的改动

`cgbmv_test.csv` 仅有一处与任务书下发的原件不同：`TC_ED_241`（`invalid_trans`，trans=999）的
`expect_result` 由 `ACLBLAS_STATUS_INVALID_ENUM` 改为 `ACLBLAS_STATUS_INVALID_VALUE`。

原因是同目录的 `aclblasSgbmv` 对同样非法的 `trans` 返回 `ACLBLAS_STATUS_INVALID_VALUE`
（`blas/gbmv/arch35/sgbmv_host.cpp`），两个 gbmv 算子在相同入参错误上应当一致。按评审意见
`aclblasCgbmv` 统一为 `ACLBLAS_STATUS_INVALID_VALUE`，该用例的期望值随之更新。

`expect_result` 列会被 `test/frame/csv_loader.h` 的 `parseStatus` 解析并逐条断言，因此这一格
必须与实现同步，否则 `TC_ED_241` 会失败。任务配套 CSV 的其余 1200 行、以及其中 14 条原本就
期望 `ACLBLAS_STATUS_INVALID_VALUE` 的负向用例均未改动。

## 已过滤的用例

任务配套测试指导允许在自动生成的用例使标杆出现异常行为时，过滤这些用例并给出说明。以下 7 条因 FP32 Netlib golden 本身不可靠而未纳入测试工程：

| 用例 | 过滤原因 |
| --- | --- |
| 补充用例 `TC_SUP_NM_EX_0744/0802/0918/0944`（任务用例 `TC_EX_0744/0802/0918/0944` 的正态分布镜像） | 长同号累加下 FP32 golden 漂移。相对双精度 `cblas_zgbmv` 参考，0918、0944 的 golden 部分分量已超出逐元素上限（1.05～1.15 倍），0744、0802 的 golden 接近上限（0.982、0.998 倍）；算子误差不超过上限的 0.151 倍 |
| 固定输入 `N/T/CGoldenPrefixOverflow` | 串行 golden 在中间累加时溢出为 Inf/NaN，而正确结果为有限值；算子输出有限且与双精度参考一致 |

对应的任务配套原用例 `TC_EX_0744/0802/0918/0944` 仍在 `cgbmv_test.csv` 中，按上述标准判定并通过。

## 补充 CSV 的可选列

| 列 | 默认 | 含义 |
| --- | --- | --- |
| `a_align_offset` / `x_align_offset` / `y_align_offset` | 0 | Device 指针相对分配起点偏移的复数元素个数（偏移前缀填 0xFF 并校验 y 前缀未被写） |

补充用例中的大值 alpha/beta 取模长约 1e2～1.5e2，输出长度 ≥224，全部用例按上面的标准原样判定。
