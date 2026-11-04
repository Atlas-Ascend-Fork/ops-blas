# aclblasStrsm 测试指导

## 测试范围

`arch35/strsm_test.csv` 为 CSV 驱动用例集，配合 `arch35/strsm_test.cpp` 中 3 个独立负向测试（`NullHandle` / `NullAlpha` / `NullB`），按 L0 / L1 / L2 三层组织：

- `TC_L0_` 功能与边界：side / uplo / trans / diag 基础组合、`m==0` / `n==0` quick-return、最小 shape、`alpha==0` 置零、非方阵 B、负向维度与 lda。
- `TC_L1_` 组合与负向扩展：UNIT / NON_UNIT 全组合、prime 维度、特殊 alpha（极大 / 极小 / 负值）、非法枚举（side / uplo / trans / diag）、lda / ldb 越界（含 0 与负值）、A 侧 NaN / Inf 传播、A / B 空指针（含 `alpha==0` 且 `A==nullptr` 的合法组合）。
- `TC_L2_` 设备路径与大 shape：SIMT / blocked 分派边界、RIGHT 大 n 的 workspace 布局（trans=N 与 trans=T/C 两种 temp 布局）、blocked 路径 UNIT / NON_UNIT 大 m 用例、lda / ldb 非对齐 padding。

用例字段（CSV 列）：`side / uplo / trans / diag / m / n / alpha / a / b / lda / ldb / random_seed / expect_result / mere_threshold / mare_multiplier`。`a`、`b` 列使用 `test/frame/fill.h` 的填充语法；`NULLPTR` 表示空指针，`VALUE_NAN` / `VALUE_INF` 表示特殊值填充。

## 用例构造约定

- 用例夹具对 NON_UNIT 用例的 A 对角元统一叠加 `max(5, aDim)` 的 boost，并按 uplo 清零非引用三角；`RANDOM_ILLCOND` 模式不做 boost。
- UNIT 用例不读取存储对角元，boost 对求解结果无影响。
- blocked 路径（`aDim > 128`）的大 m 用例 A 采用 `RANDOM_NORM_{0.4/aDim}` 对角占优构造：非对角元量级 `±0.4/aDim`，列和上界约 0.4，远离奇异与病态区域，保证 CPU golden 与设备结果在同一精度框架下可比。不要用「随机满幅 A 的 UNIT 大 m」做通过判据——unit 对角三角求解在随机满幅输入下条件数随 m 增长，golden 与实现的差异由问题病态性主导（可复现 MARE 1e-2 量级），不反映算子缺陷。

## 精度标准

golden 使用 `strsm_golden.h` 的 `aclblasStrsm_cpu`（经 `cblas_compat.h` 走系统 BLAS 的 `cblas_strsm`）。校验顺序与算子实现 `blas/trsm/arch35/strsm_host.cpp` 的 `aclblasStrsm` / `ValidateStrsmParams` 一致：句柄 → 枚举与维度 → alpha → `m==0||n==0` quick-return → B / A → lda / ldb 下界。

精度判定采用 `MereMareStrategy`（`test/frame/verify.h`）：MERE < 2^-13 且 MARE < 10 × 2^-13，阈值可按用例通过 `mere_threshold` / `mare_multiplier` 列覆写；golden 与输出均为 NaN / 均为 Inf 时按相等跳过，单侧非有限值计入失配；|golden| 低于 MERE 阈值的近零元素跳过统计，但输出超过绝对兜底（默认 1e-2）会计入违例。

## 环境

- 硬件：Ascend 950PR / Ascend 950DT（strsm 仅有 arch35 实现，Atlas A2 / A3 等 arch22 SoC 无对应算子，构建时自动跳过，UT 需在 950 实机执行）
- 参考 ctrsm 测试指导：`test/trsm/ctrsm/README.md`

## 编译

在 ops-blas 根目录执行：

```bash
bash build.sh --soc=ascend950 --ops=strsm
```

或使用 CMake 直接构建测试目标（Netlib 路径按实际环境调整，系统 `libblas.so.3` 可能通过 alternatives 指向 OpenBLAS）：

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DREFBLAS_LIB=/usr/lib/x86_64-linux-gnu/blas/libblas.so.3 \
  -DREFBLAS_INCLUDE_DIR=/usr/include/x86_64-linux-gnu \
  -DSOC_VERSION=ascend950 \
  -DBUILD_TEST=ON \
  -DTEST_NAMES=strsm
cmake --build build --target strsm_test -j2
```

## 运行

```bash
export ASCEND_RT_VISIBLE_DEVICES=0

# 全量 CSV 用例 + 独立负向测试
./build/test/trsm/strsm/strsm_test

# 仅 CSV 驱动用例
./build/test/trsm/strsm/strsm_test --gtest_filter='Strsm/StrsmArch35Test.CsvDriven/*'

# 单用例排查（示例）
./build/test/trsm/strsm/strsm_test --gtest_filter='*TC_L2_042*'
```

若设备不是逻辑 0 卡，请将 `ASCEND_RT_VISIBLE_DEVICES` 改为验收环境分配的设备号。
