# aclblasCtbmv 测试说明

本目录同时包含 Ascend 950PR（arch35）与 Atlas A2/A3（arch22）两套测试。两者共用顶层的
`ctbmv_param.h` 与 `ctbmv_golden.h`，各自的测试源、NPU wrapper 与用例数据位于对应的
`arch35/`、`arch22/` 子目录。`CMakeLists.txt` 按构建时的 `SOC_ARCH_DIRS` 自动选择架构目录，
同一份配置在两类设备上都能直接运行。

| 路径 | 内容 |
| --- | --- |
| `ctbmv_param.h`、`ctbmv_golden.h` | 两侧共用的 CSV 参数结构与 golden 入口 |
| `arch35/` | Ascend 950PR：`ctbmv_test.cpp`、`ctbmv_npu_wrapper.h`、`ctbmv_test.csv`、`gpu_baseline.csv` |
| `arch22/` | Atlas A2/A3：`ctbmv_test.cpp`、`ctbmv_npu_wrapper.h`、`ctbmv_test.csv`、`ctbmv_smoke.csv` |

## 精度判定口径

- golden：调用公共 CBLAS 的 `cblas_ctbmv`（工程按 `libblas` 自动解析），输出整个 x（n 元素）
  全量验证，实部、虚部分别按 FLOAT32 判定。
- 阈值：`atol = 2⁻¹⁶ ≈ 1.5259e-5`、`rtol = 2⁻¹⁰ ≈ 9.7656e-4`、`matched_ratio ≥ 0.99`、
  `max_abs_error ≤ 1e-2 或 32×ULP`，与生态算子开源精度标准一致。
- 逐元素通过条件：`|actual - golden| ≤ atol + rtol × |golden|`，实部、虚部分别计算。
- 带状约束：A 仅带内元素有效，golden 与 NPU 两侧均不读取带外区域。
- `diag=UNIT` 语义：两侧均将对角视为 `(1, 0)`，不读取 A 的对角元素。
- 复数类型：`aclblasComplex { float real; float imag; }`，与 CBLAS complex 布局二进制兼容，
  `cblas_ctbmv` 直接以 `void*` 调用。
- 特殊值：测试工程对两侧同为 NaN、两侧同为同号 Inf 的位置做中性化处理后再按有限值容差判定，
  见 `ctbmv_test_util.h`。
- 非法参数用例检查返回码；不将其视为数值误差为零的用例。

CBLAS 的查找和链接由 `ops_blas_add_gtest_tests` 统一处理，不再单独构建 Reference-LAPACK，
不需要 `CTBMV_REFERENCE_ROOT`，不修改公共 CI 入口。依赖安装沿用仓库的 `install_deps.sh`。

公共配置可能找到不同实现或版本的 BLAS。复测时记录 CMake 中 `REFBLAS_LIB` /
`REFBLAS_INCLUDE_DIR` 和实际动态链接库；历史独立参考库的测试结果不能代替此配置的设备复测结果。

## arch35（Ascend 950PR）

### 用例构成

`arch35/ctbmv_test.csv` 包含任务要求的 1200 条用例：1000 条精度、200 条性能；
另有三个独立参数测试。

| 类别 | 前缀 | 条数 | 说明 |
|------|------|------|------|
| L0 基础 | TC_L0 | 12 | uplo×trans×diag 全枚举组合（2×3×2=12）× 小尺寸 (n=8, k=2) |
| L1 尺寸 | TC_SQ | 92 | 23 种尺寸（1→2048）× 枚举组合轮转 |
| L2 带宽 | TC_BD | 22 | k=0/1/小值/半带/满带 n-1 × n∈{4,8,16,32,64} |
| L4 前导维/步长 | TC_LD/TC_INC | 18 | lda=k+1+padding；incx ∈ ±1/±2/±3 |
| L5 填充 | TC_FL | 14 | 均匀/全零/交替/极端值/Inf/NaN；diag=UNIT 对角 NaN/Inf 不读验证 |
| L5b 覆盖 | TC_CV | 24 | 中等尺寸（24/96）× 全枚举组合 |
| L6 边界 | TC_ED | 10 | n=0 quick return；空指针/非法前导维/负维度/负带宽/零步长；非法枚举 999 |
| EX 扩展 | TC_EX | 808 | 尺寸池×带宽×枚举×步长×padding 确定性采样 |
| PF 性能 | TC_PF | 200 | 4 条任务书 §3.3 典型 case + 小尺寸延迟区 + 规模/带宽扫描 |

### 编译与运行

在 ops-blas 仓根目录编译：

```bash
bash build.sh --soc=ascend950 --ops=ctbmv
```

运行全部 CSV 用例：

```bash
export LD_LIBRARY_PATH="$PWD/out/lib64:${LD_LIBRARY_PATH:-}"
./build/test/tbmv/ctbmv/ctbmv_test
```

可用 GTest 过滤器只运行某一类：

```bash
./build/test/tbmv/ctbmv/ctbmv_test --gtest_filter='*TC_L0*'
```

### 独立参数测试

以下三个测试不包含在 1200 条 CSV 用例内：

- `CtbmvArch35Test.UnitDiagonalSpecialValue`：12 组组合（UPPER/LOWER × N/T/C × Inf/NaN）。
  仅把存储的 UNIT 对角写成特殊值，非对角有效元素与 x 保持有限，并用未污染的有限矩阵生成
  golden，用于验证 `diag=UNIT` 时 A 的对角元素不被读取。
- `CtbmvArch35Test.TransposeUbBoundaries`：单个 GTest 内含 72 组参数。覆盖转置/共轭转置、
  上下三角、单位/非单位对角、带宽阈值与尾部边界；在未使用的 padding 与单位对角中填入
  NaN/Inf，要求输出保持有限。
- `CtbmvArch35Test.ZeroOperandSpecialValues`：x 全零、带内全 Inf，incx 取 1 与 2，分别走
  并行 UB 内核与串行 Netlib 内核。参考实现在 x(j)==0 时仍执行该列的乘加与对角缩放，
  `0*Inf` 因此必须传播为 NaN，故断言输出无任何元素保持有限。该断言基于性质而非 golden。

```bash
./build/test/tbmv/ctbmv/ctbmv_test --gtest_filter='CtbmvArch35Test.UnitDiagonalSpecialValue'
./build/test/tbmv/ctbmv/ctbmv_test --gtest_filter='CtbmvArch35Test.TransposeUbBoundaries'
./build/test/tbmv/ctbmv/ctbmv_test --gtest_filter='CtbmvArch35Test.ZeroOperandSpecialValues'
```

### 性能用例

`arch35/ctbmv_test.csv` 中的 200 条 `TC_PF` 用例以 `arch35/gpu_baseline.csv` 记录的 GPU 基线
为参照，每条先预热 10 次、再有效采样 100 次，比较 GPU 基线与 NPU 实测平均耗时。

```bash
./build/test/tbmv/ctbmv/ctbmv_test --gtest_filter='CtbmvArch35Test.PerformanceTaskCases'
./build/test/tbmv/ctbmv/ctbmv_test --gtest_filter='CtbmvArch35Test.PerformanceCsvCases'
```

任务书 §3.3 对以下 4 条给出 Ascend 950PR 标杆耗时：

| case | n | k | uplo | trans | diag | incx | 标杆耗时（us） |
|------|---:|---:|:--|:--|:--|:--|---:|
| TC_PF_1001 | 512 | 8 | UPPER | N | NON_UNIT | 1 | 17.480 |
| TC_PF_1002 | 1024 | 16 | UPPER | N | NON_UNIT | 1 | 27.475 |
| TC_PF_1003 | 2048 | 32 | LOWER | T | NON_UNIT | 1 | 31.872 |
| TC_PF_1004 | 4096 | 64 | LOWER | C | UNIT | 1 | 45.832 |

`CtbmvArch35Test.PerformanceFiniteInputCases` 使用同样的四组尺寸与限值，但每组提前准备
110 份相同的初始 x，10 份用于预热、100 份用于计时，每份 x 只调用一次原地 API，避免反复
覆写同一个 x 导致后续输入溢出。计时为 host 提交至最终 stream 同步的平均值。

## arch22（Atlas A2/A3）

### 正式测试

`arch22/ctbmv_test.csv` 保留任务提供的 1200 条官方用例（含 200 条 `TC_PF_*`），不修改参数、
输入生成或精度阈值。按此前 PreSmoke 时长整改方案，默认使用 `arch22/ctbmv_smoke.csv` 中的
前 50 条官方用例，加上两项独立回归，共 52 项 GTest。通过 `CTBMV_TEST_CSV` 指定完整 CSV，
可执行全部 1202 项。

| 验证内容 | 入口 |
| --- | --- |
| CSV 功能及精度 | `arch22/ctbmv_test.cpp` 的 `CsvDriven` |
| 空 handle | `NullHandle` |
| 非法枚举、尺寸、指针、步长及零维 | 官方 CSV 对应用例 |
| 单位对角、带外 NaN、负步长、未引用槽位 | `SupplementalUnitDiagBandSentinelsAndNegativeStride` |
| CPU golden / NPU 调用 | `ctbmv_golden.h` / `arch22/ctbmv_npu_wrapper.h` |

### 编译与运行

先加载目标机器的 CANN 环境，按仓库说明安装公共 CBLAS、LAPACK 和 GTest。
在仓库根目录运行标准入口：

```bash
bash build.sh --ops=ctbmv --soc=ascend910b3 --run
```

独立 Release 构建及完整回归：

```bash
cmake -S . -B build-release -DSOC_VERSION=ascend910b3 \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=ON -DTEST_NAMES=ctbmv
cmake --build build-release --target ctbmv_test -j8
export LD_LIBRARY_PATH="$PWD/build-release:${LD_LIBRARY_PATH:-}"
CTBMV_TEST_CSV="$PWD/test/tbmv/ctbmv/arch22/ctbmv_test.csv" \
  ./build-release/test/tbmv/ctbmv/ctbmv_test --gtest_color=no
```

### 设备内存检查

内存检查使用独立的本地插桩构建，按目标设备安装版本的 msSanitizer 指引配置，不在公共构建中
新增 Ctbmv 专用开关。

保存完整日志、GTest 结果、实际链接库及设备/工具版本。普通 GTest 通过或检测进程退出码为零，
不等同于无内存诊断；还须核对 ERROR、WARNING、测试覆盖及检测是否完整执行。内存检查开销不计入
正式性能数据。

## 已知失败用例与豁免范围

`TC_FL_151`、`TC_FL_152`、`TC_FL_153`、`TC_FL_158` 四条用例在当前判定下无法与 golden 逐元素
对齐，不计入精度通过统计。**原始用例、golden 与容差均未改动**，每次完整运行仍会照常报出这
四条失败，不做跳过或屏蔽。

四条用例均为 n=32、k=4、lda=5、incx=1：

| 用例 | uplo/trans/diag | 输入 |
|------|-----------------|------|
| TC_FL_151 | UPPER / N / NON_UNIT | A 为 `RANDOM_EXTREME`（含 ±FLT_MAX 与 0），x 为常规随机 |
| TC_FL_152 | LOWER / N / NON_UNIT | x 为 `RANDOM_EXTREME`，A 为常规随机 |
| TC_FL_153 | UPPER / N / NON_UNIT | A 整条带为 `VALUE_NORM_INF` |
| TC_FL_158 | LOWER / C / UNIT | A 整条有效带为 `VALUE_NORM_INF` |

**失败原因**：这些输入在 fp32 下必然产生非有限的中间结果，而 `NaN` 与 `±Inf` 落在哪些元素上，
由两侧的求和次序与 FMA 收缩共同决定。设备侧显式关闭 FMA 收缩（`blas/tbmv/arch35/`
`ctbmv_fallback_kernel.cpp` 中的 `#pragma clang fp contract(off)`），golden 侧则调用工程按
`libblas` 解析到的 `cblas_ctbmv`；该库由系统解析（`cmake/test.cmake`），版本与实现随环境而异，
其非有限值传播路径并不固定，因此逐元素类别一致无法保证。测试工程的中性化只覆盖“两侧同为
NaN”与“两侧同为同号 Inf”（见 `ctbmv_test_util.h`），对 `Inf` 与 `NaN` 互异的情形按失败计。

**算子侧的对照证据**：评审阶段的独立复刻验证（以 float32 按 small_ub 路径的运算次序实现，
与参考实现 Netlib `ctbmv.f` 逐元素比对）表明，上述四条数据下算子输出的类别与符号与参考实现
完全一致，按现有中性化与容差可通过。即该差异来自 golden 侧的可移植性，而非算子实现。

**根治方式**：任务书 §3.1 / §3.2.1 / §3.5.1 规定 golden 由 cblas（Netlib BLAS 复数实现）生成，
而本环境实际链接的是系统解析到的 cblas 实现。若能固定 golden 实现（例如在测试工程内构建参考
实现），这四条即可纳入常规精度统计。
