# aclblasCgemv 测试说明

本目录包含 Ascend 950PR（arch35）上的 `aclblasCgemv` 功能与精度测试。测试以列主序
CBLAS 结果为参考，覆盖 `ACLBLAS_OP_N`、`ACLBLAS_OP_T` 和 `ACLBLAS_OP_C`、复数
`alpha`/`beta`、前导维填充以及正负向量步长。

## 目录结构

```text
test/gemv/cgemv/
├── CMakeLists.txt
├── cgemv_param.h
├── cgemv_golden.h
└── arch35/
    ├── cgemv_test.cpp
    ├── cgemv_test.csv
    └── cgemv_npu_wrapper.h
```

`cgemv_test.csv` 包含 1035 条功能/精度用例和 200 条 `TC_PF` 性能用例。
`cgemv_test.cpp` 另含 7 条独立边界用例，因此精度集为 1042 条，静态测试总数为 1242 条。

## 构建和运行

需要 Ascend 950PR、支持 ascend950 的 CANN 工具链、GTest，以及**导出 `cblas_cgemv` 的
Netlib CBLAS**。

> **参考实现必须是 Netlib，不能用 OpenBLAS。** 任务书 3.1/3.2 规定 golden 由 Netlib cblas
> 生成。两者的复数求和次序不同：同一份算子在 OpenBLAS 参考下曾出现 23 条 max-abs 失败，
> 换回 Netlib 后同一集合全部通过。注意很多发行版的 `libblas.so`（Netlib）**不导出**
> `cblas_*` 符号，系统里唯一提供 `cblas_cgemv` 的往往是 OpenBLAS，直接链接会静默用错参考实现。

若系统没有 Netlib CBLAS，可从 Reference-LAPACK 源码构建：

```bash
git clone --depth 1 --branch v3.12.0 \
  https://gitcode.com/Reference-LAPACK/lapack.git lapack-3.12.0
cmake -S lapack-3.12.0 -B lapack-build -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_SHARED_LIBS=ON -DCBLAS=ON -DLAPACKE=OFF -DBUILD_TESTING=OFF \
  -DCMAKE_INSTALL_PREFIX="$PWD/netlib"
cmake --build lapack-build -j 2 --target cblas
cmake --build lapack-build --target install
export NETLIB="$PWD/netlib"     # 提供 libcblas.so / libblas.so / cblas.h
```

构建测试工程（`TEST_DEVICE_ID` 是编译期宏，换卡必须重新构建）：

```bash
export ASCEND_HOME_PATH=/path/to/cann-9.1.0
source "${ASCEND_HOME_PATH}/set_env.sh"
export CGEMV_DEVICE=0           # 必须是 Health OK 且空闲的卡

cmake -S <ops-blas> -B <ops-blas>/build \
  -DSOC_VERSION=ascend950 -DBUILD_TEST=ON -DTEST_NAMES=cgemv \
  -DTEST_DEVICE_ID="${CGEMV_DEVICE}" \
  -DASCEND_CANN_PACKAGE_PATH="${ASCEND_HOME_PATH}" \
  -DREFBLAS_LIB="${NETLIB}/lib/libcblas.so" \
  -DREFBLAS_INCLUDE_DIR="${NETLIB}/include"
cmake --build <ops-blas>/build -j 2 --target cgemv_test

# 确认链接到 Netlib 而不是 OpenBLAS
ldd <ops-blas>/build/test/gemv/cgemv/cgemv_test | grep -i blas
```

仓库原有的 `bash build.sh --ops=cgemv --soc=ascend950 --run --device=...` 同样可用，
但它执行静态清单中的 1242 条测试且 `TC_PF` 默认只做精度检查；适合功能回归，
不等同于正式的“1042 条精度 + 200 条性能倍率”验收。

### 正式验收（推荐直接用任务自带脚本）

```bash
python3 task_cases/verify_accuracy.py    --repo <ops-blas> --soc ascend950 --skip-build --device "${CGEMV_DEVICE}"
python3 task_cases/verify_performance.py --repo <ops-blas> --soc ascend950 --skip-build --device "${CGEMV_DEVICE}"
```

两个脚本都会自行核对清单、返回码、设备号与基线，任一项不符即非零退出。
`--skip-build` 要求产物位于 `<ops-blas>/build`，脚本会校验二进制编译期的 `TEST_DEVICE_ID`。

手工执行等价于：

```bash
# 精度：1042 条
./build/test/gemv/cgemv/cgemv_test --gtest_filter='*-Cgemv/CgemvArch35Test.CsvDriven/TC_PF_*'
# 性能：200 条，warmup=30 + 计时 100 次
CGEMV_ENABLE_PERF=1 ./build/test/gemv/cgemv/cgemv_test \
  --gtest_filter='Cgemv/CgemvArch35Test.CsvDriven/TC_PF_*'
```

性能判据为 `NPU_us <= gpu_baseline.csv 的 gpu_ms * 1000 / 0.4`。任务书 3.3 表格中的
2.18/3.86/3.51us 与该公式相差 6.25 倍，官方脚本已将其标注为勘误、不参与判定。

### 选卡注意

运行前用 `npu-smi info` 确认目标卡 Health OK **且没有其他进程**，运行后再确认一次。
NPU 可能与其他容器/租户共享；若运行中有他人进程进入同一张卡，会出现约 1.3 倍甚至更大的
孤立慢点，表现为个别用例倍率骤降，而重跑即恢复。

## 覆盖范围

- `TC_L0`：基础功能与 N/T/C 操作。
- `TC_SQ`、`TC_RC`：方阵和非方阵尺寸组合。
- `TC_AB`：`alpha`、`beta` 的零、一和一般复数值。
- `TC_LD`：`lda` 大于有效行数的列主序矩阵。
- `TC_INC`：正、负及非单位 `incx`、`incy`。
- `TC_FL`：零、交替值、极值、Inf 和 NaN。
- `TC_CV`、`TC_EX`：中等尺寸与扩展随机采样。
- `TC_ED`：边界参数、quick return 和非法参数。
- `TC_XT`：大尺寸、奇数维、窄矩阵、宽矩阵及退化计算路径。
- `TC_PF`：代表性尺寸和布局组合。

`TC_PF` 默认与其他 CSV 行一样只执行精度检查。设置 `CGEMV_ENABLE_PERF=1` 后，测试程序才会
对这些用例执行 warmup 和重复采样，并输出 `CGEMV_PERF_RESULT` 记录。

```bash
CGEMV_ENABLE_PERF=1 ./build/test/gemv/cgemv/cgemv_test --gtest_filter='*TC_PF*'
```

每条性能用例先 warmup 30 次，再连续执行 100 次并统一同步，输出单次平均耗时。内存分配、
数据传输、CPU golden 和结果比较不计入计时区间。

常规随机输入同时覆盖均匀分布与正态分布。填充器支持
`RANDOM_GAUSSIAN_<mean>_<standard-deviation>`，例如
`RANDOM_GAUSSIAN_N2.5_0.75`；省略参数时使用标准正态分布。矩阵、输入向量和输出向量
使用彼此独立的种子，复数实部与虚部也独立采样。

7 条独立边界用例为：

- `NullHandle`：空句柄返回码。
- `QuickReturnNullDataPointers`：不访问数据的合法 quick return。
- `ExtremeNegativeStrideLengthOne`：长度为 1 时的极值负步长。
- `FiniteCancellationOrderedAccumulation`：大数抵消、非有限值分类及复乘舍入边界。
- `SegmentedDirectMtePathLocalBoundaries`：T/C 分段 direct-MTE 路径的分段分界、非 32 字节对齐 x 搬运、完整输出与尾部 guard；另覆盖 N slab 的行尾、1/8 个 MTE tile、最后一个 x 广播及超出 slab 容量时的回退，使用可精确表示的输入对照 Netlib，并检查 A/x 只读、lda padding 不参与计算和 y 前后 guard。
- `FiniteCancellationPartitionBoundaries`：跨分段边界仍保持 Netlib 逻辑顺序结果。
- `SmallWorkspaceCompatibilityN`：小用户 workspace 下的 UB/GM 输入路径。

## 精度判定

COMPLEX64 的实部和虚部分别按 FLOAT32 判定：

```text
abs(actual - expected) <= atol + rtol * abs(expected)
rtol = 2^-10
atol = 2^-16
matched_ratio >= 0.99
max_abs_error <= max(1e-2, 32 * ULP)
```

输出向量先按逻辑步长还原，再对实部和虚部分别执行混合容差比较。特殊值用例同时校验
NaN、Inf 和符号分类。有限抵消用例要求每个输出元素按照逻辑点积顺序累加，防止并行归约
改变非有限值分类。

## 返回码

| 场景 | 返回码 |
| --- | --- |
| `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| `trans` 不是 N/T/C | `ACLBLAS_STATUS_INVALID_ENUM` |
| 维度、`lda`、步长或必需指针非法 | `ACLBLAS_STATUS_INVALID_VALUE` |
| `m == 0` 或 `n == 0` | `ACLBLAS_STATUS_SUCCESS` |
| `alpha == 0` 且 `beta == 1` | `ACLBLAS_STATUS_SUCCESS`，且不访问 A/x/y |

当 `alpha == 0` 且 `beta != 1` 时只计算 `y = beta * y`，不读取 A 和 x；当
`beta == 0` 时不读取 y 的原值。
