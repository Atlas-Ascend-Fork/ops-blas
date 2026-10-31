# ctrsv tests

The `arch22` test is a CSV-driven GTest for `aclblasCtrsv` on Atlas 800I
A2/A3 (910B3/910B4). The supplied `arch22/ctrsv_test.csv` contains the
accuracy and `TC_PF` performance cases from the task package.

Additional `arch22` sources are collected by the shared `ctrsv_test*.cpp`
pattern: `ctrsv_test_scalar.cpp` covers scalar arithmetic and
`ctrsv_test_boundaries.cpp` covers diagonal range and dispatch boundaries,
including padded leading dimensions, negative strides, and untouched vector
gaps. Their direct GTests are included in the accuracy command below.

Build the operator and test with:

```bash
bash build.sh --soc=ascend910b3 --ops=ctrsv
```

Run accuracy cases (excluding performance cases) with:

```bash
ASCEND_DEVICE_ID=0 build/test/trsv/ctrsv/ctrsv_test --gtest_filter='*-*TC_PF*'
```

Run performance cases with the normal warmup/sample configuration used by
the task test scripts:

```bash
ASCEND_DEVICE_ID=0 build/test/trsv/ctrsv/ctrsv_test --gtest_filter='*TC_PF*'
```

## Netlib Correctness Reference

The complex reference calls Netlib BLAS `ctrsv_` directly. It does not use a
handwritten triangular solve or the OpenBLAS implementation. Configure
`CTRSV_NETLIB_LIBRARY` to an absolute path to the Netlib shared library. The
adapter uses the GNU Fortran LP64 ABI (32-bit integers and trailing `size_t`
character lengths), and rejects libraries exporting `openblas_get_config`.
The first numerical correctness case prints the resolved library path.

For the local Release/O2 build on 910B:

```bash
source /usr/local/Ascend/cann-9.1.0-beta.3/set_env.sh
export LINUX_INCLUDE_PATH=/usr/local/Ascend/cann-9.1.0-beta.3/aarch64-linux/include
export EAGER_LIBRARY_PATH=/usr/local/Ascend/cann-9.1.0-beta.3/aarch64-linux/lib64
cmake -S . -B build-o2 -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_ASC_FLAGS_RELEASE='-O2 -DNDEBUG' \
  -DCTRSV_NETLIB_LIBRARY=/lib64/libblas.so.3.12.0
cmake --build build-o2 --target ctrsv_test -j8
CTRSV_TEST_CSV=/absolute/path/to/official/ctrsv_test.csv \
  ASCEND_DEVICE_ID=0 build-o2/test/trsv/ctrsv/ctrsv_test \
  --gtest_filter='Trsv/CtrsvArch22Test.CsvDriven/*-*TC_PF_*' \
  --gtest_output=xml:accuracy_netlib.xml
```

`CTRSV_TEST_CSV` overrides the CSV input without modifying the repository's
local cases. The filter selects only CSV correctness cases, excluding both
performance cases and additional direct GTests. Invalid-argument cases check
the ACLBLAS return code against the CSV; valid numerical cases call Netlib
and retain the existing MERE/MARE thresholds and nonfinite-value checks.

---

# Ascend 950PR 功能与精度 ST

使用 Ascend 950PR、CANN 9.1.0、仓库 GTest/CSV 框架和 Netlib BLAS `ctrsv_`。
本任务指定 Netlib Golden，不能用 OpenBLAS 替换。以下是 Ubuntu x86_64 示例；
其他系统须调整路径并核对实际库实现。

## 构建与运行

在仓库根目录以 Bash 执行，任一步失败立即停止：

```bash
set -e -o pipefail
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export EAGER_LIBRARY_PATH=/usr/local/Ascend/cann-9.1.0/x86_64-linux/lib64
export LINUX_INCLUDE_PATH=/usr/local/Ascend/cann-9.1.0/x86_64-linux/include
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DREFBLAS_LIB=/usr/lib/x86_64-linux-gnu/blas/libblas.so.3 \
  -DCTRSV_NETLIB_LIBRARY=/usr/lib/x86_64-linux-gnu/blas/libblas.so.3 \
  -DREFBLAS_INCLUDE_DIR=/usr/include/x86_64-linux-gnu \
  -DSOC_VERSION=ascend950 -DBUILD_TEST=ON -DTEST_NAMES=ctrsv \
  -DTEST_DEVICE_ID=0
cmake --build build --target ctrsv_test -j6
export LD_LIBRARY_PATH="$PWD/build:/usr/lib/x86_64-linux-gnu/blas:$LD_LIBRARY_PATH"
readlink -f /usr/lib/x86_64-linux-gnu/blas/libblas.so.3
ldd build/test/trsv/ctrsv/ctrsv_test
export BLAS_DEVICE_ID=0
unset BLAS_TEST_CASES GTEST_FILTER
mkdir -p results/ctrsv
./build/test/trsv/ctrsv/ctrsv_test 2>&1 | tee results/ctrsv/st.log
```

运行前核对 libblas 的实际路径确为 Netlib。`CTRSV_NETLIB_LIBRARY` 明确指定 Golden，
不使用可能自动选择 OpenBLAS 的简写步骤。`BLAS_DEVICE_ID` 优先于编译期
`TEST_DEVICE_ID`，与仓库 `build.sh --device` 和多卡分片一致；选择空闲设备。
测试继承公共 `BlasTest`，不单独维护 ACL 初始化/销毁逻辑。

## 文件与覆盖

- `ctrsv_param.h`：CSV 参数解析。
- `ctrsv_golden.h`、`ctrsv_cases.h`、`ctrsv_test_helpers.h`：CPU Golden、输入生成和精度统计。
- `arch35/ctrsv_npu_wrapper.h`：设备调用、RAII 缓冲区和同步读回。
- `arch35/ctrsv_test.cpp`、`arch35/ctrsv_test.csv`：GTest 与官方用例参数。

CSV 保留任务提供的 1000 条功能用例和 200 条性能形状。正式 ST 对这些输入验证精度，
不在 CI 中用共享机器的耗时判定功能测试。完整性能计时及统计脚本单独保留在验收材料中。

1176 条随机输入另补相同参数的正态分布测试，与均匀分布输入一一配对。
正态实部/虚部分别采样，均值在 [-5,5]、标准差在 [0.1,2]。
NON_UNIT 对角按任务书加符号保持的 `max(5,n)` 偏移；不缩放非对角元素或缩小输入区间。
UNIT 对角和未引用三角填 NaN。NPU 和 Netlib 使用同一份输入。

额外覆盖 4097 阶、受限工作区、空 handle、零尺寸参数和合法偏移指针。
检查 A 只读、x 步长间隙及缓冲区保护值，并覆盖复数除法中间乘积下溢和分量指数差距。
直接参数验证与 CSV 用例统一注册在 `Ctrsv/` 前缀下，
`BLAS_TEST_CASES=ctrsv:ALL ./build/test/blas_all_test` 包含这些检查。
不能只运行 CI smoke 子集后声称全部自验通过。

### 官方 CSV 勘误

原始 `TC_ED_095` 描述为 `lda0_n1`，lda 实际为 1，期望 INVALID_VALUE。
这与 `lda >= max(1,n)` 冲突。工作副本只将该行 lda 改为 0，保留描述和期望；
任务下载原件不变，不将合法 n=1/lda=1 人为判错。

## 判定标准

实部、虚部分别要求 matched_ratio >= 0.99，逐元素容差为
`2^-16 + 2^-10 * abs(golden)`，误差上限为 `max(1e-2,32*ULP(golden))`。
NaN 仅双方同为 NaN 时匹配；Inf 符号须一致；有限/非有限不匹配失败。
异常参数检查返回码。GTest 非零退出、失败或运行不完整均不能视为通过。

验收独立性能测试仍须运行全部 200 条数据：10 次预热、51 次有效采样，
每次恢复输入后用 ACL event 测量整个算子调用，取平均值。逐条 GPU/NPU > 0.4，
前五条同时满足 146.33、272.53、526.22、1177.84、1030.61 μs 上限。
官方 GPU 基准不改写，采样前后均检查精度。GTest 墙钟和 profiler 耗时不代替验收计时。
