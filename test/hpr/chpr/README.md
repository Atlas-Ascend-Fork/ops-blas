# aclblasChpr 测试说明

Ascend 950PR / Ascend 950DT（arch35），CANN 9.1.0，C++17。精度 golden 必须链接 **Netlib BLAS 3.10.0** 的 `cblas_chpr`；构建时显式指定该版本的头文件与库，并保留 CMake 配置、动态库依赖及版本来源记录。

## 测试内容

arch35 采用 arch22 的 `chpr_test.cpp`、`chpr_npu_wrapper.h`、`chpr_test.csv` 布局。共用 `chpr_param.h` 同时支持旧列 `x/ap` 与任务包列 `x_fill/ap_fill`，支持 `null` / `NULLPTR` alpha。arch22 的测试实现保持原状。

`arch35/chpr_test.csv` 原样来自任务包，共 1200 行（1000 精度 + 200 `TC_PF_`）。测试加载时检查行数与重复 ID；不重新生成或删减原始用例。精度类别为 TC_L0 4、TC_SQ 46、TC_AB 16、TC_INC 12、TC_FL 12、TC_CV 16、TC_ED 25、TC_EX 869。

额外 GTest 单测独立命名：

- `HandleAndQuickReturnPrecedence`：空 handle 优先返回句柄错误；有效 handle 下验证负 n 返回参数错误、n=0 携带空指针/非法属性仍快速返回成功。
- `GmFallbackAndWideStride`：n=4097、UPPER/incx=+17 与 LOWER/incx=-17 覆盖超过 UB 缓存上限的 GM 读取路径；n=1/incx=INT_MAX，以及 n=2/incx=±65537 覆盖大步长地址计算。使用固定 seed 的有限适中值、Netlib 全量 golden 与 x 字节保持检查。
- `ScalarNonfiniteAndZeroColumns`：alpha 为 NaN、+Inf、-Inf，UPPER/incx=+3 与 LOWER/incx=-3，逻辑 x 混合零与非零复数，验证 Netlib 零列跳过语义、非有限结果及对角虚部清零。
- `SeededUniformNormalAndAsyncStreams`：固定 seed 的 8 组均匀和 8 组正态输入，实部/虚部独立采样，实数 alpha 同样各占 50%；覆盖正负步长至 7、对角虚部垃圾值/NaN、逻辑 x 零元素、alpha=±0 精确保持，以及两条独立 stream 上连续异步调用。

全部 packed 输出按实部、虚部分别验证。权威判据为 rtol=atol=2^-13，有限分量 matched_ratio≥0.99，逐点误差≤max(0.01,32×ULP(golden))；非有限分量按任务书 NaN/NaN、同号 Inf/Inf、NaN/Inf 等效规则单独统计。有限/非有限混合判失败。更新后的对角虚部必须为零；alpha=0 时整个 ap 字节保持；x 全物理缓冲字节保持。CSV MERE/MARE 仅输出补充结果。无效 n/incx 在分配前直接调用实际 API 验证返回码。

## 编译及精度复现

以下按 Ubuntu x86_64 的隔离 Netlib 目录布局配置；`NETLIB_ROOT` 指向依赖解压目录。示例依赖来自 `libblas3`、`libblas-dev`、`liblapack3` 的 `3.10.0-2ubuntu1` 包，可用 `apt download` 下载至独立目录，再用 `dpkg-deb -x` 解压到 `NETLIB_ROOT`。保持系统 alternatives 配置。`libblas-dev` 提供 `cblas-netlib.h`，在该隔离 include 目录创建 `cblas.h -> cblas-netlib.h` 符号链接。

先加载实际 CANN 9.1.0 环境脚本，确认 `ASCEND_HOME_PATH` 已设置，再从仓库根目录运行：

```bash
export NETLIB_ROOT=/path/to/deps/netlib
export NETLIB_INCLUDE="${NETLIB_ROOT}/usr/include/x86_64-linux-gnu"
export NETLIB_BLAS="${NETLIB_ROOT}/usr/lib/x86_64-linux-gnu/blas"
export NETLIB_LAPACK="${NETLIB_ROOT}/usr/lib/x86_64-linux-gnu/lapack"
test -f "${NETLIB_INCLUDE}/cblas-netlib.h"
test -e "${NETLIB_INCLUDE}/cblas.h" || ln -s cblas-netlib.h "${NETLIB_INCLUDE}/cblas.h"
export EAGER_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64"
export LD_LIBRARY_PATH="${NETLIB_BLAS}:${NETLIB_LAPACK}:${EAGER_LIBRARY_PATH}:${PWD}/build:${LD_LIBRARY_PATH}"
cmake -S . -B build \
  -DBUILD_TEST=ON -DTEST_NAMES=chpr -DSOC_VERSION=ascend950 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH="${ASCEND_HOME_PATH}/lib64/cmake" \
  -DREFBLAS_INCLUDE_DIR="${NETLIB_INCLUDE}" \
  -DREFBLAS_LIB="${NETLIB_BLAS}/libblas.so.3.10.0" \
  -DREFLAPACK_LIB="${NETLIB_LAPACK}/liblapack.so.3.10.0"
cmake --build build --target chpr_test chpr_bench -j8
ldd build/test/hpr/chpr/chpr_test
export BLAS_DEVICE_ID=0
export CHPR_TEST_CSV="${PWD}/test/hpr/chpr/arch35/chpr_test.csv"
./build/test/hpr/chpr/chpr_test \
  --gtest_filter='*ChprArch35Test.*' \
  --gtest_output=xml:chpr_gtest.xml
```

检查 `ldd` 中 `libblas.so.3`、`liblapack.so.3` 解析到上述隔离目录，并保留该输出及依赖包版本记录。上述过滤条件运行全部 1200 条 CSV 及 4 个补充测试；功能/精度可使用 `--gtest_filter='*ChprArch35Test.*-*TC_PF_*'`（1000+4），性能用例正确性可使用 `--gtest_filter='*ChprArch35Test.CsvDriven/TC_PF_*'`（200）。检查进程退出码、GTest 总数、失败数及跳过数，并保留 XML、控制台日志和逐分量 `CHPR_METRIC` 记录。通过 `CHPR_TEST_CSV` 给二进制传入 CSV 绝对路径，设备号使用框架的 `BLAS_DEVICE_ID`。

## 官方性能复现

`chpr_bench` 为独立应用，已在本目录 CMake 为 arch35 注册 `ops_blas_add_tests(${OPS_BLAS} chpr_bench)`。它读取全部 200 个 PF case，提前完成每条的数据准备与 H2D，再调用 5 次 warmup 和 51 次采样；每次 API 必须返回 SUCCESS，并在输出 `CHPR_BENCH` JSON 前同步成功。应用没有 golden 计算。`--all` 便于普通运行；`--case ID` 可重复，`--case-list FILE` 接收每行一个 ID，统一按原始 CSV 顺序执行。

```bash
./build/test/hpr/chpr/chpr_bench --all
msprof op --help
msprof op --warm-up=5 --launch-count=56 \
  --kernel-name=_Z11chpr_kernelPhS_14ChprTilingData \
  --aic-metrics=BasicInfo --output=/path/to/profile \
  ./build/test/hpr/chpr/chpr_bench \
  --case TC_PF_1001 --warmup 5 --iterations 51
```

`--kernel-name` 填实际编译产物中的 CHPR kernel 名称，CSV 中 `Op Name` 为 `chpr_kernel`。示例采集单条 case 的 56 次调用，剔除应用前 5 次，分别统计后续前 10 次和全部 51 次 `OpBasicInfo.csv` 的 `Task Duration(us)`。profiler 自身 5 次 replay warmup 不计作应用样本。按任务书和任务附件的 GPU 标杆对全部 200 条性能用例分别判定 GPU_us/NPU_us≥0.4，保留实际命令、应用日志和原始 profiler 数据；GTest walltime 不用于性能判定。

工具行为依据：[官方 msOpProf 参数文档](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_user_guide.md)、[官方 OpBasicInfo 字段定义](https://github.com/Ascend/msopprof/blob/master/docs/en/user_guide/msopprof_performance_data.md)。默认 kernel replay 会清理 L2；保留实际安装工具的 `--help` 与采集参数，报告中注明该模式。
