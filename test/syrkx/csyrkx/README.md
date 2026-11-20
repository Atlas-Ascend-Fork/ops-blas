# aclblasCsyrkx 测试

本目录提供 Ascend 950 的 `aclblasCsyrkx` 测试。测试从 `arch35/csyrkx_test.csv` 加载参数，并使用 Netlib BLAS 3.10.0 的 CBLAS 计算参考结果。构建时通过 `REFBLAS_INCLUDE_DIR` 和 `REFBLAS_LIB` 指定参考库头文件及 `libcblas.so`；运行时还需确保其依赖的 Netlib `libblas.so` 可被动态链接器找到。

设置 `NETLIB_INCLUDE_DIR` 和 `NETLIB_LIB_DIR` 分别指向 Netlib 的头文件目录和库目录，在仓库根目录配置并构建测试：

```bash
cmake -S . -B build -DSOC_VERSION=ascend950 -DBUILD_TEST=ON \
  -DTEST_NAMES=csyrkx -DREFBLAS_INCLUDE_DIR="${NETLIB_INCLUDE_DIR}" \
  -DREFBLAS_LIB="${NETLIB_LIB_DIR}/libcblas.so"
cmake --build build --target csyrkx_test -j4
```

可通过 `CSYRKX_TEST_CSV` 指定完整的用例 CSV；否则测试依次查找可执行文件旁及源码目录中的 CSV。设备由 `BLAS_DEVICE_ID` 指定。

执行用例完整性自检及内存检查：

```bash
python3 test/syrkx/csyrkx/run_self_test.py \
  --binary build/test/syrkx/csyrkx/csyrkx_test --output output/csyrkx-self-test
python3 test/syrkx/csyrkx/run_memcheck.py \
  --binary build/test/syrkx/csyrkx/csyrkx_test --output output/csyrkx-memcheck
```

输出目录必须不存在；可通过 `--csv` 和 `--device` 指定 CSV 与设备。自检分别运行非性能用例的精度检查和性能用例的执行检查，后者不作为精度或性能验收；`--include-pf-golden` 可额外运行性能输入的 golden 比对。内存检查需要安装与设备环境匹配的 `mssanitizer`。更多参数见各脚本的 `--help`。

内存检查覆盖参数校验、direct/ordered/Cube 路由、长 K 双分区、非连续布局，以及 NaN、Inf、幅值阈值和原始 C 风险修复。脚本按用例分别启动进程，校验 GTest 执行结果、对应 kernel 的成对插桩记录和完整诊断日志，输出 `memcheck_summary.json`；缺少插桩记录或出现内存诊断均判为失败。自检输出 `self_test_summary.json`，逐项核对 CSV、发现与执行的测试名称，拒绝失败、跳过或未执行的用例。

跨形状 workspace 复用测试也可在同一进程中单独检查。以下命令以 `MixedOverflowChunkRepair` 为例：

```bash
binary=$(realpath build/test/syrkx/csyrkx/csyrkx_test)
mkdir -p output
mkdir -m 700 output/csyrkx-workspace-memcheck
(
  cd output/csyrkx-workspace-memcheck
  CSYRKX_MODE=accuracy GTEST_FILTER=CsyrkxIndependentTest.MixedOverflowChunkRepair \
    mssanitizer --tool=memcheck --leak-check=yes --log-file="$PWD/memcheck.log" "$binary"
)
```

可将测试名替换为 `NonfiniteComplexTermsDespiteFiniteRealPlanes` 或 `InputRiskThresholdBoundary`；使用 `--tool=racecheck` 可执行竞争检查。手工执行时应核对完整诊断日志、插桩起止记录和 GTest 结果，不能仅凭进程退出码判断通过。逐形状内存检查与跨形状内存检查的覆盖范围不同。
