# aclblasCtbmv arch22 测试

## 正式测试

测试使用仓库公共 CSV/GTest 框架，golden 调用公共 CBLAS 的 `cblas_ctbmv`。
CBLAS 的查找和链接由 `ops_blas_add_gtest_tests` 统一处理；
不再单独构建 Reference-LAPACK，不需要 `CTBMV_REFERENCE_ROOT`，
不修改公共 CI 入口。依赖安装沿用仓库的 `install_deps.sh` 和测试说明。

`arch22/ctbmv_test.csv` 保留任务提供的 1200 条官方用例（含 200 条
`TC_PF_*`），不修改参数、输入生成或精度阈值。
按此前 PreSmoke 时长整改方案，默认使用 `arch22/ctbmv_smoke.csv`
中的前 50 条官方用例，加上两项独立回归，共 52 项 GTest。
通过 `CTBMV_TEST_CSV` 指定完整 CSV，可执行全部 1202 项。

实部和虚部分别验证：
`matched_ratio >= 0.99 && (max_abs_error <= 0.01 || max_ulp <= 32)`，
逐元素容差为 `atol=2^-16, rtol=2^-10`。
非法参数用例检查返回码；不将其视为数值误差为零的用例。

| 验证内容 | 入口 |
| --- | --- |
| CSV 功能及精度 | `arch22/ctbmv_test.cpp` 的 `CsvDriven` |
| 空 handle | `NullHandle` |
| 非法枚举、尺寸、指针、步长及零维 | 官方 CSV 对应用例 |
| 单位对角、带外 NaN、负步长、未引用槽位 | `SupplementalUnitDiagBandSentinelsAndNegativeStride` |
| CPU golden / NPU 调用 | `ctbmv_golden.h` / `arch22/ctbmv_npu_wrapper.h` |

## 编译与运行

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

公共配置可能找到不同实现或版本的 BLAS。复测时记录 CMake 中
`REFBLAS_LIB` / `REFBLAS_INCLUDE_DIR` 和实际动态链接库。
历史独立参考库的测试结果不能代替此配置的设备复测结果。

## 设备内存检查

内存检查使用独立的本地插桩构建，按目标设备安装版本的 msSanitizer
指引配置，不在公共构建中新增 Ctbmv 专用开关。

保存完整日志、GTest 结果、实际链接库及设备/工具版本。
普通 GTest 通过或检测进程退出码为零，不等同于无内存诊断；
还须核对 ERROR、WARNING、测试覆盖及检测是否完整执行。
内存检查开销不计入正式性能数据。

本地性能采集、精度报告和证据打包工具不属于公共库交付，
已从本提交移出；正式 GTest、CSV、golden 和 NPU wrapper 保留。
