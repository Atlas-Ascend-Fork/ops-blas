# STRMM 测试

## Atlas A2/A3（arch22）

`arch22/strmm_test.csv` 保留 9 月 aclblasStrmm 任务包的前 25 条用例，
用于 CI 冒烟测试。完整 1200 条 CSV（1000 条精度用例和 200 条性能用例）的固定版本
及校验信息见[需求 Issue #525](https://gitcode.com/cann/ops-blas/issues/525)。
测试使用仓库 CSV/GTest 框架，另包含三角读取、单位对角、padding、原地写出、特殊值
和有限大数溢出等专项回归；专项测试不受 CSV 精简影响。

### 构建与 Netlib 依赖

任务书指定 Netlib reference BLAS 的 STRMM 为 Golden。部分系统的 `libblas.so`
实际指向 OpenBLAS/BLIS，而 openEuler 的 Netlib 库只导出 Fortran `strmm_`，
没有 CBLAS 包装。因此保留 `arch22/strmm_test_netlib.cpp` 作为测试适配器：
从独立库句柄解析 `strmm_`，拒绝 OpenBLAS/BLIS，并检查三角 Inf 和零 B 语义，
避免同进程其他 BLAS 或 `LD_PRELOAD` 替换 Golden。

依赖校验直接放在本目录已有的 `CMakeLists.txt`，仅对 arch22 生效，不增加独立
CMake 文件或构建入口。配置阶段用同一适配器探测参考库，设置实际库路径和动态加载
链接依赖；这是保证任务指定 Golden 可用的测试配置，不是算子构建配置。
arch35 仍使用原有测试和构建路径。

在 CANN 9.1.0 的 A2/A3 环境中，从仓库根目录执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export INCLUDE_PATH="$ASCEND_HOME_PATH/include"
export LINUX_INCLUDE_PATH="$ASCEND_HOME_PATH/$(uname -m)-linux/include"
export EAGER_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64"
cmake -S . -B build -DSOC_VERSION=ascend910b3 -DBUILD_TEST=ON -DTEST_NAMES=strmm \
  -DREFBLAS_LIB=/usr/lib64/libblas.so -DREFBLAS_INCLUDE_DIR=/usr/include/openblas \
  -DSTRMM_NETLIB_LIB=/usr/lib64/libblas.so
cmake --build build --target strmm_test -j
```

以上路径为 openEuler 示例，其他系统应指定实际的 `cblas.h` 和 **LP64 Netlib
reference BLAS 共享库**。显式指定无效库时配置失败，不回退到其他实现。
未指定 `STRMM_NETLIB_LIB` 时检查公共 `REFBLAS_LIB` 和编译器默认库目录。
也可通过环境变量传给仓库构建入口：

```bash
STRMM_NETLIB_LIB=/usr/lib64/libblas.so bash build.sh --ops=strmm --soc=ascend910b3 --device=0
```

配置日志输出 `STRMM Netlib golden: <真实路径> (sha256=...)`；
运行首次生成 Golden 时输出 `[STRMM golden] Netlib strmm_: <实际加载路径>`。
该校验要求在测试主机原生配置，不支持交叉编译运行探测。

### 精度测试

默认 CSV 仅运行前 25 条用例及专项测试。进行全量精度验证或下节性能采集前，
请从[需求 Issue #525](https://gitcode.com/cann/ops-blas/issues/525) 获取完整 CSV 并核对
SHA-256，在本地测试工作区替换 `test/trmm/strmm/arch22/strmm_test.csv`，再重新配置和构建。
完整 CSV 不应重新提交到本 PR；默认 25 条 CSV 不包含 `TC_PF_*` 性能用例。

```bash
./build/test/trmm/strmm/strmm_test --gtest_filter='*-*TC_PF*' \
  --gtest_output=xml:build/strmm_accuracy.xml
./build/test/trmm/strmm/strmm_test --gtest_filter='*TC_PF*' \
  --gtest_output=xml:build/strmm_perf_accuracy.xml
```

恢复完整 CSV 后，第一条命令运行 1000 条精度用例及专项测试，第二条命令对
200 条性能用例也执行 Netlib 精度比对，耗时较长。
仅精度测试可使用 GTest 分片，每片写独立日志/XML；性能采集时不要并行运行其他 NPU 任务。

### 性能采集

性能测试默认禁用；先按上节恢复完整 CSV，再显式运行 `DISABLED_PerfAll`，
可遍历其中的 200 条性能用例。
每条执行 5 次预热和默认 20 次有效调用，不在测量循环中重复 CPU Golden。
通过 `STRMM_PERF_FIRST`、`STRMM_PERF_COUNT` 分批采集，输出目录应尚不存在：

```bash
for first in 0 25 50 75 100 125 150 175; do
  STRMM_PERF_SAMPLES=20 STRMM_PERF_FIRST="$first" STRMM_PERF_COUNT=25 \
    msprof op --aic-metrics=BasicInfo --warm-up=5 \
    --launch-skip-before-match=0 --launch-count=5000 \
    --output="build/strmm_basicinfo_${first}" \
    --application='./build/test/trmm/strmm/strmm_test --gtest_filter=*DISABLED_PerfAll --gtest_also_run_disabled_tests'
done
```

统计时按用例执行顺序排除每条前 5 次调用，将一次调用的所有必要 kernel 耗时相加，
再对 20 次有效调用取平均；不能只计 GEMM、剔除高值或混入其他 NPU 任务。
性能结果、验证报告及个人分析脚本不作为上游源码提交内容。

测试使用容器内逻辑设备 0，运行前确认设备映射。本说明提供复现步骤，不代表
A3 实机验证或官方验收已经完成。

## Ascend 950（arch35）

继续使用 `arch35/strmm_test.cpp` 和 `arch35/strmm_test.csv`，
通过仓库构建入口选择对应 SOC；不受 arch22 的 Netlib 专用配置影响。
