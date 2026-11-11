# aclblasCsyr2 自测

目标为 Ascend 950PR、CANN 9.1.0、complex64。使用 ops-blas 的 CSV 参数化 GTest 框架调用公共 `aclblasCsyr2`，独立 CPU Golden 按 Netlib ssyr2 的复数扩展生成。仅检查指定三角的数学结果，实部和虚部分别判定；另一三角、padding、guard 与只读输入检查位级不变。

## 构建

在 ops-blas 根目录运行（以下目录为 CANN 9.1.0 的常见安装位置，使用实际安装路径）：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export EAGER_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64"
cmake -S . -B build-release -DSOC_VERSION=ascend950 \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TEST=ON -DTEST_NAMES=csyr2 \
  -DTEST_DEVICE_ID=0 -DCMAKE_PREFIX_PATH="$ASCEND_HOME_PATH/lib64/cmake"
cmake --build build-release --target csyr2_test csyr2_golden_test -j8
cd build-release/test/syr2/csyr2
./csyr2_golden_test --gtest_output=xml:golden.xml
./csyr2_test --gtest_output=xml:results.xml > stdout.log 2>&1
```

CSV 由 CMake 复制到测试二进制所在目录。须检查实际退出码及 XML，不能用 grep 出现 PASS 代替全量成功。Release 的实际编译选项和源码/CSV 哈希应随报告保留。

## 用例清单与执行

`arch35/csyr2_test.csv` 含 1000 条非 PF 精度用例和 200 条 `TC_PF_` 性能用例，每条名称唯一。CSV 同时列明 uplo、n、alpha 实/虚部、incx/incy、lda、填充分布、预期返回码和 seed。

```bash
# 1000 条精度及其他专项，排除 PF
./csyr2_test --gtest_filter=-*TC_PF* --gtest_output=xml:accuracy.xml
# 200 条 PF 和四条固定门槛专项；每条 PF 先检查正确性
./csyr2_test --gtest_filter=*TC_PF* --gtest_output=xml:performance.xml
# 单例复现（完整参数由 CSV 提供）
./csyr2_test --gtest_filter=*TC_PF_1001*
```

专项覆盖手算复数、对角虚部、负步长、空 handle、结构非法参数、合法 quick return、INT_MIN/INT_MAX 步长、连续尾部和非对齐起址，以及 4096 优化边界之外的通用路径、两个独立 stream 和连续原地更新。专项测试的最新执行结果以 XML 为准。

随机扩展 TC_EX 的 UNIFORM/NORMAL 各占 400 条。NORMAL 使用 CSV 的 μ/σ；alpha 由生成器独立采样写入 CSV，x/y/A 的实部与虚部使用不同随机子序列。矩阵由选中三角定义复对称矩阵，未选中的物理存储故意独立填充，以检测错误访问/改写；不将它作为第二个数学输入三角。

## 精度判定

逐分量 `abs(actual-golden) <= 2^-16 + 2^-10*abs(golden)`，实/虚部各自要求匹配比例≥0.99，且每个分量误差≤`max(1e-2,32*ULP(golden))`。NaN 分类不符、有限/非有限不符或 Inf 符号不符为硬失败。同步保留 CSV 指定 MERE/MARE 判据。阈值、参数、原始数据分布不得为通过测试而放宽。

`[CSYR2_ACCURACY]` 输出每个分量的 total、matchedRatio、maxAbsErr、MERE、MARE、cap_failures、special_mismatch；`[ACCURACY]` 提供验收脚本兼容字段。非法参数和 no-op 用例用返回码/位级不变判断，不伪造精度分母或数值。

## 性能与内存

原始设备事件计时：warmup=10；60 个批次，每批100次公共 API 调用；每批事件时间除100后，对全部60个样本取算术平均。A 在各批开始前恢复，恢复和数据搬运在计时区间外。不得删除异常批次或将中位数当任务平均值。`CSYR2_SAMPLE` 保留原始数据及主机/线程诊断，`[PERF]` 给出平均 us、采样数、timer 和进程 RSS 历史峰值。

四条固定门槛：512 UPPER≤11 us；1024 LOWER≤12.17 us；2048 UPPER≤24.89 us；4096 LOWER≤131.97 us。其他 PF 依据随任务提供的 GPU 基线逐例对账。GTest 的 CSV 正确性通过不代表性能已达标，应运行验收脚本。

测试使用完整 A0、完整输出和一个 Golden 列；RSS 字段为进程 Host 历史峰值，不是单例独立峰值或 Device HBM。任务书 §3.4 不设硬性内存门槛，报告仍提供实际内存数据。

## 验收脚本

完整交付包附带 `CANN训练营东南大学-aclblasCsyr2算子开发(950)/test_cases/` 中的 `gen_csv.py`、`verify_accuracy.py`、`verify_performance.py` 和基线。可在测试实际退出码已确认后，用其 `--input-log`、`--input-xml`、`--input-returncode` 参数对同一次测试离线对账；所有计划用例必须唯一完成，不允许缺失、跳过、无基线或非零退出被当作成功。

详细命令、冻结环境与已测数值见交付包的 `docs/validation/csyr2-20260908/README.md`。正式验收须使用最终冻结源码重新构建与测试。
