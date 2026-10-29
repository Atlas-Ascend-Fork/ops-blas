# aclblasCher2 arch35 自验证

Ascend 950PR 上的 COMPLEX64 Hermitian 秩-2 更新。实现方案见
[设计文档](../../../docs/zh/develop/aclblasCher2_design.md)，构建与复现步骤见下文。
自测报告、逐例结果、原始采样和工具版本随交付包保存，评审记录见
[PR #713 讨论](https://gitcode.com/cann/ops-blas/pull/713/discuss)。开发者自测结果与社区官方验收状态分别记录。

## 环境与编译

依赖 CANN 9.1.0、CMake、C++17、GTest、Netlib BLAS/LAPACK 3.10.0。
生产源码不依赖 CPU BLAS；golden 链接 Netlib `cblas_cher2`。

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --soc=ascend950 --ops=cher2
```

设备通过 `BLAS_DEVICE_ID` 环境变量选择，默认为逻辑设备 0。

## 精度与覆盖自检

```bash
python3 test/syr2/cher2/run_self_test.py --output /tmp/cher2_accuracy
# 日常回归可排除性能形状；发布前使用上面的完整自检：
./build/test/syr2/cher2/cher2_test --gtest_filter='-*TC_PF*'
```

完整自检执行原始 1200 条 CSV（1000 精度、200 性能形状的单次更新结果验证）、144 条
混合分布用例和 5 项补充测试，总计 1349 项 GTest。补充测试覆盖空句柄、30 个特殊
标量场景、24 个大步长场景、20 个错位地址/分块边界场景、32 个混合特殊值场景和
12 个 SIMD 路由边界。大步长覆盖 n=65/513 时 x/y 的全部 ±1000 组合，以及 n=1 时
INT_MAX、INT_MIN+1 的全部组合，两种三角均与 Netlib golden 比较。
测试注册和脚本均检查 CSV 数量，脚本还核对逐 case 覆盖、GTest 状态及误差记录。

实部/虚部分别统计混合容差、非有限值、MERE/MARE。检查未引用三角、lda padding、
前后保护区及只读 x/y。默认 64 字节保护区覆盖 8 字节对齐路径；68 字节偏移覆盖
public complex ABI 的 4 字节对齐回退。零列与 Inf/NaN 混合、脏对角虚部、负步长、
1023/1024/1025、4095/4096/4097 的路径边界均有回归用例。

## 正式口径性能采集

使用 `msprof op` 后端 `msopprof` 的 BasicInfo / Task Duration(us)，默认 kernel replay。
每个有效采样由工具完成 5 次 warmup；测试自身不再额外 warmup，且不做 CPU golden。
任务 §3.3 的 10 次有效样本取算术均值，逐 case 判断 `GPU/NPU >= 0.4`。

```bash
python3 test/syr2/cher2/run_op_profile.py \
  --tool /usr/local/Ascend/cann/tools/msopprof/bin/msopprof \
  --output /tmp/cher2_op_profile --samples 10
```

`--samples 51` 可采集更长样本，同时报告前 10 次和全部 51 次均值。
必须使用新的输出目录。脚本每批最多采集 1000 条，避开当前工具的 CSV 导出上限；
汇总时严格检查全部 200 个用例、采样数量、返回码、GTest 通过记录、正且有限的耗时，
以及 uplo/n/alpha/incx/incy/lda 的全部基线参数。结果包含每个原始样本，缺失记录会失败。

本次容器预装 op 采集库存在物理设备 3/逻辑设备 0 的映射问题。使用上游未修改的
msopcom `d1564487d9c0b27033f6bb51b19b21bec7be6aa5` 构建注入库后解决。
可在独立目录复现（不替换系统安装，需要 `nlohmann-json3-dev`）：

```bash
bash test/syr2/cher2/prepare_op_profiler.sh /tmp/cher2_op_tool
python3 test/syr2/cher2/run_op_profile.py \
  --tool /tmp/cher2_op_tool/msopprof/bin/msopprof \
  --output /tmp/cher2_op_profile_fixed --samples 10
```

## 额外稳定性采集

普通 msprof 的 task duration 用于额外 5+51 次稳定性检查，单独标注，不能冒充 op 结果：

```bash
set -o pipefail
CHER2_PROFILE=1 msprof --ai-core=off --output=/tmp/cher2_task_profile \
  --application='./build/test/syr2/cher2/cher2_test --gtest_filter=*TC_PF*' \
  2>&1 | tee /tmp/cher2_task_profile.log
python3 test/syr2/cher2/summarize_task_profile.py \
  --profile /tmp/cher2_task_profile --log /tmp/cher2_task_profile.log \
  --baseline test/syr2/cher2/gpu_baseline.csv --output /tmp/cher2_task_performance.csv
```

禁止使用 GTest 用例总耗时作为 kernel 耗时，也不能把 profiling replay 后的矩阵与
单次更新 golden 混合校验。

## 内存检查

任务 §3.4 写明“不涉及”，仍额外执行带插桩的 mssanitizer 检查：

```bash
export LINUX_INCLUDE_PATH="${ASCEND_HOME_PATH}/include"
export EAGER_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64"
cmake -S . -B build-sanitize -DSOC_VERSION=ascend950 -DBUILD_TEST=ON -DTEST_NAMES=cher2 \
  -DCANN_3RD_LIB_PATH="$PWD/build/third_party" \
  '-DCMAKE_ASC_FLAGS=-g --cce-enable-sanitizer'
cmake --build build-sanitize --target cher2_test -j8
python3 test/syr2/cher2/run_memcheck.py --output /tmp/cher2_memcheck
```

脚本按测试套件分别启动原装 mssanitizer，避免工具跨 `aclFinalize/aclInit` 复用失效
kernel 句柄。29 项测试覆盖负向契约、特殊标量、混合特殊值和 512/1024/4096 规模。
检查 GTest 数量、实际 kernel 检查记录、`No error detected.` 及所有 WARNING/ERROR，
不能仅凭工具退出码认定通过。

INT_MIN 步长按随附负向 CSV 返回 INVALID_VALUE；其与任务书“任意非零步长”的差异
明确保留供评审确认。
