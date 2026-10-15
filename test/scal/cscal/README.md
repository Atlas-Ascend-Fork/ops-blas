# aclblasCscal 950PR 测试

本目录测试公共接口 `aclblasCscal(handle, n, alpha, x, incx)`。`alpha` 是 Host complex64 指针，`x` 是 Device complex64 向量；950PR 实现位于 `blas/scal/arch35/`。构建和运行需 Ascend 950PR、CANN 9.1.0 及完整 Ascend C/ACL 开发环境，不能使用 CPU stub 代替设备验收。

## 构建与运行

在仓库根目录执行：

```bash
source /usr/local/Ascend/cann/set_env.sh
bash build.sh --ops=cscal --soc=ascend950
CSCAL_BIN="$PWD/build/test/scal/cscal/cscal_test"
"$CSCAL_BIN" --gtest_list_tests
"$CSCAL_BIN" --gtest_filter='-*TC_PF*'
"$CSCAL_BIN" --gtest_filter='*TC_PF*'
```

默认读取与 `cscal_test.cpp` 同目录的 `cscal_test.csv`。`CSCAL_DEVICE` 可设置设备 ID；`CSCAL_CASES` 可指定另一份同格式 CSV，例如 `CSCAL_CASES="$PWD/test/scal/cscal/arch35/cscal_extra.csv" "$CSCAL_BIN"`。`CSCAL_REPORT` 可指定逐例结果 CSV 路径。GTest 每行注册一个以 `TC_*` 结尾的测试名，便于官方验收脚本逐条抽取；两个过滤器分别运行非性能和性能用例。仅 GTest 返回 0 不等于性能门槛通过，需用任务书的 GPU 基线及三个主点上限逐例复核。

`CscalPlanTest.*` 是不初始化 NPU 的 host 规划回归测试，覆盖向上取整的整数上界、浮点位模式分类、尾部分核覆盖以及无效核数。可用 `"$CSCAL_BIN" --gtest_filter='CscalPlanTest.*'` 单独运行；这三项不计入官方 CSV 用例数量，也不替代设备验收。

## 用例与判定

| 文件 | 数量 | 用途 |
|---|---:|---|
| `arch35/cscal_test.csv` 非 `TC_PF_` 项 | 1000 | 官方精度、特殊值、no-op、负向参数 |
| `arch35/cscal_test.csv` 的 `TC_PF_` 项 | 200 | 官方性能及计算正确性 |
| `arch35/cscal_extra.csv` | 181 | 独立分布、尾部、stride、偏移和 API 边界 |
| `arch35/gpu_baseline.csv` | 200 | 任务包的 GPU 基线，不是本次重跑的 GPU 数据 |

实部和虚部分别对照 Netlib CBLAS：`atol=2^-16`、`rtol=2^-10`、匹配率至少 0.99；每个有限分量还须满足绝对误差不超过 0.01 或 ULP 不超过 32。Inf 比较符号，NaN 比较类别，不比较 payload。任务要求 `alpha=(0,0)` 将选中元素写为正零，因此测试保留与原始 CBLAS 特殊值行为的差异记录。no-op、identity、负向参数检查状态码和原存储字节不变；哨兵与 stride 空隙也会检查。

性能用例每条预热 10 次、采集 64 次有效 ACL timeline 样本，取全部均值。每次在同一 stream 上于 start event 前从 pristine Device 缓冲恢复输入；计时后的输出再次校验。该指标是 warm-reset stream 区间，不是 Host 墙钟或 msprof 纯 kernel 时间。逐例上限为 `gpu_ms * 1000 / 0.4`，三个主点再与任务书上限取较严者：13.73、21.2175、43.3075 微秒。不得删掉失败项或用最小值代替均值。

## 历史验收记录

原交付分支 `delivery/cscal950-acceptance` 的 2026-09-15 设备复测记录了官方精度 1000/1000、补充精度 181/181、性能 200/200。详细日志、官方脚本输出、构建哈希及复测波动说明保留在[验收附件](https://gitcode.com/zee_920/ops-blas/tree/delivery/cscal950-acceptance/docs/cscal950/acceptance/20260915)。这些记录属于当时的交付基线，不能代替本 PR 基于最新上游 `master` 的新设备复测；HBM/寄存器峰值和专用设备 sanitizer 未运行。
