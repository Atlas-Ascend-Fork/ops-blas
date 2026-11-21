# aclblasCtpttr 实卡测试

从 ops-blas 仓库根目录运行，使用 Ascend 950PR 和 CANN 9.1。

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
TRIANGULAR_VERIFY_PERF=1 bash build.sh --ops=ctpttr --soc=ascend950 --run
```

用例表和 H100 基线位于 `arch35/`。测试按实部和虚部的位模式比较输出，并检查输入、未选三角区、padding 和尾部哨兵。

使用 `msprof op` 采样时，取消设置 `TRIANGULAR_VERIFY_PERF`，通过 `--gtest_filter` 选择单个性能用例。每个用例发射 10 次算子。

设置 `TRIANGULAR_VERIFY_PERF=1` 时，`PerformanceBaselineCases` 会对 `arch35/gpu_baseline.csv` 全部 200 行做 NPU 计时（预热 10 次、计时 100 次取平均）并打印诊断；其中 `ctpttr-base-*` 四个任务书 shape 断言 NPU 平均用时 ≤ H100 基线 × 2.5（与 ctbmv/cgerc 的阈值口径一致），`ctpttr-perf-*` 行暂不断言，待目标实机按 size class 校准预算后再启用。
