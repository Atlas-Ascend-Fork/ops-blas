# aclblasCtpttr 实卡测试

从 ops-blas 仓库根目录运行，使用 Ascend 950PR 和 CANN 9.1。

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
TRIANGULAR_VERIFY_PERF=1 bash build.sh --ops=ctpttr --soc=ascend950 --run
```

用例表和 H100 基线位于 `arch35/`。测试按实部和虚部的位模式比较输出，并检查输入、未选三角区、padding 和尾部哨兵。

使用 `msprof op` 采样时，取消设置 `TRIANGULAR_VERIFY_PERF`，通过 `--gtest_filter` 选择单个性能用例。每个用例发射 10 次算子。
