# aclblasCgemmEx 测试说明

## 编译与运行

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cd ops-blas
bash build.sh --soc=ascend950 --ops=cgemm_ex --run --device=0
```

设备号：`npu-smi` 显示 NPU 0；若 `aclrtSetDevice(0)` 失败，改 `--device=5`。

## 精度

```bash
./build/test/gemm/cgemm_ex/cgemm_ex_test --gtest_filter='-:*TC_PF*'
./build/test/gemm/cgemm_ex/cgemm_ex_test --gtest_filter='*TC_HX*'
```

## 自测报告（逐条表）

精度逐条：`--gtest_filter='-:*TC_PF*'` 日志解析为 `tc_acc_results.csv`（1019 行）。
性能逐条：`docs/bench_tc_pf.cpp` 对 `gpu_baseline.csv` 200 条做 aclrtEvent 计时
（warmup 8 + 有效采样 50），输出 `tc_pf_results.csv`。
汇总 xlsx：`docs/aclblasCgemmEx算子自验证报告.xlsx`（工作表「精度明细」「性能明细」）。

## 性能

任务书 §3.3 前 4 条标杆（Avg time，us）：

1. 1024 NN C_32: 415.95
2. 2048 NN C_32: 3243.65
3. 1024 TN C_32: 411.27
4. 2048 NN R_32: 851.86

warmup 后有效采样 >50 次取平均。
