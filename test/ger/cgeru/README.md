# aclblasCgeru 测试说明

本目录包含 Ascend 950PR（arch35）与 Atlas A2/A3（arch22）上 `aclblasCgeru` 的功能、边界、精度与性能测试。CSV 测试共 1204 组，覆盖正负步长、`lda` 填充、零维、非法参数、Inf、NaN、极端值、复乘消去、乘积溢出以及连续和非连续访问；另外包含 5 个专项 GTest，用于校验空句柄、quick return、零 `y` 列特殊值语义、无共轭语义，以及连续 Reg 路径的窄高矩阵和行数边界。后者执行 50 次形状组合验证，覆盖 `4096×9`、非向量对齐行数及 `m=4097` 的执行路径边界。

```text
test/ger/cgeru/
├── cgeru_golden.h / cgeru_param.h / cgeru_perf_common.h   # 公共 golden / 参数 / 性能工具
├── arch22/   # Atlas A2/A3（910B/910_93，CANN 9.1.0）
└── arch35/   # Ascend 950PR（950）
```

## 构建

在 ops-blas 仓库根目录执行：

```bash
# Ascend 950PR（arch35）
bash build.sh --soc=ascend950 --ops=cgeru
# Atlas A2/A3（arch22）
bash build.sh --ops=cgeru
```

## 功能与精度测试

```bash
./build/test/ger/cgeru/cgeru_test
```

测试以 Netlib CBLAS `cgeru` 为 golden，Complex64 的实部和虚部分别按 FLOAT32 标准比较：`rtol=2^-10`、`atol=2^-16`、匹配比例不低于 0.99，且最大绝对误差不超过 `max(1e-2, 32*ULP)`。全部 1209 个测试均通过时，功能与精度验收通过。

arch22 用例按同名 CSV 驱动，另外覆盖 `m > 2048` 的行片段边界、非单位/负步长与 `lda > m` 的填充列，全量 1214 条（1200 条任务书 CSV + 14 条专项）通过即验收。

## 性能测试

```bash
./build/test/ger/cgeru/cgeru_perf \
  ./test/ger/cgeru/arch35/gpu_baseline.csv
# arch22
./build/test/ger/cgeru/cgeru_perf \
  ./test/ger/cgeru/arch22/gpu_baseline.csv
```

性能程序对每组用例先预热 10 次，再以 ACL Event 测量 100 次异步调用的平均耗时。`gpu_baseline.csv` 包含附件给出的 200 组 GPU 基线；当 `GPU 平均耗时 / NPU 平均耗时 >= 0.4` 时该组通过（arch22、arch35 相同）。程序逐组输出 NPU 平均耗时、GPU/NPU 比值和判定，并在标准错误输出汇总结果；仅当 200 组全部通过时返回 0。arch22 程序同时打印任务书 910B3 的单次调用上限对照列 `target_us`，供参考。

其中任务书的典型门槛为：512×512 不高于 8.6075us、1024×1024 不高于 12.82us、2048×2048 不高于 44.5275us。性能测试应在空闲的目标设备上执行。
