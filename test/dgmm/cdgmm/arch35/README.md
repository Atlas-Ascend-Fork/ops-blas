# aclblasCdgmm arch35 测试

目标环境为 Ascend 950PR 和 CANN 9.1.0。以下命令均从 ops-blas 仓库根目录运行。

## 构建与精度

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
bash build.sh --soc=ascend950 --ops=cdgmm
export LD_LIBRARY_PATH="$PWD/build:${LD_LIBRARY_PATH:-}"
BLAS_DEVICE_ID=0 ./build/test/dgmm/cdgmm/cdgmm_test \
    --gtest_filter='-*TC_PF*' \
    --gtest_output=json:./accuracy.json
```

CSV 与 `cdgmm_test.cpp` 同目录，按 `__FILE__` 定位。1000 条精度 + 200 条性能保持任务附件清单不变。CSV 必填列缺失、空值、非法数字、重复名称会立即报告错误；mode 同时解析 LEFT/RIGHT 文本与非法枚举数值。

Golden 为任务附件提供的 `cdgmm_golden.h`，函数 `aclblasCdgmm_cpu`，采用列主序、float64 中间结果、LEFT/RIGHT 和正负步长。参考计算逻辑沿用任务附件，排版遵循仓库规范。

实部和虚部分别采用任务要求的 `atol=rtol=2^-13`，匹配率 >=0.99，并检查 `max(1e-2,32*ULP)` 硬误差界。所有非有限值按任务规则逐项检查。CSV 的旧 MERE/MARE 两列显式读取，实际判定采用上述任务标准。

分布：A 与 x 选择独立的均匀或正态填充；实虚部分别使用 seed 与 seed+1000。正态 μ∈[-5,5]、σ∈[0.1,2]，样本截断 [-5,5]，保留附件定义，与 N(0,1) 截断分布存在差异。复数缓冲直接生成，避免额外整矩阵实部/虚部临时数组。

测试检查全部逻辑元素、输出 padding 哨兵、输入 A/x 只读和确定性。原地场景覆盖两种 mode 与 incx=1,2,-1,-2，重跑前恢复原始输入。

## 性能形状功能校验

```bash
BLAS_DEVICE_ID=0 CDGMM_VERIFY_PERF=1 ./build/test/dgmm/cdgmm/cdgmm_test \
    --gtest_filter='*TC_PF*' \
    --gtest_output=json:./performance_shapes.json
```

此阶段检查 200 条原始性能形状和 2 条补充场景的输出与保护区域。

## 正式性能采集

每个 TC_PF 在应用中发射 10 次 kernel，每次均检查接口成功并同步；`msprof op` 对捕获的发射执行 5 次 warmup。`--launch-count=10` 是捕获数量上限，不是将一次应用发射变成 10 个独立样本。分别采集单 case，并要求导出至少 10 条有效 Task Duration 记录：

```bash
unset CDGMM_VERIFY_PERF
BLAS_DEVICE_ID=0 msprof op \
    --warm-up=5 \
    --launch-count=10 \
    --kernel-name=cdgmm_aiv_kernel \
    --aic-metrics=BasicInfo \
    --output=./prof_cdgmm \
    ./build/test/dgmm/cdgmm/cdgmm_test \
    '--gtest_filter=*CsvDriven/TC_PF_1001'
```

实际 case 名称见 CSV 或 `--gtest_list_tests`。读取 `OpBasicInfo*.csv` 的 `Task Duration(us)`，按每例有效样本的算术平均计算 NPU_us。附件基线为本目录的 `gpu_baseline.csv`，先将 gpu_ms 转为 us，再计算 H100_us/NPU_us。200 条原始性能用例逐例倍率应 ≥0.4，算术平均应 ≥0.5。

另有 `TC_PF_EXTRA_LEFT` / `TC_PF_EXTRA_RIGHT` 两个手工性能场景，覆盖 incx>1 与不同前导维 padding，这两例需要同参数的 H100 实测数据才能计算倍率。
