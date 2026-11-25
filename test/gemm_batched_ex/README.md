# GemmBatchedEx 测试

## Atlas A2/A3（arch22）

主测试沿用 `arch22/gemm_batched_ex_test.csv` 的上游 2393 条用例，并保留固定标量、随机标量和特殊值的补充 CSV。八个 `GemmBatchedExContractTest` 专项补充覆盖短 K 的 TT/TN 条带、padding、beta=0、流内 Device 标量更新、长归约、alpha=0/k=0、固定工作区和混合 batch 精度选路。专项与上游 CSV 使用独立 GTest 套件，原自测脚本的用例选择和数量保持一致。

### 编译

加载 CANN 9.1.0 环境，按仓库依赖说明准备 ops-tensor、tensor_api、GTest 和 Netlib BLAS 3.10.0。指定 Netlib 头文件和 `libblas.so.3.10.0`：

```bash
source <cann>/set_env.sh
cmake -S . -B <build> -DBUILD_TEST=ON -DTEST_NAMES=gemm_batched_ex \
  -DSOC_VERSION=ascend910b3 -DCANN_3RD_LIB_PATH=<dependencies> \
  -DCMAKE_PREFIX_PATH=<cann>/aarch64-linux/lib64/cmake \
  -DREFBLAS_INCLUDE_DIR=<netlib-include> \
  -DREFBLAS_LIB="<netlib-lib>/libcblas.so.3.10.0;<netlib-lib>/libblas.so.3.10.0"
cmake --build <build> --target gemm_batched_ex_test -j8
```

### 精度

使用运行环境已映射的逻辑设备。单卡容器使用 `BLAS_DEVICE_ID=0`。进入测试程序所在目录，确认 CSV 可读取。

```bash
export BLAS_DEVICE_ID=0
GEMM_PHASE=precision ./gemm_batched_ex_test --gtest_output=xml:<accuracy.xml>
```

上游 CSV 的完整精度和内存检查见 [arch22 测试说明](arch22/README.md)。八个专项可用 `--gtest_filter=GemmBatchedExContractTest.*` 单独执行。

Golden 使用 Netlib 3.10.0 `cblas_sgemm`，逐 batch 检查全矩阵。FLOAT32 使用 atol=rtol=2^-13、matched_ratio≥0.99、max_abs_error≤max(0.01,32ULP)。非有限值按任务书单独判定；缩放路径按位比较。C 的 padding 必须保持不变。

### 性能

设置 `GEMM_PHASE=profile` 后，每条 `TC_PF_*` 用例只调用一次 `aclblasGemmBatchedEx`，确认返回 SUCCESS 并同步 stream。使用 `msprof op` 的预热参数采集；`GEMM_PHASE=performance` 保留原扩展计时模式。

进程内使用已映射的逻辑设备 0。`msopprof` 是 `msprof op` 的独立入口。以下命令采集一个用例的一个有效样本：

```bash
source <cann-9.1.0>/set_env.sh
export BLAS_DEVICE_ID=0
export GEMM_PHASE=profile
msopprof --warm-up=5 --launch-count=5000 --aic-metrics=BasicInfo --output=<sample>/profile \
  <binary> --gtest_filter='*/TC_PF_1064' --gtest_output=xml:<sample>/gtest.xml
```

按任务书 §3.3，每条用例连续采集 10 次，每次使用独立输出目录。5 次预热不计入样本。每个样本累加该次接口调用全部 kernel 的 `Task Duration(us)`，覆盖输入整理、矩阵乘和写回；MIX 根任务不重复累加 AIC/AIV 子任务。将 10 个有效样本取算术平均，得到 Avg time（us）。

`--launch-count` 是采集 kernel 数量的上限，不是接口调用次数或采样次数。采集后核对完整发射序列和有效记录数。输出目录放在源码仓之外，保存命令、日志、GTest XML 和原始 CSV。异常样本记录原因；有效样本不足 10 次时不作验收判定。

用同一方法逐条验证 200 条性能用例。与附件 `gpu_baseline.csv` 比较，Avg time 不高于 GPU 耗时的 1.25 倍。Golden 精度通过上一节的独立测试验证。
