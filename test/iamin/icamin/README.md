# aclblasIcamin 自验证

## A2/A3 构建与精度

在 ops-blas 仓库根目录执行，先加载目标机器的 CANN 9.1.0 环境。

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
bash build.sh --ops=icamin --soc=ascend910b3
export BLAS_DEVICE_ID=0 ASCEND_DEVICE_ID=0 OPENBLAS_NUM_THREADS=1
export LD_LIBRARY_PATH="$PWD/build:$LD_LIBRARY_PATH"
./build/test/iamin/icamin/icamin_test --gtest_output=xml:accuracy.xml
```

A3 构建使用对应的 `ascend910_93` SOC 型号。测试依赖 GTest 和 CBLAS 开发包；OpenBLAS 头文件位于非默认目录时，通过 `CMAKE_INCLUDE_PATH` 指定目录。

`arch22/icamin_test.csv` 包含任务提供的1200例，`arch22/icamin_test_contract.csv` 补充924例，另有2个独立GTest。补充用例覆盖正态分布、非对齐地址、步长、相同模值、Inf/NaN、quick return、参数错误及跨度边界。每例的输入模式、seed和参数写在CSV中。

Golden 使用共享的 `icamin_golden.h` CPU参考循环。FP32计算 `abs(real)+abs(imag)`，严格小于才更新索引；结果为1-based INT32，使用 `EXPECT_EQ` 精确比对。首元素为NaN时结果为1，其他NaN不更新索引。负向用例按期望状态码判定。

## A2 性能

性能用例为 `TC_PF_1001` 至 `TC_PF_1200`，GPU基线位于 `arch22/gpu_baseline.csv`。每个GTest仅调用一次接口，预热由 `msprof op` 完成。

```bash
for round in $(seq 1 10); do
    mkdir -p "perf/round-$round"
    GTEST_FILTER='*/TC_PF_*' msprof op \
        --warm-up=5 --launch-count=400 --aic-metrics=BasicInfo \
        --output="perf/round-$round" \
        ./build/test/iamin/icamin/icamin_test \
        > "perf/round-$round/msprof.log" 2>&1
done
```

从每轮 `OpBasicInfo_*.csv` 读取 `Task Duration(us)`。按GTest与kernel实例顺序关联用例；两阶段路径将 `icamin_arch22_vector_kernel` 和 `icamin_arch22_finish_kernel` 的耗时相加，作为一个有效样本。每例10个有效样本取算术平均，工具的5次预热不计入样本。`--launch-count=400` 覆盖整轮kernel采集，10个样本来自外层10轮。

GPU基线由毫秒换算为微秒。平均耗时不高于GPU基线的1.25倍即通过。任务书的内存要求为不涉及。

## A2/A3 向量接口约束

- `Compare` 将每个比较结果写为 1 bit；`LocalTensor<uint8_t>` 是承载这些 bit 的字节视图。掩码按 `ceil(count / 256) * 32` 字节分配，host UB 预算与 kernel 一致。掩码对齐尾字节在使用前清零。
- `Brcb(..., 1, {1, 8})` 将 8 个 float 分别广播到 8 个 32-byte 块，共写入 64 个 float。A2/A3 的 `Compare` 前 n 元素接口按 256 字节执行，保留 64 个 float 的比较长度；后续 `Select` 使用前 8 bit。
- A2/A3 的 `ReduceMin` 前 n 元素接口支持不足一个 repeat 的输入及尾块。归约传入 `valid`，使 +Inf 填充不参与索引选择；相同最小值保留最小索引。
- Gather 偏移视图的完整向量读取范围预留并初始化，host 与 kernel 共用 `ICAMIN_METADATA_FLOATS` 容量。kernel 引用启动参数，避免额外的 tiling 私有副本。
- `TC_REDUCTION_*` 的 36 例覆盖 n=2/8/63/65、三种访存布局、跨核尾块、两 tile 合并、全 Inf/NaN 和末元素最小值。每例与 CPU Golden 精确比对，并检查输出前后哨兵。
## 本次评审验证

Ascend910B3、CANN 9.2.1。当前实现的 36 条 `TC_REDUCTION_*` 专项与 CPU Golden 精确一致，普通执行及 `mssanitizer --tool=initcheck` 均通过，initcheck 无 error/warning。外置记录为 `no-reset-accuracy.xml`、`no-reset-initcheck.xml` 和 `no-reset-initcheck-report.log`。

当前实现的完整 2126 项回归在设备初始化阶段停止，未计为通过；FFTS_BASE_ADDR 的 memcheck 告警尚未闭合。
