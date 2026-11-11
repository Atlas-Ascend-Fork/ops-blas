# aclblasCsymv 测试

加载 CANN 9.1.0 的 `set_env.sh` 后，在仓库根目录构建并运行：

```bash
# Atlas A2
bash build.sh --soc=ascend910b3 --ops=csymv --run --device=0

# Atlas A3
bash build.sh --soc=Ascend910_9391 --ops=csymv --run --device=0

# Ascend 950PR
bash build.sh --soc=ascend950 --ops=csymv --run --device=0
```

构建后可分别运行精度和性能用例：

```bash
export BLAS_DEVICE_ID=0
build/test/symv/csymv/csymv_test --gtest_filter='-*TC_PF*'
build/test/symv/csymv/csymv_test --gtest_filter='*TC_PF*'
```

A2/A3 构建完成后，通过 CMake 目标执行自测和内存检查：

```bash
BLAS_DEVICE_ID=0 cmake --build build --target csymv_self_test
BLAS_DEVICE_ID=0 cmake --build build --target csymv_memcheck
```

这两个目标分别调用本目录的 `run_self_test.py` 和 `run_memcheck.py`，并依赖
`csymv_test`。自测核对 1244 条 CSV、200 条 GPU 基准、GTest 注册与实际执行清单，
默认运行 1044 条精度用例及独立接口、边界测试。失败、跳过、漏执行或零用例均返回非零。
结果写入 `build/test/symv/csymv/self_test/self_test_summary.json`。

内存检查要求已加载 CANN 环境且 `mssanitizer` 在 PATH 中，以 `--tool=memcheck`
执行 TC_ED/TC_INC、GENERIC 的 incx=±2/±3、n=16/32/64、T64 的 n=697、
T128 的 n=2048/4096、n=8192、Device 标量和 workspace 回退。
不同 fixture 分进程运行，逐 kernel 配对检查 `Start memcheck` 与 `No error detected`；
插桩缺失或诊断异常均失败。结果写入 `build/test/symv/csymv/memcheck/memcheck_summary.json`。

直接调用可指定设备、测试程序和输出目录；完整性能门禁额外检查全部 200 条性能判定：

```bash
python3 test/symv/csymv/run_self_test.py \
  --binary build/test/symv/csymv/csymv_test --device 0 \
  --output build/test/symv/csymv/full_self_test --include-performance
python3 test/symv/csymv/run_memcheck.py \
  --binary build/test/symv/csymv/csymv_test --device 0 \
  --output build/test/symv/csymv/memcheck
```

A2/A3 的每条性能用例预热 5 次，再运行 100 次，取每次调用的 kernel 总耗时的平均值。全部 200 条用例报告 `GPU_us / NPU_us >= 0.8` 的达标情况，GPU 基准见 `arch22/gpu_baseline.csv`。前五条按任务书时延上限和该比例断言，其余用例仅报告性能判定。

外部 kernel-only 采集器通过 `CSYMV_EXTERNAL_PROFILER=1` 接管计时：

```bash
CSYMV_EXTERNAL_PROFILER=1 \
CSYMV_KERNEL_CSV=./kernel_activity.csv \
LD_PRELOAD=/path/to/libcsymv_kernel_collector.so \
  build/test/symv/csymv/csymv_test --gtest_filter='*TC_PF*'
```

外部采集器只订阅 MSPTI 的 kernel activity，不记录 Host 全量任务，不重放 kernel；直接输出起止时间 CSV。测试程序关闭内部 MSPTI，但保留完整公共 API 调用、数值校验、5 次真实预热和连续 100 次调用。100 次调用之间不插入同步，批末统一同步并校验结果。

每条用例依次产生 1 次数值校验、5 次预热和 100 次批量调用的 kernel 记录。按用例执行顺序分组，每组跳过前 6 条，以后 100 条 kernel duration 的平均值作为 `NPU_us`，沿用原 GPU 基准和门限。记录不完整或数值校验失败时，该次结果无效；外部模式的 GTest 通过仅表示执行和数值校验通过，性能判定使用外部记录。

`msprof op` 注入时也会关闭内部 MSPTI，避免采集器冲突；其逐 kernel 调优采集与上述 kernel-only 批量采集不同，不混用结果。

Ascend 950PR 的全部 200 条性能用例要求 `GPU_us / NPU_us >= 0.4`，GPU 基准见 `arch35/gpu_baseline.csv`。
