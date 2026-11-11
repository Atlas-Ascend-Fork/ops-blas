# aclblasCgetriBatched 自测说明

## 覆盖范围与输入

`arch35/cgetri_batched_test.csv` 保留任务附件的 1200 条原始用例，并补充 2 条 Inf/NaN、12 条强制循环换行主元、14 条混合奇异对角矩阵、8 条前代求解和 16 条尾部列求解回归用例，共 1252 条。原始用例包括 1000 条 API/精度用例和 200 条性能用例。`arch35/gpu_baseline.csv` 保存任务提供的全部 200 条 GPU 性能基线；测试检查基线条数、用例对应关系、n 和 batchSize，缺失或无效基线直接失败。

PIVOT 输入通过 LAPACK `cgetrf_` 得到 LU 和 1-based pivot，NO_PIVOT 输入使用工程内的无主元 LU。有限数 golden 使用 LAPACK `cgetri_`。一般矩阵和三角矩阵加强实对角线，并检查无穷范数条件数不超过 1e6。

性能输入使用 CSV 指定的矩阵类型。测试在 CPU 上生成、分解一份可复现矩阵，将 LU 和 pivot 复制到该 case 的所有 batch。每个矩阵均在 NPU 上执行并逐 batch 检查结果，CPU 数据准备、分配、传输和 golden 计算不计入性能区间。IDENTITY 和 ILL_CONDITIONED 精度用例同样复用完全相同输入的 CPU 分解和 golden。

## 精度门禁

- 实部、虚部分别满足 `rtol=2^-10`、`atol=2^-16`，匹配比例至少 0.99。
- 最大绝对误差不超过 `max(1e-2, 32 ULP)`。
- 奇异矩阵检查 `infoArray`，不比较未定义的逆矩阵内容。
- 输出逻辑矩阵在调用前置零；leading dimension 大于 n 时检查输出 padding 哨兵。
- Inf/NaN 用例检查接口和流同步成功、`infoArray == 0` 及非有限值传播。
- `TC_RG_PIVOT_*` 在良态稠密矩阵上循环交换行，断言 CPU getrf 确实产生非平凡主元，并比较完整逆矩阵；覆盖 SIMT、SIMD 和分块路径。
- `TC_RG_INFO_*` 在同一批次混合可逆矩阵与不同对角位置的精确奇异矩阵，检查每个 info。n8/n16/n32 分别使用 batch=513/257/129，覆盖 8/4/2 个矩阵交错处理和尾批次；补充用例均含 lda/ldc padding。
- n128、n129、n256、n257、n288、n289 的补充用例覆盖完整 LU 与分片 LU 的切换，以及 16、8、4 列 LU 面板之间的边界，包含强制主元交换及成功/奇异矩阵混合批次。分块路径从 n321 开始，验收时还应核查 n320/n321 的主元、混合奇异状态和非对齐矩阵指针。
- `TC_RG_FORWARD_*` 使用单位下双对角矩阵，覆盖 n8/n16/n32/n64、pivot/null pivot、padding 和尾批次。除完整 LAPACK 比对外，逐元素检查逆矩阵的第一条次对角线等于 `-0.1-0.2i`，捕获 Debug 前代漏算等可能被 99% 总体比例掩盖的问题。
- `TC_RG_TAIL_*` 覆盖 n65..n72 的全部尾部列数，包含强制主元交换、无主元下双对角矩阵、padding 和独立批次。尾部每个实部/虚部元素都检查原有 rtol/atol，不依赖整体 99% 匹配比例。

每个完成比较的精度用例输出 `ACCURACY_RESULT`；逐 batch 的 GTest 断言仍是判定依据。

## Release 构建和运行

在 Ascend 950PR 上加载 CANN 环境，使用对应 CANN 支持的 CMake：

```bash
source /usr/local/Ascend/cann-9.1.0-beta.3/set_env.sh
export PATH=/workspace/venvs/pytorch-npu/bin:$PATH
export EAGER_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64"
bash build.sh --soc=ascend950 --ops=cgetri_batched --device=0
```

CMake 未显式指定构建类型时默认 Release，也可在直接调用 CMake 时指定 `-DCMAKE_BUILD_TYPE=Release` 或 Debug。正式验收使用全量重建的 Release 产物，保存 `CMakeCache.txt`、构建日志及源码完整 SHA。测试设备由构建时的 `--device` 参数选择。

对构建得到的 `cgetri_batched_test` 分别运行：

```bash
./cgetri_batched_test --gtest_filter='*-*TC_PF_*'
./cgetri_batched_test --gtest_filter='*TC_PF_*'
```

第一条应执行 1052 条用例，第二条应执行 200 条用例。可按用例名称选择失败项诊断，但最终必须运行全量。

## Debug 功能验证

导师在 [验收 Issue](https://gitcode.com/kevin_lee1231/ops-blas/issues/1) 中指定后续验收使用 Release。Debug 使用独立目录和 `-DCMAKE_BUILD_TYPE=Debug` 全量构建，保留 CANN 的 `-O0 -g`，不覆盖设备优化选项。

Debug 功能验证设置 `CGETRI_TEST_FUNCTIONAL_ONLY=1` 运行全量用例：1052 条 API/精度用例照常验证；200 条性能输入保持相同形状、矩阵类型和 batch，每条执行一次并检查所有输出及 info，不做性能判定。此模式日志输出 `FUNCTIONAL_RESULT ... performance_measured=0`，XML 记录 `performance_measured=0`，不能用作性能达标证据。

Debug 性能复测仍按下述全部 200 条基线和 5+51 次完整调用判定。正式性能运行须清除上述功能模式开关，并检查 XML 中 `performance_measured=1`，防止功能结果被误报为性能通过。

## 性能门禁与 msprof 口径

全部 200 条性能用例都要求：

```text
GPU 基线耗时 / NPU 耗时 >= 0.4
NPU 阈值(us) = gpu_ms * 1000 / 0.4
```

开发阶段可使用更严格的时限目标（例如 GPU/NPU >= 0.8，即设备平均耗时不超过原时限的一半），保持同样的采样与完整 kernel 聚合规则；验收门禁不得低于 0.4。

每个 case 预热 5 次，随后连续调用 51 次，最后同步一次。GTest 用 host 单调时钟记录这 51 次调用和同步的总耗时，除以 51 得到 `avg_us`，并对全部 200 条用例检查上述阈值。该值包含主机发射开销，不能直接称为 msprof 设备 kernel 时间。

```text
PERF_RESULT case=<id> n=<n> batch=<batch> warmup=5 samples=51 avg_us=<value> formal=1 threshold_us=<value> ratio=<value>
```

导师的正式设备计时使用 msprof：每个 case 恰有 56 次算子调用，前 5 次是 warmup，后 51 次有效。对于分块实现，一次 API 调用包含初始化、三角求解、矩阵乘更新和收尾等多个 kernel；必须将同一次调用内的所有 kernel 时长相加，再对 51 次有效调用取平均。不得只统计其中一个 kernel，亦不得把整组 56 次取平均后当作有效样本。

在测试二进制所在目录执行（`msprof-output` 应是本次运行的独立目录）：

```bash
OPENBLAS_NUM_THREADS=8 msprof --output=./msprof-output --task-time=on --ascendcl=off \
  --python-path=/workspace/venvs/pytorch-npu/bin/python \
  ./cgetri_batched_test --gtest_filter='*TC_PF_*' --gtest_output=xml:performance.xml \
  > performance.log 2>&1
```

完整调用口径校验以该次运行 `mindstudio_profiler_output/op_summary_*.csv` 为依据：要求 200 个不同用例、每个用例 56 个完整调用、正确的 kernel 顺序和有限正耗时；缺项、额外调用或任一性能不达标均判失败。n<=320 每次调用一个 kernel；n>=321 令 T=ceil(n/64)，每次调用包含 14T-10 个 kernel。解析历史提交的数据时，必须使用与采集源码对应的校验口径。

## 验收证据

保存命令、完整 SHA、构建配置、stdout/stderr、GTest XML、msprof 原始输出及逐 case 聚合结果。Release 验收成功要求无 failed/skip，精度/API 为 1052/1052（含原始 1000 条），性能为 200/200，且 msprof 按上述完整调用口径聚合后全部满足 0.4。Debug 功能通过与 Debug 性能通过分别报告。构建成功、零用例运行或仅代表用例通过均不代表验收完成。
