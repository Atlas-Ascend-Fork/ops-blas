# aclblasHgemmStridedBatched 测试

适用 Ascend 950PR、CANN 9.1.0。测试通过仓库 CSV/GTest 框架调用公共接口，输入输出为 FP16，逐 batch 全矩阵比较 Netlib BLAS 3.10.0 的 FP32 `cblas_sgemm` golden，经 RNE 转 FP16 后判定。

| 文件 | 用途 |
| --- | --- |
| `hgemmStridedBatched_param.h` | CSV参数解析和可复现输入生成 |
| `hgemmStridedBatched_golden.h` | Netlib golden、任务精度配置及公共Verifier调用 |
| `arch35/hgemmStridedBatched_npu_wrapper.h` | 设备调用、工作区、缓冲哨兵和输入输出回放 |
| `arch35/hgemmStridedBatched_test.cpp` | CSV/GTest注册与测试流程 |
| `arch35/hgemmStridedBatched_test.csv` | 原始1200例：1000功能/精度、200性能；性能例也可执行golden检查 |
| `arch35/hgemmStridedBatched_supplement.csv` | 880例：663参数/布局边界、180分组流水、37输出重叠边界 |
| `gpu_baseline.csv` | 200条H100原始基线，`gpu_ms`单位ms |
| `run_self_test.py` | 构建并执行所选CSV，检查退出码、超时、XML及实际用例集合 |
| `run_memcheck.py` | 分组运行mssanitizer，检查设备kernel实际检测、告警、泄漏及GTest结果 |
| `reference/build_reference.py`、`reference/sources.json` | 下载或读取离线Netlib输入，按固定版本的字节数和SHA-256验证后编译 |
| `reference_parallel.h`、`reference/reference_worker.py`、`reference/reference_environment.c` | 大矩阵CPU参考的列切分、共享内存和浮点环境恢复 |

## 准备与构建

需要 CMake≥3.16、GCC/G++、gfortran、Python3、GTest；按仓库说明准备CANN及公共依赖。所有命令在ops-blas仓库根执行，下面输出目录须事先不存在。

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
TEST=test/gemm/hgemmStridedBatched
python3 "$TEST/run_self_test.py" --filter 'TC_L0_*' --device 0 --output /tmp/hgemm-l0
```

入口先将固定Netlib源编译为`build/reference/libnetlib_sgemm.so`，再配置`SOC_VERSION=ascend950`、`BUILD_TEST=ON`、`TEST_NAMES=hgemmStridedBatched`，显式设置`REFBLAS_LIB`和`REFBLAS_INCLUDE_DIR`并构建`hgemmStridedBatched_test`。不以系统OpenBLAS替代任务要求的Netlib。参考源使用`-fno-fast-math -ffp-contract=off`；大矩阵worker加载测试程序实际链接的同一动态库。

无网络时，将`reference/sources.json`列出的原始文件（含许可证）放到同一输入目录，再设置`export HGEMM_REFERENCE_INPUTS=/absolute/path/to/netlib-inputs`。校验仍强制执行，缺失或哈希不匹配即失败。也可先独立准备参考库：

```bash
python3 "$TEST/reference/build_reference.py" build/reference
```

二进制位于`build/test/gemm/hgemmStridedBatched/hgemmStridedBatched_test`。默认申请高带宽Device内存，分配策略由日志`[MEMORY]`和`summary.json`记录；`--device-memory normal`仅作普通页对照，精度前置与性能采集须采用同一配置。

## 完整精度与补充测试

```bash
python3 "$TEST/run_self_test.py" --skip-build --device 0 --workers 16 \
  --timeout 43200 --output /tmp/hgemm-original
python3 "$TEST/run_self_test.py" --skip-build --device 0 --workers 4 \
  --csv "$TEST/arch35/hgemmStridedBatched_supplement.csv" --dump \
  --output /tmp/hgemm-supplement
```

默认全量执行1200例，包括性能分类的正确性；补充CSV单独执行880例。`--filter 'TC_OV_*'`等可选取单类。原始CSV不因补充用例改变。任何失败、跳过、缺失、重复、非零退出或超时均判失败，不能仅看终端的“0 failures”。`--workers`仅用于正确性进程分片，性能采集须串行。

输入按CSV seed及fill产生，A/B/C使用seed、seed+100、seed+200；需记录C++标准库版本以复现正态分布。golden使用本次实际FP16量化后的输入。每个batch分别检查有限元素满足`abs(actual-golden) <= 2^-10*(1+abs(golden))`的比例≥0.99，且最大绝对误差≤`max(0.1,32*ULP(g_low))`，其中`g_low`为最大误差位置的FP16 golden。NaN/Inf按任务书等效规则单独检查；C padding、batch间隙及哨兵不得被改写。

每次保存`cases.json`、`accuracy.log`、`gtest.xml`、`batches.jsonl`和`summary.json`。`--dump`另存本次实际`A.f16.bin`、`B.f16.bin`、`C_input.f16.bin`和`C_actual.f16.bin`；用`--input-attachments /path/to/attachments`配合同一CSV可读取实际输入重跑，旧C_actual仅供对照。不要对全部大矩阵无选择地导出数据。

CPU参考在大矩阵上按独立输出列并行，保持完整K归约及同一Netlib计算。`HGEMM_REFERENCE_WORKERS=1`可验证串行路径；`HGEMM_REFERENCE_MIN_VOLUME=0 HGEMM_REFERENCE_WORKERS=4`可在小例上覆盖并行路径。这些变量只影响CPU golden，不影响NPU算法或性能计时。

## 内存检查

在950环境加载CANN后使用`mssanitizer --help`核对工具安装。单独构建带`--sanitizer -g`的内存检查版本，保留正常构建用于精度与性能；检查版本仅通过mssanitizer启动。以下命令在仓库根目录执行，离线Netlib输入的配置同上：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
TEST=test/gemm/hgemmStridedBatched
python3 "$TEST/reference/build_reference.py" build-memcheck/reference
export EAGER_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64"
export LINUX_INCLUDE_PATH="$ASCEND_HOME_PATH/x86_64-linux/include"
cmake -S . -B build-memcheck -DSOC_VERSION=ascend950 -DCMAKE_BUILD_TYPE=Release \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
  -DCMAKE_PREFIX_PATH="$ASCEND_HOME_PATH/lib64/cmake" \
  -DCMAKE_ASC_FLAGS='--sanitizer -g' \
  -DBUILD_TEST=ON -DTEST_NAMES=hgemmStridedBatched \
  -DREFBLAS_INCLUDE_DIR="$PWD/build-memcheck/reference" \
  -DREFBLAS_LIB="$PWD/build-memcheck/reference/libnetlib_sgemm.so" \
  -DREFLAPACK_LIB="$PWD/build-memcheck/reference/libnetlib_sgemm.so"
cmake --build build-memcheck --target hgemmStridedBatched_test -j8
python3 "$TEST/run_memcheck.py" --device 0 --output /tmp/hgemm-memcheck
```

默认二进制为`build-memcheck/test/gemm/hgemmStridedBatched/hgemmStridedBatched_test`；`--binary`、`--tool`可指定实际路径，`--timeout`为每组超时秒数。`--dry-run`只保存计划，`passed`始终为false。工具或二进制缺失、超时和检测失败返回非零，输出目录必须不存在。

入口按contracts、direct、workspace、grouped、scale顺序串行运行，每组单独使用`mssanitizer --tool=memcheck --leak-check=yes`，保留工具默认warn级别，日志目录以0750权限创建。contracts复用两份CSV的全部负向和空操作用例；成功样本覆盖N/T/C九种组合、65/127/129/255尾块、lda/ldb/ldc填充与batch间隙、非对齐C、65×9输出受限工作区的7+2列strip、2+1个batch的group复用及scale-only。紧凑8/16/32/64方阵采用257个batch，在本任务28个Cube核上覆盖8槽环形复用、延迟Fixpipe和不足4个batch的尾组。所有成功样本仍执行Netlib golden及padding检查，不启用profiling跳过分支。

输出`plan.json`保存完整输入与源CSV哈希；各组保存`cases.csv`、`command.json`、`process.log`、`sanitizer.log`、`gtest.xml`、`batches.jsonl`（有golden输出时）和`summary.json`。顶层`summary.json`汇总通过状态及未测组。必须同时满足：进程成功、GTest用例集合完整且无失败/跳过、无sanitizer告警或错误/泄漏、所有kernel的start与无错误finish数量按名称配对；各成功组所需`hgemm_*` kernel检测次数至少等于该组用例数。仅统计独立sanitizer日志以避免stdout重复记录；成功样本组的空日志、零实际kernel或未完成检测不能判成功。contracts不应发射kernel，允许无告警时报告文件为空，但报告文件必须存在，且49条GTest契约用例必须完整通过；不能据此替代四组设备检测。失败后停止后续组并明确记为未测。

## 性能采集

先完成对应用例正确性检查，再串行采集200条`TC_PF_*`。使用本机`msprof op --help`核对工具参数；以下为单例入口：

```bash
export HGEMM_CSV="$(pwd)/$TEST/arch35/hgemmStridedBatched_test.csv"
export HGEMM_DEVICE_MEMORY=high-bandwidth
export HGEMM_PROFILE_REPEATS=10
export BLAS_DEVICE_ID=0
msprof op --application="$(pwd)/build/test/gemm/hgemmStridedBatched/hgemmStridedBatched_test --gtest_filter=*/*/TC_PF_1001" \
  --output=/tmp/hgemm-profile-1001
unset HGEMM_PROFILE_REPEATS HGEMM_CSV HGEMM_DEVICE_MEMORY BLAS_DEVICE_ID
```

profiling分支先执行一次准备调用并同步，再发射指定次数；该分支跳过golden，不能代替先前的正确性测试。剔除准备调用及工具warmup，关联每次API调用的全部必要kernel（含临时结果和合并），以实际Task Duration(us)求和；不能把GTest wall-time、单个最快kernel或空采样当作接口耗时。保存工具版本、完整命令、原始采样、关联规则及有效条数，缺失、NA、零耗时或解析失败记为采集失败。

任务书§3.3要求msprof自身5次warmup及10次有效样本，逐例GPU/NPU≥0.4，200例倍率算术平均≥0.5；§7.4及测试指导另含更高采样次数/性能口径，须分别记录并按实际验收要求执行。GPU表ms换算为us后比较。任务级采样编排、报告生成与验收打包工具独立随复现资料提供，不作为本测试工程的构建依赖。

精度校验复用 `frame/verify.h` 的 `Verifier`，显式设置任务要求的 rtol=atol=2^-10；非有限值按任务书单独统计并判断，有限值单独计算匹配率，保留最大误差锚点的硬限值。公共接口的 strideA/strideB/strideC 使用 `int64_t`，与同族跨步批量接口一致。
