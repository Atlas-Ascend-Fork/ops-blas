# GemmBatchedEx arch22 自验证

环境：Ascend910B3、CANN9.1.0，独立 Netlib BLAS3.10 cblas_sgemm Golden。沿用仓库 gemm_batched_ex_test 目标和原 CMake；构建时需通过 CMAKE_CXX_FLAGS 提供 Netlib cblas.h include，并通过 CMAKE_EXE_LINKER_FLAGS / CMAKE_CXX_STANDARD_LIBRARIES 提供 Netlib cblas/blas 及 Fortran 运行时。不得以其它实现或候选输出冒充已测 Netlib。

按仓库构建说明选择 gemm_batched_ex 和 ascend910b，构建 gemm_batched_ex_test。先完成性能，再运行精度。以下命令中的 TEST_BIN 为该目标的实际绝对路径，使用全新的证据目录。证据盘需要至少8GiB余量，测试提供32MiB guarded workspace，不把此容量当作算子峰值占用。

```bash
source <CANN9.1.0>/set_env.sh
export GEMM_EVIDENCE_DIR=<全新的绝对证据目录>
unset GEMM_REFERENCE_CACHE GEMM_OUTPUT_BLOB_POOL
export GEMM_PHASE=performance
"$TEST_BIN" --gtest_filter='*TC_PF*'
```

性能阶段每条保存85次输入/输出及原始计时。使用 msprof op 采集完整发射序列的 TaskDuration，MIX根任务不把 AIC/AIV 重复相加。主窗口固定为原始第6～15次（预热5次后连续10次）算术均值，逐例检查GPU/NPU≥0.8；原始第26～85次60样本补充统计单列。只有性能全过后继续精度：

```bash
export GEMM_PHASE=precision
"$TEST_BIN"
export GEMM_TEST_CSV=<仓库绝对路径>/test/gemm_batched_ex/arch22/fixed_scalar144.csv
"$TEST_BIN"
export GEMM_TEST_CSV=<仓库绝对路径>/test/gemm_batched_ex/arch22/random_scalar72.csv
"$TEST_BIN"
unset GEMM_TEST_CSV
```

主CSV2393条含原附件1200条、32边界及1161 Gaussian扩展。补充144固定标量和72随机标量合计2609条；随机72为均匀/正态各2组标量复用到9转置×2指针模式，按用例各36条。补充用例不冒充已删除历史72条的精确seed身份恢复。

原FP32合同：atol=rtol=2^-13，有限值匹配率≥.99，硬界max(.01,32ULP)，alpha0缩放遵守EXACT规则。非有限值必须同为NaN或同号Inf，不能把NaN与Inf混为等价。输入、指针、padding及workspace guards同时核验。

评审回归 `special_zero432.csv` 新增432条：1×1×1、8³、16³、17×19×23，9转置×host/device标量×A为±Inf/NaN×B为±0，带非对齐地址和padding。它们验证零乘特殊值仍传播NaN；依据为[Netlib 3.10 SGEMM](https://github.com/Reference-LAPACK/lapack/blob/v3.10.0/BLAS/SRC/sgemm.f)，不替换独立Netlib Golden。

先完成本版本200条性能验收，再自动运行完整精度：

```bash
python3 test/gemm_batched_ex/arch22/run_self_test.py \
  --binary "$TEST_BIN" --output <全新的绝对精度证据目录>
```

脚本逐进程执行2393、144、72、432四份CSV，共3041条参数用例。每进程另执行`CsvCaseCount`和`SpecialValueComparison`两条契约测试。脚本核对CSV名称唯一性/固定条数、XML的逐例覆盖与完成状态、退出码，以及每个batch的数值普查；漏跑、重复、跳过或缺失记录都会失败。数值失败保留后继续后续套件；运行时/guards失败立即停止并记录未测范围。`--suite main|fixed|random|special`可缩小范围；`--dry-run`只验证CSV并打印计划，不构成NPU验证。

按SDK与仓库说明准备可被mssanitizer插桩的测试二进制，然后执行内存检查：

```bash
# 私有构建使用原CMake和前述Netlib编译/链接配置，另增加：
# -DCMAKE_ASC_FLAGS="-g --cce-enable-sanitizer"
python3 test/gemm_batched_ex/arch22/run_memcheck.py \
  --binary <测试二进制绝对路径> --output <全新的绝对memcheck证据目录>
```

memcheck覆盖全部参数错误用例、按shape/转置/标量位置/beta/ldc/batch提取的代表用例和全部432条特殊值用例。每组独立进程，保留mssanitizer完整日志、XML、退出码及选例清单；必须观察到实际Gemm内核插桩开始及匹配的无错误结束，缺少工具、插桩证据、用例或发生诊断均不能PASS。当前Host的beta≠0路径走参考Vector，不能将其冒称为temporaryOutput路径覆盖；不可由公共Host入口触发的内部分支仍是覆盖边界。

历史R50曾有17000计时输出引用验证、200条85次输出逐位一致及串行/双流探针246次调用通过；这些记录绑定原始R50，探针未证明物理时间重叠。原测试基线commit为05b96ea22cea926ecfd43fc7ac691a795522bd61。CI重构、同步改造、特殊值及测试判定修复均已改变源码身份，不将历史结果自动计为当前版本验证。本轮新验证以独立目录内实际日志为准。

性能0.8主窗口200/200；1.0主窗口189/200。GPU基线型号存在A100/P102-100来源冲突，workspace无GPU原数据且任务书§3.4写不涉及，均保持未验收。正式XLSX及日志随PR附件提供。
