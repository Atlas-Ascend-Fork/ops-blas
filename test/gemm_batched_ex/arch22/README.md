# GemmBatchedEx arch22 验证

环境：Ascend910B3、CANN 9.1.0、独立 Netlib BLAS 3.10.0 `cblas_sgemm` Golden。构建目标为 `gemm_batched_ex_test`，编译和单次接口性能采集方法见 [测试说明](../README.md)。

## 精度范围

| CSV | 用例数 | 范围 |
|---|---:|---|
| `gemm_batched_ex_test.csv` | 2393 | 任务附件 1200 条、32 条边界、1161 条 Gaussian 扩展 |
| `fixed_scalar144.csv` | 144 | 固定标量、9 种转置组合、Host/Device 标量 |
| `random_scalar72.csv` | 72 | 均匀/正态分布、随机标量、Host/Device 标量 |
| `special_zero432.csv` | 432 | ±Inf/NaN 与 ±0 相乘、非对齐地址、padding |

共 3041 条参数用例。每个进程另执行 `CsvCaseCount` 和 `SpecialValueComparison` 两条契约测试。独立的 `GemmBatchedExContractTest` 包含八个专项：TT/TN 短 K、padding、Device 标量更新、长归约、只缩放 C、固定工作区和混合 batch。

FLOAT32 使用 atol=rtol=`2^-13`、有限值匹配率不低于 0.99、逐元素硬界 `max(0.01,32ULP)`。alpha=0 或 k=0 的缩放结果按位比较。非有限值必须同为 NaN 或同号 Inf。输入、指针数组、C padding 及工作区 guards 同时核验。

```bash
source <cann>/set_env.sh
export BLAS_DEVICE_ID=0
export GEMM_PHASE=precision
export GEMM_EVIDENCE_DIR=<新的绝对证据目录>
unset GEMM_REFERENCE_CACHE GEMM_OUTPUT_BLOB_POOL
python3 test/gemm_batched_ex/arch22/run_self_test.py \
  --binary <测试程序绝对路径> --output <新的完整精度输出目录>
<测试程序绝对路径> --gtest_filter='GemmBatchedExContractTest.*'
```

自测脚本逐进程执行四份 CSV，核对 XML 的逐例覆盖、退出码及逐 batch 数值普查。`--suite main|fixed|random|special` 可选择范围。数值失败保留并继续；运行时或 guards 失败立即停止。`--dry-run` 只显示计划。

固定工作区专项覆盖 256 B 和 32 MiB，检查容量不足时的直接计算或零工作区逐序路径，并核验工作区两端 guards。混合 batch 专项和流内标量更新用于检查每次调用的数据与标量独立性。

## 本轮精度验证记录

环境为 Ascend910B3、CANN 9.1.0、Netlib BLAS 3.10.0。恢复参考序路径、数据统计和固定工作区选路后的计算实现完成以下回归；Golden 和容差保持原样。

| 范围 | 通过 / 总数 | 外置证据 |
|---|---:|---|
| 主 CSV | 2393 / 2393 | `full-precision-v5/main-*.xml` |
| 固定标量 | 144 / 144 | `full-precision-v5/fixed.xml` |
| 随机标量 | 72 / 72 | `full-precision-v5/random.xml` |
| 特殊值 | 432 / 432 | `full-precision-v5/special.xml` |
| 专项 | 8 / 8 | `full-precision-v5/contracts.xml` |

另有四份 CSV 各两条契约测试通过，合计 3057 项。汇总见 `full-precision-v5/summary.json`。最终源码重新构建的计算库与上述全量验证库的 SHA-256 一致。CI 报告的十条 Gaussian 失败用例均在主 CSV 的通过记录中。增强后的混合 batch 专项另见 `v6-mixed.xml`。最终构建摘要见 `final-validation-summary.json`。

验证材料保存在源码仓之外的任务证据目录，未将 XML、日志或采样数据提交到产品仓库。

## 内存检查

使用独立构建目录，保留普通构建的 Netlib 配置，并增加 `-DCMAKE_ASC_FLAGS="-g --cce-enable-sanitizer"`：

```bash
python3 test/gemm_batched_ex/arch22/run_memcheck.py \
  --binary <插桩测试程序绝对路径> --output <新的内存检查输出目录>
```

脚本保留参数错误用例，按 shape、转置、标量位置、beta、ldc 和 batch 选择代表用例，并覆盖全部 432 条特殊值用例。它核对实际 Gemm 内核插桩的开始/结束记录、诊断、XML 和数值结果。只有全部匹配且没有诊断时才判定通过。

本轮五类代表路径执行了七个测试，数值检查通过；工具记录九次内核启动、八次无错误结束，另有 FFTS_BASE_ADDR 寄存器未恢复告警和六条工作区越界告警，内存检查尚未通过。完整参数错误与特殊值内存检查门禁尚未完成。原始记录为 `memcheck-v6-wait-first.xml`、`memcheck-v6-wait-first-detail.log`。退出清理实验触发同步错误，保留在外置实验目录，最终源码采用已通过完整精度验证的实现。

## 性能

正式采集设置 `GEMM_PHASE=profile`：每条用例只调用一次接口，使用工具的 5 次预热，并连续采集 10 个有效样本。每个样本累加接口的完整 kernel 序列，包含精度判定、输入整理、计算与写回，MIX 根任务不重复累计子任务。10 个样本的算术平均为 Avg time。与任务附件 GPU 基线相比，`GPU / NPU >= 0.8`。

`GEMM_PHASE=performance` 是保留的 85 次扩展计时模式，不用于上述单次接口验收。验证日志、XML、profiler 数据和性能表保存在源码仓之外；结果应绑定实际构建的提交或源码摘要。
