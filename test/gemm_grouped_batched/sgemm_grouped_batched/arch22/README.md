# arch22 CSV / GTest 测试

本目录保留完整的 `sgemm_grouped_batched_test_full.csv`（1000 条精度、200 条性能）和补充 306 条的 `supplemental.csv`。默认读取的 `sgemm_grouped_batched_test.csv` 只含完整任务 CSV 的前 50 条，用于仓库 PreSmoke；完整文件及既有全量运行证据不受影响。显式设置 `GB_CASES_CSV` 为 `_full.csv` 路径时，GTest 注册全部非 PF 精度用例及 200 条 PF 性能用例；`GB_INCLUDE_PERF_ACCURACY=1` 额外将 PF 形状纳入精度测试。`GB_JSONL` 保存逐例 JSONL；`GB_DEVICE` 指定设备；`GB_MAX_BYTES` 限制测试 Device 分配。CI 冒烟通过不能代替完整精度与性能证据。

上游 `test/gemm_grouped_batched/arch22/` 的包装源接入本目录精度与性能 GTest。独立验证工程另提供 `gb_runner`、`gb_async_runner` 和 `gb_gtest`，在真实 NPU 模式联编时须使用明确的产品库路径，并确保 `GB_CPU_MOCK=OFF`。A2 / Ascend 910B3 已实际完成上游 GTest 构建与原始非性能 1000 条、补充 306 条验证；具体结果以对应提交的证据为准，不表示全部验收项完成。

缓冲区前缀保护区为 128 个 float，以保持 512 字节对齐；后缀为 8 个 float。CSV 显式 `offset_floats` 仍会施加偏移，整个前后保护区仍参与检查。性能测试使用原始 CSV 规模，5 次预热及 20 次有效采样；必须对一次 API 的全部设备 kernel 耗时求和。完整数值验证与性能测量分别判定，测出耗时不等于数值通过。

示例：
```bash
GB_CASES_CSV=/absolute/path/sgemm_grouped_batched_test_full.csv GB_INCLUDE_PERF_ACCURACY=1 \
 GB_JSONL=/absolute/path/results.jsonl ./gb_gtest \
 '--gtest_filter=TaskCSV/SgemmGroupedBatchedArch22.*' --gtest_output=xml:result.xml
```

结合 GTest XML 与 `GB_JSONL` 检查用例总数、失败、跳过和逐例结果；不能只根据进程退出码判断完整测试通过。精度及特殊值判定使用仓库测试框架的既有规则，NOT_RUN/SKIP 不得记录为 PASS。

## 评审回归与独立门禁

默认 50 条 CSV 继续按维护者要求保持原样。额外的 `review_regression.csv` 含 3 条 TN96 精度用例，通过 `Review/` 套件始终注册并执行（不依赖 `GB_CASES_CSV`）：512×96×16/TN、2048×96×32/TN 多 batch、TN96 与普通 group 混合调用，均由原有 `cblas_sgemm` golden 和前后保护区校验。`CsvCaseCount` 每次检查 default=50、full=1000+200、supplemental=306、review=3，防止文件被截断而静默通过。8×8 路径还增加输入/输出部分重叠的主机规划测试和 A/B 跨组依赖的设备数值回归。

在仓库根目录运行（输出目录必须尚不存在；指定实际构建的二进制和设备）：

```bash
python3 test/gemm_grouped_batched/sgemm_grouped_batched/run_self_test.py \
  --binary build/test/gemm_grouped_batched/gemm_grouped_batched_test --device 0 --output results/full_accuracy
python3 test/gemm_grouped_batched/sgemm_grouped_batched/run_memcheck.py \
  --binary build/test/gemm_grouped_batched/gemm_grouped_batched_test --device 0 --output results/memcheck
```

`run_self_test.py` 默认独立运行完整 1000 条精度与 306 条补充精度（每次同时包含强制 Review、契约及异步测试），逐一核对所有 CSV 的计数和 GTest 发现映射，并比较发现名单、实际执行 XML 与逐例 JSONL；失败、跳过、缺失、CPU mock 或零测试均退出非零。`--smoke` 显式缩短为 50 条；`--include-performance` 另启 200 条原始性能测时，此选项不替代 profiler/GPU 基线验收。输出完整命令日志、XML、JSONL、CSV/二进制 SHA-256、动态库搜索路径和 `self_test_summary.json`。

`run_memcheck.py` 使用 `mssanitizer --tool=memcheck` 执行完整 CSV 中的 TC_ED/TC_L0 队列、强制 TN96 和 8×8 依赖回归，保存工具原始日志与 `memcheck_summary.json`。必须实际发现并执行这些测试，且工具日志存在正数且配对的 `Start memcheck` / `No error detected` 记录；启动失败、插桩未生效或日志异常均不能标为 PASS。内存门禁需在安装 mssanitizer 的 A2 环境另行运行，普通 CI 不等于已做有效插桩。


