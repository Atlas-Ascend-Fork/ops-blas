# Cgemm arch35 validation (issue #458 regression guard)

## Why golden comparison cannot guard this defect

`CgemmCombineKernel::ProcessTile` 的交织复数 C 路径按 `2 * count` 个 float 读写
`cQue_`/`outQue_`，而修复前两个队列仅按 `COMBINE_TILE * sizeof(float)`（1024 字节）
分配。`count = min(m - rowOffset, COMBINE_TILE)`，当 `m > 128`（即
`count > COMBINE_TILE / 2 = 128`，首个越界点为 m=129）时发生越界：

- 越界写落在同一 `TPipe` 中**相邻队列的缓冲区**内（不越出 UB 边界），
  数据仍被后续流程消费，golden 比对全部通过——
  `cgemm_test_full.csv` 中 317 条 m>128（最大 m=2048）用例在修复前同样全绿。
- 因此本缺陷只能靠**设备侧内存检查**锁定，不能靠金标回归。

## What to run

1. 构建（sanitizer 插桩）：

   ```bash
   bash build.sh --soc=ascend950 \
       -DCMAKE_ASC_FLAGS_RELEASE='-O2 -g --cce-enable-sanitizer'
   ```

   （仅 Arch35 存在 cgemm combine kernel；需要 950PR 设备执行。）

2. 金标回归（确认功能无损）：

   ```bash
   bash test/gemm/cgemm/arch35/run_validation.sh <blas_all_test> <report_dir> full
   ```

3. 设备内存检查（本缺陷的防回归手段，`mssanitizer` 需与目标设备配套）：

   ```bash
   bash test/gemm/cgemm/arch35/run_validation.sh <blas_all_test> <report_dir> memcheck
   ```

## Expected outcomes

| 构建 | memcheck 结果 | 说明 |
| --- | --- | --- |
| 修复前 | `invalid \_\_gm\_\_ write` / 越界访问 findings（count>128 的 tile 首次触发） | 证明缺陷存在 |
| 修复后（本 PR） | `No error detected.`，无 ERROR/WARNING findings | 证明 UB 已消除 |

判定口径与 `test/her2k/cher2k/arch22/run_validation.sh` 一致：测试进程退出码为 0
不代表检查通过，必须同时满足 memcheck-report.log 含 `No error detected.` 且无
ERROR/WARNING findings（含 `FFTS_BASE_ADDR` 之类的寄存器态告警，需复核后重跑）。

## Coverage notes

`full` 模式覆盖 cgemm_test_full.csv 全量用例（含 317 条 m>128）；`memcheck`
模式同样以 `*Cgemm*CsvDriven*` 过滤跑全量 CSV——越界从 m=129 起每个 tile 都会
触发，全量运行可确保覆盖首个越界点与最大规模（m=2048）场景。
