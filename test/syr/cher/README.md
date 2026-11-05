# aclblasCher 自测说明

## 目录

| 路径 | 说明 |
|------|------|
| `cher_param.h` / `cher_golden.h` | CSV 参数解析与手写 CPU 参考实现（含负 incx） |
| `arch35/cher_npu_wrapper.h` | NPU 调用封装 |
| `arch35/cher_test.cpp` / `cher_test.csv` | GTest + 1200 用例表 |
| `run_self_test.py` | CSV 条数 + GTest 全绿自检（输出 `self_test_summary.json`） |
| `run_memcheck.py` | mssanitizer memcheck 门禁（`starts>0` 防假绿，输出 `memcheck_summary.json`） |
| `test_validation.py` | Host-only 单元测试，真实 import 上述双脚本（BL-34） |

## 环境

- 硬件：Ascend 950PR（soc=`ascend950`，arch35）
- CANN：与 ops-blas 主仓要求一致

## 编译与运行

在仓库根目录：

```bash
bash build.sh --ops=cher --soc=ascend950 --run --device=0
# 推荐显式 Release（与任务书性能门禁一致）
# CMAKE_OPTIONS 或 build.sh 内确保 -DCMAKE_BUILD_TYPE=Release
```

期望输出包含：

```text
[  PASSED  ] 120N tests.
```

（1200 条 CSV 参数化用例 + 固定契约用例：`NullHandle` / `CsvCaseCount` / Host·Device alpha 等）

## 双脚本门禁（合入检视口径）

```bash
# Host-only：CSV / memcheck 判定逻辑（不依赖 NPU）
python3 test/syr/cher/test_validation.py

# 用例数 + GTest 全绿自检（输出 self_test_summary.json）
python3 test/syr/cher/run_self_test.py --binary ./build/test/syr/cher/cher_test

# mssanitizer 内存安全（需安装 CANN mssanitizer；缺工具时退出码 3=SKIP）
python3 test/syr/cher/run_memcheck.py --binary ./build/test/syr/cher/cher_test
```

## 用例分类（CSV）

- 精度 / 功能：基础尺寸、uplo 组合、alpha 特殊值、步长、lda padding、填充模式、边界与负向（含 `n=0`、`alpha=0`、非法枚举、`INT_MIN` incx）、Inf/NaN
- 性能：CSV 中 `TC_PF_*` 前缀用例（与精度同批 ST 跑通；正式 perf 采样请按任务书 msprof op 流程）

## 精度标准

实部 / 虚部分别按 ops-blas / opbase **mixed tolerance**（matched_ratio + max_abs_error）校验；对角虚部须接近 0；非 uplo / padding 区域保持不变（NaN-aware）。
