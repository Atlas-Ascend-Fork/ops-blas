# aclblasCtrmm 测试工程说明（Ascend 950PR / arch35）

本文档说明如何复现 `aclblasCtrmm`（complex64 三角矩阵-矩阵乘）的**精度自测**与**性能自测**。
测试工程以 CSV 驱动 C++ GTest，与随任务提供的用例文件及仓内 trmm 族结构一致。

## 1. 目录与文件

```text
test/trmm/ctrmm/
├── README.md                 # 本文件（测试步骤与复现说明）
├── CMakeLists.txt            # GTest 目标注册（随 test/ 目录统一构建）
├── ctrmm_param.h             # CSV 行 → 用例参数（列名对齐 strmm_param.h；复数标量拆 alpha_real/alpha_imag）
├── ctrmm_golden.h            # 参考实现封装：委托 Netlib BLAS 的 cblas_ctrmm 生成 golden
└── arch35/
    ├── ctrmm_test.cpp        # CSV 驱动精度用例（Ctrmm/CtrmmArch35Test.CsvDriven/<case>）
    ├── ctrmm_npu_wrapper.h   # 用例侧数据准备与结果回收（含 B/C 别名等专项场景）
    ├── ctrmm_test.csv        # 随任务提供的用例文件（精度 + TC_PF 性能用例同文件）
    ├── ctrmm_test_bench.cpp  # 性能采集（CtrmmBench/CtrmmBenchTest.DataTrack/<case>）
    └── ctrmm_test_bench.csv  # 性能用例列表（TC_PF_*）
```

> 用例文件由随任务的 `test_cases/` 提供；`verify_accuracy.py` / `verify_performance.py`
> 会在运行时把 `ctrmm_test.csv` 安装到 `test/trmm/ctrmm/arch35/` 下。

## 2. 环境要求

| 项 | 要求 |
| --- | --- |
| 硬件 | Ascend 950PR（SOC `ascend950`，arch35） |
| CANN | 9.1.0（`source /usr/local/Ascend/ascend-toolkit/set_env.sh`） |
| 其他 | 支持 C++17 的编译器、CMake、GTest（由仓内 `install_deps.sh` 统一准备） |

## 3. 编译

在 ops-blas 仓根目录执行：

```bash
bash install_deps.sh            # 首次：准备测试依赖（GTest / 参考 BLAS 等）
bash build.sh --soc=ascend950 --ops=ctrmm
```

产物与测试二进制：

```text
out/lib64/libops_blas.so                              # 算子库（需加入 LD_LIBRARY_PATH）
build/test/trmm/ctrmm/ctrmm_test                      # GTest 可执行文件
```

## 4. 精度自测（复现步骤）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd <repo>/build/test/trmm/ctrmm
./ctrmm_test --gtest_filter=-*TC_PF*        # 精度用例（排除性能用例）
```

判定口径（与任务书 §3.2 及仓内框架一致）：

- golden 由 **Netlib BLAS `cblas_ctrmm`** 生成（见 `ctrmm_golden.h`），输出矩阵 C **实部、虚部分别**按 FLOAT32 判定；
- 逐元素：`|actual − golden| ≤ atol + rtol × |golden|`，`rtol = 2^-10`、`atol = 2^-16`；
- 用例级：`matched_ratio ≥ 0.99` 且 `max_abs_error ≤ 1e-2 或 32·ULP`；
- 仓内框架另有 MERE/MARE 汇总门（`mere_threshold = 2^-13`、`mare_multiplier = 10`），CSV 中按列配置；
- 汇总输出：`[==========] N tests ... [  PASSED  ] N tests.`；任一条 FAILED 会给出用例名与
  实部/虚部对比样例（可用 `CTRMM_DUMP=1` 打印更多逐元素明细）。

对照官方脚本：`python verify_accuracy.py --repo <repo> --soc ascend950 --csv ./ctrmm_test.csv`

## 5. 性能自测（复现步骤）

性能用例（`TC_PF_*`）由 bench 套件采集：**先 warmup（10 次），再有效采样 50 次**，
采样默认**批量下发 + 单次同步**（相邻调用的发射开销被流水隐藏，报告设备侧执行时间）；
如需与"整条墙钟"口径对齐，设 `CTRMM_BENCH_STRICT=1` 改为逐次同步。
以 `aclrtEvent` 记录每次调用的算子区间耗时，输出平均与中位（单位 µs）：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd <repo>/build/test/trmm/ctrmm
CTRMM_BENCH=1 ./ctrmm_test --gtest_filter=*DataTrack*
```

输出形如（每个用例一行）：

```text
[BENCH] TC_PF_1001 m=256 n=256 avg_us=63.30 med_us=63.31 gpu_us=48.07 budget_us=120.18 ratio=0.759 PASS
```

`ratio = gpu_us / med_us`（`gpu_us` 来自随任务的 `gpu_baseline.csv` 的 `gpu_ms` 列，
`budget_us = 2.5 × gpu_us`）；`ratio ≥ 0.4` 为该用例达标。

可选环境开关（不影响默认行为）：

| 变量 | 作用 |
| --- | --- |
| `CTRMM_BENCH_BREAKDOWN=1` | 打印用例体分解（数据准备 / 数据搬运 / warmup / 采样循环） |
| `CTRMM_BENCH_STRICT=1` | 采样改为**逐次同步**（每次调用后立即同步再计时，含 host 发射开销；默认口径为批量下发 + 单次同步） |
| `CTRMM_FAST_FILL=0` | 关闭用例体的快速数据准备（用于对照数据准备开销） |
| `CTRMM_DUMP=1` | 精度用例失败时打印逐元素对比与错误坐标分布 |
| `CTRMM_DEBUG=1` | **诊断总闸**（算子侧）：启用相位计时（P0 分解表）与平面落盘；**会插入额外 stream 同步、扰乱异步语义，生产环境不得开启** |
| `CTRMM_DUMP_PLANES=<dir>` | Phase-A 平面按 `.bin` 落盘，供融合/非融合路径逐字节对比；**仅在 `CTRMM_DEBUG=1` 时生效** |
| `CTRMM_DUMP_CSV=<file>` | **测试专用**：导出 (actual, golden) 逐元素对（实/虚部并列，上限 4096 元素），供自测报告精度图表渲染 |

对照官方脚本：`python verify_performance.py --repo <repo> --soc ascend950`
（该脚本解析 GTest 的**整条用例整数毫秒**，其中包含数据准备与 50 次采样循环，脚本自注为
"保守上界"；逐用例的 kernel 级耗时建议用 `msprof` 采集，见下节。）

## 6. 内存自测

任务书 §3.4 不涉及内存指标。测试工程不额外申请 host 侧大块内存，用例数据由 CSV 参数按需分配后释放。

## 7. 补充工具：msprof 采集 kernel 级耗时

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd <repo>/build/test/trmm/ctrmm
CTRMM_BENCH=1 msprof --output=<out_dir> \
  --application="./ctrmm_test --gtest_filter=*DataTrack*TC_PF_1001"
```

结果位于 `<out_dir>/PROF_*/mindstudio_profiler_output/task_time_*.csv`，
按 `kernel_name` 分组求 `task_time(us)` 均值即可得到各相位 kernel 的耗时。

## 8. 用例覆盖

- **精度（随任务提供）**：24 组枚举（side × uplo × trans × diag）全组合、尺寸扫描（1 → 4096，
  含质数、2 的幂 ±1、非对齐值）、非方阵（fat/thin）、前导维 padding（+4/+8）、
  alpha 特殊值（(0,0)/(1,0)/(−1,0)/纯虚/大值）、填充模式（均匀/交替/极值/Inf/NaN）、
  边界与负向（零维、空指针、非法枚举、非法前导维、负维度）；
- **性能**：`TC_PF_*` 共 200 条，含任务书 §3.3 的 5 条典型用例（`TC_PF_1001`~`TC_PF_1005`）。

## 9. 常见问题

| 现象 | 处理 |
| --- | --- |
| 找不到 `ctrmm_test` | 先确认 `bash build.sh --soc=ascend950 --ops=ctrmm` 成功，且路径为 `build/test/trmm/ctrmm/` |
| 运行报找不到 `libops_blas.so` | `export LD_LIBRARY_PATH=<repo>/out/lib64:$LD_LIBRARY_PATH` |
| bench 用例全部 SKIP | 需设置 `CTRMM_BENCH=1` |
| 精度用例全部 SKIP | 未传 `--gtest_filter` 时 bench 与精度套件同时加载；请按第 4/5 节分别过滤 |
