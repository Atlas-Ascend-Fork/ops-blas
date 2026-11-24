# aclblasCsyr2k 测试说明

本目录提供 CSV 驱动 GTest，覆盖 Ascend 950PR/950DT（arch35）与 Atlas A2/A3（arch22）两个平台。测试调用正式库中的 `aclblasCsyr2k`，并以 Netlib CBLAS 的 `cblas_csyr2k` 生成 CPU golden。`ACLBLAS_OP_C` 按无共轭的 `ACLBLAS_OP_T` 语义验证。

两个平台的测试源按架构目录隔离编译（见 `cmake/test.cmake` 的 `ops_blas_get_test_target_sources`）：`--soc=ascend950` 只编译 `arch35/`，`--soc=ascend910b3` / `ascend910b4` 只编译 `arch22/`，因此两侧的 wrapper 各自独立、行为一致。

## 目录结构

```text
csyr2k/
├── CMakeLists.txt
├── README.md
├── csyr2k_golden.h
├── csyr2k_param.h
├── arch22/
│   ├── csyr2k_npu_wrapper.h
│   ├── csyr2k_test.cpp
│   └── csyr2k_test.csv
└── arch35/
    ├── csyr2k_npu_wrapper.h
    ├── csyr2k_test.cpp
    ├── csyr2k_test.csv
    └── csyr2k_test_smoke.csv
```

## 用例与判定

- `arch35/csyr2k_test.csv` 为任务随附的官方 1200 条用例：1000 条功能/精度用例和 200 条性能用例。
- `arch35/csyr2k_test_smoke.csv` 保留完整用例文件的前 50 条数据，默认由 GTest 加载，以缩短冒烟流水线运行时间。设置 `CSYR2K_FULL_CSV=1` 可切回完整的 1200 条用例。
- 官方固定种子必须保持不变，其中 `TC_EX_0431` 为 `20260431`，`TC_EX_0539` 为 `20260539`。
- 指定三角的实部、虚部分别按 FLOAT32 标准判定：`rtol=2^-10`、`atol=2^-16`、匹配率不低于 99%，并满足最大绝对误差限制。
- 未指定三角与输入 C 进行逐位比较，确保其保持不变。
- 覆盖 `N/T/C`、`UPPER/LOWER`、复数 alpha/beta、leading-dimension padding、零维与 quick return、非法参数、Inf/NaN 和任务书性能尺寸。

## arch22 用例与判定

`arch22/csyr2k_test.csv` 为 Atlas A2/A3 的 79 条用例，覆盖 `N/T/C` × `UPPER/LOWER`、1..256 的方阵规模（含 127/128/129 的 Cube 瓦片边界）、`k != n` 的矩形、复数 alpha/beta、`alpha=0` 与 `k=0` 的退化路径、`alpha=0,beta=1` 的位精确 no-op、leading-dimension padding、`k` 越过内核编译期 singleK（8192）的分段累加，以及非法参数返回码。

除 CSV 外，`arch22/csyr2k_test.cpp` 还包含若干独立用例：空句柄、校验先于 `n==0` 快速返回、负数维度与非法 leading dimension、空矩阵指针、`n=0` 的 no-op、host 侧 alpha/beta（不经设备拷贝）、非指定三角（含 NaN/Inf）逐位保持不变。

## 编译与运行

先配置 CANN 9.1.0 环境，再使用仓库统一构建入口：

```bash
cd /path/to/ops-blas
source "${ASCEND_HOME_PATH}/set_env.sh"

# 编译正式 ops_blas 和 csyr2k_test
bash build.sh --ops=csyr2k --soc=ascend950

# 默认运行前 50 条 CSV 用例及独立测试
bash build.sh --ops=csyr2k --soc=ascend950 --run --device=0

# Atlas A2/A3（arch22）
bash build.sh --ops=csyr2k --soc=ascend910b3
bash build.sh --ops=csyr2k --soc=ascend910b3 --run --device=0

# 需要完整验收时显式运行全部 1200 条 CSV 用例
CSYR2K_FULL_CSV=1 bash build.sh --ops=csyr2k --soc=ascend950 --run --device=0

# Atlas A2/A3（arch22）
bash build.sh --ops=csyr2k --soc=ascend910b3
bash build.sh --ops=csyr2k --soc=ascend910b3 --run --device=0
```

如需筛选用例，可在完成构建后直接运行测试程序：

```bash
# 排除性能用例，执行功能与精度回归
CSYR2K_FULL_CSV=1 ASCEND_DEVICE_ID=0 build/test/syr2k/csyr2k/csyr2k_test \
  '--gtest_filter=*-Csyr2k/Csyr2kArch35Test.CsvDriven/TC_PF_*'

# 执行任务书三项性能用例；测试内部先 warmup，再采样 51 次
CSYR2K_RUN_PERF=1 ASCEND_DEVICE_ID=0 \
  build/test/syr2k/csyr2k/csyr2k_test \
  --gtest_filter=Csyr2kArch35Test.PerformanceRequiredCases \
  --gtest_color=no
```

性能验收必须使用 `-O3` 编译的 Ascend 设备代码；显式 `-DCMAKE_BUILD_TYPE=Debug`（`-O0`）的结果不可作为任务书性能结论。仓库未指定构建类型时默认 Release（#363 修复后口径，#831 同步修正本处过期表述）。
