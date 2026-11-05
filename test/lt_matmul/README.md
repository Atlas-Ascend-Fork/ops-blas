# aclblasLtMatmul 测试口径

测试工程使用仓库统一的 CSV 驱动 GTest 注册方式，目标名为 `blasLtMatmul_test`。
构建、设备选择和执行入口遵循[仓库 README](../../README.md)，测试实现位于 `arch22/`。

## 精度

| 用例集合 | 参数与范围 |
|---|---|
| `arch22/blasLtMatmul_test.csv` | 任务原始 1200 行：1000 条精度用例和 200 条性能用例；全部检查数值结果 |
| `arch22/blasLtMatmul_gap_cases.csv` | 36 条补充输入分布、标量与布局边界用例 |
| `SupplementalCases`、`RuntimeFixture` | 参数、workspace、guard、算法回退及 stream 等运行时边界 |

原任务 CSV 的 SHA256 为 `26eb13eccd2b1450fe98d4fcf4b620c7b9ac80e1a902f8449261f5e4b8cab233`；GPU 表的 SHA256 为 `a04368d02715c6f4f106c4700d1aba7904efc763bbc739677ac74cb44adf7f32`。
重复 shape 按各自 case ID 保留，原始参数、随机种子和精度阈值保持任务定义。

标准仓库测试使用已安装的 CBLAS 作为参考，运行日志打印实际参考库路径。
任务书验收使用预置的 Netlib BLAS 3.10.0：`LT_MATMUL_USE_TASK_NETLIB=ON`，
`LT_MATMUL_NETLIB_ROOT` 指定其安装前缀；该模式检查版本标识、实际加载库和 SHA256。
`LT_MATMUL_USE_TASK_NETLIB=OFF` 使用仓库 CBLAS，不下载依赖、不安装软件，也不跳过数值用例。

有限值逐元素采用 `abs(actual-reference) <= 2^-13 + 2^-13*abs(reference)`，
匹配比例不低于 99%；最大绝对误差还需满足任务的 `max(0.01, 32*ULP)` 上限。
非有限值按既定任务规则单独核对；零乘积、零维和退化复制检查精确位模式。
同时检查 D 的 padding、输入保护区、workspace guard、容量和 16B 对齐边界。

## 性能

性能用例为原 CSV 中的 200 条 `TC_PF_*`，GPU 数值基线为 `arch22/gpu_baseline.csv`。
使用 `msprof op` 采集完整公开 API 调用涉及的全部 kernel，预热 5 次后保留 60 个有效样本。
每次 API 的各 kernel 耗时先求和，再计算 60 次平均值；GTest 总运行时间不作为 kernel 耗时。

逐例判据为 `GPU_us / NPU_mean60_us >= 0.8`。前 10 个有效样本的均值仅作诊断。
计时须对应同一源码、构建产物、加载库和设备；缺样、调用失败或 kernel 不完整不计为达标。
GTest 的 `LT_MATMUL_BENCHMARK` 模式提供该固定预热与采样次数；数值校验、公开参数和输入分布相同。
