# aclblasCher2k（Ascend 950PR）测试说明

## 1. 环境要求

| 项 | 要求 |
|---|---|
| 硬件 | Ascend 950PR（arch35 / DAV_3510） |
| CANN | asc-devkit >= 9.1（`ASC_DEVKIT_MAJOR >= 9 && ASC_DEVKIT_MINOR >= 1`，低于该版本编译与运行自动跳过本算子） |
| 工具 | cmake >= 3.16、bisheng/ccec（随 CANN 工具链安装） |

## 2. 文件结构

```
test/her2k/cher2k/
├── README.md                     # 本文档（arch35 侧测试说明）
├── CMakeLists.txt                # ops_blas_add_gtest_tests(${OPS_BLAS} cher2k_test) 挂载（arch22/arch35 双 SOC）
├── cher2k_golden.h               # CPU golden：double 主判据（升精度 cblas_zher2k）+ float 交叉校验 + 参数镜像校验（arch22/arch35 共用）
├── cher2k_param.h                # CSV 列解析（alpha_real/alpha_imag/beta/nullA/B/C 等，arch22/arch35 共用）
├── arch22/                       # arch22（Atlas A2）侧测试，验证流程见 arch22/VALIDATION.md
│   ├── cher2k_npu_wrapper.h
│   ├── cher2k_scalar_regression.h
│   ├── cher2k_test.cpp           # CSV 驱动 GTest（精度 + 标量回归）
│   ├── cher2k_test.csv           # 默认 50 条
│   ├── cher2k_test_full.csv      # 全量 1214 条（CHER2K_FULL_REGRESSION=1）
│   ├── run_validation.sh
│   └── VALIDATION.md
└── arch35/                       # arch35（Ascend 950PR）侧测试（本文档所述）
    ├── cher2k_npu_wrapper.h      # Device 装载/同步/回读 wrapper
    ├── cher2k_perf.h             # TC_PF 性能分支（aclrtEvent 计时）
    ├── cher2k_test.cpp           # CSV 驱动 GTest（精度 + 性能 + Host 单测）
    └── cher2k_test.csv           # 主 CSV 1203 条（1000 精度 + 3 TC_WB K4 路径锁定 + 200 性能）
```

## 3. 编译

> 注意：本机/目标为 Ascend 950PR（arch35）时**必须显式指定 `--soc=ascend950`**（build.sh 默认 SOC 为 ascend910b3，会构建 arch22 侧测试而非本文档所述的 arch35 用例）；`--ops` 名为算子名 `cher2k`（不带 `_test` 后缀，CMake 经家族嵌套目录 `test/her2k/cher2k/` 定位）。

```bash
# 全仓编译（库 + 全部测试）
bash build.sh --soc=ascend950

# 只编译 cher2k 及其测试
bash build.sh --ops=cher2k --soc=ascend950
```

编译产物：`build/test/her2k/cher2k/cher2k_test`（GTest 二进制）；库安装于 `out/lib64/libops_blas.so`。

## 4. 运行测试

### 4.1 方式一：build.sh 直跑

```bash
bash build.sh --ops=cher2k --soc=ascend950 --run              # 设备 0
bash build.sh --ops=cher2k --soc=ascend950 --run --device=1   # 指定设备
```

### 4.2 方式二：随任务 verify 脚本（推荐，与自测报告口径一致）

脚本位于任务包 `test_cases/` 目录（CSV 已安装到本目录，脚本会自动对接）：

```bash
# 精度全量回归（自动排除 TC_PF 性能用例）
python verify_accuracy.py --repo /path/to/ops-blas --soc ascend950

# 只跑某类用例（如冒烟）
python verify_accuracy.py --repo /path/to/ops-blas --soc ascend950 --filter TC_L0

# 性能采集与对标（选取 TC_PF 用例，aclrtEvent 计时）
python verify_performance.py --repo /path/to/ops-blas --soc ascend950

# 跳过重新编译 / 指定设备
python verify_accuracy.py --repo /path/to/ops-blas --soc ascend950 --skip-build --device 1
```

### 4.3 方式三：直接执行 GTest（过滤用例）

```bash
./build_out/cher2k_test --gtest_filter='CsvDriven/0'            # 全部
./build_out/cher2k_test --gtest_filter='*TC_L0*'                # 冒烟 8 条
./build_out/cher2k_test --gtest_filter='-*TC_PF*'               # 排除性能
./build_out/cher2k_test --gtest_filter='*TC_PF_1001*'           # 单条性能 case
```

## 5. 用例说明与判定口径

### 5.1 用例清单（按任务书 §4 第 2 项：精度 case 与性能 case 分列）

**精度测试 case（1000 条，主 CSV `cher2k_test.csv`）**

| 前缀 | 条数 | 内容 |
|---|---|---|
| TC_L0 | 8 | 冒烟：uplo×trans 四组合 × n=4/8 |
| TC_SQ | 92 | 尺寸扫描 1→2048（含奇数/2^n±1/非对齐）× 4 组合 |
| TC_AB | 32 | alpha/beta 特殊值（0/1/负/复数/纯虚/大值） |
| TC_TK / TC_BK | 24 | k≪n 低秩（N）/ k≫n（C） |
| TC_LD | 6 | 前导维 padding |
| TC_FL | 6 | 填充模式（rand/zero/alter/extreme/Inf/NaN） |
| TC_CV | 32 | 中等尺寸覆盖 |
| TC_ED | 23 | 边界/负向：quick return、空指针、非法枚举、OP_T、非法 ld、负维度 |
| TC_EX | 777 | 扩展精度采样（尺寸池×组合×标量×pad） |
| **小计** | **1000** | 全部为精度用例，判定 golden=cblas_zher2k（double 主判据，float cblas_cher2k 链为交叉校验） |

**性能测试 case（200 条，主 CSV 中的 TC_PF 前缀）**

| 前缀 | 条数 | 内容 |
|---|---|---|
| TC_PF | 200 | 4 条任务书典型 case（UPPER/N/1024、UPPER/N/2048、LOWER/C/1024、LOWER/C/2048）+ 小尺寸 + 方阵扫描 + k 网格 + 低秩大 n + 预算内混合；判定门槛取自 `gpu_baseline.csv` 的 `gpu_ms / 0.4` |

**白盒锁定 case（3 条，主 CSV 中的 TC_WB 前缀，随 a5ecd9e 引入）**

| 前缀 | 条数 | 内容 |
|---|---|---|
| TC_WB | 3 | K4 SIMT 路径（n≤8）β=0 行为锁定：β=0 时算子跳过 C_old 读取直接写零，C 中预置的 NaN/Inf（VALUE_NORM_NAN / VALUE_NORM_INF）不得传播到输出 uplo 三角，对角虚部按置 0 口径比对；判定口径同精度用例 |

合计主 CSV 1203 条 = **精度 1000 条 + TC_WB 白盒锁定 3 条 + 性能 200 条**（固定随机种子，可复现）。

### 5.2 精度判定（golden = cblas_zher2k 双精度主链，仅比对 uplo 三角含对角）

- 实部、虚部分别按 FLOAT32 判定：`|actual - golden| ≤ atol + rtol × |golden|`
- rtol = atol = 2^-13（1.22e-4，框架 verify.h FLOAT32 档现行口径）、matched_ratio ≥ 0.99、max_abs_error ≤ 1e-2 或 32 ULP
- 对角虚部按强制置 0 口径参与比对；非 uplo 三角与 ldc padding 做 EXACT（逐位不变）校验
- 返回码断言：负向用例按参数校验顺序表逐条断言（INVALID_ENUM / INVALID_VALUE / HANDLE_IS_NULLPTR）

### 5.3 性能判定（aclrtEvent 事件计时，非 GTest host wall time）

- 先 warmup 5 次，再有效采样 60 次（>50）取平均；验收以事件平均值为准
- 门槛：NPU 平均单次耗时 ≤ `gpu_baseline.csv` 对应 `gpu_ms / 0.4`（= 任务书 §3.3 四 case 标杆）
- 四条标杆 case：UPPER/N/1024 ≤ 641.43us；UPPER/N/2048 ≤ 4157.94us；LOWER/C/1024 ≤ 680.47us；LOWER/C/2048 ≤ 4258.12us

## 6. 预期结果（2026-09-20 终态代码实测）

| 项 | 结果 |
|---|---|
| 精度·主表（随任务 1000 条精度 case） | **984/984 = 100% PASS**（余 16 条为无张量行的负向/quick-return，gtest OK 单列 SKIP-OK，不计入精度分母） |
| 性能·4 条任务书标杆 | **4/4 达标**：427.21 / 3140.06 / 426.11 / 3150.77 us（标杆 641.43 / 4157.94 / 680.47 / 4258.12，余量 +33.4% / +24.5% / +37.4% / +26.0%） |
| 性能·参考 200 条 | **200/200 PASS**（判定 `gpu_ms / npu_ms ≥ 0.4`；最紧余量 TC_PF_1093 = 0.4111） |
| 内存·Host 峰值 RSS | n=1024 ≈190 MiB；n=2048 ≈301 MiB（N/C 路径持平） |

- 产物：`cher2k_verify_results.csv`（逐 case 逐 tensor 精度明细，可用 `CHER2K_RESULT_DIR` 指定落盘目录）
  与 `cher2k_perf_results.csv`（逐 case 性能明细），二者即自测报告的数据源，可逐条复核
- 精度回归自动排除 `TC_PF` 前缀（性能用例不参与精度判定）

## 7. 常见问题

| 现象 | 处置 |
|---|---|
| 编译时 cher2k 被跳过 | 检查 asc-devkit 版本 ≥ 9.1（见 §1）；确认 `cmake/asc_devkit_version.cmake` 已注册 CHER2K |
| TC_FL NaN 用例 MERE 统计异常 | NaN 输入 golden 与 NPU 归约序差异属预期，框架按 nan-equal 跳过；必要时过滤该 case 并在报告说明 |
| 2048 档性能批间波动 | 属已知监控项（极差 ~4%，中位余量 7%+），复测建议 ≥3 批×5 轮取中位 |
