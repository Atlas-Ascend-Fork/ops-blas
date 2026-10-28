# aclblasCtbsv 测试说明

> 适配硬件：Ascend 950PR（arch35）、Atlas A2/A3（arch22）
>
> CANN：9.1.0  
> 接口：`aclblasCtbsv`  
> golden：cblas（Netlib `cblas_ctbsv`）

本目录是 ops-blas 仓内 CSV + GTest 测试工程，覆盖任务配套的全部自测用例。验收人按下面步骤即可复现精度与性能结果。

## Ascend 950（arch35）

### 目录结构

```text
test/tbsv/ctbsv/
├── CMakeLists.txt
├── README.md                 # 本文件
├── ctbsv_param.h             # CSV 参数解析
├── ctbsv_golden.h            # cblas_ctbsv golden
└── arch35/
    ├── ctbsv_npu_wrapper.h   # NPU 调用封装
    ├── ctbsv_test.cpp        # 精度 / 负向 GTest
    ├── ctbsv_test.csv        # 1200 条用例（1000 精度 + 200 性能）
    └── ctbsv_bench.cpp       # 任务书 §3.3 四个典型 case 的独立计时
```

### 用例清单

CSV：`arch35/ctbsv_test.csv`，共 **1200** 条。

#### 精度测试 case（1000 条，排除 `TC_PF_`）

| 前缀 | 条数 | 场景 |
|------|------|------|
| TC_L0 | 12 | 小 shape 基础：n=8,k=2，uplo×trans×diag 全组合 |
| TC_SQ | 23 | 尺寸扫描 1→2048（含奇数 / 边界 / 非对齐） |
| TC_BD | 22 | 带宽 0/1/小值/半带/满带 |
| TC_DG | 12 | UNIT 对角不读 vs NON_UNIT |
| TC_LD | 6 | lda = k+1+padding |
| TC_INC | 12 | incx ∈ {±1,±2,±3} |
| TC_FL | 11 | 填充 / Inf/NaN（UNIT 对角不读） |
| TC_CV | 24 | 中等尺寸全枚举 |
| TC_ED | 13 | 零维、空指针、非法枚举(INVALID_ENUM)、非法维度 / 步长(INVALID_VALUE) |
| TC_EX | 865 | 扩展精度采样 |

#### 性能测试 case（200 条，`TC_PF_`）

CSV 中 `TC_PF_1001`～`TC_PF_1200` 为性能/内存用例。其中前 4 条与任务书 §3.3 完全一致，由独立二进制 `ctbsv_bench` 用 `aclrtEvent` 计时（warmup 20 + 采样 100）：

| Case | n | k | uplo | trans | diag | lda | incx | 标杆 Avg (us) |
|------|---|---|------|-------|------|-----|------|---------------|
| TC_PF_1001 | 512 | 8 | UPPER | N | NON_UNIT | 9 | 1 | 917.5 |
| TC_PF_1002 | 1024 | 16 | LOWER | N | NON_UNIT | 17 | 1 | 1831.11 |
| TC_PF_1003 | 2048 | 16 | UPPER | T | NON_UNIT | 17 | 1 | 3753.09 |
| TC_PF_1004 | 4096 | 32 | LOWER | C | UNIT | 33 | 1 | 4898.68 |

精度判定：工程侧实部/虚部分别走仓库 `verifyMereMareComplexFloat`（mere=2⁻¹³，mare×10）。双方均溢出（Inf/NaN/超大有限值）的分量按 special-value 对齐；大带宽病态回代若 MERE/MARE 仍超阈值，再用 `op(A)·x ≈ b` 残差兜底。本地复测精度 **1000/1000 通过**。

### 编译

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cd /path/to/ops-blas
bash build.sh --ops=ctbsv --soc=ascend950
```

产物：

- `build/test/tbsv/ctbsv/ctbsv_test`
- `build/test/tbsv/ctbsv/ctbsv_bench`（若已按 CMake 注册）

### 精度复现

```bash
export LD_LIBRARY_PATH=$PWD/build:/usr/local/Ascend/cann-9.1.0/x86_64-linux/lib64:$LD_LIBRARY_PATH

# 全量精度 1000 条（排除性能 TC_PF_）
./build/test/tbsv/ctbsv/ctbsv_test --gtest_filter='-*TC_PF*'

# 核心套件 135 条
./build/test/tbsv/ctbsv/ctbsv_test \
  --gtest_filter='*TC_L0*:*TC_SQ*:*TC_BD*:*TC_DG*:*TC_LD*:*TC_INC*:*TC_FL*:*TC_CV*:*TC_ED*'

# 扩展精度 865 条
./build/test/tbsv/ctbsv/ctbsv_test --gtest_filter='*TC_EX*'

# 也可使用任务包脚本
python3 /path/to/test_cases/verify_accuracy.py --repo $PWD --soc ascend950 --skip-build
```

### 性能复现

```bash
# 推荐：任务书 4 个典型 case，warmup 20 + 采样 100，单位 us
./build/test/tbsv/ctbsv/ctbsv_bench

# 或直接编译独立 bench
g++ -O2 -std=c++17 test/tbsv/ctbsv/arch35/ctbsv_bench.cpp \
  -Iinclude -I$ASCEND_HOME_PATH/include \
  -Lbuild -L$ASCEND_HOME_PATH/x86_64-linux/lib64 \
  -lops_blas -lascendcl -o /tmp/ctbsv_bench
/tmp/ctbsv_bench
```

判定：平均单次耗时 ≤ 任务书标杆即 PASS。

### 内存

不单独申请 workspace。单用例 Device 占用约为：

```text
A: lda * n * sizeof(aclblasComplex)
x: (1 + (n - 1) * |incx|) * sizeof(aclblasComplex)
```

最大典型 case（n=4096, k=32, lda=33, incx=1）：

```text
A = 33 * 4096 * 8 = 1,081,344 B
x = 4096 * 8      = 32,768 B
合计 ≈ 1.06 MiB
```

Host 侧单用例设计预算 ≤ 512 MiB。

## Atlas A2/A3（arch22）

A2/A3 与 arch35 共用本目录的 `ctbsv_param.h` 和 `ctbsv_golden.h`。参数结构支持可选的 `data_distribution` 字段，未提供时保持原有默认行为。CPU golden 用于成功求解的数值对比；负向用例直接核对 NPU 返回码与各架构 CSV 中的预期值，不调用 CPU golden。

### 目录结构

```text
test/tbsv/ctbsv/
├── CMakeLists.txt
├── README.md
├── ctbsv_param.h
├── ctbsv_golden.h
└── arch22/
    ├── ctbsv_npu_wrapper.h
    ├── ctbsv_test.cpp
    ├── ctbsv_test.csv
    ├── ctbsv_bench.cpp
    ├── ctbsv_bench_cases.h
    └── ctbsv_gpu_baseline.csv
```

### 覆盖范围

`ctbsv_test.csv` 保留任务附件 1200 条用例的全部原始字段和参数，并增加 `data_distribution` 字段。前 1000 条为精度、边界和参数校验用例，后 200 条 `TC_PF_` 用例用于性能和内存验证。另有 3 条独立 GTest，验证空句柄和两类 GM 兜底路径。

972 条有效随机精度用例按 `data_distribution` 分为 486 条 `UNIFORM` 和 486 条 `NORMAL`。均匀分布范围为 `[-5,5]`；正态分布使用 `μ=0、σ=0.1`，位于任务书要求的 `μ∈[-5,5]、σ∈[0.1,2]` 范围内。为避免 UNIT 三角系统在正常精度测试中因条件数失控而掩盖实现误差，482 条 UNIT 随机用例使用正态分布，另取 4 条 NON_UNIT 用例补足正态组；其余 486 条 NON_UNIT 用例使用均匀分布。特殊值、负向和零维用例沿用原始填充方式；200 条性能用例保持均匀分布，以匹配任务提供的 GPU 基线。

#### 精度与边界用例

| 前缀 | 数量 | 覆盖内容 |
|---|---:|---|
| `TC_L0` | 12 | `uplo × trans × diag` 全组合 |
| `TC_SQ` | 23 | `n=1..2048`，含质数、2 的幂及非对齐值 |
| `TC_BD` | 22 | `k=0/1`、小带宽、半带宽和满带宽 |
| `TC_DG` | 12 | `UNIT` 与 `NON_UNIT` 对角语义 |
| `TC_LD` | 6 | `lda` padding |
| `TC_INC` | 12 | `incx=±1/±2/±3` |
| `TC_FL` | 11 | 均匀、零值、交替、极值、Inf/NaN |
| `TC_CV` | 24 | 中等规格下的枚举组合覆盖 |
| `TC_ED` | 14 | 零维、空指针、非法枚举、非法维度和非法步长 |
| `TC_EX` | 864 | shape、带宽、枚举、步长和 padding 的确定性扩展采样 |

独立用例：

- `NullHandle`：空句柄返回码；
- `LargeNFallback`：`n=4097,k=0`；
- `LargeKFallback`：`n=2050,k=2049,UPPER,C,UNIT,incx=-2`。

#### 性能用例

200 条 `TC_PF_` 用例与 `ctbsv_gpu_baseline.csv` 按顺序一一对应。前五条为任务书指定的核心性能点：

`TC_PF_` 用例保留在 GTest 参数化注册中，便于按单条用例执行和使用 msprof 采集；GTest 路径校验接口返回与 kernel 执行完成，不重复执行精度判定。`ctbsv_bench` 按 20 次 warmup、100 次有效调用进行事件计时，用于全量筛查；最终 kernel 耗时按任务书要求使用 msprof `Task Duration(us)` 统计。

| case | n | k | uplo | trans | diag | GPU 基线（us） | NPU 达标上限（us） |
|---|---:|---:|---|---|---|---:|---:|
| `TC_PF_1001` | 256 | 8 | UPPER | N | NON_UNIT | 203.616 | 254.520 |
| `TC_PF_1002` | 512 | 32 | LOWER | N | NON_UNIT | 395.072 | 493.840 |
| `TC_PF_1003` | 1024 | 16 | UPPER | T | UNIT | 556.373 | 695.466 |
| `TC_PF_1004` | 2048 | 64 | LOWER | C | NON_UNIT | 1812.350 | 2265.438 |
| `TC_PF_1005` | 4096 | 128 | UPPER | N | UNIT | 1974.020 | 2467.525 |

复数实部、虚部分别按任务书规定的 FLOAT32 Mixed Tolerance 验证：

- `rtol = atol = 2^-13`；
- `matched_ratio >= 0.99`；
- `max_abs_error <= max(1e-2, 32 * ULP)`。

994 条常规精度/边界用例直接按任务书规定的 Mixed Tolerance 判定。6 条显式特殊值用例包含 `RANDOM_EXTREME`、Inf 和 NaN 输入，按特殊值传播语义对齐后，有限分量仍按 Mixed Tolerance 验证。

### 编译

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cd /path/to/ops-blas

export LINUX_INCLUDE_PATH="$ASCEND_HOME_PATH/include"
export EAGER_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64"

# Atlas A2（Ascend 910B3）
bash build.sh --ops=ctbsv --soc=ascend910b3

# Atlas A3（示例设备 Ascend910_9382；按实际设备选择 SOC）
bash build.sh --ops=ctbsv --soc=ascend910_9382
```

`ascend910b*` 与 `ascend910_93*` 均映射到 `arch22`，两种构建使用相同的测试目录。

主要产物：

```text
build/test/tbsv/ctbsv/ctbsv_test
build/test/tbsv/ctbsv/ctbsv_bench
```

### 精度复现

```bash
export LD_LIBRARY_PATH=$PWD/build:$PWD/out/lib64:$ASCEND_HOME_PATH/lib64:$LD_LIBRARY_PATH

# 1000 条精度/边界用例，另加 NullHandle 和两条 GM 兜底用例
./build/test/tbsv/ctbsv/ctbsv_test \
  --gtest_filter='-*TC_PF*'

# 核心覆盖集
./build/test/tbsv/ctbsv/ctbsv_test \
  --gtest_filter='*TC_L0*:*TC_SQ*:*TC_BD*:*TC_DG*:*TC_LD*:*TC_INC*:*TC_FL*:*TC_CV*:*TC_ED*'
```

### 性能复现

`ctbsv_bench` 为每次 kernel 调用准备独立的 x 副本，避免原地重复求解改变后续调用的工作量。程序执行 20 次 warmup，并对 100 次有效调用进行事件计时。

```bash
# 全部 200 条性能用例
./build/test/tbsv/ctbsv/ctbsv_bench

# 五个任务书基准点
./build/test/tbsv/ctbsv/ctbsv_bench --five

# 单个用例的 kernel profiling；替换 TC_PF_1001 可复现其他性能用例
msprof op \
  --application="./build/test/tbsv/ctbsv/ctbsv_test --gtest_filter=*TC_PF_1001*" \
  --output=./prof_ctbsv_1001
```

`OpBasicInfo.csv` 的 `Task Duration(us)` 为最终 kernel 平均耗时。判定条件为 `GPU 基线 / NPU 耗时 >= 0.8`。

A3 的 `n=1`、`n=2` 微型用例分别对应 `TC_PF_1006`、`TC_PF_1007`。这两条使用独立 AIC kernel。事件计时包含固定调度开销，最终性能判定使用 msprof 的 `Task Duration(us)`，有效采样大于 10 次取平均。

### 内存

算子不申请 workspace。单用例 Device 数据量为：

```text
A: lda * n * sizeof(aclblasComplex)
x: (1 + (n - 1) * abs(incx)) * sizeof(aclblasComplex)
```
