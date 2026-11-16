# aclblasCgeru 测试说明

本目录包含 `aclblasCgeru`（complex64 无共轭秩 1 更新，`A = alpha * x * y^T + A`，列主序）在
Atlas A2/A3（arch22）与 Ascend 950PR（arch35）上的功能、边界、精度与性能测试。两个架构目录
结构一致：`cgeru_test.cpp`（CSV 驱动 GTest + 专项用例）、`cgeru_perf.cpp`（性能采集）、
`cgeru_npu_wrapper.h`（NPU 调用包装）、`cgeru_test.csv`（1200 条用例）、`gpu_baseline.csv`
（200 条性能基线，`gpu_ms` 列对标 GPU A100）。

## 用例构成（arch22，任务书配套 1200 条 + 13 个专项）

| 类别 | 前缀 | 条数 | 说明 |
|------|------|------|------|
| L0 基础 | TC_L0 | 6 | 小尺寸基础步长组合 |
| L1 尺寸 | TC_SQ | 23 | 1 → 2048 方阵扫描（奇数/边界/非对齐） |
| L2 标量 | TC_AB | 8 | alpha 零/一/复数/大值/纯虚 |
| L3 矩形 | TC_RC | 12 | 宽/窄矩阵 |
| L4 步长 | TC_INC | 36 | incx×incy ∈ ±1/±2/±3 全组合 |
| L5 填充 | TC_FL | 18 | 全零/交替/极端值/Inf/NaN（含 `TC_FL_098` A 极端值、`TC_FL_101..` Inf/NaN） |
| L6 边界 | TC_ED | 16 | 零维 quick return、alpha=(0,0)、空指针、非法步长/lda |
| lda | TC_LD | 3 | lda padding 场景 |
| 扩展 | TC_EX | 878 | 尺寸×标量×步长×lda 确定性采样 |
| 性能 | TC_PF | 200 | 含 4 条任务书典型 case（512²/1024²/2048²/4096²） |

合计 1200 条（与任务书配套 CSV 逐行一致，无过滤）。专项 GTest（`CgeruArch22Test`）：空句柄、
quick return 不触碰设备指针、非法参数、无共轭符号约定、大负步长、零 `y` 列 Inf/NaN 保持
（Netlib 列门控）、退化 alpha（纯实/纯虚乘 Inf 必须按普通复数乘积传播 NaN）、非有限 alpha ×
零 `y` 列仍逐位保持原值（门控按 y 判零、与 alpha 无关）、混合列 chunk 零 y 列逐位保持、
`lda` 填充行不被改写、通用路径形状×步长扫描。

## 构建

```bash
# Atlas A2/A3（arch22）
bash build.sh --soc=ascend910b3 --ops=cgeru
# Ascend 950PR（arch35）
bash build.sh --soc=ascend950 --ops=cgeru
```

## 功能与精度测试

```bash
BLAS_DEVICE_ID=<设备号> ./build/test/ger/cgeru/cgeru_test
# 或仅跑精度类（排除性能 case）
BLAS_DEVICE_ID=4 ./build/test/ger/cgeru/cgeru_test --gtest_filter=*TC_L0*:*TC_SQ*:*TC_AB*:*TC_RC*:*TC_LD*:*TC_INC*:*TC_FL*:*TC_ED*:*TC_EX*
```

golden 由 `aclblasCgeru_cpu` 调用宿主 cblas（`cblas_cgeru`，Netlib BLAS 3.10.0 语义：
列门控 `y == (0,0)` 的列保持原值；`TEMP = ALPHA*Y(JY)` 按普通复数乘积）生成，实部/虚部
分别按 FLOAT32 比较。任务书口径：`|a-g| ≤ 2^-13 + 2^-13*|g|`，matched_ratio ≥ 0.99 且
max_abs_error ≤ 1e-2（或 32*ULP）。全部 1213 个测试（1200 CSV + 13 专项）通过即精度
验收通过。

## 性能测试

```bash
BLAS_DEVICE_ID=<设备号> ./build/test/ger/cgeru/cgeru_perf \
  ./test/ger/cgeru/arch22/gpu_baseline.csv
```

性能程序对每组用例预热 5 次，再以 ACL Event 测量 100 次异步调用的平均耗时（任务书 §3.3），
逐组输出 `gpu_us / npu_us / ratio / threshold_us`。验收判定：`NPU 耗时 ≤ gpu_ms/0.8`
（倍率阈值 0.8）。任务书 4 条典型 case 的门槛：512² ≤ 5.7µs、1024² ≤ 10.1µs、
2048² ≤ 55.71µs、4096² ≤ 219.35µs。

精确 kernel 耗时（不含 host 下发开销）用 msprof 采集，判读 `OpBasicInfo.csv` 的
`Task Duration(us)` 列：

```bash
msprof op --application="./build/test/ger/cgeru/cgeru_perf \
  ./test/ger/cgeru/arch22/gpu_baseline.csv" --output=./prof_cgeru
```

性能测试应在空闲设备上执行；多卡环境可用 `BLAS_DEVICE_ID` 选择空闲卡。
