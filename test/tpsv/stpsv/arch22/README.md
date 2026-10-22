# aclblasStpsv arch22 测试说明

三角 packed 方程组求解算子 `aclblasStpsv` 在 arch22（Atlas A2 / Atlas A3）上的自测说明。
验收人可按本文步骤完整复现测试结果。

## 1. 目录结构

```
test/tpsv/stpsv/
├── CMakeLists.txt                 测试编译配置（注册 GTest）
├── stpsv_param.h                  CSV 参数解析（ReadMap）
├── stpsv_golden.h                 CPU golden（cblas_stpsv）+ packed 索引 CPU 版
└── arch22/
    ├── stpsv_npu_wrapper.h        NPU 调用封装（H2D / 下发 / 同步 / D2H）
    ├── stpsv_test.cpp             GTest 主文件（CSV 驱动 + 负向用例 + 性能用例）
    ├── stpsv_test.csv             官方 CSV 驱动用例（1200 条，冻结不改）
    ├── stpsv_test_supplement.csv  补充 CSV 驱动用例（8 条，packed AP 填充 NaN / Inf）
    └── README.md                  本文件
```

> **用例表加载方式**：测试工程按 `__FILE__` 推导同名 CSV（`stpsv_test.csv`）作为主用例表，
> 并在同目录存在 `stpsv_test_supplement.csv` 时**追加**其中的用例，两者合并成一个参数化用例集。
> 这样官方用例表保持**逐字节不变**（便于与任务配套用例比对），补充用例单独成表。

## 2. 前置依赖

- CANN 9.1.0 已安装并 `source set_env.sh`。
- 参考 BLAS（golden 依赖 `cblas_stpsv`）：编译时需能在 `/usr/local/include`、`/usr/include`
  下找到 `cblas.h` 及其包含的 `openblas_config.h`，并能链接 `libblas`。
  - openEuler（A2）：`dnf install -y blas-devel lapack-devel openblas-devel`
  - Ubuntu（A3）：`apt-get install -y libopenblas-dev libblas-dev liblapack-dev`
  - 注意：仓库 CMake 只搜索 `/usr/local/include`、`/usr/include`，不认 aarch64 multiarch
    子目录 `/usr/include/aarch64-linux-gnu/openblas-pthread/`，需**逐个**软链全部头文件：

    ```bash
    sudo mkdir -p /usr/local/include /usr/local/lib
    for f in /usr/include/aarch64-linux-gnu/openblas-pthread/*.h; do
      sudo ln -sf "$f" "/usr/local/include/$(basename "$f")"
    done
    sudo ln -sf /usr/lib/aarch64-linux-gnu/libopenblas.so /usr/local/lib/libblas.so
    sudo ln -sf /usr/lib/aarch64-linux-gnu/liblapack.so  /usr/local/lib/liblapack.so
    ```

    只软链 `cblas.h` 会因其内部 `#include "openblas_config.h"` 找不到而编译失败。

## 3. 编译

```bash
cd /path/to/ops-blas
bash build.sh --soc=ascend910b3 --ops=stpsv     # Atlas A2
bash build.sh --soc=ascend910_93 --ops=stpsv    # Atlas A3（与 A2 共用 arch22 源码）
```

产物：`build/test/tpsv/stpsv/stpsv_test`。

## 4. 运行

> **必须在二进制所在目录执行**：用例 CSV 在运行期按二进制同目录查找，
> 否则会出现「无用例可执行」的假象。

```bash
cd build/test/tpsv/stpsv
./stpsv_test --gtest_color=no 2>&1 | tee stpsv_full.log
```

- 全量：1208 条 CSV 用例（官方 1200 + 补充 8）+ 3 条手工用例（`NullHandle`、
  `OrderAboveSupportedMaximum`、`PerfBenchmark`），共 1211 条。
- 只看补充的特殊值用例：`./stpsv_test --gtest_filter='*TC_SP*'`
- 只看功能与精度（排除性能用例）：`./stpsv_test --gtest_filter=-*TC_PF*`
- 只看性能用例：`./stpsv_test --gtest_filter=*TC_PF*`
- 只跑性能专项：`./stpsv_test --gtest_filter=TpsvTest.PerfBenchmark`
- 指定 NPU：`ASCEND_RT_VISIBLE_DEVICES=<id> ./stpsv_test`

## 5. CSV 列格式

```
case_name,description,uplo,trans,diag,n,incx,ap_fill,x_fill,random_seed,expect_result,mere_threshold,mare_multiplier
```

| 列 | 说明 |
| --- | --- |
| `case_name` | 用例名，前缀标识用例族（TC_L0 / TC_SQ / TC_INC / TC_FL / TC_CV / TC_ED / TC_EX / TC_PF） |
| `uplo` / `trans` / `diag` | 枚举，非法值用字面量 `INVALID` 表达 |
| `n` / `incx` | 阶数与步长 |
| `ap_fill` / `x_fill` | 填充模式；`NULLPTR` 表示空指针负向用例 |
| `expect_result` | 期望返回码；空表示期望 `ACLBLAS_STATUS_SUCCESS` |
| `mere_threshold` / `mare_multiplier` | 测试工程口径的平均 / 最大相对误差阈值 |

## 6. 判定口径

**精度**（任务书 §3.2，FLOAT32 生态精度标准）：

- `atol = 2^-16 ≈ 1.5259e-5`，`rtol = 2^-10 ≈ 9.7656e-4`
- 逐元素：`|actual - golden| ≤ atol + rtol × |golden|`
- 用例级：`matched_ratio ≥ 0.99` 且 `max_abs_error ≤ 1e-2`（或 32×ULP）
- golden 由 cblas（Netlib `stpsv`）生成，对输出 x 的 n 个元素全量逐元素比对。

**性能**（任务书 §3.3）：

- 先 warmup，再有效采样 > 50 次取平均（本性能用例 warmup=10、采样 60 次）。
- 判定：`NPU 平均单次耗时 ≤ 标杆耗时`（等价于倍率 = 标杆 / NPU ≥ 1.0）。
- 标杆耗时的口径：`标杆 = A100 实测耗时 / 0.8`（见 `test_cases/gpu_baseline.csv` 的 `gpu_ms`），
  即允许 NPU 比 A100 原始耗时慢至多 25%。**标杆值本身已含 0.8 因子**，
  故不可再对标杆除一次 0.8。

| case | n | uplo | trans | diag | incx | 标杆（us） |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 512 | LOWER | N | NON_UNIT | 1 | 432.71 |
| 2 | 1024 | UPPER | N | NON_UNIT | 1 | 941.23 |
| 3 | 2048 | LOWER | T | NON_UNIT | 1 | 1690.94 |
| 4 | 4096 | UPPER | C | NON_UNIT | 1 | 5185.20 |

## 7. 测试填充策略（求解类特有）

- `diag = NON_UNIT`：填充 AP 后对**被引用的对角元**加符号保持偏移量
  `boost = amp * max(2, n) + 1`（`amp` 为填充模式的幅值上界，`RANDOM_NORM_5_5`
  下 amp = 5，即 boost = 5n + 1），强于任务书 §7.4 的 `max(5, n)` 下界且语义一致，
  保证矩阵远离奇异；golden 使用同一偏移后的矩阵，两侧输入严格一致。
- `diag = UNIT`：对角位置不读取，golden 按单位对角计算；同时将整个三角按
  `min(1, 1/(amp*n))` 缩放，使隐式单位对角占优。
- 另一三角不读取，由三角性隐含。

## 8. 补充用例：packed AP 的 Inf / NaN（`TC_SP`）

任务书 §3.5.3 要求 AP **含 Inf / NaN 特殊值用例**。官方用例表对 x 已覆盖
`VALUE_NORM_INF` / `VALUE_NORM_NAN`，但 `ap_fill` 只用到
`RANDOM_ALTER / RANDOM_EXTREME / RANDOM_NORM_5_5 / NULLPTR`，而极端值生成器
`ExtremeGenerator`（`test/frame/fill.h`）也只含 `±FLT_MAX / FLT_MIN / denorm`，**不含 Inf / NaN**。
因此以补充用例表补齐 8 条（`VALUE_*` 直接产出常量 NaN / Inf）：

| 用例族 | AP 填充 | diag | 覆盖 |
| --- | --- | --- | --- |
| `TC_SP_1201 ~ 1204` | `VALUE_NORM_NAN` | `NON_UNIT` / `UNIT` | LOWER/UPPER × N/T/C，n = 16 / 31 / 33 / 64 |
| `TC_SP_1205 ~ 1208` | `VALUE_NORM_INF` | `UNIT` | LOWER/UPPER × N/T/C，n = 16 / 17 / 31 / 33 |

**判定口径**：AP 整体为 `NaN` 时两侧输出全为 `NaN`，与求和顺序无关，判定器对「两侧同为 NaN」
的元素按一致处理；AP 整体为 `Inf` 时必须配合 `diag = UNIT`，原因见下。

**已知口径差异（非算子缺陷）**：Netlib 参考实现 `stpsv` 的回代循环带
`if (x[j] != 0)` 的**零值跳步**。当 `diag` 为 `Inf` 时 `x[j] = b[j] / Inf = 0`，参考实现随即跳过
整列更新，结果整体退化为 ±0；而向量化 kernel 不做该分支，按 IEEE 语义继续按
`Inf × 0 = NaN` 累加，输出为 `NaN`。两者在**非有限输入**下必然分叉，但在任何有限输入上逐位一致。
这是参考实现为有限输入所做的优化，不是本算子的精度问题；因此 `Inf` 组用例统一采用
`diag = UNIT`（对角不参与除法，参考实现不触发零值跳步），此时两侧传播路径一致。

## 9. 返回值与判定顺序

`aclblasStpsv` 的返回值按**固定优先级**判定，靠前项命中即返回、不再继续：

| 优先级 | 判定条件 | 返回码 |
| --- | --- | --- |
| 1 | `handle == nullptr` | `ACLBLAS_STATUS_HANDLE_IS_NULLPTR` |
| 2 | `n == 0`（合法 quick-return，不读写 AP / x） | `ACLBLAS_STATUS_SUCCESS` |
| 3 | `n < 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 4 | `n > ACLBLAS_STPSV_MAX_N`（32768） | `ACLBLAS_STATUS_NOT_SUPPORTED` |
| 5 | `uplo` / `trans` / `diag` 为非法枚举 | `ACLBLAS_STATUS_INVALID_VALUE` |
| 6 | `incx == 0` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 7 | `AP == nullptr` 或 `x == nullptr` | `ACLBLAS_STATUS_INVALID_VALUE` |
| 8 | `(n-1)*\|incx\|` 超出 `uint32` 范围（x 索引跨度溢出，设备侧按 `uint32` 计算偏移） | `ACLBLAS_STATUS_NOT_SUPPORTED` |

**关键顺序**：`handle` 检查（优先级 1）位于 `n == 0` quick-return（优先级 2）**之前**。因此 `n == 0 且 handle == nullptr` 一律返回 `HANDLE_IS_NULLPTR`，与 arch35 路径及参考实现（cuBLAS / Netlib）行为一致；arch22 与 950（arch35）在该组合上不再出现返回值分歧。优先级 8 的跨度溢出判定在 host 侧提前拒绝，避免设备侧 `uint32` 偏移回绕导致的静默错误。
