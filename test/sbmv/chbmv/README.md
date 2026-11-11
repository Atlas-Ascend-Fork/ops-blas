# aclblasChbmv 测试说明

`aclblasChbmv`（单精度复数 Hermitian 带状矩阵-向量乘，`y = alpha * A * x + beta * y`）在 Ascend 950PR
（arch35 / DAV-3510）上的精度与性能测试。

## 目录结构

| 文件 | 说明 |
| --- | --- |
| `CMakeLists.txt` | 注册 `chbmv_test` 测试目标 |
| `chbmv_param.h` | CSV 行到用例参数的解析 |
| `chbmv_golden.h` | Netlib `cblas_chbmv` 参考实现封装 |
| `arch35/chbmv_npu_wrapper.h` | Host/Device 搬运与算子调用 |
| `arch35/chbmv_test.cpp` | CSV 驱动的 GTest 用例与专项 `TEST_F` |
| `arch35/chbmv_test.csv` | 任务配套用例：1000 条精度 + 200 条性能（`TC_PF_` 前缀） |
| `arch35/chbmv_test_supplement.csv` | 补充用例 18 条：`lda > k + 1` 的带宽前导维 padding（主 CSV 全为紧凑 `lda == k + 1`，无法暴露列步长误用 `k + 1`） |

## 编译

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
cd <ops-blas 仓库根目录>
bash build.sh --soc=ascend950 --ops=chbmv
```

产物：

- 算子库 `build/libops_blas.so`
- 测试程序 `build/test/sbmv/chbmv/chbmv_test`

## 精度测试

```bash
BLAS_DEVICE_ID=0 ./build/test/sbmv/chbmv/chbmv_test --gtest_filter='-*TC_PF*'
```

`chbmv_test_supplement.csv` 与主 CSV 同目录时会被自动追加到参数化用例集；主 CSV 保持与任务包
逐字节一致，补充用例单独成表，覆盖 `lda = k + 2 / k + 3 / k + 4`、正负步长与两种三角。
用例集完整性由 GTest 启动时的 CSV 契约校验保证（列名、1200/200 条数量、`case_name` 唯一），
契约不满足时直接以可读信息失败，不会静默跳过。

判定口径：golden 由 Netlib BLAS 3.10.0 `cblas_chbmv(CblasColMajor, ...)` 生成，实部、虚部分别按 FLOAT32
标准比较（`rtol = atol = 2^-13`、`matched_ratio >= 0.99`、`max_abs_error <= max(1e-2, 32*ULP)`）。
NaN/Inf 按任务书 §3.2 的等效规则判定：同为 NaN、同为同号 Inf、或一方 NaN 一方 ±Inf 视为等效，
有限与非有限不等效，非有限值不计入匹配率统计。

除 CSV 用例外，`chbmv_test.cpp` 另含专项回归：空句柄、对角虚部 Inf/NaN 毒值、正态分布与随机复数
标量、大值防溢出反例（`A = 1e38` 配 `alpha = 0.25`）、以及 `lda > k + 1` 的 padding 覆盖。

## 性能测试

性能以 `msprof op` 采集的 Kernel `Task Duration(us)` 为准。GTest 直接运行性能用例时使用
ACL Event 计时（先 5 次 warmup，再 55 次采样取平均），输出 `[PERF_EVENT] <case> <avg_us>`：

```bash
CHBMV_EVENT_TIMING=1 BLAS_DEVICE_ID=0   ./build/test/sbmv/chbmv/chbmv_test --gtest_filter='*TC_PF*'
```

需要提交 profiler 原始数据时，可对同一用例使用：

```bash
msprof op --application="./build/test/sbmv/chbmv/chbmv_test --gtest_filter=*TC_PF_1001*" \
  --aic-metrics=BasicInfo --warm-up=5 --launch-count=10 --output=./chbmv_prof
```

判定口径：任务书 §3.3 要求 `GPU 平均耗时 / NPU 平均耗时 >= 0.4`（即 NPU 不高于 GPU 的 2.5 倍）。
GTest 逐例打印 NPU 平均耗时，与提交包中的 GPU 基线表按 `case_name` 对齐后即可核算比值。
