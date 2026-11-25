# aclblasHgemm 测试说明（Ascend 950PR / arch35）

> **适配硬件**: Ascend 950PR（arch35）
> **CANN 版本**: CANN 9.1.0（asc-devkit >= 9.1）
> **数据类型**: FLOAT16（`aclblasHalf`，IEEE 754 binary16 位模式）

## 目录结构

```
test/gemm/hgemm/
├── CMakeLists.txt           # 注册 hgemm_test 目标
├── hgemm_param.h            # CSV 用例参数解析（任务书 §3.5 列格式）
├── hgemm_golden.h           # FLOAT32 参考（cblas_sgemm）+ §3.2 混合容差判定
├── run_self_test.py         # 全量精度回归：CSV 行数/执行数核对 + summary.json
├── run_memcheck.py          # mssanitizer memcheck 分组检查（契约/负向/成功路径）
└── arch35/
    ├── hgemm_npu_wrapper.h  # 设备内存搬运 + aclblasHgemm 调用封装
    ├── hgemm_test.cpp       # GTest：精度用例 + TC_PF 性能测量 + 专项断言
    └── hgemm_test.csv       # 1200 条 CSV 用例（1000 精度 + 200 性能）
```

## 环境准备

```bash
source /usr/local/Ascend/ascend-toolkit/latest/set_env.sh   # 或等价环境脚本
# 依赖：gtest、Netlib cblas（libblas / libopenblas）
```

## 编译

```bash
bash build.sh --soc=ascend950 --ops=hgemm
```

产物：`build/test/gemm/hgemm/hgemm_test`（CSV 由 CMake 自动拷贝到可执行文件目录，运行时从源码目录读取）。

## 精度验证（1000 条 CSV 用例 + 2 条专项断言，排除 TC_PF）

```bash
./build/test/gemm/hgemm/hgemm_test --gtest_filter='-*TC_PF*'
```

- **用例构成**：1000 条精度用例 = 980 条成功路径（golden 逐元素比对）+ 20 条负向校验
  （参数校验返回码断言）；另有 2 条专项 TEST_F（`CsvCompositionAssert` 用例数完整性断言、
  `ZeroingOrderedAfterInFlightGemm` stream 顺序回归），合计执行 1002 条

- **golden 口径**（任务书 §3.1/§3.2/§3.5）：输入量化为 FLOAT16 后以 FLOAT32（Netlib `cblas_sgemm`）计算
  `C = alpha*op(A)*op(B) + beta*C`，结果按 IEEE 754 就近舍入转回 FLOAT16，`ldc*n` 全缓冲逐元素比对
  （前导维填充行按"不被算子改写"断言）
- **判定标准**（FLOAT16 档）：rtol = atol = 2^-10、matched_ratio >= 0.99、max_abs_error <= max(1e-1, 32*ULP)；
  Inf/NaN 等效规则：双方同为 NaN、同号 Inf、或一方 NaN 另一方 ±Inf 判等效，非有限值不计入比率统计
- 覆盖：转置全组合（L0）、尺寸扫描 1→2048 含质数/2 幂 ±1（MN）、标量特殊值（AB）、前导维 padding（LD）、
  Inf/NaN 填充（FL）、中等覆盖（CV）、边界负向（ED：m/n=0 no-op、k=0、空指针、非法枚举、负维度、
  beta=0 含 NaN C 等）、扩展采样（EX）
- 预期：`[  PASSED  ] 1002 tests.`（CSV 用例数由 `CsvCompositionAssert` 硬断言兜底：总行数 1200、
  成功 980、负向 20、TC_PF 200、case_name 唯一）

## 自检脚本

```bash
python3 test/gemm/hgemm/run_self_test.py --output /tmp/hgemm_selftest
#   全量精度回归 + CSV 行数/执行数核对，输出 summary.json（accuracy_cases/max_abs_error 等）
python3 test/gemm/hgemm/run_memcheck.py --binary build/test/gemm/hgemm/hgemm_test --output /tmp/hgemm_memcheck
#   mssanitizer --tool=memcheck --leak-check=yes：
#   contracts（专项断言）/ csv（负向校验）各一个进程，smoke 的 84 条成功路径用例
#   （九种转置组合、LD padding、alpha=0/beta=0 退化分支、K 尾尺寸、alpha/beta workspace
#   中转）每用例独立进程执行——aclblas handle 的 workspace 为进程级共享，独立进程
#   保证 sanitizer 的堆块记录与单次 launch 对应，避免跨用例复用噪声；
#   要求三类 kernel 全部被插桩且零告警，输出 summary.json
```

## 性能验证（200 条 TC_PF）

```bash
./build/test/gemm/hgemm/hgemm_test --gtest_filter='*TC_PF*' | grep PERFORMANCE_SUMMARY
```

- 每条用例：设备端常驻缓冲，warmup 5 次后以 NPU event 有效采样 50 次取均值，输出
  `PERFORMANCE_SUMMARY_JSON={...,"avg_us":...}`（kernel 侧耗时）
- 对照基线：随任务提供的 `gpu_baseline.csv`（H100 锁频实测，前 4 条与任务书 §3.3 典型 case 一致）
- 判据：单 case 倍率（gpu_ms / npu_ms）>= 0.4；全部 TC_PF 平均倍率 >= 0.5（L3 算子要求）

## 官方口径采样（msprof）

任务书 §3.3 以 msprof 采集单算子 task duration 为准（warmup 5 + 有效采样 10 取均值）：

```bash
msprof --application="./build/test/gemm/hgemm/hgemm_test --gtest_filter=*TC_PF_1001*:*TC_PF_1002*:*TC_PF_1003*:*TC_PF_1004*" \
       --output=/tmp/msprof_hgemm
# 解析 mindstudio_profiler_output/op_summary_*.csv 的 Task Duration(us) 列（按 Task Start Time 排序，
# 每条用例 55 次下发，去掉前 5 次 warmup 取中位数）
```

## 用例 CSV 列格式

```
case_name,description,transA,transB,m,n,k,alpha,a_fill,lda,b_fill,ldb,beta,c_fill,ldc,
expect_result,mere_threshold,mare_multiplier,random_seed
```

- `alpha`/`beta` 写 `null` 表示空指针探测；`a_fill`/`b_fill`/`c_fill` 写 `NULLPTR` 表示设备指针为空
- `lda/ldb/ldc` 留空按最小合法值 `max(1, 物理行数)` 推导
- `transA/transB` 非法枚举写 `999`（透传到参数层触发 `ACLBLAS_STATUS_INVALID_ENUM`）
