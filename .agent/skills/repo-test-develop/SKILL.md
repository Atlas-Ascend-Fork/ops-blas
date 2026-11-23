---
name: repo-test-develop
description: |
  仓库测试开发指导，介绍 ops-blas 测试框架的使用与测试代码的开发方法。
  为 BLAS 算子开发 GTest + CSV 驱动的精度 ST：param / golden / npu_wrapper / test.cpp / test.csv / CMake。
  触发：实现 golden、编写分级功能用例、补全白盒测试、执行精度测试，或改写旧式 TEST_F 为 CSV 参数化测试时加载。
  代码模板位于 references/ 目录，由调用方按需复制。
---

# BLAS 算子 ST 开发技能

ops-blas 精度 ST 采用 **GTest 参数化 + CSV 用例表** 驱动。开发新算子测试时，以 `references/test/{op}/` 下的模板为起点，复制到算子目录后按 API 填充。

## 测试交付件目录结构

一个算子的测试交付件位于 `test/<family>/<operator_name>/`，共 6 个文件（3 个芯片无关 + 3 个在 `arch35/`）：

```
test/<family>/<operator_name>/
├── <operator_name>_param.h          -- 参数结构体（与芯片无关）
├── <operator_name>_golden.h         -- CPU golden（与芯片无关），签名与 BLAS API 一致
├── CMakeLists.txt                   -- 测试注册
└── arch35/
    ├── <operator_name>_npu_wrapper.h  -- NPU wrapper（芯片相关 ACL 操作）
    ├── <operator_name>_test.cpp       -- GTest 入口（TEST_F 空 handle + TEST_P CSV 驱动）
    └── <operator_name>_test.csv       -- CSV 用例表，列名 = API 参数名
```

> **提醒**：性能测试用例、白盒测试用例等非 ST 交付件，应在提交 PR 前从测试目录中移除，避免混入正式交付。

## 代码模板

模板目录 `references/test/{op}/` 按上表布局组织，文件名以 `op_` 为占位前缀，内容用 `{{op}}` / `{{Op}}` 占位符与 `// TEMPLATE:` 注释标注需替换处：

| 交付文件 | 模板 |
|---------|------|
| `<op>_param.h` | [references/test/{op}/op_param.h](references/test/{op}/op_param.h) |
| `<op>_golden.h` | [references/test/{op}/op_golden.h](references/test/{op}/op_golden.h)（含 CBLAS / LAPACK 两变体） |
| `CMakeLists.txt` | [references/test/{op}/CMakeLists.txt](references/test/{op}/CMakeLists.txt) |
| `arch35/<op>_npu_wrapper.h` | [references/test/{op}/arch35/op_npu_wrapper.h](references/test/{op}/arch35/op_npu_wrapper.h) |
| `arch35/<op>_test.cpp` | [references/test/{op}/arch35/op_test.cpp](references/test/{op}/arch35/op_test.cpp) |
| `arch35/<op>_test.csv` | [references/test/{op}/arch35/op_test.csv](references/test/{op}/arch35/op_test.csv) |

## ST 框架头文件（`test/frame/`）

| 头文件 | 职责 |
|--------|------|
| `csv_loader.h` | `csv_map`、`ReadMap`、`GetCasesFromCsv`、`PrintCaseInfoString`、枚举解析、`BlasTestParamBase`、`isNullHandleCase`、`parseInt/parseFloat/parseDouble/parseUint` |
| `blas_test.h` | `BlasTest<ParamType>` 模板基类（含 SetUpTestSuite/TearDownTestSuite/handle_/stream_） |
| `fill.h` | `BlasFillMode` 结构体、`makeBlasArray`、`makeBlasTriangular`、`makeBlasBanded`、`makeBlasStrided`、`makeBlasMatrix` |
| `verify.h` | `Verifier` 精度比对类 |
| `types.h` | `VerifyConfig`、`PrecisionMode` |
| `cblas_compat.h`（`test/utils/`） | `aclblas` 枚举到 CBLAS/LAPACK 的映射（`ToCblasOp`/`ToCblasUplo`/`ToCblasDiag`）及 Fortran BLAS/LAPACK 函数声明 |

**依赖**：CMake 自动链接 `libblas`（OpenBLAS）和 `liblapack`，golden 可直接调用 `cblas_*` 和 Fortran 函数。

## 六步开发流程

1. **分析 API**：读 `include/cann_ops_blas.h` 与算子实现，确认签名与精度模式。NPU wrapper 封装全部 ACL 操作，测试侧只准备 host `std::vector`。
2. **写 Param**（`op_param.h`）：字段按 API 参数顺序，数组参数用 `BlasFillMode`（见下方命名规则）。
3. **写 cpu / npu**（`op_golden.h` / `op_npu_wrapper.h`）：golden 签名与 API 一致、保留校验、调 CBLAS/LAPACK；wrapper 每个 ACL 调用校验返回值、失败即清理。
4. **写 CSV + GTest**（`op_test.csv` / `op_test.cpp`）：CSV 分级用例（L0/L1/L2），GTest `TEST_F` 空 handle + `TEST_P` 5 步流程。
5. **CMake**（`CMakeLists.txt`）：`test/frame/test_main.cpp` 提供统一 main，勿自写 `main()`。
6. **构建验证**：
   ```bash
   source <CANN>/set_env.sh
   cd ops-blas
   bash build.sh --ops=<op> --run              # 默认卡0
   bash build.sh --ops=<op> --run --device=1   # 指定卡1
   ```
   通过标准：`[  PASSED  ] N tests.`，Summary 中 `Failed: 0`。

## BlasFillMode 命名规则

`METHOD_PATTERN_VAL...`，从某位起可省略（取默认），不允许跳位。参数化 PATTERN（如 BANDED）先消耗结构参数，剩余 VAL 用于填充值。

| 位 | 可选值 | 说明 |
|----|--------|------|
| METHOD（必填） | `NULLPTR` / `INDEX` / `RANDOM` / `VALUE` | 值获取方式 |
| PATTERN | `NORM` / `UPPER` / `LOWER` / `DIAG` / `ALTER` / `EXTREME` / `ILLCOND` / `BANDED` | 矩阵形状/分布模式 |
| VAL... | 数值（`N`前缀=负，`P`可省略）或特殊标记 | 结构参数 + 填充值参数 |

常用写法：`INDEX`（1,2,3…）、`INDEX_ALTER`（正负交替）、`RANDOM_1_3`（[−1,3]）、`RANDOM_NORM_1E6`、`RANDOM_UPPER_0.5_2.0`、`RANDOM_BANDED_2_3_N5_5`（kl=2 ku=3 band 内 [−5,5]）、`VALUE_NORM_0/1`、`VALUE_NORM_N999`（哨兵）、`VALUE_NORM_INF/NAN`、`VALUE_DIAG_1`（单位阵）。

> **强制（代码检视必查 HIGH）**：CSV 中所有 `RANDOM` 必须显式指定值域范围（`RANDOM_lo_hi` 或 `RANDOM_PATTERN_lo_hi`），**禁止**裸 `RANDOM`（默认范围 [−FLT_MAX, FLT_MAX] 不可控）。无法确定时发问卷确认，默认 `RANDOM_NORM_1`。

## 精度模式选择

`VerifyConfig.mode` 必须**显式设置**，不得依赖默认值。

| 算子类型 | 推荐模式 | 配置方式 |
|----------|----------|----------|
| 格式转换 / pack-unpack | EXACT | `TEST_P` 内设 `cfg.mode = PrecisionMode::EXACT` |
| Level-1 向量 | ABS | `cfg.mode = ABS`；`cfg.absThreshold = 1e-5f` |
| Level-2/3 浮点、矩阵分解 | MERE_MARE | param 加 `mereThreshold`/`mareMultiplier` 字段，CSV 加 `mere_threshold`/`mare_multiplier` 列 |

## 常见问题

| 现象 | 处理 |
|------|------|
| CSV 读取失败 | 确认 CSV 与 .cpp 同名同目录，`ReplaceFileExtension2Csv(__FILE__)` 自动定位 |
| null handle 测试多余代码 | 改用 `TEST_F` 单独测，不下 CSV |
| 数组填充不匹配 | 检查 `BlasFillMode` 字符串，三角用 `makeBlasTriangular`，带状用 `makeBlasBanded` |
| 精度 fail | 看 Verifier 日志中的 MERE/MARE 或 exact mismatch 计数 |
| `gtest_main` 链接冲突 | 框架统一用 `test/frame/test_main.cpp`，勿自写 `main()` |
