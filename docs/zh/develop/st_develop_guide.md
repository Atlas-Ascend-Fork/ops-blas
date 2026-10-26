# 算子 ST 设计方案与代码编写指导

本文档描述 gbmv、stpttr、strttp 已落地的新一代 GTest + CSV 精度 ST 框架，并给出新算子接入时的设计与编码规范。

---

## 1 框架方案设计

### 1.1 核心设计思想

**CSV 定义用例参数 → GTest 参数化读取并驱动 → 统一 5 步测试流程**

| 层次 | 职责 | 由谁承担 |
|------|------|----------|
| 用例参数 | 维度、系数、步长、fill mode、期望返回码、精度阈值等 | `<op>_test.csv` |
| 框架公共能力 | CSV 解析、枚举解析、数组填充、精度验证、GTest 基类 | `test/frame/*.h`（9 个头文件） |
| 算子专属逻辑 | 参数结构体、CPU golden、NPU wrapper | `<op>_param.h` / `<op>_golden.h` / `<op>_npu_wrapper.h` |

设计原则：

1. **CSV 列名 = API 参数名**，列顺序与接口声明一致，枚举值使用 `ACLBLAS_` 完整前缀。
2. **数组参数用 `BlasDataFill` 枚举**（`index`/`random`/`zeros`/`ones`/`nullptr`/`sentinel`），无需在 CSV 中冗余描述数据生成逻辑。
3. **`_golden.h` 签名与 BLAS API 完全一致**，返回 `aclblasStatus_t`，包含完整参数校验。
4. **`_npu_wrapper.h` 封装全部 ACL 操作**（malloc/H2D/kernel/D2H/free），空指针透传，测试侧只需准备 host `std::vector`。
5. **`BlasTest<Param>` 模板基类**统一管理 ACL 生命周期，测试文件不复写 `SetUpTestSuite`。
6. **共享 `test_main.cpp`** 提供 `main()`，测试文件不自行定义。

### 1.2 5 步测试流程

```
makeBlasArray / makeBlasTriangular / makeBlasBanded / makeBlasStrided  (host 数据)
    → _npu (NPU 执行，内部 H2D→kernel→D2H)
    → 失败: EXPECT_EQ(ret, p.expectResult)
    → _cpu (CPU golden，同签名)
    → Verifier::verifyVector (精度比对)
```

### 1.3 CSV 用例表设计

每个算子一个 CSV，命名 `<op>_test.csv`，与 `<op>_test.cpp` 同名同目录，由 `ReplaceFileExtension2Csv(__FILE__)` 自动定位。

#### stpttr 示例

```csv
case_name,description,uplo,n,ap,a,lda,expect_result
TC_L0_06,n1_lower,ACLBLAS_LOWER,1,index,sentinel,1,ACLBLAS_STATUS_SUCCESS
TC_L0_01,handle_null,ACLBLAS_LOWER,5,nullptr,nullptr,5,ACLBLAS_STATUS_NOT_INITIALIZED
TC_L1_19,zeros_lower,ACLBLAS_LOWER,8,zeros,sentinel,8,ACLBLAS_STATUS_SUCCESS
```

#### gbmv 示例

```csv
case_name,description,trans,m,n,kl,ku,alpha,a,lda,x,incx,beta,y,incy,seed,expect_result
TC_L0_001,Small square banded matrix,ACLBLAS_OP_N,8,8,2,2,1.5,random,5,random,1,0.8,random,1,20260516,ACLBLAS_STATUS_SUCCESS
```

#### 通用列定义

| 列名 | 类型 | 说明 |
|------|------|------|
| case_name | string | 用例编号（GTest 参数名） |
| description | string | 语义描述（供 exotic fill / nullHandle 判断） |
| expect_result | string | 期望返回码，完整枚举名如 `ACLBLAS_STATUS_SUCCESS` |
| uplo | string | `ACLBLAS_UPPER` / `ACLBLAS_LOWER` / 数值 |
| trans | string | `ACLBLAS_OP_N` / `T` / `C` |
| m, n, kl, ku | int | 矩阵维度 |
| lda | int | leading dimension |
| alpha, beta | float | 标量系数 |
| incx, incy | int | 向量步长 |
| ap, a, x, y | BlasDataFill | 数组填充模式：`index`/`random`/`zeros`/`ones`/`nullptr`/`sentinel` |
| seed | uint32 | 随机种子（`random` fill 时使用） |

### 1.4 项目目录结构

```
ops-blas/
  test/
    frame/                          # ST 框架公共头文件（9 个，header-only）
      csv_loader.h                  # csv_map、ReadMap、GetCasesFromCsv、PrintCaseInfoString、枚举解析
      blas_test.h                   # BlasTest<Param> 模板基类（SetUpTestSuite/TearDownTestSuite）
      fill.h                        # BlasDataFill、makeBlasArray、makeBlasTriangular、makeBlasBanded、makeBlasStrided
      verify.h                      # Verifier 精度引擎
      types.h                       # VerifyConfig、PrecisionMode
      data.h                        # DataGenerator（旧式算子兼容）
      device.h                      # DeviceBuffer、allocAndCopyToDevice、adjustStridedBase（旧式算子兼容）
      dtype_utils.h                 # dtype 工具（bf16/fp8 转换等）
      npu_event_timer.h             # NPU 事件计时（性能用例）
      test_main.cpp                 # 共享 main() 入口
    tpttr/                          # 家族目录（family）
      stpttr/                       # 接口子目录（精度变体），param/golden 在本层
        stpttr_param.h              # StpttrParam（继承 BlasTestParamBase）
        stpttr_golden.h             # aclblasStpttr_cpu() golden
        arch35/
          stpttr_npu_wrapper.h      # aclblasStpttr_npu() NPU wrapper
          stpttr_test.cpp           # GTest 入口
          stpttr_test.csv           # CSV 用例表
    gbmv/                           # 同 tpttr：家族目录下按精度变体嵌套接口子目录
      sgbmv/
      cgbmv/
```

---

### 1.5 框架公共头文件说明

#### csv_loader.h

核心类型与函数：

- `csv_map` — `unordered_map<string, string>`
- `ReadMap(m, key, default)` — 安全查表
- `GetCasesFromCsv<T>(path)` — CSV → `vector<T>`，每行调用 `T(csv_map)`
- `PrintCaseInfoString<T>` — GTest 参数名生成
- `ReplaceFileExtension2Csv(__FILE__)` — 自动推导 CSV 路径
- `BlasTestParamBase` — param 基类（`caseName`, `description`, `expectResult`）
- 枚举解析：`parseStatus`, `parseFillMode`, `parseOpTrans`, `parseDiagType`, `parseSideMode`, `parseComputeType`
- 安全数值解析：`parseInt`, `parseFloat`, `parseDouble`, `parseUint`

**parseUint 安全说明**：`std::stoul` 返回 `unsigned long`（64 位系统为 uint64_t），`parseUint` 内部检查 `val > UINT32_MAX`，超出范围时返回默认值而非静默截断。CSV 中的 `random_seed` 等 uint32 字段应确保值在 `[0, 4294967295]` 范围内。

#### blas_test.h

```cpp
template <typename ParamType>
class BlasTest : public ::testing::TestWithParam<ParamType> {
protected:
    static void SetUpTestSuite() {
        // 全局单例：首次调用时初始化，后续调用复用
        // aclInit 允许 ACL_ERROR_REPEAT_INITIALIZE（多测试套件场景）
        // 所有 ACL 调用均有 ASSERT 检查，环境异常时明确报错
        // 通过 std::atexit 注册 cleanupAcl，进程退出时释放
    }
    static void TearDownTestSuite() { /* 空实现，由 atexit 统一清理 */ }
    static aclblasHandle_t handle_;
    static aclrtStream stream_;
};
```

设备 ID 由 cmake 宏 `TEST_DEVICE_ID` 控制（`build.sh --device=N`）。

**多测试套件支持**：同一测试文件中可包含多个 `TEST_F` 和 `TEST_P`，`BlasTest` 通过全局 handle 指针判断是否已初始化，避免重复 `aclInit/aclFinalize` 导致的崩溃。

#### fill.h

| 函数 | 适用场景 |
|------|----------|
| `makeBlasArray(size, fill, desc)` | 通用一维数组，size 为 int64_t（支持负值返回空），NULLPTR 返回空 |
| `makeBlasTriangular(n, upper, fill, desc)` | 打包三角矩阵，n*(n+1)/2 元素 |
| `makeBlasBanded(m,n,kl,ku,lda,fill,seed)` | 带状矩阵，按 kl/ku 填对角线区域 |
| `makeBlasStrided(count,inc,fill,seed)` | 步长向量 |

`BlasDataFill` 枚举：`INDEX`, `RANDOM`, `ZEROS`, `ONES`, `NULLPTR`, `SENTINEL`

**desc 优先逻辑**：`makeBlasArray` 和 `makeBlasTriangular` 先检查 `desc` 参数中的关键字（`large`/`neg`/`inf`/`nan`/`extr`），匹配时按特殊模式填充；未匹配时才走 `fill` 分支。这样避免先按 fill 填充再被 desc 覆盖的浪费。

---

### 1.6 GTest 参数化测试入口

标准模板：

```cpp
class StpttrArch35Test : public BlasTest<StpttrParam> { };

TEST_F(StpttrArch35Test, NullHandle) {
    aclblasStatus_t ret = aclblasStpttr_npu(nullptr, ACLBLAS_LOWER, 5, nullptr, nullptr, 5);
    EXPECT_EQ(ret, ACLBLAS_STATUS_NOT_INITIALIZED);
}

INSTANTIATE_TEST_SUITE_P(
    Stpttr, StpttrArch35Test,
    ::testing::ValuesIn(GetCasesFromCsv<StpttrParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<StpttrParam>);

TEST_P(StpttrArch35Test, CsvDriven) {
    const auto& p = GetParam();

    std::vector<float> apHost = makeBlasTriangular(p.n, p.uplo == ACLBLAS_UPPER, p.ap, p.description);
    std::vector<float> aHost  = makeBlasArray(static_cast<int64_t>(p.lda) * p.n, p.a);

    const float* apPtr = apHost.empty() ? nullptr : apHost.data();
    float*       aPtr  = aHost.empty()  ? nullptr : aHost.data();

    aclblasStatus_t ret = aclblasStpttr_npu(StpttrArch35Test::handle_, p.uplo, p.n, apPtr, aPtr, p.lda);
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult));
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS) return;

    std::vector<float> golden(aHost.size());
    aclblasStpttr_cpu(StpttrArch35Test::handle_, p.uplo, p.n, apHost.data(), golden.data(), p.lda);

    VerifyConfig cfg;
    cfg.mode = PrecisionMode::EXACT;
    EXPECT_TRUE(Verifier::verifyVector(aPtr, golden.data(), aHost.size(), 1, cfg, p.caseName));
}
```

**约定**：null handle 用 `TEST_F` 单独测不下 CSV；测试文件不写 `main()`。

---

### 1.7 构建与运行

```bash
source <cann>/set_env.sh
cd ops-blas
bash build.sh --ops=gbmv,stpttr --run              # 默认卡0
bash build.sh --ops=gbmv,stpttr --run --device=1   # 指定卡1
```

CMake：`ops_blas_add_gtest_tests` 自动发现 `arch35/<op>_test.cpp`，链接 `test/frame/test_main.cpp` 作为共享入口。

---

## 2 参考实现对照

| 项目 | gbmv | stpttr | strttp |
|------|------|--------|--------|
| param | `gbmv_param.h` | `stpttr_param.h` | `strttp_param.h` |
| CPU golden | `gbmv_golden.h` | `stpttr_golden.h` | `strttp_golden.h` |
| NPU wrapper | `gbmv_npu_wrapper.h` | `stpttr_npu_wrapper.h` | `strttp_npu_wrapper.h` |
| 测试入口 | `gbmv_test.cpp` | `stpttr_test.cpp` | `strttp_test.cpp` |
| CSV | `gbmv_test.csv` (51行) | `stpttr_test.csv` (55行) | `strttp_test.csv` (55行) |
| 指针模型 | device | device | device |
| 精度 | MIXED_TOLERANCE | EXACT | EXACT |
| 数组填充 | `makeBlasBanded`/`makeBlasStrided` | `makeBlasTriangular`/`makeBlasArray` | `makeBlasArray` |

---

## 3 新算子 ST 开发步骤

### 3.1 分析 API

阅读 `include/cann_ops_blas.h`，明确 API 签名、参数顺序、device 指针约定、推荐精度模式。

### 3.2 编写 `<op>_param.h`

继承 `BlasTestParamBase`，字段按 API 参数顺序排列，数组参数用 `BlasDataFill`。

**命名约定**：构造函数参数使用 `map` 而非 `m`，避免与成员变量 `m`（矩阵行数）冲突：

```cpp
struct GbmvParam : public BlasTestParamBase {
    int m = 0;
    int n = 0;
    // ...
    GbmvParam(const csv_map& map) : BlasTestParamBase(map) {
        m = parseInt(ReadMap(map, "m", "0"));
        n = parseInt(ReadMap(map, "n", "0"));
        // ...
    }
};
```

### 3.3 编写 `<op>_golden.h` + `<op>_npu_wrapper.h`

- `_golden.h`：签名与 BLAS API 一致，含参数校验和 CPU 计算
- `_npu_wrapper.h`：封装 ACL 操作，处理 nullptr 透传和 `n<=0` 情况

### 3.4 编写 CSV + `<op>_test.cpp`

- CSV 列名 = API 参数名，顺序与接口一致，枚举值写 `ACLBLAS_` 完整前缀
- 测试使用 `BlasTest<Param>` fixture，遵循 5 步流程
- null handle 用 `TEST_F` 单独测

### 3.5 添加 CMakeLists.txt

```cmake
ops_blas_add_gtest_tests(${OPS_BLAS} <op>_test)
```

### 3.6 构建验收

```bash
bash build.sh --ops=<op> --run
```

---

## 4 文件清单（新算子交付）

| 文件 | 必需 | 说明 |
|------|------|------|
| `test/<family>/<op>/<op>_param.h` | 是 | 参数结构体 |
| `test/<family>/<op>/<op>_golden.h` | 是 | CPU golden |
| `test/<family>/<op>/arch35/<op>_npu_wrapper.h` | 是 | NPU wrapper |
| `test/<family>/<op>/arch35/<op>_test.cpp` | 是 | GTest 入口 |
| `test/<family>/<op>/arch35/<op>_test.csv` | 是 | CSV 用例表 |
| `test/<family>/<op>/CMakeLists.txt` | 是 | `ops_blas_add_gtest_tests` |
| `blas/<family>/<op>/README.md` | 推荐 | 算子说明 |
