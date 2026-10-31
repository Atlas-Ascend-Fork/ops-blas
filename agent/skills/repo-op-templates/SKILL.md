---
name: repo-op-templates
description: 算子代码模板库，提供代码模板与模板选择规则，作为算子代码开发的起点。
---

# Skill: blas-op-templates

ops-blas 算子代码模板库，为不同编程模型和目标架构提供标准化的代码骨架。Agent 在开发新算子时，应以对应类型的模板为起点，按算子需求填充业务逻辑。

---

## 算子分类体系

ops-blas 仓中的算子先按 **是否 Ex 变体** 分流，再按 **编程模型** 和 **目标架构** 两个维度分类：

- **基础算子**（固定 dtype 的 `sxxx` 系列）：按下方编程模型选模板。
- **Ex 变体**（多 dtype + 计算精度可配，如 `gemm_ex` / `gemm_batched_ex` / `scalex`）：一律以 `references/ex-variant/` 为起点，详见 [references/ex-variant/README.md](references/ex-variant/README.md)。

### 编程模型（基础算子）

| 编程模型 | 关键特征 | 适用场景 | 模板目录 |
|---------|---------|---------|---------|
| **SIMD membase** | `TPipe` / `TQue` / `DataCopyPad` / `SetFlag/WaitFlag<HardEvent::...>` | 数据搬运为主、Vector 计算为辅的算子（Level-1 BLAS 等） | `references/simd-membase/` |
| **SIMD regbase** | `__VEC_SCOPE__` / `RegTensor` / `MicroAPI::DataCopy/Mul/Add/ReduceSum` | 寄存器级 SIMD 算子，在 UB 内使用寄存器张量计算 | `references/simd-regbase/` |
| **SIMT** | `__simt_vf__` / `asc_vf_call` / `threadIdx.x` / `blockDim.x` | 线程级并行算子（Level-2/3 BLAS 等，仅 arch35） | `references/simt/` |

> **编程模型选择**：算子开发模板选择时，应优先评估 SIMD 方案（SIMD membase / SIMD regbase）的可行性。仅在经评估后当前目标架构或算子数据依赖模式确实无法使用 SIMD 表达时，才允许退化为 SIMT 方案。

### 目标架构

| 架构 | SOC_VERSION | NPU_ARCH | 说明 |
|------|------------|----------|------|
| arch20 | ascend310p* | dav-1101 | 推理芯片 |
| arch22 | ascend910b* / ascend910_93* | dav-2201 | 训练/推理芯片 |
| **arch35** | **ascend950*** | **dav-3510** | **Atlas A5 系列（当前重点）** |

> **架构默认反选**：目标 arch 目录默认由 SOC 参数反选（ascend950 → arch35，ascend910b/910_93 → arch22，ascend310p → arch20），优先使用该默认值。

---

## 模板目录结构

每个编程模型的模板文件按仓库实际目录层级组织，从 `blas/` 一级开始，仅包含算子实现代码（**2 层**，无 `{family}/` 中间层）：

```
references/<programming-model>/
  blas/{op}/
    archxx/
      op_tiling_data.h       -- Tiling 数据结构（host/kernel 共享）
      op_kernel.h            -- kernel_do 签名（host.cpp / kernel.cpp 共用）
      op_kernel.cpp          -- Device 侧 kernel 实现
      op_host.cpp            -- Host 侧 API 入口 + Validate/Launch 拆分 + dlog 集成
```

说明：
- `simd-membase` 模板适用于 arch22 和 arch35，使用 `archxx/` 作为通用目录名
- `simd-regbase` 模板仅适用于 arch35（DAV_3510 RegBase 模式），使用 `arch35/` 目录名
- `simt` 模板仅适用于 arch35（arch22 不支持 SIMT 编程模型）
- `{op}` 为 snake_case 格式的算子名（如 `sswap`、`sgeqrf_batched`）

---

## 算子交付件目录结构

本 skill 覆盖算子的 **`blas/` 代码交付件**（下方文件布局）。一个完整算子在仓库中的代码布局：

```
blas/<operator_name>/
  README.md            -- 算子使用文档（交付件，见下方说明）
  arch35/
    <operator_name>_host.cpp
    <operator_name>_kernel.cpp
    <operator_name>_kernel.h
    <operator_name>_tiling_data.h
```

- **README.md 的编写模板不在本 skill**：算子 README 是面向用户的**交付件**，按仓内指引 `docs/zh/develop/readme_develop_guide.md` 编写（含占位符约定、RAII 调用示例、产品支持情况等）。它与 AGENT 工作产生的中间文档本质不同，勿混用。
- 本 skill 只覆盖 `blas/` 代码交付件，不含算子测试代码。

---

## 命名规范

**重要**：算子目录和文件名使用 **snake_case**（下划线分隔），API 名和结构体名使用 **PascalCase**。

| 元素 | 规范 | 简单示例 | 复合词示例（如 _batched） |
|------|------|---------|-------------------------|
| 算子目录 | `blas/<operator_name>/`（snake_case） | `blas/sswap/` | `blas/sgeqrf_batched/` |
| Kernel 文件 | `<operator_name>_kernel.cpp` | `sswap_kernel.cpp` | `sgeqrf_batched_kernel.cpp` |
| Kernel 头文件 | `<operator_name>_kernel.h` | `sswap_kernel.h` | `sgeqrf_batched_kernel.h` |
| Host 文件 | `<operator_name>_host.cpp` | `sswap_host.cpp` | `sgeqrf_batched_host.cpp` |
| Tiling 头文件 | `<operator_name>_tiling_data.h` | `sswap_tiling_data.h` | `sgeqrf_batched_tiling_data.h` |
| Tiling 结构体 | `<OpName>TilingData`（PascalCase） | `SswapTilingData` | `SgeqrfBatchedTilingData` |
| Kernel 类 | `<OpName>AIV` 或 `<OpName>Kernel<T>`（SIMD） | `SswapAIV` | `SgeqrfBatchedAIV` |
| SIMT 计算函数 | `<OpName>SimtCompute` + `__simt_vf__` | `SspmvSimtCompute` | `SgemvBatchedSimtCompute` |
| Kernel 入口 | `<operator_name>_kernel`（`__global__`） | `sswap_kernel` | `sgeqrf_batched_kernel` |
| Kernel 启动器 | `<operator_name>_kernel_do` | `sswap_kernel_do` | `sgeqrf_batched_kernel_do` |
| 公共 API | `aclblas<OpName>`（PascalCase） | `aclblasSswap` | `aclblasSgeqrfBatched` |
| 测试 fixture | `<OpName>Arch35Test` | `SswapArch35Test` | `SgeqrfBatchedArch35Test` |

**命名规则说明**：
- `<operator_name>`：snake_case 格式，用于目录名和文件名（如 `sswap`、`sgeqrf_batched`、`sgetrf_batched`）
- `<OpName>`：PascalCase 格式，用于结构体名、类名、API 名（如 `Sswap`、`SgeqrfBatched`、`SgetrfBatched`）

---

## host.cpp 强制规范

- **函数拆分**：host.cpp 必须拆分为两个静态函数：
  - `static aclblasStatus_t Validate{Op}Params(...)` — 参数校验（包含空指针、维度、lead dim 等）
  - `static aclblasStatus_t Launch{Op}Kernel(...)` — tiling 计算 + workspace 获取 + kernel launch
  - API 入口 `aclblas{OpName}(...)` 仅做 handle 校验 + 快速返回 + 调用上述两个函数
- **dlog 强制集成**：host.cpp 必须 `#include "log/log.h"`，按以下约定使用：
  - `OP_LOGE` — 参数校验失败、`aclrtMalloc`/`aclrtMemcpy` 失败、workspace 不足
  - `OP_LOGD` — tiling 数据详细字段
  - `OP_LOGI` — kernel launch 信息（block 数、core 数）
  - 禁止使用 `printf`、`std::cout`
- **kernel.h 共用头文件**：host.cpp 通过 `#include "{op}_kernel.h"` 引入 `kernel_do` 签名，**禁止**在 host.cpp 中写 extern 前向声明

---

## 使用方法

1. 根据算子的编程模型和目标架构，选择对应的模板目录
2. 将模板文件复制到算子目录，按命名规范重命名
3. 按模板中的 `// TEMPLATE:` 注释指引，替换占位逻辑为实际业务逻辑
4. 模板中的代码已包含标准框架（Init/Process、参数校验、Tiling 计算等），只需填充核心算法部分

---

## 参考资源

| 资源 | 路径 | 说明 |
|------|------|------|
| SIMD membase 模板 | [references/simd-membase/](references/simd-membase/) | 以 sswap 为原型抽取的 SIMD membase 算子全套模板（arch22/arch35 通用） |
| SIMD regbase 模板 | [references/simd-regbase/](references/simd-regbase/) | 以 gemv_batched 为原型抽取的 SIMD regbase 算子全套模板（仅 arch35） |
| SIMT 模板 | [references/simt/](references/simt/) | 以 sspmv/sgemv 为原型抽取的 SIMT 算子全套模板（仅 arch35） |
| Ex 变体模板 | [references/ex-variant/](references/ex-variant/) | 多 dtype + 计算精度可配的 Ex 算子共性骨架（dtype 派发三件套 / DTypeCase 枚举 / alpha·beta 处理），原型 gemm_ex/scalex 等；用法见同目录 README.md |

### 原始参考算子

> 注：以下算子是历史遗留的 3 层目录结构（`blas/{family}/{op}/`），仅供**算法思路**参考（Tiling 切分、计算逻辑）。**新算子必须按新工作流的 2 层目录 + kernel.h + host 拆分结构开发**，禁止复制其代码骨架或目录布局。

| 编程模型 | 参考算子 | 路径 |
|---------|---------|------|
| SIMD membase | sswap | `blas/swap/sswap/arch35/` |
| SIMD membase | sdot | `blas/dot/sdot/arch35/` |
| SIMD regbase | gemv_batched | `blas/gemv_batched/arch35/` |
| SIMT | sspmv | `blas/spmv/sspmv/arch35/` |
| SIMT | sgemv | `blas/gemv/sgemv/arch35/` |
| **SIMT** | **getri_batched** | **`blas/getri_batched/arch35/`**（新结构标杆算子，host.cpp/kernel.h/kernel.cpp/tiling_data.h 拆分完整） |
