# Ex 变体算子模式说明

> 适用模板：`references/ex-variant/`
> 原型算子：`gemm_ex`、`gemm_batched_ex`、`gemm_grouped_batched_ex`、`scalex`

## 1. 什么是 Ex 变体

「Ex」(Extended) 变体是 ops-blas 仓中相对于**基础算子**（固定 dtype 的 `sxxx` 系列）的扩展接口，核心特征是**多数据类型 + 计算精度可配**：

| 特征 | 基础算子（如 sgemm） | Ex 变体（如 gemm_ex） |
|------|--------------------|---------------------|
| 数据类型 | 固定（s=FP32） | A/B/C 各自带 `aclDataType`（Atype/Btype/Ctype） |
| 计算精度 | 固定 | `aclblasComputeType_t computeType`（COMPUTE_16F / COMPUTE_32F） |
| alpha/beta | `float` | `const void*`，宿主类型由 computeType 决定 |
| dtype 组合 | 无校验 | 白名单校验，非法返回 `ACLBLAS_STATUS_NOT_SUPPORTED` |
| kernel 派发 | 单实例 | `DTypeCase` 枚举驱动多模板实例 |

## 2. 已知 Ex 变体形态

Ex 变体**不限于 GEMM 系列**，参数列表 / 计算模型差异极大：

| 形态 | 代表算子 | API 参数特征 | 计算模型 | 模板适用 |
|------|---------|-------------|---------|---------|
| 单矩阵 | `gemm_ex` | A/B/C 单指针 + m,n,k + algo | Cube BlockMmad | ✅ |
| 批量 | `gemm_batched_ex` | Aarray/Barray/Carray + batchCount | Cube BlockMmad | ✅ |
| 分组 | `gemm_grouped_batched_ex` | 逐组数组 + groupCount/groupSize | Cube + Epilogue | ✅ |
| 向量 | `scalex` | alpha + x + incx + executionType | Vector SIMD/SIMT | ✅ |

> **重要**：ex-variant 模板只固化「ex 共性骨架」（见下节），参数列表与计算逻辑用占位符标注，按算子 API 填充。**不照抄任一形态的具体参数签名**。

## 3. Ex 共性骨架（所有形态必备）

### 3.1 dtype 工具三件套

每个 ex 变体 host.cpp 必须包含这三个静态函数，构成 dtype 派发的 host 侧基础：

```
IsValidDtypeCombination(...)  → bool    // 白名单校验，非法返回 false
IsFP8Type(aclDataType)        → bool    // FP8 判定（E4M3/E5M2）
GetDtypeCase(...)             → DTypeCase  // aclDataType 组合 → 枚举
```

**形态差异**：参数个数随算子调整
- 矩阵类：`(Atype, Btype, Ctype, computeType)` 四入参
- 向量类：`(xType, alphaType, executionType)` 等按算子 API 调整

### 3.2 DTypeCase 枚举 + kernel 派发链路

```
host GetDtypeCase()  →  DTypeCase 枚举值
  ↓ 传入 kernel_do(..., DTypeCase)
kernel.cpp switch(dtypeCase)  →  各模板 kernel 实例
```

- 枚举值与 `GetDtypeCase()` 返回值、`kernel_do` switch case **三者一一对应**
- 矩阵类有「OUT_F32」中间态枚举（alpha/beta 后处理时 FP16/BF16 落 FP32 中间盘）
- 向量类无 OUT_F32，枚举按 `xType` 单值

### 3.3 alpha/beta 处理

| 形态 | alpha/beta 类型 | 处理方式 |
|------|----------------|---------|
| 矩阵类 | `const void*`，按 computeType 读 half/float | `alpha!=1‖beta!=0` 时 Cube 落中间态，Vector kernel 做 `C=alpha*temp+beta*C` |
| 向量类 | `const void*`，固定 float | 原地 `Muls`，无后处理 kernel |

### 3.4 Validate / Launch 拆分

与 simd-membase 一致，host.cpp 强制拆分：
- `Validate{Op}Params` — 含 dtype 组合校验
- `Launch{Op}Kernel` — tiling 计算 + dtype 派发 + kernel launch（异步）

## 4. 各形态实现要点

### 4.1 矩阵类（gemm_ex 系列）

**Tiling 字段**：矩阵维度 m/n/k + leading dim + 多核切分 mBlocks/nBlocks/singleCoreM/N + Cube tile baseM/N/K/c0Size + 转置标志 + alpha/beta + outputFp32

**Cube tile 随 dtype 变化**：
| dtype | baseM | baseN | baseK | c0Size | QuantMode |
|-------|-------|-------|-------|--------|-----------|
| FP16/BF16 | 128 | 128 | 16 | 16 | F322F16 / F322BF16 |
| FP32 | 32 | 16 | 8 | 8 | NoQuant |
| FP8 | 32 | 16 | 32 | 32 | F322F16 |

**列主序 swap**：把列主序 GEMM 转行序 `B^T * A^T`
```cpp
std::swap(tiling.m, tiling.n);
std::swap(tiling.lda, tiling.ldb);
std::swap(tiling.isTransA, tiling.isTransB);
std::swap(A指针, B指针);  // aDevicePtr=B, bDevicePtr=A
```

**Cube kernel 实例化**：用宏 `GEMM_CUBE_KERNEL(FUNC, A_TYPE, B_TYPE, C_GM_TYPE, BM, BK, BN, C0, QUANT)` 一次性生成所有变体，每个 DTypeCase 一个实例。arch35 限制：Cube kernel 必须为 standalone `__aicore__` 函数，禁止类模板方法（Mmad hang）。

**alpha/beta 后处理**：`template<typename TEMP_TYPE, typename C_TYPE, RoundMode> class AlphaBetaKernel`，按 (dtypeCase, useFP32Temp) 派发。列方向多核切分。

### 4.2 批量类（gemm_batched_ex）

在单矩阵基础上：
- API：`Aarray/Barray/Carray` 为**设备侧指针数组**
- Tiling 追加：`batchCount`、`totalTasks = batchCount * mBlocks * nBlocks`、`cElemSize`
- 后处理 workspace 布局：`[tempPtrArray | tempABData]`，host 构造 tempPtr 数组 H2D 拷贝
- 早退路径（k=0 / alpha=0）走 device kernel 而非 host memset（因 C 在 device）

### 4.3 分组类（gemm_grouped_batched_ex）

与单矩阵/批量差异最大：
- Tiling 不走 const 引用，改为 **GM 平铺结构**：`[TilingHeader | GroupData[]]`
  - host `aclrtMalloc` + `aclrtMemcpy` H2D 拷贝到 GM
  - kernel 从 GM 读 tiling
- 每组独立 m/n/k/lda/ldb/ldc/alpha/beta，逐组算 mBlocks/nBlocks/cubeTaskCount
- 两段 kernel：`cube_kernel_do`（Cube）+ `epilogue_kernel_do`（Vector，做 alpha/beta + 列主序回写）
- **唯一允许同步的形态**：因 tiling/workspace 由 host malloc，launch 后需 `aclrtSynchronizeStream` 确保完成再释放

### 4.4 向量类（scalex）

与矩阵类完全不同：
- 无 Cube，无列主序 swap，无 alpha/beta 后处理 kernel
- `template<typename XType> class {Op}AIV`，按 xType 实例化 float/half/bfloat16_t
- 双路径：`incx==1` 走 SIMD 连续（DataCopy+Muls+DataCopy），`incx!=1` 走 SIMT stride
- 混合精度：FP16/BF16 输入经 `Cast->FP32->Muls->Cast` 回原类型（midBuf 中转）
- alpha 可能是 Host 或 Device 指针，用 `aclrtPointerGetAttributes` 判定

## 5. 选型决策

```
新算子是否多 dtype + computeType？
├─ 否 → 用 simd-membase / simd-regbase / simt 模板
└─ 是 → 用 ex-variant 模板
    ├─ 矩阵乘法类（GEMM 变体）→ 参考 4.1/4.2/4.3
    └─ 向量类（scale 类变体）→ 参考 4.4
```

## 6. 原始参考算子

| 形态 | 参考算子 | 路径 |
|------|---------|------|
| 单矩阵 | gemm_ex | `blas/gemm/arch35/gemm_ex_*.cpp` |
| 批量 | gemm_batched_ex | `blas/gemm_batched_ex/arch35/` |
| 分组 | gemm_grouped_batched_ex | `blas/gemm_grouped_batched_ex/arch35/` |
| 向量 | scalex | `blas/scalex/arch35/` |
