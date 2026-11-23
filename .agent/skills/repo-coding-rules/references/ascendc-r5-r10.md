# AscendC 编码规则 R5-R10

## R5: 圈复杂度 ≤ 20

函数的 Cyclomatic Complexity 不超过 20，超过需拆分子函数。

**错误示例：圈复杂度超标的长函数**

```cpp
// CCN = 1（基础）+ 1（if）+ 1（for）+ 1（if）+ ... = 23+
void ProcessMatrix(int m, int n, int k,
                   const float* A, const float* B, float* C)
{
    if (m > 0) {
        for (int i = 0; i < m; i++) {
            if (i % 2 == 0) {
                ProcessEvenRow(i, A, B, C, n, k);
            } else {
                ProcessOddRow(i, A, B, C, n, k);
            }
            // ... 还有 20 行条件判断
        }
    } else if (m == 0) {
        ClearZero(C, n, k);
    } else {
        HandleError();
    }
}
```

**正确示例：拆分成子函数**

```cpp
// CCN = 3 (m > 0, i%2==0, m == 0)
void ProcessMatrix(int m, int n, int k,
                   const float* A, const float* B, float* C)
{
    if (m > 0) {
        ProcessRows(0, m, A, B, C, n, k);
    } else {
        HandleMatrixZeroOrError(m, C, n, k);
    }
}

// CCN = 2 (i%2==0 分支)
void ProcessRows(int start, int end, ...) {
    for (int i = start; i < end; i++) {
        (i % 2 == 0) ? ProcessEvenRow(...) : ProcessOddRow(...);
    }
}
```

**计算方式**（按分支数量）：
- `if` / `else if`：每出现一次 +1
- `for` / `while`：每出现一次 +1
- `case`（switch 分支）：每出现一次 +1
- `&&` / `||`（逻辑运算）：每出现一次 +1

---

## R6: 嵌套深度 ≤ 5

最大嵌套层级不超过 5，超过需提取内层循环/分支为独立函数。

**错误示例：嵌套深度 7 层**

```cpp
void CopyBlocks(...) {
    for (int batch = 0; batch < numBatches; batch++) {                 // 1
        for (int block = 0; block < numBlocks; block++) {                // 2
            for (int tile = 0; tile < numTiles; tile++) {                // 3
                for (int row = 0; row < tileHeight; row++) {             // 4
                    for (int col = 0; col < tileWidth; col++) {          // 5
                        if (mask[batch][block][tile][row][col]) {        // 6
                            if (validElement(batch, block, tile, row, col)) {  // 7 ← 超标
                                dst[...] = src[...];
                            }
                        }
                    }
                }
            }
        }
    }
}
```

**正确示例：拆分为单层处理函数**

```cpp
void CopyBlocks(...) {
    for (int batch = 0; batch < numBatches; batch++) {
        for (int block = 0; block < numBlocks; block++) {
            CopyBlockBatch(batch, block, ...);
        }
    }
}

// 嵌套深度：3（tile + row + col）
void CopyBlockBatch(int batch, int block, ...) {
    for (int tile = 0; tile < numTiles; tile++) {
        CopyTile(batch, block, tile, ...);
    }
}

// 嵌套深度：4（row + col + mask + valid）
void CopyTile(int batch, int block, int tile, ...) {
    for (int row = 0; row < tileHeight; row++) {
        CopyTileRow(batch, block, tile, row, ...);
    }
}

// 嵌套深度：4（col + mask + validCheck + valid）
void CopyTileRow(int batch, int block, int tile, int row, ...) {
    for (int col = 0; col < tileWidth; col++) {
        if (mask[...]) {
            CopyIfValid(batch, block, tile, row, col, src, dst);
        }
    }
}
```

---

## R7: 函数行数 ≤ 50 (NBNC)

NBNC（Non-Blank Non-Comment lines，非空非注释行）不超过 50，超过需拆分为多个函数。

**统计方式**：
- 空行不算
- 纯注释行（以 `//` 或 `/* */` 开头）不算
- 混合行（代码 + 注释）只计代码部分
- 函数签名 + 函数体开闭括号 `{` / `}` 各算 1 行
- 模板函数 `template <...>` 不计行

**错误示例：NBNC = 65 行**

```cpp
template <typename T>
void ComputeGEMV(
    __gm__ T* a, __gm__ T* x, __gm__ T* y,
    const SgemvTilingData& tiling, int batchCount)
{
    __gm__ T* outPtr = y;
    __gm__ T* matrixPtr = a;
    __gm__ T* vectorPtr = x;
    int m = tiling.m;
    int n = tiling.n;
    T alpha = tiling.alpha;
    T beta = tiling.beta;
    // ... 60 行逻辑处理
    return;
}
```

**正确示例：按功能拆分**

```cpp
template <typename T>
void ComputeGEMV(
    __gm__ T* a, __gm__ T* x, __gm__ T* y,
    const SgemvTilingData& tiling, int batchCount)
{
    // NBNC = 15
    PreparePointers(a, x, y, tiling);
    for (int b = 0; b < batchCount; b++) {
        ComputeOneBatch(b);
    }
    WriteBackResults(y);
}
```

---

## R8: 除零防御

除法/取模运算的被除数必须校验非零，特别是来自外部输入的变量（如 `coreNum`、`blockCount`）。

**错误示例：`coreNum` 可能为 0**

```cpp
void TilingKernel(int totalElements, int coreNum)
{
    int perCore = totalElements / coreNum;   // ← 风险！coreNum 可能为 0
    int remainder = totalElements % coreNum; // ← 同样风险
    // ...
}
```

**正确示例：先校验再运算**

```cpp
void TilingKernel(int totalElements, int coreNum)
{
    if (coreNum <= 0) {
        OP_LOGE("TilingKernel", "coreNum must be positive, got %d", coreNum);
        return;
    }
    int perCore = totalElements / coreNum;
    int remainder = totalElements % coreNum;
    // ...
}
```

**特别关注**：所有通过 `GetAivCoreCount` 获取的数值都可能返回 0（获取失败），必须先判零再用于除法。

---

## R9: 许可证头

所有源码文件（.cpp / .h / .c）必须包含标准许可证头，CSV 文件除外。

**标准许可证头模板**：

```cpp
/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
```

**例外文件**：
- `.csv`（测试用例表）
- `.gitignore` / `.clang-format` 等配置文件
- 自动生成的文件（在文件顶部注明自动生成来源）

---

## R10: 禁止 extern 引用

禁止在 Kernel/Host 代码中使用 `extern "C"` 声明外部函数接口，应通过头文件 include 引入。

**错误示例：直接 extern 声明 kernel_do**

```cpp
// 在 host.cpp 中直接 extern 声明
extern "C" void saxpy_kernel_do(uint8_t* x, uint8_t* y, uint8_t* workSpace,
                                uint32_t numBlocks, const SaxpyTilingData& tiling,
                                void* stream);
```

**正确示例：通过头文件引入**

```cpp
// saxpy_host.cpp
#include "saxpy_kernel.h"   // kernel_do 在该头文件中声明

// saxpy_kernel.h
#pragma once
void saxpy_kernel_do(/* ... */);
```

**例外**：在 kernel.cpp 中声明 kernel 入口函数时可以使用 `extern "C"`（这是 Ascend C 的要求）。
