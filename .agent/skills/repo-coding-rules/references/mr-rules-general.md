# MR 安全编码规则 - 一般/建议级（代码检视深度审查）

本文件汇总所有**一般级（Normal）**和**建议级（Advisory）**规则。代码检视时作为参考。

---

## 1. 表达式规则 (G.EXP)

### G.EXP.26 [一般] 整型表达式先转型再运算
整型表达式比较或赋值为更大类型前，必须用该更大类型求值。

**错误**：
```cpp
int a = 1000000, b = 1000000;
long long product = a * b;   // ← a*b 溢出后才转 long
```

**正确**：
```cpp
long long product = (long long)a * b;   // 先转型再运算
```

---

## 2. 控制流规则 (G.CTL)

### G.CTL.03 [建议] 循环必须安全退出
禁止死循环，循环变量必须可控。

**错误**：
```cpp
while (true) {
    // 忘记 break 条件 → 死循环
}
```

**正确**：
```cpp
while (condition) {
    if (break_condition) break;
    update_condition();
}
```

---

## 3. 断言规则 (G.AST)

### G.AST.03 [一般] 禁止用断言检测运行时可能发生的错误
运行时错误使用错误处理代码（返回值校验/异常），而非 `assert`。

**错误**：
```cpp
assert(file_handle != nullptr);   // ← file_handle 可能为空（运行时错误）
```

**正确**：
```cpp
if (file_handle == nullptr) {
    OP_LOGE("ReadFile", "file_handle is null");
    return ACLBLAS_STATUS_INVALID_VALUE;
}
```

---

## 4. 安全函数规则 (G.FUU)

### G.FUU.11 [一般] 必须检查安全函数返回值
所有安全函数的返回值必须校验并正确处理。

### G.FUU.14 [一般] 不能用宏重命名安全函数
保持安全函数原名，不要通过 macro 改名。

**错误**：
```cpp
#define my_memcpy memcpy_s
```

---

## 5. 数组规则 (G.ARR)

### G.ARR.03 [一般] 禁止 sizeof(ptr) 求数组大小
`sizeof(ptr)` 返回指针大小（8 字节）而非数组大小。

**错误**：
```cpp
void process(int* arr) {
    size_t n = sizeof(arr) / sizeof(arr[0]);   // ← 错误，arr 是指针
    // ...
}
```

**正确**：
```cpp
void process(int* arr, size_t n) {   // 显式传递大小
    // ...
}
```

---

## 6. 库规则 (G.STD)

### G.STD.18 [一般] 多线程场景注意竞争条件
调用库函数时使用线程安全版本。

---

## 7. 其他安全规则 (G.OTH)

### G.OTH.03 [一般] 禁用 rand 做安全用途
安全场景使用密码学安全随机数（如 `/dev/urandom`），而非 `rand()`/`srand()`。

---

## 8. 代码质量规则 (CQ)

### CQ.04 [一般] 超大源文件拆分
源文件总行数有上限要求（建议 ≤ 1500 行），超大应拆分模块。

### CQ.05 [一般] 超大头文件精简
头文件应精简，避免膨胀（建议 ≤ 500 行）。

### CQ.06 [一般] 目录文件数合理
目录下文件数有上限要求（建议 ≤ 50 个），过多应分层组织。

### CQ.07 [一般] 禁止重复代码块
相同逻辑应提取为公共函数，禁止复制粘贴。

### CQ.08 [一般] 禁止重复文件
发现内容相同的文件应合并。

### CQ.09 [一般] 禁止未使用的代码
包括未使用的 `#include` / 变量 / 函数 / 死代码等。

### CQ.10 [一般] 禁止不安全函数
使用安全函数版本替代不安全函数（详见 mr-rules-essential.md 的 G.FUU.15）。

### CQ.11 [一般] 告警应修复而非抑制
禁止无理由抑制编译告警（如 `#pragma GCC diagnostic ignored`），告警应修复。

---

## Reviewer 检视深度自查清单

### 一般级检查

- [ ] G.EXP.26 整型表达式先转型
- [ ] G.AST.03 断言不检测运行时错误
- [ ] G.FUU.11 检查安全函数返回值
- [ ] G.FUU.14 不重命名安全函数
- [ ] G.ARR.03 sizeof(ptr) 不用求数组
- [ ] G.STD.18 多线程竞争条件
- [ ] G.OTH.03 不用 rand 做安全用途

### 代码质量（CQ）检查

- [ ] CQ.04 / CQ.05 源文件/头文件大小在合理范围
- [ ] CQ.06 目录文件数合理
- [ ] CQ.07 无重复代码块
- [ ] CQ.08 无重复文件
- [ ] CQ.09 无未使用的代码
- [ ] CQ.10 全部使用安全函数
- [ ] CQ.11 无无理由压制告警

---

## 建议级检查

- [ ] G.CTL.03 所有循环有安全退出条件
