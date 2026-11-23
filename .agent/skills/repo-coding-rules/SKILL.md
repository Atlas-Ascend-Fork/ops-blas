---
name: repo-coding-rules
description: |
  ops-blas 仓 AscendC 编码规范 + MR 安全编码规则速查索引。
  作为编码与代码检视的对照清单，按需读取，不常驻上下文。
  触发：编写/修复算子或测试代码后自查、代码检视逐条核对规则、修复 codecheck 告警时加载。
  详细规则内容位于 references/ 目录，由调用方按需加载相关文档。
---

# AscendC 编码规范速查索引

本规则为**自查/检视清单**，不加载为常驻上下文。按场景由 agent 按需调用（本 skill 不感知自己在工作流中的位置，仅按场景提供规则）：

| 场景 | 加载方式 |
|------|---------|
| 编码/修复后自查 | 加载本 skill，按 `references/checklist.md` 8 步流程自查一次；发现违规查 `references/fix-guide.md` |
| 代码检视（逐条核对） | 对照 `references/mr-rules-essential.md`（严重/致命，须零违规）+ `references/mr-rules-general.md`（深度审查）+ `references/ascendc-r5-r10.md`（质量指标）做多维度检视 |
| codecheck 告警修复 | 按告警定位违规规则，依 `references/fix-guide.md` + `references/mr-rules-*.md` 修复 |

## references 索引

### AscendC 编码规则（R1-R10）

- 每条规则独立成一个 reference 文件，包含错误/正确示例：

| 编号 | 规则 | reference |
|------|------|-----------|
| R1 | 禁止逐元素操作 | `references/R1-禁止逐元素操作.md` |
| R2 | 动态获取 CoreNum（禁止硬编码） | `references/R2-动态获取CoreNum.md` |
| R3 | TPipe 禁止作为成员变量 | `references/R3-TPipe禁止成员变量.md` |
| R4 | TilingData 禁止使用数组做核间分配 | `references/R4-TilingData禁止数组.md` |
| R5-R10 | 圈复杂度/嵌套深度/函数行数/除零防御/许可证头/extern 引用 | `references/ascendc-r5-r10.md` |

### MR 安全编码规则

- 按**严重等级**分组，方便代码检视时按优先级逐条核对：

| reference | 包含内容 | 适用场景 |
|-----------|---------|---------|
| `references/mr-rules-essential.md` | 严重/致命级规则（G.PRE.05、G.INC.*、G.FUU.09/10/12/13/15、G.MEM.04、G.STD.*、OAT 等 ~18 条） | MR 提交前必查 |
| `references/mr-rules-general.md` | 一般/建议级规则（G.EXP.*、G.CTL.03、G.AST.03、G.FUU.11/14、CQ.*、CIP.01 等 ~17 条） | 代码检视深度审查 |

### 检查流程与修复指南

| reference | 内容 | 适用场景 |
|-----------|------|---------|
| `references/checklist.md` | 8 步检查流程（文件级 → 头文件 → 函数级 → 表达式 → 安全函数 → 内存 → 标准库 → 冗余告警） | 提交前系统自查 |
| `references/fix-guide.md` | 14 种常见违规的修复方法对照表 | 发现违规后查修复方案 |

## 使用示例

### 场景 1：编码自查

代码/测试编写或修复完成后调用本 skill 执行自检：

```
1. 加载 `references/checklist.md`，按 8 步流程逐条检查代码
2. 发现 R6 嵌套深度超标 → 加载 `references/ascendc-r5-r10.md` 查看修复方法
3. 发现 G.FUU.09 使用了 realloc → 加载 `references/mr-rules-essential.md` 确认级别与修复方案
4. 修复完成后再次执行 checklist.md 直至通过
```

### 场景 2：代码检视

对变更文件做逐条核对时，加载：

```
1. `references/mr-rules-essential.md` 对照严重/致命规则做检视（必须零违规）
2. `references/mr-rules-general.md` 做深度审查（建议修复）
3. `references/ascendc-r5-r10.md` 检查代码质量指标（圈复杂度/嵌套深度/行数）
```

---

**注意**：本 skill 的 SKILL.md 仅作为索引与触发说明。Agent 在执行自查/检视时，应直接读取相应的 references 文件获取完整规则，**不要**在开发过程中常驻本 skill 的全部内容。
