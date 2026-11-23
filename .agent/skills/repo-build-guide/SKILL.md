---
name: repo-build-guide
description: BLAS 算子仓编译与验证命令参考。编译算子、运行测试、打包的常用命令。触发：需要编译算子、运行测试、查看 build.sh 用法时。
---

# BLAS 仓编译与验证命令

## 算子验证流程

验证一个算子需要两步：**编译** → **运行测试**

```bash
# 步骤 1：编译算子（仅编译，不运行）
bash build.sh --ops=<算子名> --soc=<芯片型号>

# 步骤 2：编译并运行测试（加 --run）
bash build.sh --ops=<算子名> --soc=<芯片型号> --run
```

## 测试失败诊断

当测试用例执行失败时，需要判断是否为**本次修改引入的问题**：

1. **定位基准仓远程名**（远程名只是本地约定，按 URL 反查最稳妥）：
   ```bash
   BASE_REMOTE=$(git remote -v | awk '/cann\/ops-blas/ {print $1; exit}')
   # 若为空说明本地未配置基准远程，临时加一个（命名为 upstream）
   if [ -z "$BASE_REMOTE" ]; then
     git remote add upstream https://gitcode.com/cann/ops-blas.git
     BASE_REMOTE=upstream
   fi
   ```

2. **获取最新基准分支代码**：
   ```bash
   git fetch "$BASE_REMOTE"
   ```

3. **切换到基准分支**，重新编译并运行相同算子的测试：
   ```bash
   git checkout "$BASE_REMOTE/master"
   bash build.sh --ops=<算子名> --soc=<芯片型号> --run
   ```

4. **对比结果**：
   - 若基准分支上测试**通过** → 本次修改引入了问题，需要排查
   - 若基准分支上测试**同样失败** → 这是算子原有的问题，非本次修改导致

5. **切回开发分支**继续工作：
   ```bash
   git checkout <你的分支名>
   ```

**注意**：切换分支前确保当前修改已 commit 或 stash，避免丢失工作进度。

## 常用命令

| 命令 | 说明 |
|------|------|
| `bash build.sh` | 只编译库，不编译测试 |
| `bash build.sh --ops=scopy` | 编译指定算子 |
| `bash build.sh --ops=scopy,sdot` | 编译多个算子（逗号分隔） |
| `bash build.sh --ops=gbmv` | 编译整个算子家族（自动展开子算子） |
| `bash build.sh --ops=scopy --run` | 编译并运行测试 |
| `bash build.sh --run` | 编译并运行所有算子测试 |
| `bash build.sh --ops=scopy --run --device=1` | 指定测试运行设备（默认 0） |
| `bash build.sh --pkg` | 编译并打包 run 包 |
| `bash build.sh --pkg --soc=ascend950` | 打包指定 SOC 的 run 包 |

## 参数说明

| 参数 | 说明 | 示例 |
|------|------|------|
| `--ops=<name>` | 指定算子名或家族名，支持逗号分隔多个 | `--ops=scopy,sdot` |
| `--soc=<soc>` | 指定目标芯片（默认 ascend910b3） | `--soc=ascend950` |
| `--run` | 编译后运行测试 | — |
| `--pkg` | 编译后打包 run 包 | — |
| `--device=<id>` | 指定测试运行的 NPU 设备 ID（默认 0） | `--device=1` |

## 支持的 SOC 版本

- `ascend910b` / `ascend910_93` → arch22（支持前缀匹配，如 `ascend910b3`）
- `ascend950` → arch35
- `ascend310p` → arch20

**注意**：`--soc` 参数支持前缀匹配，例如 `ascend910b3`、`ascend910b4` 都会匹配到 `ascend910b` 对应的 arch22 架构。

### 架构默认反选

根据 SOC 参数确定目标 arch 目录的默认值：

| SOC | 默认 arch |
|-----|----------|
| `ascend950` | `arch35` |
| `ascend910b` / `ascend910_93` | `arch22` |
| `ascend310p` | `arch20` |

此映射为默认值，开发者若有对应板子可根据实际目标调整。

## 算子名解析

`--ops` 参数支持两种格式：

1. **具体算子名**（如 `scopy`、`sswap`）→ 直接编译该算子
2. **家族名**（如 `gbmv`、`geqrf_batched`）→ 自动展开为家族下所有有当前 SOC arch 实现的子算子

## 输出目录

- 编译产物：`build/`
- 安装输出：`out/`（库文件在 `out/lib64/`，头文件在 `out/include/`）
- 测试二进制：`build/test/<family>/<op>/<op>_test`

## 提交前检查

> **提醒**：提交 PR 前检查并移除工作流中产生的临时文件（如 `*_perf.cpp`、`*_perf.csv`、`run_perf.py` 等性能迭代产物），这些文件不应带入 PR。
