# aclblasIcamax 测试说明

## 测试范围

正式测试位于 `arch35/`，对应 Ascend 950PR：

- `icamax_test.csv`：全量文件按原顺序保留的前 50 条任务用例，供默认 CSV 驱动测试读取；
- `icamax_test_full.csv`：保留原始 1200 条任务用例，其中 1000 条精度与边界用例、200 条性能数据用例，设置 `ICAMAX_FULL_CSV=1` 时加载；
- `icamax_test.cpp`：CSV 驱动测试和 2 项补充测试：空 handle、空输出地址。

CPU 参考实际调用指定库的 `cblas_icamax`，将其 0-based 结果加 1 后作为唯一 golden。
参考库使用 Netlib BLAS，不能替换为 NaN 处理规则不同的其他 BLAS 实现。
默认注册 50 条 CSV 用例和 2 项补充测试，共 52 项。全量 CSV 不参与默认运行。
设置 `ICAMAX_FULL_CSV=1` 后注册全部 1200 条 CSV 用例和 2 项补充测试，共 1202 项。
该选项在测试程序启动时读取；未设置或值不为 `1` 时均使用前 50 条，无需替换文件或重新编译。
NPU 与 golden 按 1-based 整数完全相等判定。全量文件中的 1052 条随机填充用例按种子稳定划分为 526 条均匀
分布和 526 条正态分布，实部和虚部分别生成。

## 重点语义

- 复数比较量为 `|Re| + |Im|`。
- 无 NaN 时，多个最大值返回最小逻辑索引。
- `n = 0` 或 `incx < 1` 时输出 0，不读取 x。
- `n < 0`、正常路径 x 为空或 result 为空时返回非法参数。
- 第一项的比较值为 NaN 时返回 1；否则忽略后续 NaN，结果与 Netlib 参考库相等。
- 无 NaN 时，第一个 Inf 胜出，后续相同 Inf 不改变索引。

## 构建与运行

在 CANN 9.1.0 环境中执行：

```bash
bash build.sh --soc=ascend950 --ops=icamax
export LD_LIBRARY_PATH="$PWD/build:$LD_LIBRARY_PATH"
./build/test/iamax/icamax/icamax_test
```

测试沿用仓库公共配置查找系统 `libblas`，不设置 Icamax 专用参考库选项。
系统 `libblas` 可能指向不同实现；按 Netlib 验证时，应通过公共变量
`REFBLAS_LIB` 指定 Netlib 库，并用 `REFBLAS_INCLUDE_DIR` 指定 CBLAS 头文件目录。
同时确认运行时加载的是同一库；必要时将其所在目录加入 `LD_LIBRARY_PATH`。
使用不同的库重新配置时须使用新的构建目录或清理对应配置缓存。
构建模式与编译选项遵循仓库统一配置，不为 Icamax 单独追加优化选项。

运行全量用例（沿用上面的环境设置）：

```bash
ICAMAX_FULL_CSV=1 ./build/test/iamax/icamax/icamax_test
```

只运行全量文件中的精度与边界用例，以及补充测试：

```bash
ICAMAX_FULL_CSV=1 ./build/test/iamax/icamax/icamax_test --gtest_filter='-*TC_PF*'
```

列出已注册测试：

```bash
./build/test/iamax/icamax/icamax_test --gtest_list_tests
ICAMAX_FULL_CSV=1 ./build/test/iamax/icamax/icamax_test --gtest_list_tests
```
