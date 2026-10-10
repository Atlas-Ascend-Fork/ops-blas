# aclblasCdotc 测试说明

## 测试环境

- 硬件：Ascend 950PR
- CANN：9.1.0
- 架构：arch35 / DAV_3510

## 测试用例

测试 CSV 包含 1000 条功能与精度用例、200 条性能用例。功能与精度用例覆盖基础功能、尺寸、正负步长、特殊值、边界、非法参数和扩展场景；测试代码另包含一条空句柄检查。

## 测试步骤

将本仓库中的 `blas/` 和 `test/` 目录按相同相对路径放入 `cann/ops-blas` 源码树，然后执行：

```bash
source /path/to/CANN/set_env.sh
cd /path/to/ops-blas
bash build.sh --ops=cdotc --soc=ascend950
```

设置运行时动态库路径：

```bash
export LD_LIBRARY_PATH="$PWD/build:$PWD/out/lib64:$ASCEND_HOME_PATH/lib64:$ASCEND_HOME_PATH/x86_64-linux/lib64:${LD_LIBRARY_PATH:-}"
```

执行精度测试：

```bash
./build/test/dot/cdotc/cdotc_test --gtest_filter='*-*TC_PF*'
```

执行性能测试：

```bash
./build/test/dot/cdotc/cdotc_test --gtest_filter='*TC_PF*'
```

精度结果的实部和虚部分别与 CPU Golden 比较。性能测试预热后使用 ACL Event 采样 100 次，输出中的 `avg_us` 为单次平均耗时；逐项判定标准为 NPU 平均耗时不大于对应 GPU 基线耗时除以 0.4。
