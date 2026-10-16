# aclblasCswap Ascend 950PR 自测说明

## 测试内容

Ascend 950PR 测试目录为 `test/swap/cswap/arch35/`，包括：

- `cswap_test.csv`：1200 条任务配套用例，其中 1000 条精度/功能用例、
  200 条 `TC_PF` 性能用例；
- `cswap_test.cpp`：CSV 驱动的 bit-exact 校验，以及 quick return、空 handle、
  NaN/Inf/有符号零位模式、混合正负步长、步长空洞和对齐偏移专项测试；
- `cswap_benchmark.cpp`：设备事件计时，固定预热 20 次、有效采样 100 次，
  并对任务书三个典型 shape 执行硬门槛断言；
- `cswap_npu_wrapper.h`、`cswap_golden.h`、`cswap_param.h`：Device 调用、
  Netlib CBLAS golden 和 CSV 参数解析。

精度测试比较 x、y 的完整物理分配，因此参与交换的实部/虚部、步长空洞
及对齐前缀均须逐位一致。

## 环境与依赖

验证环境为 Ascend 950PR、CANN 9.1.0。构建测试还需要 CMake、GTest
（仓库构建系统自动获取/配置）、BLAS 和 LAPACK 开发包。Ubuntu 22.04 可
安装：

```bash
apt-get update
apt-get install -y cmake libblas-dev liblapack-dev
```

## 编译

在 ops-blas 仓库根目录执行：

```bash
source /usr/local/Ascend/cann/set_env.sh
rm -rf build
bash build.sh --soc=ascend950 --ops=cswap
```

预期生成：

```text
build/test/swap/cswap/cswap_test
build/test/swap/cswap/cswap_benchmark
```

## 精度与功能测试

运行 1000 条非性能 CSV 用例和 4 条专项测试：

```bash
./build/test/swap/cswap/cswap_test \
  --gtest_filter='-*TC_PF*' \
  --gtest_brief=1
```

如使用任务包中提供的精度脚本：

```bash
python3 /path/to/test_cases/verify_accuracy.py \
  --repo "$PWD" --soc ascend950 --skip-build --timeout 3600
```

预期汇总分别为 `1004 tests ... PASSED`，以及脚本输出
`PASS=1000 FAIL=0`。

## 性能测试

运行全部 200 条性能用例：

```bash
./build/test/swap/cswap/cswap_benchmark --gtest_brief=1
```

只运行一个典型用例：

```bash
CSWAP_BENCHMARK_CASE=TC_PF_1001 \
  ./build/test/swap/cswap/cswap_benchmark --gtest_brief=1
```

典型用例与门槛如下：

| case | n | incx | incy | 最大平均耗时 |
| --- | ---: | ---: | ---: | ---: |
| TC_PF_1001 | 1048576 | 1 | 1 | 23.11 us |
| TC_PF_1002 | 2097152 | 1 | 1 | 63.82 us |
| TC_PF_1003 | 4194304 | 1 | 1 | 166.11 us |

计时使用同一 stream 上的 ACL 设备事件，只覆盖 100 次 kernel 下发；Host
侧内存申请、初始化和结果处理不在计时区间。每条输出包含 `warmup=20`、
`repeats=100`、`avg_us`，三个典型用例还包含 `threshold_us`。

任务包自带 `verify_performance.py` 从 GTest 的整数毫秒 wall time 解析耗时，
其中还包含 Host 准备、Device 内存申请/拷贝、golden 和比对，不适用于本
任务的微秒级 kernel 门槛。正式性能验收请使用上述设备事件 benchmark；
这是任务书“先 warmup、有效采样 >50 次”的直接实现。

## 内存说明

算子自身 workspace 为 0 Byte。最大典型用例 `n=4194304` 时，一个
COMPLEX64 向量为 32 MiB，测试同时分配 x、y 共 64 MiB；该输入/输出内存
不属于算子 workspace。
