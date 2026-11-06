# GemmEx / SgemmEx / CgemmEx arch22 自验证

## 三接口测试入口

任务书§5将三个接口的测试代码统一放在`test/gemm_ex/`。arch22测试共享`../gemm_ex_param.h`和`../gemm_ex_contract_test.h`，分别从三个CSV装载用例，调用各自的公共API；无需按接口复制一套同内容的参数与包装文件。

| 接口 | 测试驱动与CSV前缀 | 共享契约入口 | 实际公共API |
|---|---|---|---|
| GemmEx | `gemm_ex_test` | `Run<0>` / `Invoke<0>` | `aclblasGemmEx`（19参） |
| SgemmEx | `sgemm_ex_test` | `Run<1>` / `Invoke<1>` | `aclblasSgemmEx`（17参） |
| CgemmEx | `cgemm_ex_test` | `Run<2>` / `Invoke<2>` | `aclblasCgemmEx`（17参） |

arch22的设备调用、拷贝和数值检查由共享契约测试完成，不使用`arch35/*_npu_wrapper.h`。检查入口完整性时应沿上述包含关系和模板实例化核对；只按`sgemm_ex_param.h`或`cgemm_ex_npu_wrapper.h`等独立文件名查找，不能判定arch22缺少测试实现。

## 用例与判据

三个接口的`test/gemm_ex/arch22/*_test.csv`及`*_gpu_baseline.csv`保持任务附件原字节（GPU表仅加接口名前缀）。每接口1000条正式精度/边界用例、200条性能用例，三接口合计3000条正式精度和600条性能。输入由CSV的fill、标量、维度、leading dimension、随机种子及仓内生成器共同确定，无须另外准备模型或媒体数据。

GemmEx/SgemmEx主路径为FLOAT32；CgemmEx为COMPLEX64，实部/虚部分别比较。非法dtype数值999返回INVALID_VALUE，合法但不支持的dtype或组合返回NOT_SUPPORTED；不扩展任务自验范围之外的混合精度。

golden必须是Netlib BLAS/CBLAS3.10.0，实数调用`cblas_sgemm`、复数调用`cblas_cgemm`，不得由同名OpenBLAS库替代。按任务标准使用rtol=atol=2^-13、matched_ratio>=0.99、max_abs_error<=max(0.01,32*ULP(g_low))，其中g_low为最大误差位置的FP32 golden。非有限值另按任务书等效规则判定；alpha=0的缩放路径按EXACT比较，C的padding不得改写。

性能按msprof op的NPU TaskDuration计时，默认使用kernel replay清L2。三个接口各执行一轮200例，每例实际执行5次预热与64次正式调用并取算术平均；kernel replay另设工具warm-up=5，application补充对照设为0，两者分别记录。按stream/last-task ID关联每次调用的全部必要kernel，并将其耗时求和；GTest日志中的毫秒数是含profiler开销的执行时间，不作为性能值。GPU表的ms转换为us，GPU/NPU越大越快，官方门槛>=0.8；本任务自验约定每例64次均值的比值严格>0.8，不将单轮说成额外完成三次独立批次。

回放方式必须与结果一起记录：kernel replay会清理L2缓存，application replay保留前序kernel的L2状态，两种模式的耗时不能当作相同缓存条件直接比较。工具版本、回放方式、额外warmup次数及应用预热次数均以实际命令为准。

kernel replay采用公共默认stream，使CANN9.1回放快照恢复与被测调用按序执行；普通精度测试使用fixture自建stream。采集完整kernel序列，并按应用输出的`GEMMEX_SAMPLE`中的stream ID与last-task ID关联调用。发现漏采、截断或无法关联时保留原数据并判为采集未完成，不能据此生成通过结论。

性能路径也在全部计时调用结束后对最终输出执行同一Netlib golden和精度断言，包括TC_PF用例。golden计算不进入NPU计时窗口；64次性能采样不等于64次独立golden检查。

## 环境与构建

使用任务规定的CANN9.1.0，显式加载实际安装路径的`set_env.sh`，同时记录`npu-smi info`、驱动、工具版本和`ldd`实际加载路径。A2的SOC设为`ascend910b3`，逻辑设备0；A3架构支持与实际实测设备分开记载，只有最终报告列出的设备具有实测结论。

准备Reference-LAPACK v3.10.0的CBLAS与GTest，CBLAS头文件位于该版本的`CBLAS/include`。复用仓库公共CBLAS配置，通过`REFBLAS_LIB`与`REFBLAS_INCLUDE_DIR`显式指定Netlib；不得仅凭`libblas.so`名称判断版本。记录参考库源码commit、实际编译器/flags与动态库哈希，不额外关闭FMA或开启fast-math。

从ops-blas仓库根目录配置，下面的依赖路径由本机已准备的对应版本目录替换：

```bash
source /path/to/cann-9.1.0/set_env.sh
export LINUX_INCLUDE_PATH="${ASCEND_HOME_PATH}/include"
export EAGER_LIBRARY_PATH="${ASCEND_HOME_PATH}/lib64"
cmake -S . -B build -DSOC_VERSION=ascend910b3 \
  -DBUILD_TEST=ON '-DTEST_NAMES=gemm_ex;sgemm_ex;cgemm_ex' \
  -DCMAKE_BUILD_TYPE=Release -DTEST_DEVICE_ID=0 \
  -DREFBLAS_LIB=/path/to/netlib-build/lib/libcblas.so \
  -DREFBLAS_INCLUDE_DIR=/path/to/lapack-3.10.0/CBLAS/include \
  -DREFLAPACK_LIB= \
  -DGTEST_LIB=/path/to/gtest-build/lib/libgtest.a \
  -DGTEST_INCLUDE_DIR=/path/to/googletest/googletest/include
cmake --build build --target gemm_ex_test sgemm_ex_test cgemm_ex_test -j6
```

CATLASS固定于`14d5536bb59cb5822ea3f3e2090b0917d7a4e7b4`，由构建模块默认获取或验证已有缓存；不依赖开发设备的绝对include路径。离线构建在配置时增加`-DOPS_BLAS_CATLASS_SOURCE_DIR=/path/to/pinned/catlass -DOPS_BLAS_CATLASS_OFFLINE=ON`。头文件或license与锁定版本不一致会报错；不要跳过校验。其他仓内依赖沿用顶层工程的固定版本及缓存机制。

## 串行执行与采集

从仓库根目录执行，创建新的证据目录，确保设备上没有其他测试或性能采集。先检查`ldd build/test/gemm_ex/gemm_ex_test`与实际输出路径，确认加载本次编译的`libops_blas.so`和指定Netlib库。

```bash
export LD_LIBRARY_PATH="$(pwd)/build:/path/to/netlib-build/lib:${LD_LIBRARY_PATH}"
mkdir /path/to/new_results
# 使用实际构建输出目录；每个接口排除TC_PF后应恰好执行1000例。
./build/test/gemm_ex/gemm_ex_test --gtest_filter='-*/TC_PF_*' \
  --gtest_output=xml:/path/to/new_results/gemm_ex_accuracy.xml
./build/test/gemm_ex/sgemm_ex_test --gtest_filter='-*/TC_PF_*' \
  --gtest_output=xml:/path/to/new_results/sgemm_ex_accuracy.xml
./build/test/gemm_ex/cgemm_ex_test --gtest_filter='-*/TC_PF_*' \
  --gtest_output=xml:/path/to/new_results/cgemm_ex_accuracy.xml
```

同时保存标准输出与错误输出，核对XML中的实际用例集合、failure/error/skipped和结束状态，不能只看进程退出码。CPU golden按独立输出列调用同一Netlib函数，不改变单个点积累加顺序；`GEMMEX_CHECK_GOLDEN_SPLIT=1`可额外比较串行CBLAS字节一致性。

对每个接口的200条TC_PF用例逐例采集。下面只展示一条GemmEx用例；使用本机已安装的`msprof op --help`核对该工具版本的参数。被测二进制路径避免空白字符，输出目录不得复用。

```bash
export GEMMEX_PROFILE=1 GEMMEX_DEFAULT_STREAM=1
export GTEST_FILTER='*/TC_PF_1001'
export GTEST_OUTPUT='xml:/path/to/new_results/gemm_ex_1001.xml'
msprof op --application=/absolute/path/to/gemm_ex_test \
  --output=/path/to/new_results/gemm_ex_1001 \
  --replay-mode=kernel --warm-up=5 --aic-metrics=BasicInfo
unset GEMMEX_PROFILE GEMMEX_DEFAULT_STREAM GTEST_FILTER GTEST_OUTPUT
```

`GEMMEX_PROFILE`让测试执行5次预热+64次正式调用，每次恢复初始C，并在最后一次调用结束后按同一Netlib标准检查最终输出。`GEMMEX_SAMPLE`时间戳用于关联采集记录，**不是纯NPU耗时**。剔除前5次应用调用后，按调用汇总所有必要kernel的TaskDuration，取64次算术平均，与同目录对应GPU表比较；完整包含输入转换、计算、归约与输出恢复。工具自身warm-up与应用预热分别记录，不以工具汇总中的单个kernel耗时代替整接口耗时。

对4096³等长用例预留足够采集资源并确认记录完整；NO_KERNEL、缺失尾部输出步骤或数值检查未结束应记为采集未完成。工具报错应保留原命令、版本、环境及日志，不能静默改用application replay或丢弃较慢记录。若另测application replay，单独保存结果与缓存条件。

历史验收采集器、原始时序解析器和容器诊断工具保存在对应验收资料中，不作为算子库的构建依赖。新代码应锁定新源码及库哈希、核实实际kernel序列后重新采集；旧报告只覆盖其记录的版本。本目录保留正式测试、CSV、GPU基线及可直接调用的公共接口测试入口。
