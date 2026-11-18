# aclblasChemv 测试

测试通过公共 `aclblasChemv` 接口运行，目标平台为 Ascend 950PR、CANN 9.1.0。
CPU 参考使用 Netlib BLAS/CBLAS 3.10.0 的 `cblas_chemv`。

## 目录结构

- `chemv_param.h`：CSV 参数解析。
- `chemv_golden.h`：参考库绑定检查和 CPU golden 调用。
- `chemv_compare.h`：实部、虚部精度比较。
- `arch35/chemv_npu_wrapper.h`：设备缓冲区管理与 NPU 接口调用。
- `arch35/chemv_test.cpp`：参数化测试、边界检查及随机标量测试。
- `arch35/chemv_test.csv`、`arch35/gpu_baseline.csv`：测试输入和 GPU 性能基线。
- `prepare_netlib.py`：从本地源码归档构建固定版本参考库。

## 构建与运行

准备 CANN、CMake、GNU C/C++/Fortran 编译器，以及仓库要求的 GTest 等依赖。
将 Netlib 3.10.0 源码归档保存在本地，归档须与脚本中固定的 SHA256 匹配。
在仓库根目录运行，环境脚本路径按实际安装位置调整：

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
export CHEMV_REF_DIR="$(pwd)/../chemv-netlib310"
python3 test/symv/chemv/prepare_netlib.py \
  --archive /path/to/lapack-3.10.0.tar.gz --work-dir "$CHEMV_REF_DIR"
export CHEMV_NETLIB_CONFIG="$CHEMV_REF_DIR/netlib-reference.cmake"
export LD_LIBRARY_PATH="$CHEMV_REF_DIR/build/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
bash build.sh --ops=chemv --soc=ascend950 --run
```

参考库构建脚本在独立目录生成 `netlib-reference.cmake` 和 `netlib-reference.json`。
`build.sh` 将 `CHEMV_NETLIB_CONFIG` 传给 CMake；测试启动时核对 CBLAS 与 Fortran 符号的实际来源。
已有参考库构建目录可直接复用其配置文件。

## 覆盖范围

测试覆盖上下三角、共轭镜像、对角虚部忽略、lda padding、正负步长、标量退化、快速返回及非法参数。
未使用三角、padding 和对角虚部填入 NaN；步长空洞按字节检查。
实部与虚部分别比较混合容差、最大绝对误差、非有限值及 CSV 指定的 MERE/MARE 指标。
随机标量测试分别使用均匀分布和正态分布。

单独运行测试项可使用 GTest 过滤器，例如：

```bash
build/test/symv/chemv/chemv_test --gtest_filter='*TC_EX_0001'
```

接口与参数约束见[算子说明](../../../../blas/symv/README.md#aclblaschemv)，实现见[设计文档](../../../../docs/zh/develop/aclblasChemv_design.md)。

## 全量自测与内存检查

构建后可运行以下门禁，输出目录中保留原始日志、GTest XML 和 `summary.json`：

```bash
python3 test/symv/chemv/run_self_test.py \
  --binary build/test/symv/chemv/chemv_test --device 0 --output chemv_self_test
```

`run_self_test.py` 校验 CSV 表头、1200 条用例、1000/200 分类、名称唯一性及 GPU 基线参数对应关系。
运行后核对每条 CSV 用例及补充测试均实际完成，汇总实部、虚部的精度记录到 `precision.json`。
CSV 装载器在 GTest 注册阶段也强制执行字段、数量、分类和唯一性检查，装载异常直接报错。

内存检查使用独立构建目录，给 kernel 添加 `--cce-enable-sanitizer -g`，供 mssanitizer 使用：

```bash
cmake -S . -B build-memcheck \
  -DASCEND_CANN_PACKAGE_PATH="$ASCEND_HOME_PATH" \
  -DSOC_VERSION=ascend950 -DBUILD_TEST=ON -DTEST_NAMES=chemv \
  -DCHEMV_NETLIB_CONFIG="$CHEMV_NETLIB_CONFIG" -DCHEMV_ENABLE_MEMCHECK=ON
cmake --build build-memcheck --target chemv_test
python3 test/symv/chemv/run_memcheck.py \
  --binary build-memcheck/test/symv/chemv/chemv_test --device 0 --output chemv_memcheck
```

`CHEMV_ENABLE_MEMCHECK` 默认关闭，正常构建保持原编译选项；检查产物仅用于 mssanitizer，不用于性能采样。
`run_memcheck.py` 要求 CANN 环境提供 `mssanitizer`，使用 `--tool=memcheck --leak-check=yes` 及工具默认日志等级。
参数错误、快速返回和成功计算分别检查；成功用例覆盖上下三角、尾块、lda padding、正负步长及标量退化。
每组成功计算必须有 Chemv kernel 的开始与无错误结束记录，工具告警、漏检和泄漏均使门禁失败。

两项门禁也可通过构建目标执行，设备号在配置时由 `-DCHEMV_TEST_DEVICE=0` 指定：

```bash
cmake --build build --target chemv_self_test
cmake --build build-memcheck --target chemv_memcheck
```
