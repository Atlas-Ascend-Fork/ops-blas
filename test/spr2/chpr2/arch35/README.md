# aclblasChpr2（Atlas 950PR）测试

本测试使用 GTest 和 `cblas_chpr2`，覆盖 arch35 上的参数校验、精度、对角虚部、负步长及设备缓冲区哨兵。构建依赖 CANN、GTest 与 CBLAS，验收需使用任务指定的 Netlib 版本。

在仓库根目录加载 CANN 环境并构建：

```bash
source /path/to/cann/set_env.sh
export EAGER_LIBRARY_PATH="$ASCEND_HOME_PATH/lib64"
export LINUX_INCLUDE_PATH="$ASCEND_HOME_PATH/include"
bash build.sh --ops=chpr2 --soc=ascend950
cmake --build build --target chpr2_test -j8
TEST=build/test/spr2/chpr2/chpr2_test
"$TEST" --gtest_filter='-*TC_PF*' --gtest_color=no
```

完整精度自测及内存检查可分别执行（输出目录须不存在）：

```bash
python3 test/spr2/chpr2/arch35/run_self_test.py --binary "$TEST" --output /tmp/chpr2-accuracy
python3 test/spr2/chpr2/arch35/run_memcheck.py --binary "$TEST" --output /tmp/chpr2-memcheck
```

精度脚本核对 CSV 数量、用例名称及执行记录，保存日志、结果和汇总。内存检查依赖环境提供的 `mssanitizer`，分别检查上、下三角以及正、负步长的真实 kernel 调用，并运行负向参数用例；每组生成独立日志和 XML，`summary.json` 汇总结果。

使用 `ldd "$TEST"` 确认实际链接的 CBLAS 与验收要求一致。`CHPR2_CSV` 可指定用例文件；`CHPR2_RESULTS` 可指定结果 CSV 路径。精度验收使用任务书规定的混合误差与绝对误差上限；CSV 阈值的 MERE/MARE 检查仅用于工程诊断，不作为该任务的验收条件。测试还检查非零 alpha 的结果对角虚部、输入向量及设备缓冲区哨兵。

性能采样时可设置 `CHPR2_PROFILE=1` 并用 `msprof op` 采集指定 `TC_PF` 用例，使用 profiler 输出的原始 Task Duration 数据计算性能，不以 GTest 运行时间代替 kernel 耗时。`CHPR2_PROFILE_CALLS` 可设置每条用例的算子调用次数（默认 10，范围 1～1000），应与 profiler 采样配置对应。Profiling 模式可能重放原地更新，不用于精度判断。
