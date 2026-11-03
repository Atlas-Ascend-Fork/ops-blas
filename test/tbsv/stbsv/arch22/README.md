# aclblasStbsv arch22 测试说明

适用环境：Ascend 910B3、CANN 9.1.0，Linux aarch64。历史验收版本da2b134的完整精度1786/1786通过、全部200项性能通过；后续代码变更需要对实际构建版本重新完整复验，不能沿用旧结果作为新版本通过结论。

## 精度用例

- 原任务CSV：`stbsv_test.csv`，1200项，包含200项性能参数的精度校验；文件及生成规则不变。
- RAW补充：576项，不强化对角，不将输入条件化。均匀[-5,5]和规定正态分布各半，覆盖12种uplo/trans/diag组合、正负步长、padding、k≥n及n=32769回退路径。
- 独立GTest：11项，涵盖空句柄、INT_MIN非法步长、UNIT对角Inf/NaN忽略、极小数、指数边界、零带系数下NaN传播、精确FMA中点及缩放缓冲切换；内部子检查不另计GTest。INT_MIN以真实非空A/x缓冲区覆盖n=0/1/2/16及全部uplo/trans/diag组合，统一要求INVALID_VALUE；CSV的INT_MIN用例也分配最小非空x缓冲区，避免空指针校验掩盖步长错误。
- golden：Netlib BLAS3.12.0的cblas_stbsv，不替换成OpenBLAS或独立近似求解。
- FLOAT32判定：rtol=atol=2^-13，matched_ratio≥0.99，逐元素误差≤max(0.01,32×golden元素ULP)。负向测试核对状态码，数值测试保留padding检查。

保留完整运行日志及实际加载库路径，完整精度测试应为1787项通过、0失败。独立测试的内部子检查不重复计为GTest。

窄带窗口边界追加n/k为3001/8、4096/8、4097/15、15873/8的4组形状，结合3种lda、上下三角、N/T/C、正负步长及两种原始分布，共288项内部检查，覆盖窗口重建、尾块与UB容量边界；不改变原CSV、576项RAW或GTest总数。
## 零值与异常对角的参考语义

Netlib 3.12.0 的 `stbsv.f` 在 NO_TRANS 下先判断 `X(J).NE.ZERO`，再在该分支内执行 NON_UNIT 对角除法；上下三角及非单位步长分支一致。T/C 路径则对零 pivot 也执行 NON_UNIT 除法。因此，`b=0` 且对角为零或NaN时，NO_TRANS可保持零，T/C可得到NaN；这与参考实现一致，不将两种转置模式的异常值结果强制统一，也不新增奇异性检测。

参考：[Netlib 3.12.0 原始实现](https://github.com/Reference-LAPACK/lapack/blob/v3.12.0/BLAS/SRC/stbsv.f#L268-L332)。NO_TRANS的零值判断包围了除法和后续更新；不能改为先除法再依据商是否为零跳过更新。

`ZeroPivotSpecialDiagonalMatchesNetlib` 增加480组内部检查，覆盖±0右端、零/NaN对角、UNIT/NON_UNIT、上下三角、N/T/C、正负步长及padding。形状为1/0、17/1、1024/256、3001/8、32769/1（n/k），在910B3的已查询预算下分别覆盖微小全局路径、缓存、多核owned、窄窗和超预算全局回退。每组先调用真实Netlib golden，并显式确认上述零/NaN分类，再逐元素核对设备结果；不使用匹配比例放过特殊值差异。

## 性能用例

TC_PF_1001至TC_PF_1200共200项，保留原任务参数，每项独立msprof op、warm-up=5、15个有效样本全部取算术平均，共3000样本。不使用GTest墙钟、设备事件计时、最小值或筛选后的样本作正式判定。

全部200项上限为GPU基线gpu_ms×1000/0.8微秒；前五项再与198、379.8、659.5、1725、2365微秒取较严格值。输入右端b每次恢复，不使用上一次的解作为新输入。只有所测版本完整200项均达标、无缺项、无NO_REF，才能声明该版本通过。

每项保留全部原始OpBasicInfo CSV及应用采集日志，按原参数关联GPU标杆。核对Op Name、实际15个正有限样本及全量平均；任一缺失、采集失败或超标均判失败。

## 文件及工程关系

`stbsv_test.cpp`为CSV驱动GTest与补充测试，直接调用公共`aclblasStbsv`接口。测试复用上级目录`stbsv_param.h`和`stbsv_golden.h`及ops-blas框架。开发期融合/非融合顺序参考已删除，正式golden仍为Netlib。

代码快照需结合完整ops-blas工程，构建目标为stbsv_test，SOC为ascend910b3，参考库选择为Netlib BLAS。完整工程及其标准测试入口以个人仓库提交分支为准。

## 测试步骤

已经配置好CANN9.1.0与Netlib参考库的完整工程，可使用以下入口：

```bash
bash build.sh --soc=ascend910b3 --ops=stbsv
cmake -S . -B build -DREFBLAS_LIB=/usr/local/lib/libblas_netlib.so \
  -DREFBLAS_INCLUDE_DIR=/usr/local/include
cmake --build build --target stbsv_test -j8
./build/test/tbsv/stbsv/stbsv_test > stbsv-precision.log 2>&1
```

参考库路径按实际Netlib安装位置调整。检查ldd和`[STBSV-LOADED-LIB]`，确保加载本次构建的算子库和Netlib。正式性能可在源码根目录、同一Shell执行：

```bash
set -euo pipefail
runtime=$(mktemp -d)
cp build/libops_blas.so build/test/tbsv/stbsv/stbsv_test "$runtime/"
export LD_LIBRARY_PATH="$runtime:${LD_LIBRARY_PATH:-}"
export STBSV_EXPECTED_LIBRARY="$runtime/libops_blas.so"
readelf -d "$runtime/stbsv_test" > "$runtime/dynamic-tags.txt"
grep -q '(RUNPATH)' "$runtime/dynamic-tags.txt"
! grep -q '(RPATH)' "$runtime/dynamic-tags.txt"
ldd "$runtime/stbsv_test" > "$runtime/ldd.txt"
"$runtime/stbsv_test" > "$runtime/precision.log" 2>&1
grep -Fq '[  PASSED  ] 1787 tests.' "$runtime/precision.log"
! grep -Fq '[  FAILED  ]' "$runtime/precision.log"
output=$(mktemp -d)
for number in $(seq 1001 1200); do
  case="TC_PF_$number"
  STBSV_EVENT_PERF=1 msprof op --aic-metrics=BasicInfo --warm-up=5 --launch-count=15 \
    --output="$output/$case" \
    --application="$runtime/stbsv_test --gtest_filter=*$case" \
    > "$output/$case.log" 2>&1
  grep -Fq "[STBSV-LOADED-LIB] $STBSV_EXPECTED_LIBRARY" "$output/$case.log"
  grep -Fq '[  PASSED  ] 1 test.' "$output/$case.log"
  ! grep -Eq '\[ERROR\]|\[  FAILED  \]' "$output/$case.log"
done
cmp build/libops_blas.so "$runtime/libops_blas.so"
cmp build/test/tbsv/stbsv/stbsv_test "$runtime/stbsv_test"
```

保持源码路径供CSV加载，采样期间不得重新构建或并行运行其他NPU作业。`STBSV_EXPECTED_LIBRARY`拒绝混合库；用相同冻结运行库完成精度和全部性能采样。msprof自身预热5次，`STBSV_EVENT_PERF=1`保留恢复原始b的额外调用和可选事件诊断，事件值标记NOT_FORMAL_ACCEPTANCE，不作正式验收。

核对每项实际原始样本数量及全量均值，不从launch-count参数或GTest总时长推算。HDC失败、0 success、空CSV、0us都不是通过。测试README独立给出步骤，不依赖未随代码PR提交的验收材料目录。
