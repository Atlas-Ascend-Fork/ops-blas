# aclblasCscal (950) 验收修订

任务：`005181d868734baebffacbe679e5a6ba`。代码分支：
https://gitcode.com/zee_920/ops-blas/tree/delivery/cscal950-acceptance

设计 PR：https://gitcode.com/cann/cann-ops-competitions/pull/1432

## 退回意见与修复

退回原文：“精度运行全部missing，请按照任务书的3.5的说明进行修改调整。”

旧入口只注册 `Cscal950.CsvAccuracy`、`Cscal950.CsvPerformance` 两个聚合 GTest。
任务包的 `verify_accuracy.py` 从每条 GTest 结果末尾的 `/TC_*` 提取用例名，
因此无法识别旧聚合日志。现按官方 CSV 每行注册一个参数化 GTest：
`Cscal950/CscalCsvTest.CsvDriven/TC_*`。默认输入从源码路径定位，支持官方
`-*TC_PF*` 和 `*TC_PF*` 过滤器及 `ASCEND_DEVICE_ID` 设备选择。

官方 CSV、GPU 基线、Netlib golden、输入生成、内核、精度门槛及设备计时方法
均保持不变。更新配套测量调用器的过滤器，历史冻结清单另存，不改写旧轮次记录。

## 2026-09-15 新构建设备复测

| 检查 | 本轮结果 |
|---|---|
| 官方原版精度脚本 | 1000 PASS，0 FAIL，ALL PASS |
| 官方精度 GTest XML | 实际执行 1000/1000，无 missing/skipped |
| 全量精度/补充精度 | 1000/1000、181/181 |
| 设备性能 | 200/200，最差 NPU/有效上限 0.9858365878 |
| 三个主性能点 | 10.7745937275 / 19.4476875768 / 34.743906348 us |
| 三点有效上限 | 13.73 / 21.2175 / 43.3075 us |
| 冻结数据/保护区 | 全部通过 |

环境为 Ascend950PR、CANN 9.1.0、driver 25.7.rc1.6，NPU0，CPU22。
每条性能用例 10 次预热、64 个 ACL event 样本，使用完整均值，不剔除离群值。
输入恢复在计时区间外，属于 warm-reset stream 区间，不是 msprof 纯 kernel 时间。
题包 GPU 基线未重跑，未独立证明 GPU/NPU 缓存口径相同。

测试二进制 SHA256：`15107161cf9329443f8f403816739654e4ea9efb32109c738fe9e27258bd0bec`。
生产库 SHA256：`4e3f63bb97dae623e5cac8f120dbb72f7f5d9ef5ef3a3d43802dd773dd97885a`。

## 复现与附件

完整源码见附件 `source/ops-blas/`；报告、逐例 CSV、XML、构建日志和原版官方
脚本输出见 `evidence/`。代码与材料哈希见 `MANIFEST.sha256`，版本见
`SOURCE_REVISION.json`。构建和逐阶段计时方式见 `test/scal/cscal/README.md`。

```bash
source /usr/local/Ascend/cann/set_env.sh
export CSCAL_BUILD_DIR="$PWD/build"
BUILD_JOBS=8 bash tools/cscal950/build.sh "$PWD" auto 1 4096
python3 docs/cscal950/acceptance/evidence/original_task/test_cases/verify_accuracy.py --repo "$PWD" --soc ascend950 --skip-build --device 0
taskset -c 22 python3 tools/cscal950/measure.py
```

先确认设备空闲、CPU22在允许集合内，并检查实际 SDK/依赖及编译参数。历史失败和
波动保留在旧附件；本轮通过不代表任意机器或任意轮次必过。HBM/寄存器峰值和设备
sanitizer 本轮 NOT_RUN。当前结论为板端自测通过，官方验收结果以平台复核为准。
