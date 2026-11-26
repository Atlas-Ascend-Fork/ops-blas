# aclblasIsamin Atlas A2/A3 测试

## 用例

`isamin_test.csv` 固定包含 1000 条精度用例和 200 条性能用例。

精度用例的输入分布为：

- `RANDOM_NORM_5_5`：500 条。
- `GAUSS_NORM_5_2`：500 条。
- 特殊输入通过 `description` 开头的 `special=<tag>;` 注入，包括 `nullptr`、`zero`、`alter`、`index`、`extreme`、`inf`、`nan_all`、`nan_first`、`nan_inf` 和 `nan_max`。

输出为 INT32 1-based 索引，NPU 结果必须与 CPU golden 完全相等。

## 精度执行

```bash
bash build.sh --soc=ascend910b3 --ops=isamin
./build/test/isamin/isamin_test \
  --gtest_filter='Isamin/IsaminTest.CsvDriven/*-*TC_PF*' \
  --gtest_color=no
```

正式精度执行 `1000/1000`。

## 性能快速筛查

GTest 默认性能分支使用 `aclrtEvent`，执行 5 次 warmup 和 20 次采样，仅用于快速筛查：

```bash
./build/test/isamin/isamin_test \
  --gtest_filter='Isamin/IsaminTest.CsvDriven/*TC_PF*' \
  --gtest_color=no
```

该计时值不作为正式性能判据。

## msprof 正式性能

`verify_performance_msprof.py` 对固定 200 条 `TC_PF_*` 用例逐例执行：

```bash
python3 test/isamin/arch22/verify_performance_msprof.py \
  --binary build/test/isamin/isamin_test \
  --output /path/to/new/evidence
```

每例独立启动 `msprof op`：

- `--aic-metrics=BasicInfo`
- `--warm-up=5`
- `--launch-count=15`
- `--replay-mode=kernel`
- `--kernel-name=isamin_kernel`

测试内部同时执行 5 次预热和 15 次有效调用。采集脚本逐例保存：

- 原始 `OpBasicInfo_*.csv` 15 份。
- `command.json`。
- `msprof.log`，包含 stdout/stderr 和 GTest 结果。
- 逐样本 `performance_samples.csv`。
- 逐例 `performance_results.csv`。
- `performance_summary.json`。

正式判据：

$$
msprof\_mean\_us \le gpu\_ms \times \frac{1000}{0.8}
$$

全部 200 例均须存在 15 个正有限样本，且逐例达标。正式采集共 3000 个 kernel 样本。
