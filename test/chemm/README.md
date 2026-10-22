# aclblasChemm 测试说明

本目录提供 `aclblasChemm` 的 CSV 驱动测试；`arch22` 对应 Atlas A2 / A3 系列产品。

## 目录结构

```text
test/chemm/
├── CMakeLists.txt
├── README.md
├── chemm_golden.h            # CPU 参考实现（调用 CBLAS 计算 golden）
├── chemm_param.h             # CSV 行 -> 用例参数的解析
└── arch22/
    ├── chemm_npu_wrapper.h   # 设备侧算子调用封装
    ├── chemm_test.cpp        # GTest 用例入口
    ├── chemm_test.csv        # CI 实跑集合（表头 + 前 50 条精度用例）
    └── chemm_test_full.csv   # 全量集合（1000 条精度 + 200 条性能）
```

## CSV 用例集合

任务随附共 1200 条用例：1000 条精度/接口用例与 200 条性能用例，完整集合保存在
`chemm_test_full.csv`。精度用例覆盖 LEFT/RIGHT、UPPER/LOWER、非对齐尺寸、前导维、
复数标量、Inf/NaN、零维快速返回与非法参数，均匀分布与正态分布各占约一半；性能
用例采用紧凑前导维，并与任务提供的 GPU 基线逐点对应。

为适配 PreSmoke 共享测试池的时间/内存预算，测试代码实跑的 `chemm_test.csv` 只保留
表头与前 50 条精度用例（`chemm_test.cpp` 通过 `ReplaceFileExtension2Csv(__FILE__)`
按同名 `.csv` 读取，故该文件名固定）。如需本地全量回归，将 `chemm_test_full.csv`
覆盖为 `chemm_test.csv` 后再运行即可。

## 构建与运行

```bash
bash build.sh --soc=ascend910b3 --ops=chemm
bash build.sh --soc=ascend910b3 --ops=chemm --run
```

## 判定标准

精度对 COMPLEX64 的实部、虚部分别校验：`rtol=2^-10`、`atol=2^-16`，匹配率不低于
99%，且最大绝对误差不超过生态 FLOAT32 标准规定的上限。

性能用例默认跳过，需设置 `CBLAS3_PERF_MODE=1` 后运行：先预热 5 次，再采样 60 次
device Event 平均耗时；任务书 §3.3 的三个必测 case 需满足其绝对耗时门槛。
