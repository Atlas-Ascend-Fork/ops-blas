# aclblasChemv 测试说明（Atlas A2 / A3）

本目录包含单精度复数 Hermitian 矩阵-向量乘算子 `aclblasChemv` 在 Atlas A2 / A3（`arch22`）平台上的 GTest 测试工程与 CSV 用例集。

## 目录结构

```text
test/symv/chemv/
├── CMakeLists.txt                     # 测试构建注册（arch22 / arch35 注册 GTest）
├── README.md                          # 本测试说明文档
├── chemv_param.h                      # CSV 参数解析结构
├── chemv_check.h                      # FLOAT32 混合容差及 Inf/NaN 等效判定器
└── arch22/
    ├── chemv_npu_wrapper.h            # 带前后哨兵的 Device 缓冲与标量封装
    ├── chemv_test.cpp                 # GTest 主测试程序（精度 + 性能 + 边界单测）
    ├── chemv_test.csv                 # 随包 1200 条用例（1000 精度 + 200 性能 TC_PF_*）
    ├── chemv_gauss_supplement.csv     # 补充 200 条 GAUSS_NORM_5_2 正态分布精度用例（TC_GS_*）
    └── gpu_baseline.csv               # NVIDIA A100 实测性能标杆数据（200 条）
```

## 环境准备

1. 加载 CANN 9.1.0 环境变量：
   ```bash
   source /home/developer/Ascend/cann/set_env.sh
   ```
2. 确保系统已安装 Netlib CBLAS/LAPACK（`libblas-dev`、`liblapack-dev`）与 GoogleTest（`libgtest-dev`）。

## 编译方法

在 `ops-blas` 仓库根目录执行：

```bash
# Atlas A2 系列（如 Atlas 800T A2 / Ascend 910B3）
bash build.sh --soc=ascend910b --ops=chemv

# Atlas A3 系列
bash build.sh --soc=ascend910_93 --ops=chemv
```

编译产物位于 `./build/test/symv/chemv/chemv_test`。

## 运行精度测试

1. 直接执行全部非性能用例（含 1000 条 `chemv_test.csv` 精度用例、200 条 `TC_GS_*` 正态分布补充用例及 4 项专项单测）：
   ```bash
   ./build/test/symv/chemv/chemv_test --gtest_filter=-*TC_PF*
   ```
2. 或使用任务书随包验收脚本验证：
   ```bash
   python3 verify_accuracy.py --repo /path/to/ops-blas --soc ascend910b --skip-build
   ```

## 运行性能测试

1. 使用随包性能验收脚本（采集 5 次 warmup 后连续下发的 `aclrtEvent` 平均耗时并与 `gpu_baseline.csv` 按 `>= 0.8x` 比对）：
   ```bash
   python3 verify_performance.py --repo /path/to/ops-blas --soc ascend910b --skip-build
   ```
2. 使用 `msprof op` 采集单算子精确 `Task Duration(us)`（设置 `CHEMV_MSPROF_PURE=1` 时跳过前置校验核，严格执行 5 次 warmup + 10 次有效采样）：
   ```bash
   CHEMV_MSPROF_PURE=1 msprof op \
     --application="./build/test/symv/chemv/chemv_test --gtest_filter=*TC_PF_1001*" \
     --output=./prof_chemv
   ```
   采集完成后在 `./prof_chemv/OPPROF_*/OpBasicInfo.csv` 中跳过前 5 条 warmup 记录，对其余 10 条有效样本的 `Task Duration(us)` 求算术平均。
