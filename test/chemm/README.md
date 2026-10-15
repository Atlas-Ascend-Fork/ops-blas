# aclblasChemm 测试

本目录包含 `aclblasChemm` 的 CSV 驱动测试。`arch22` 对应 Atlas A2/A3 系列产品。

## 文件结构

```text
test/chemm/
├── CMakeLists.txt
├── README.md
├── chemm_golden.h
├── chemm_param.h
└── arch22/
    ├── chemm_npu_wrapper.h
    ├── chemm_test.cpp
    └── chemm_test.csv
```

头文件保留测试接口声明和参数类型；CPU golden、NPU 封装、设备 Event 计时及参数构造器
在 `arch22/chemm_test.cpp` 中实现，与测试主体共用一个编译单元。

CSV 共包含随任务提供的 1200 条用例：1000 条精度与接口用例、200 条性能用例。精度用例覆盖
LEFT/RIGHT、UPPER/LOWER、非对齐尺寸、前导维度、复数标量、Inf/NaN、零维快速返回和非法参数。
精度输入中均匀分布与正态分布各占 50%。性能用例使用紧凑前导维度，并与任务提供的 GPU
基线逐点对应。

## 构建与运行

按照仓库快速入门完成环境配置后执行：

```bash
bash build.sh --soc=ascend910b3 --ops=chemm
bash build.sh --soc=ascend910b3 --ops=chemm --run
```

精度按 COMPLEX64 的实部、虚部分别验证：`rtol=2^-10`、`atol=2^-16`、匹配率不少于
99%，且每个有限分量的绝对误差不超过 `max(1e-2, 32×ULP(golden))`。CPU golden
每次由 CBLAS 直接计算，不读写文件缓存。

性能用例先预热 5 次，再采样 60 次设备 Event 平均耗时。200 个点均须满足
`GPU 耗时 / NPU 耗时 >= 0.8`；任务书列出的绝对耗时限制同时生效。
