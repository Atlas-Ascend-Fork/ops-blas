# R4: TilingData 禁止使用数组

TilingData 结构体中不可使用数组来为每个核分配独立地址。

## 错误示例

```cpp
// ❌ 错误：数组长度依赖核数，核数不确定
struct FooTilingData {
    uint32_t startOffset[MAX_CORE_NUM];
    uint32_t calNum[MAX_CORE_NUM];
};
```

问题：核数由运行时动态获取，编译期不可知，数组长度无法合法定义；同时 TilingData 体积膨胀。

## 正确示例

```cpp
// ✅ 正确：记载核间间隔，Kernel 自己算
struct FooTilingData {
    uint32_t totalN;     // 总元素数
    uint32_t perCoreN;   // 每核基础分配量
    uint32_t remainder;  // 余数（前 remainder 个核多分 1 个）
};

// Kernel 侧自行计算：
// uint32_t myOffset = blockIdx * perCoreN + min(blockIdx, remainder);
// uint32_t myCount  = perCoreN + (blockIdx < remainder ? 1 : 0);
```
