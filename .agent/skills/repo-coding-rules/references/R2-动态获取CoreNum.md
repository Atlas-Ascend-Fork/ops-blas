# R2: 动态获取 CoreNum

## 说明

禁止硬编码核数。不同芯片型号、不同形态（PCIE/EP）的 AI Core 数量不同，硬编码会导致跨平台兼容性问题。

## 错误示例

```cpp
constexpr int numBlocks = 28;  // 硬编码核数
scopy_kernel<<<numBlocks, nullptr, stream>>>(x, y, nullptr, tilingGm);
```

## 正确示例

```cpp
int32_t deviceId = 0;
aclrtGetDevice(&deviceId);
uint64_t numBlocks = 0;
aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, sizeof(uint64_t), &numBlocks, nullptr);
scopy_kernel<<<numBlocks, nullptr, stream>>>(x, y, nullptr, tilingGm);
```
