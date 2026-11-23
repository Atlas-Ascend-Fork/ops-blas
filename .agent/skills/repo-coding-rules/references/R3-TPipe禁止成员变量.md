# R3: TPipe 禁止作为成员变量

## 说明

`TPipe` 管理硬件流水线队列，其生命周期必须与单次 Kernel 调用绑定。作为类成员变量可能导致多次调用间状态残留或重复初始化。

## 错误示例

```cpp
class CopyAIV {
public:
    TPipe pipe_;  // 禁止：作为类成员变量
};
```

## 正确示例

```cpp
// Kernel 入口函数中创建 TPipe，通过指针传入 Init
extern "C" __global__ __aicore__ void scopy_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR tilingGm) {
    TPipe pipe;
    CopyTilingData tilingData;
    ParseTilingData(tilingData, tilingGm);
    CopyAIV op;
    op.Init(x, y, &pipe);
    op.Process();
}
```
