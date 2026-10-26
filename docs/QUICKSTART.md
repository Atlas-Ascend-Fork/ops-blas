# 快速入门：基于ops-blas仓

## 使用须知

本指南旨在帮助您快速上手CANN和`ops-blas`算子仓的使用。为方便快速了解算子开发全流程，将以**scopy**算子为实践对象，算子源码按芯片架构区分目录（A2/A3 平台对应 `arch22/`，Ascend 950 PR/DT 对应 `arch35/`），以下以 A2/A3 硬件平台为例，其源文件位于`ops-blas/blas/copy/arch22/`，具体操作流程如下：

1. **[环境部署](zh/install/quick_install.md)**：完成软件包安装和源码下载，此处不再赘述。快速入门场景下，**推荐WebIDE或Docker环境**，安装操作简单。

   > **说明**：当前WebIDE或Docker环境默认最新商发版CANN包；如需体验master分支最新能力，可手动安装CANN包，注意软件与源码版本配套。

2. **[编译运行](#一编译运行)**：编译自定义算子包并安装，实现快速调用算子。

3. **[算子开发](#二算子开发)**：通过修改现有算子Kernel，体验开发、编译、验证的完整闭环。

4. **[算子调试](#三算子调试)**：掌握算子打印和性能采集方法。

## 一、编译运行

本阶段目的是**快速体验项目标准流程**，验证环境能否成功进行算子源码编译、打包、安装和运行。

### 1. 编译scopy算子

环境准备好后（注意软件与源码版本配套），进入环境并访问项目源码根目录，编译指定算子。

通用编译命令格式：`bash build.sh --pkg --soc=<芯片版本> --ops=<算子名>`。以scopy算子为例，编译命令如下：

```bash
bash build.sh --pkg --soc=${soc_version} --ops=scopy
```

${soc_version}设置方法如下：

访问[CANN下载中心](https://www.hiascend.com/cann/download)，根据页面提示复制硬件查询命令，在当前环境中执行，返回芯片ID信息，再回填到官网按Enter键获取产品名，产品名对应的${soc_version}取值如下，请按实际场景传参。

- Atlas A2 训练系列产品/Atlas A2 推理系列产品：取值为ascend910b
- Atlas A3 训练系列产品/Atlas A3 推理系列产品：取值为ascend910_93
- 950系列产品：取值为ascend950
- Ascend 310P 系列产品（如 Atlas 300I 推理卡）：取值为ascend310p

<img src="./zh/figures/socInfo.png" alt="芯片版本" width="800px" height="160px">

若提示如下信息，说明编译成功。
```bash
Self-extractable archive "cann-<芯片名>-ops-blas_${cann_version}_linux-${arch}.run" successfully created.
```
编译成功后，run包存放于项目根目录的build_out目录下。文件名中的 <芯片名> 随 --soc 参数变化（如 ascend910b 对应 910b、ascend950 对应 950），可直接用通配 cann-*-ops-blas_*.run 定位安装包。

### 2. 安装scopy算子包

> **说明**：run包必须通过`--install`参数执行安装，不带参数仅显示帮助信息不会安装。可通过`--install-path=<路径>`指定安装目录，默认安装路径：root用户为`/usr/local/Ascend`，普通用户为`~/Ascend`。
```bash
./build_out/cann-*-ops-blas_*.run --install --quiet
```

### 3. 快速验证：运行算子样例

通用的运行命令格式：`bash build.sh --soc=${soc_version} --ops=<算子名> --run`。

以scopy算子为例，其提供了简单算子样例`test/copy/scopy/arch22/scopy_test.cpp`，运行该样例验证算子功能是否正常。

```bash
bash build.sh --pkg --soc=${soc_version} --ops=scopy --run
```
预期输出：打印算子`scopy`的计算结果，表明算子已成功部署并正确执行。

```
Running scopy_test...
Output: 1.2 1.2 1.2 1.2 1.2 1.2 1.2 1.2 ...
Golden: 1.2 1.2 1.2 1.2 1.2 1.2 1.2 1.2 ...
[Success] Case accuracy is verification passed.
```

> **提示**：若已完成编译且仅需重新验证功能，可直接运行已编译的二进制文件，无需再次编译。编译完成后，测试二进制文件位于`build/test/<家族>/<算子名>/<算子名>_test`，例如scopy算子的测试文件为`build/test/copy/scopy/scopy_test`，直接执行即可：
>
> ```bash
> ./build/test/copy/scopy/scopy_test
> ```

## 二、算子开发

本阶段目的是对已成功运行的scopy算子尝试**修改核函数代码**。

### 1. 修改Kernel实现

找到scopy算子的核心kernel实现文件`blas/copy/arch22/scopy_kernel.cpp`。scopy的搬运分两条路径：

- `SingleIteration`：每次搬运一个完整分块（90KB / 4B = 23040个元素），仅在单核数据量达到23040个元素时执行；
- `SingleIterationAligned`：搬运不足一个分块的剩余元素，**默认样例（16384个元素、8核，每核2048个元素）只会走这条路径**。

以`SingleIterationAligned`为例，尝试注释掉其中的输出搬运`DataCopyPad`操作：

```cpp
template <typename T>
__aicore__ inline void CopyAIV<T>::SingleIterationAligned(uint32_t curOffset, uint32_t dataCount)
{
    uint8_t paddingNum = elementsPerBlock - dataCount % elementsPerBlock;
    DataCopyExtParams copyParams{1, dataCount * BYTENUM_PER_FLOAT32, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, paddingNum, 0};

    LocalTensor<T> inLocal = inQueue.AllocTensor<T>();
    DataCopyPad(inLocal, inGM[curOffset], copyParams, padParams);
    inQueue.EnQue<T>(inLocal);

    int32_t eventIDMTE2ToMTE3 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_MTE3));
    AscendC::SetFlag<AscendC::HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);
    AscendC::WaitFlag<AscendC::HardEvent::MTE2_MTE3>(eventIDMTE2ToMTE3);

    LocalTensor<T> outLocal = inQueue.DeQue<T>();
    // 注释掉输出搬运，重新编译运行，观察样例的比对结果
    // DataCopyPad(outGM[curOffset], outLocal, copyParams);
    inQueue.FreeTensor(outLocal);

    int32_t eventIDMTE3ToMTE2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
    AscendC::SetFlag<AscendC::HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
    AscendC::WaitFlag<AscendC::HardEvent::MTE3_MTE2>(eventIDMTE3ToMTE2);
}
```

### 2. 编译与验证

重复[编译运行](#一编译运行)章节中的步骤：

1. **重新编译**：
    先回到项目根目录，编译命令如下：

    ```bash
    bash build.sh --pkg --soc=${soc_version} --ops=scopy
    ```

2. **重新安装**：
    ```bash
    ./build_out/cann-*-ops-blas_*.run --install --quiet
    ```

3. **重新验证**：
    ```bash
    bash build.sh --soc=${soc_version} --ops=scopy --run
    ```

4. **预期结果**：注释掉输出搬运后，样例会打印比对失败，说明修改已编译生效，且测试能够捕获问题（样例的输出向量初始化为2.3、输入向量为1.2，输出搬运被注释后输出保持初始值2.3，与期望值1.2不一致）：
    ```
    Running scopy_test...
    Output: 2.3 2.3 2.3 2.3 2.3 2.3 2.3 2.3 ...
    Golden: 1.2 1.2 1.2 1.2 1.2 1.2 1.2 1.2 ...
    [Failed] Case accuracy is verification failed!
    ```

5. **恢复验证**：将注释恢复为原代码后重新编译、安装、运行，输出恢复为精度比对成功，完成「修改→编译→验证」闭环：
    ```
    Running scopy_test...
    Output: 1.2 1.2 1.2 1.2 1.2 1.2 1.2 1.2 ...
    Golden: 1.2 1.2 1.2 1.2 1.2 1.2 1.2 1.2 ...
    [Success] Case accuracy is verification passed.
    ```

> **说明**：样例中scopy与ccopy用例复用同一份kernel，注释后两个用例都会打印比对失败，属预期现象。
>
> 若改为注释`SingleIteration`中的输出搬运，默认样例（每核2048个元素）不经过该路径，测试仍会打印`[Success]`；需要把样例数据量放大到单核达到23040个元素（如`test/copy/scopy/arch22/scopy_test.cpp`中8核合计达到184320个元素）才能观察到影响。

## 三、算子调试
本阶段以scopy算子为例，在算子中添加打印并采集算子性能数据，以便后续问题分析定位。

### 1. 打印
算子如果出现执行失败、精度异常等问题，添加打印进行问题分析和定位。

请在`blas/copy/arch22/scopy_kernel.cpp`中进行代码修改（若你已在「二、算子开发」中注释过输出搬运，请先恢复代码再继续本节）。下面两个示例分别加在`Process()`与`SingleIterationAligned()`中，两者都会被默认样例执行到；`SingleIteration`仅在单核数据量达到23040个元素时执行，如需在该路径加打印请先放大样例数据量。

* **printf**

  该接口支持打印Scalar类型数据，如整数、字符型、布尔型等，详细介绍请参见[《Ascend C API》](https://hiascend.com/document/redirect/CannCommunityAscendCApi)中“算子调测API > printf”。在`CopyAIV<T>::Process()`中添加：

  ```cpp
  // 打印当前核的数据量与起始偏移
  AscendC::PRINTF("Block %d: n is %u, calNum is %u, startOffset is %u\n", this->vecIdx, this->n, this->calNum,
                  this->startOffset);
  ```
* **DumpTensor**

  该接口支持Dump指定Tensor的内容，同时支持打印自定义附加信息，比如当前行号等，详细介绍请参见[《Ascend C API》](https://hiascend.com/document/redirect/CannCommunityAscendCApi)中“算子调测API > DumpTensor”。在`SingleIterationAligned()`中已有语句`LocalTensor<T> outLocal = inQueue.DeQue<T>();`之后添加一行：

  ```cpp
  // 打印outLocal前32个元素的信息
  AscendC::DumpTensor(outLocal, 0, 32);
  ```
### 2. 性能采集

当算子功能验证正确后，可通过`msprof`工具采集算子性能数据。

-  **生成可执行文件**

    调用scopy算子的test样例，生成可执行文件（scopy_test），该文件位于项目`ops-blas/build/test/copy/scopy`目录。
    ```bash
    bash build.sh --soc=${soc_version} --ops=scopy
    ```

-  **采集性能数据**

    进入scopy算子可执行文件目录`ops-blas/build/test/copy/scopy`，执行如下命令：

    ```bash
    msprof --application="./scopy_test"
    ```
采集结果在项目`ops-blas/build/test/copy/scopy`目录，msprof命令执行完后会自动解析并导出性能数据结果文件，详细内容请参见[msprof](https://www.hiascend.com/document/detail/zh/mindstudio/82RC1/T&ITools/Profiling/atlasprofiling_16_0110.html#ZH-CN_TOPIC_0000002504160251)。
