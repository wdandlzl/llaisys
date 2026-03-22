# LLAISYS：基于 MPI 的 CPU 张量并行实现

> 这是我在 **LLAISYS** 项目基础上做的一个课程/练习性质的扩展实现。  
> 这次主要完成的内容是：**在 CPU 场景下引入 MPI，支持将模型分片到多个进程上进行分布式推理**。

## 1. 项目说明

LLAISYS（Let's Learn AI SYStem）是一个偏教学性质的 AI 系统项目，目标是帮助学习者从底层理解 AI 推理系统的组成，包括：

- Tensor 与内存组织
- 基础算子实现
- Runtime / Device 抽象
- Python 前端调用 C++ 后端
- 大模型推理过程

我这次在原项目基础上，尝试补充了一个相对完整的 **CPU 分布式推理版本**，重点放在 **张量并行（Tensor Parallelism）** 上。

## 2. 我完成的主要工作

这次改动的目标是：

**把一个 Qwen2 系列模型按张量维度切分到多个 CPU 进程上，通过 MPI 完成进程间通信，实现分布式推理。**

目前已经完成并成功运行的部分有：

1. **引入 MPI 分布式接口**
   - 增加了 rank / world size 获取
   - 增加了 barrier / init / finalize 等接口
   - 让 Python 侧也能调用这些接口

2. **在 CPU 后端实现张量并行**
   - `q_proj / k_proj / v_proj / gate_proj / up_proj` 按输出维切分
   - `o_proj / down_proj` 按输入维切分
   - 局部结果通过 `MPI_Allreduce` 汇总

3. **KV Cache 分片**
   - 按 `num_key_value_heads` 对 KV cache 做分片
   - 每个 rank 只保存自己负责的那一部分 KV

4. **模型权重按 rank 加载**
   - Python 侧在读取 safetensors 时，只加载当前 rank 需要的 shard
   - 避免每个进程都加载一整份完整权重

5. **测试脚本适配 MPI**
   - 增加了 `--skip_hf` 选项，便于只测试 LLAISYS 自己的推理路径
   - 避免多进程下每个 rank 都额外加载一份 Hugging Face 模型，减少内存占用
   - 控制只有 root rank 打印主要输出

## 3. 我对张量并行的理解

我对这次实现的理解比较朴素，核心思想就是：

### 3.1 为什么可以做张量并行

在线性层中，本质上是矩阵乘法。如果把权重矩阵按某个维度切开，那么不同进程就可以分别计算自己负责的那一部分，最后再把结果拼接或求和。

例如：

- 对 `q/k/v` 这类投影层，可以按输出通道切分
- 每个进程各算一部分 head
- 最后继续在各自的 head 上做 attention

而对于：

- `o_proj`
- `down_proj`

这类层，它们需要把各个分片的结果聚合回来，所以需要做一次通信，这里我使用的是 `MPI_Allreduce`。

### 3.2 这次实现里通信发生在哪里

我这次实现中，通信主要发生在下面两类位置：

1. **分片结果需要合并时**  
   例如 `o_proj/down_proj` 的局部输出，需要通过 `MPI_Allreduce` 汇总。

2. **推理流程需要同步时**  
   例如模型初始化之后，或者某些测试阶段，需要所有 rank 走到同一位置，这时用 `barrier`。

### 3.3 这种实现的优点和不足

优点：

- 可以让单机多核 / 多进程一起参与推理
- 能更直观理解分布式推理和张量并行的原理
- 对学习 MPI 和大模型推理结构很有帮助

不足：

- 目前只做了 **CPU + MPI**
- 通信开销还没有做更细的优化
- 代码风格和工程细节还有改进空间
- 目前更偏“教学/实验实现”，不是生产级优化版本

## 4. 关键修改点

我主要修改了下面这些部分：

- `src/llaisys/distributed.cc`  
  增加 MPI 初始化、同步、rank/world size 查询等接口。

- `src/llaisys/qwen2.cc`  
  修改 Qwen2 推理路径，使其支持 CPU 张量并行。

- `python/llaisys/` 相关代码  
  增加 Python 对分布式接口的封装，并在加载权重时支持按 rank 切片。

- `test/test_infer.py`  
  适配 MPI 测试流程。

- `xmake.lua`  
  增加 MPI 构建选项，并修复了 MPI 头文件和链接库的问题。

## 5. 运行环境

我本地测试使用的是类似下面的环境：

- Ubuntu / WSL2
- Python 3.10
- xmake
- GCC / G++
- OpenMPI
- PyTorch
- transformers
- safetensors

安装 OpenMPI（Ubuntu）可以参考：

```bash
sudo apt update
sudo apt install -y openmpi-bin libopenmpi-dev
```

## 6. 编译方法

### 6.1 配置并编译

```bash
xmake f -c --mpi=y
xmake
xmake install
```

### 6.2 安装 Python 包

建议在项目根目录下执行：

```bash
pip install -e python
```

## 7. 运行方法

### 7.1 单进程测试

```bash
python test/test_infer.py \
  --device cpu \
  --model /your/model/path \
  --skip_hf
```

### 7.2 MPI 多进程测试

```bash
mpirun -np 2 python test/test_infer.py \
  --device cpu \
  --model /your/model/path \
  --skip_hf \
  --max_steps 16
```

如果想做更严格的对照测试，也可以不加 `--skip_hf`，但这样会占用更多内存。

## 8. 运行时需要注意的条件

为了保证当前这版张量并行能够正常运行，需要下面几个参数能被进程数整除：

- `num_attention_heads`
- `num_key_value_heads`
- `intermediate_size`

也就是说，如果使用 `mpirun -np 2`，那么这些维度最好都能被 2 整除。

## 9. 我遇到的问题

这次实现过程中，我遇到的几个主要问题有：

### 9.1 MPI 头文件找不到

一开始编译时报 `mpi.h: No such file or directory`，后来发现是：

- 系统没有安装 MPI 开发包
- 构建脚本也没有正确把 MPI 的 include/link 参数传给编译器

最后通过安装 `libopenmpi-dev`，并在 `xmake.lua` 中解析 `mpicxx --showme:compile` 和 `mpicxx --showme:link` 解决了。

### 9.2 Python 导入共享库失败

中间还出现过：

- `undefined symbol: _ZN3MPI8Datatype4FreeEv`
- `undefined symbol: ompi_mpi_op_sum`

这些问题本质上是 **MPI C++ bindings 和动态链接没有处理干净**。

后来通过：

- 在 `distributed.cc` 中跳过旧的 MPI C++ bindings
- 在 `xmake.lua` 中把 MPI 的 linkdirs / links 正确加入共享库依赖

最终解决了这个问题。

### 9.3 多进程测试时内存占用过大

一开始 `mpirun` 时，每个 rank 都会加载一份 Hugging Face 模型做参考推理，导致内存压力很大，甚至被系统直接 kill。

后来我增加了 `--skip_hf` 选项，先只验证自己实现的 MPI 推理路径，问题就缓解了很多。

## 10. 当前结果

目前这版代码已经可以：

- 正常编译通过
- 正常导入 Python 包
- 成功链接 MPI 动态库
- 使用 `mpirun -np 2` 进行 CPU 分布式推理

对我来说，这个结果说明：

- 张量并行的基本思路已经跑通
- MPI 通信链路已经接通
- 模型分片加载与推理主流程能够工作

## 11. 后续还能继续改进的地方

如果后面还有时间，我觉得还可以继续完善：

1. 支持 **GPU + NCCL** 路径
2. 对通信过程做更细的性能优化
3. 增加更多单元测试和一致性测试
4. 减少 Python 侧和 C++ 侧之间的一些重复逻辑
5. 做更完整的 benchmark，对比单进程和多进程速度

## 12. 仓库说明

这个仓库是我自己基于 LLAISYS 做的学习和实现记录，主要是为了：

- 练习阅读 C++ / Python 混合工程
- 理解大模型推理系统的基本结构
- 动手实现一次比较基础的张量并行

如果老师或同学看到这个仓库，欢迎指出我的问题。我目前的理解和工程能力都还在学习过程中，所以 README 和代码里如果有不严谨的地方，我后面也会继续修改。

## 13. 致谢

感谢原始 LLAISYS 项目提供的整体框架和学习材料，让我能够在这个基础上继续做分布式推理方面的尝试。

---

## 附：常用命令

### 编译

```bash
xmake f -c --mpi=y
xmake
xmake install
pip install -e python
```

### 运行

```bash
mpirun -np 2 python test/test_infer.py \
  --device cpu \
  --model /home/wang/models/DeepSeek-R1-Distill-Qwen-1.5B \
  --skip_hf \
  --max_steps 16
```

### 检查 MPI 是否正确链接

```bash
ldd python/llaisys/libllaisys/libllaisys.so | grep mpi
```