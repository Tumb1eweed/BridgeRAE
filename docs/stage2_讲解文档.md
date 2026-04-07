# Stage2的讲解文档

这份文档整理的是前面按“像讲 Stage-1 那样，一步一步拆”的 Stage-2 讲解。内容从 Stage-2 的整体目标开始，依次讲到：partial/complete 的编码对齐、bridge 中间状态采样、transport 模型、Euler 积分、多损失训练、denormalize 接 decoder、decoder 在 Stage-2 里的训练方式，以及推理时完整主路径。

相关源码主要在：

- [latent_transport.py](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py)
- [stage2_train.py](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py)
- [latent_normalizer.py](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_normalizer.py)
- [completion_decoder.py](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py)
- [pointmae_encoder.py](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/pointmae_encoder.py)

---

## 第零步：Stage-2 到底在做什么

Stage-1 学的是：

`complete points -> encoder -> complete latent -> decoder -> complete points`

也就是说，Stage-1 的 decoder 只学了一件事：

“如果你给我一个像样的 complete latent，我就把它解码成完整点云。”

但是 Stage-1 并没有解决最关键的问题：

**partial latent 怎么变成 complete latent？**

这就是 Stage-2 的任务。

### Stage-2 的核心目标

Stage-2 学的是：

`partial points -> encoder -> partial latent -> transport -> complete-like latent -> decoder -> complete points`

也就是：

1. 先把残缺点云编码成 latent
2. 再在 latent 空间里，把这个残缺 latent 推到完整 latent 的分布附近
3. 最后把它交给 Stage-1 已经训练好的 decoder

所以 Stage-2 不是直接在点云坐标上“补洞”，而是在 **latent 空间里做补全**。

### 为什么要分成两阶段

因为如果你一上来就想学：

`partial points -> complete points`

这个映射太难了。

原因是中间混着很多任务：

1. 你要理解 partial 的局部几何
2. 你要猜缺失部分
3. 你要学会完整形状的生成规律
4. 你还要把这些规律变成稠密点云

这会让一个模型同时做太多事。

所以这里拆成两步：

### Stage-1
先单独把“完整形状生成器”练好

也就是：
`complete latent -> complete points`

### Stage-2
再单独学“怎么把 partial latent 变成 complete latent”

也就是：
`partial latent -> complete latent`

这样职责就很清楚。

你可以把它理解成：

- Stage-1 先训练一个很会“雕完整形状”的老师傅
- Stage-2 再训练一个“搬运 latent 的人”，把残缺 latent 搬到老师傅看得懂的位置

### Stage-2 和 Stage-1 最大的结构区别

Stage-1 的主角是：

- 冻结 encoder
- 可训练 decoder

Stage-2 的主角变成：

- 冻结 encoder
- 加载 Stage-1 的 decoder
- 新训练一个 `transport model`

代码在：
[stage2_train.py#L275](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py#L275)
[stage2_train.py#L276](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py#L276)
[stage2_train.py#L278](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py#L278)

```python
encoder = PointMAEEncoder(..., freeze=True).to(device)
decoder, normalizer = load_frozen_decoder(args.stage1_ckpt, device, output_points=args.num_complete_points)
transport = LatentTransportModel(hidden_dim=384, depth=6, num_heads=6).to(device)
```

所以 Stage-2 一上来就把 Stage-1 训练好的 decoder 载入了。

这很关键，因为它说明：

**Stage-2 默认相信 decoder 已经会“完整形状生成”了。**

现在只差把 partial latent 送到正确位置。

### Stage-2 里 decoder 是不是完全冻结的

默认不是完全死冻，但它的基础角色是“从 Stage-1 继承来的完整形状解码器”。

代码里先把它从 Stage-1 checkpoint 读出来：
[stage2_train.py#L73](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py#L73)

然后默认配置会只微调 decoder 的后几层，见：
[stage2_train.py#L96](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py#L96)

默认参数是：

- `decoder-train-mode = last_n`
- `decoder-train-last-n = 2`

也就是说，Stage-2 的重点还是 transport，decoder 只是轻微配合一下，不是从头重学。

你可以把这个理解成：

老师傅已经会干活了，Stage-2 不会把他整个重练一遍，只是让他稍微适应一下新的 latent 输入。

### Stage-2 最关键的新东西是什么

就是这个：

[latent_transport.py](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py)

它学的不是“直接输出完整点云”，而是学一个 **latent 里的速度场** 或 **运输规则**。

也就是：

“当前 latent 在这个时间点，应该往哪个方向走一点，才能逐步从 partial latent 走向 complete latent。”

所以 Stage-2 不是一步跳过去，而是：

- 先定义起点：partial latent
- 再定义终点：complete latent
- 然后学习一条中间的桥

这也是它和“薛定谔桥/bridge”有关的地方。

### 先用最白话的话理解 Stage-2

你可以把 Stage-2 想成这样：

- Stage-1 教会了 decoder：什么样的 latent 能解码出完整飞机
- 但 partial 点云编码出来的 latent 还不是这种“完整飞机 latent”
- 所以 Stage-2 要学会：怎么把“残缺飞机 latent”一步步推成“完整飞机 latent”

就像：

- Stage-1 训练好了一个会根据“完整图纸”造房子的工人
- Stage-2 的任务不是重新训练工人
- Stage-2 的任务是把“缺页的草图”整理、补齐、搬运成“工人能看懂的完整图纸”

### Stage-2 的总数据流先提前看一眼

后面我们会一步一步拆，但你先有个大框架：

`partial_points: (B,2048,3)`
`-> encoder`
`src_tokens: (B,64,384)`

`complete_points: (B,8192,3)`
`-> encoder(with same centers)`
`tgt_tokens: (B,64,384)`

然后：

`src_tokens -> transport -> pred_tokens`
`pred_tokens -> decoder -> pred_complete_points`

最后和：

`complete_points`

算重建 loss。

这里最关键的一个新点是：

**Stage-2 会同时编码 partial 和 complete，而且 complete 会复用 partial 的 centers。**

这个点后面我会单独详细讲，因为它非常重要。

### 一句总结这一小步

Stage-2 的本质不是“再训练一个补全 decoder”，  
而是“在 Stage-1 已经学好的完整形状解码器前面，再插入一个 latent transport 模块，让 partial latent 能走到 complete latent 的分布上”。

---

## 第一步：partial 和 complete 怎么编码，为什么 complete 要复用 partial 的 centers

核心代码在：
[stage2_train.py#L357](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py#L357)
[stage2_train.py#L358](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py#L358)
[stage2_train.py#L359](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py#L359)

```python
src = encoder(partial_points)
tgt = encoder(complete_points, centers=src.centers)
src_tokens = normalizer.normalize(src.tokens) if normalizer is not None else src.tokens
tgt_tokens = normalizer.normalize(tgt.tokens) if normalizer is not None else tgt.tokens
```

### 1. Stage-2 一开始拿到什么输入

训练时 batch 里会同时有：

`partial_points: (B, 2048, 3)`
`complete_points: (B, 8192, 3)`  
如果是 PCN，完整点可能是 `(B, 16384, 3)`

也就是说，Stage-2 每次看到的是一对：

- 残缺点云
- 对应的完整点云

这和 Stage-1 不一样。  
Stage-1 只看 `complete_points`，Stage-2 一上来就同时看两边。

### 2. 先编码 partial

第一句：

```python
src = encoder(partial_points)
```

这一步和 Stage-1 里 encoder 的流程一样：

`partial_points: (B,2048,3)`
`-> Group`
`centers: (B,64,3)`
`neighborhoods: (B,64,32,3)`
`-> LocalEncoder + Transformer`
`src.tokens: (B,64,384)`

所以 partial 编码后得到：

`src.centers: (B,64,3)`
`src.tokens: (B,64,384)`

这里的 `src` 可以理解成 source，也就是桥的起点。

### 3. 再编码 complete，但这里有个非常关键的操作

第二句：

```python
tgt = encoder(complete_points, centers=src.centers)
```

注意这里不是普通的：

`encoder(complete_points)`

而是明确传了：

`centers=src.centers`

这意味着：

**complete 点云在编码时，不再自己重新做 FPS 选中心点，而是直接复用 partial 那边选出来的 64 个 centers。**

这个设计非常关键。

### 4. 为什么不能让 partial 和 complete 各自独立选 centers

如果各选各的，就会这样：

partial 可能选出一组中心：
`centers_partial = (B,64,3)`

complete 也会选出另一组中心：
`centers_complete = (B,64,3)`

虽然形状一样，但这 64 个位置通常不一一对应。

那后面就会出一个大问题：

`src_tokens[:, i, :]` 和 `tgt_tokens[:, i, :]`

不一定表示同一个空间区域。

比如：

- partial 的第 17 个 token 可能对应机翼前缘
- complete 的第 17 个 token 可能对应机尾附近

那你还怎么学：

`partial latent -> complete latent`

这就像你想学“把 A 地图改成 B 地图”，结果两张地图的格子编号都没对齐。

所以必须让两边 token 尽量空间对齐。

### 5. 复用 partial centers 后，会发生什么

现在 complete 编码时，grouping 不再自己采样中心，而是用：

`src.centers: (B,64,3)`

对应 encoder 里的逻辑是：
[pointmae_encoder.py#L245](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/pointmae_encoder.py#L245)
[pointmae_encoder.py#L248](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/pointmae_encoder.py#L248)
[pointmae_encoder.py#L249](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/pointmae_encoder.py#L249)

```python
if centers is None:
    neighborhoods, centers = self.group_divider(points)
else:
    neighborhoods, centers = self.group_divider.forward_with_centers(points, centers)
```

也就是说：

- partial：自己选 64 个 center
- complete：沿用 partial 的 64 个 center，再各自去 complete 点云里找邻居

于是 complete 这边会得到：

`tgt.centers = src.centers`
`tgt.tokens: (B,64,384)`

这样 `src_tokens` 和 `tgt_tokens` 的第 `i` 个 token，就大致都在描述同一个空间位置附近的局部结构。

### 6. 这个“对齐”到底有多重要

非常重要，几乎是 Stage-2 能不能学好的基础。

因为后面 bridge 里最关键的监督之一是：

`velocity_target = tgt_tokens - src_tokens`

见：
[stage2_train.py#L175](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py#L175)

如果 `src_tokens` 和 `tgt_tokens` 没对齐，那这个减法就会变得没有明确几何意义。

但如果两边都对应同一组 centers，那么：

`tgt_tokens - src_tokens`

就可以理解成：

“同一个空间局部，从 partial 表示变成 complete 表示，需要补上的 latent 差值”

这就很有意义了。

### 7. 你可以把它类比成什么

可以把 partial 和 complete 都想成同一个城市的两张地图。

- partial 是残缺地图
- complete 是完整地图

如果你想比较两张地图上“同一个地块”的差别，最重要的是先把网格对齐。

这里的 `src.centers` 就像那套统一网格。

于是：

- partial token i 表示第 i 个网格里的残缺信息
- complete token i 表示第 i 个网格里的完整信息

这样 transport 才知道自己是在补哪个位置，不是在乱搬。

### 8. 然后再做 normalizer

接下来两句：

```python
src_tokens = normalizer.normalize(src.tokens) if normalizer is not None else src.tokens
tgt_tokens = normalizer.normalize(tgt.tokens) if normalizer is not None else tgt.tokens
```

这和 Stage-1 的 latent normalization 是同一个 normalizer，来自 Stage-1 checkpoint。

所以：

`src.tokens: (B,64,384) -> src_tokens: (B,64,384)`
`tgt.tokens: (B,64,384) -> tgt_tokens: (B,64,384)`

形状不变，只是数值进入了 decoder/transport 熟悉的 latent 坐标系。

这里也再次说明：

Stage-2 不是另起一套 latent 空间，  
而是在 Stage-1 已经定义好的 complete latent 坐标系里工作。

### 9. 到这一步结束后，Stage-2 手里有什么

现在已经有：

`src.centers: (B,64,3)`
`src_tokens: (B,64,384)`
`tgt_tokens: (B,64,384)`

语义上分别是：

- `src.centers`：统一的空间锚点
- `src_tokens`：partial latent
- `tgt_tokens`：在同一组锚点下编码出的 complete latent

这三样东西就是后面 bridge/transport 学习的基础。

### 10. 这一小步的完整维度链

`partial_points: (B,2048,3)`
`-> encoder`
`src.centers: (B,64,3)`
`src.tokens: (B,64,384)`

`complete_points: (B,8192,3)`
`-> encoder(with centers=src.centers)`
`tgt.centers: (B,64,3)`  
实际上和 `src.centers` 对齐
`tgt.tokens: (B,64,384)`

再经过 normalizer：

`src_tokens: (B,64,384)`
`tgt_tokens: (B,64,384)`

### 一句白话总结

Stage-2 的第一步不是直接让 partial 去补全，而是先把 partial 和 complete 都编码到 **同一组空间中心对齐的 latent token** 上。  
这样后面的 transport 学的才是“同一局部位置从残缺到完整该怎么变”，而不是在两套乱序 token 之间硬做映射。

---

## 第二步：构造 bridge 中间状态 `bridge_state`

已经有了

`src_tokens: (B,64,384)`  
`tgt_tokens: (B,64,384)`

接下来不会直接把 `src_tokens` 扔进 transport 然后要求一次变成 `tgt_tokens`。  
而是会先在它们中间随机取一个“桥上的中间状态”。

代码在：
[stage2_train.py#L168](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py#L168)

```python
def sample_bridge_state(source_tokens, target_tokens, noise_std):
    batch_size = source_tokens.shape[0]
    t = torch.rand(batch_size, device=source_tokens.device, dtype=source_tokens.dtype)
    interp = (1.0 - t)[:, None, None] * source_tokens + t[:, None, None] * target_tokens
    if noise_std > 0:
        noise_scale = noise_std * torch.sqrt((t * (1.0 - t)).clamp_min(1e-4))[:, None, None]
        interp = interp + noise_scale * torch.randn_like(interp)
    velocity_target = target_tokens - source_tokens
    return interp, velocity_target, t
```

这一步一定要慢慢理解，因为它就是 Stage-2 “桥”的味道最浓的地方。

### 1. 先说它在干什么

它做的是：

给定起点 `src_tokens` 和终点 `tgt_tokens`，  
不让模型只看起点或终点，  
而是随机采样一个中间时刻 `t`，再构造那个时刻对应的中间 latent 状态 `interp`。

所以 transport 学到的不是：

“从头到尾一把跳过去”

而是：

“如果我现在在桥上的某个位置、某个时间，我下一步该往哪里走”

### 2. 先随机采样时间 `t`

代码：

```python
t = torch.rand(batch_size, device=source_tokens.device, dtype=source_tokens.dtype)
```

所以：

`t: (B,)`

这表示：

每个样本都会随机取一个 0 到 1 之间的时间值。

例如某个 batch 里可能有：

- 样本 1：`t = 0.1`
- 样本 2：`t = 0.7`
- 样本 3：`t = 0.45`

这是什么意思？

你可以把桥想成一条从 partial latent 走到 complete latent 的路。

- `t = 0` 表示刚出发，在起点
- `t = 1` 表示到终点
- `t = 0.5` 表示走到一半

所以这个 `t` 就是在问：

“当前这个样本，我现在要把你放在桥上的哪个时刻来训练？”

### 3. 然后做线性插值，得到中间状态 `interp`

代码：

```python
interp = (1.0 - t)[:, None, None] * source_tokens + t[:, None, None] * target_tokens
```

我们来看形状。

原本：

`source_tokens: (B,64,384)`  
`target_tokens: (B,64,384)`  
`t: (B,)`

这里的：

`(1.0 - t)[:, None, None]`

会变成：

`(B,1,1)`

然后和 `(B,64,384)` broadcast 相乘。

所以：

`(B,1,1) * (B,64,384) -> (B,64,384)`

同样：

`t[:, None, None] * target_tokens -> (B,64,384)`

最后两项相加：

`interp: (B,64,384)`

### 4. 这一步语义上是什么意思

这就是一个非常标准的“起点到终点之间的中间状态”。

比如：

- 如果 `t = 0`，那 `interp = source_tokens`
- 如果 `t = 1`，那 `interp = target_tokens`
- 如果 `t = 0.5`，那就是两者正中间
- 如果 `t = 0.2`，那就更靠近 source
- 如果 `t = 0.8`，那就更靠近 target

所以它相当于在 latent 空间里画了一条从 partial 到 complete 的直线，  
然后随机从这条线上取一个点。

### 5. 为什么不是直接学 `src -> tgt`，而要学这些中间状态

因为如果你只训练“起点到终点”的一次映射，模型只会学一个粗暴的整体变换。  
但桥模型想学的是一个 **随时间变化的速度场**。

也就是：

“在不同时间、不同位置，latent 应该怎么流动”

这就要求训练时不能只看起点和终点，  
而要让模型经常看到“桥上的中间位置”。

不然它就像只知道起点和终点，但不会走中间的路。

### 6. 你可以把它类比成什么

想象你在教一个机器人从 A 走到 B。

如果你只告诉它：

- 这是 A
- 那是 B
- 你自己跳过去吧

它可能学不到稳定路径。

但如果你经常把它随机放在：

- 靠近 A 的位置
- 中间位置
- 靠近 B 的位置

然后问它：

“你现在应该往哪走？”

它就更容易学会一条连续的行进规则。

这里的 `interp` 就是在做这种事情。

### 7. 然后为什么还要加噪声

代码：

```python
if noise_std > 0:
    noise_scale = noise_std * torch.sqrt((t * (1.0 - t)).clamp_min(1e-4))[:, None, None]
    interp = interp + noise_scale * torch.randn_like(interp)
```

这里先看形状：

`t * (1.0 - t)` 是 `(B,)`

再 `[:,None,None]` 后变成：

`(B,1,1)`

于是：

`noise_scale: (B,1,1)`

然后：

`torch.randn_like(interp): (B,64,384)`

所以最后加上去的噪声仍然是：

`(B,64,384)`

输出形状不变：

`interp: (B,64,384)`

### 8. 这个噪声为什么要乘 `sqrt(t(1-t))`

这个设计很有意思。

因为：

- 当 `t` 接近 0 时，离起点很近
- 当 `t` 接近 1 时，离终点很近
- 当 `t` 在 0.5 左右时，处在中间最不确定的位置

而：

`t(1-t)`

在 `t=0` 和 `t=1` 时接近 0，  
在 `t=0.5` 时最大。

所以这个噪声安排的意思是：

- 起点附近噪声小
- 终点附近噪声小
- 中间区域噪声大一点

这很符合“桥”的直觉：

中间状态通常更模糊、更不确定，  
所以训练时允许它周围有更多扰动。

### 9. 这噪声在语义上像什么

它相当于不是只让模型学习一条“特别细、特别硬”的直线桥，  
而是让它学会一条带一点厚度的走廊。

也就是：

“你不一定非得恰好站在线上，只要在这条桥附近，也得知道该往哪走”

这会让 transport 学得更稳，不会特别脆。

### 10. 然后定义目标速度 `velocity_target`

代码：

```python
velocity_target = target_tokens - source_tokens
```

所以：

`velocity_target: (B,64,384)`

这一步很关键。

它表示的是：

**从 partial latent 到 complete latent 的总位移方向。**

如果把 latent 当成空间中的点，那：

`target - source`

就是一根从起点指向终点的箭头。

### 11. 为什么这里叫 velocity，不叫 displacement

严格说，从代码看它更像“目标位移方向”，  
因为这里没有再乘时间长度，直接就是：

`target - source`

之所以叫 velocity，是因为整个模型想学的是一个“流动方向场”。

你可以把它理解成：

虽然代码里写的是一根从起点指向终点的箭头，  
但训练目标是让模型在桥上的任意时刻，都能预测“此时应该朝哪个方向流动”。

所以工程上它被当成 velocity target。

### 12. 这一步整体返回了什么

函数最后返回：

`interp`
`velocity_target`
`t`

也就是：

`bridge_state: (B,64,384)`  
`velocity_target: (B,64,384)`  
`t: (B,)`

后面 transport 网络的输入，就是这三样里的前两样再加上 `src.centers`。

### 13. 到这里你可以怎么理解 Stage-2 的训练样本

每个训练样本，现在不只是：

“给你 partial latent 和 complete latent”

而是变成：

“我随机把你放到 partial 和 complete 之间桥上的某个位置 `bridge_state`，再告诉你当前时间 `t`，请你预测应该往哪个方向继续流动，最终朝 complete latent 靠近。”

这个思路和普通 one-shot regression 很不一样。  
它更像是在学习一个动态系统。

### 14. 这一小步的完整维度链

输入：

`src_tokens: (B,64,384)`  
`tgt_tokens: (B,64,384)`

随机采样：

`t: (B,)`

广播后插值：

`(B,1,1) * (B,64,384)`  
`+ (B,1,1) * (B,64,384)`  
`= bridge_state: (B,64,384)`

加噪声后仍然：

`bridge_state: (B,64,384)`

目标速度：

`velocity_target = tgt_tokens - src_tokens`
`-> (B,64,384)`

最终输出：

`bridge_state: (B,64,384)`  
`velocity_target: (B,64,384)`  
`t: (B,)`

### 一句白话总结

这一步是在 latent 空间里，从 partial 和 complete 之间随机抽一个“桥上的中间状态”，再告诉模型当前时间 `t`，让它学习：  
**如果我现在站在桥上的这里，我接下来应该朝哪个方向走。**

---

## 第三步：把 `bridge_state`、`src_tokens`、`centers`、`t` 送进 `LatentTransportModel`

代码在：
[stage2_train.py#L366](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py#L366)

```python
out = transport(bridge_state, src_tokens, src.centers, t)
flow_loss = F.mse_loss(out.velocity, velocity_target)
```

对应模型 forward 在：
[latent_transport.py#L123](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L123)

```python
def forward(self, state_tokens, source_tokens, centers, time_steps):
```

这一步你可以把它理解成：

**transport 模型要根据“当前位置”“起点是什么”“空间位置在哪”“现在是什么时刻”，预测当前该往哪个方向流动。**

### 1. 先分清四个输入各自代表什么

#### `state_tokens: (B,64,384)`
这就是刚才采样出来的 `bridge_state`

表示：
**我现在在桥上的哪里**

不是起点，也不是终点，而是当前状态。

#### `source_tokens: (B,64,384)`
这就是 `src_tokens`

表示：
**我是从哪里出发的**

也就是 partial latent。

这个输入很关键，因为 transport 不是无条件流动。  
它始终记得自己的起点是谁。

#### `centers: (B,64,3)`
这就是 partial 那边那组统一的空间中心点。

表示：
**每个 token 在物体空间里的锚点位置**

也就是这个 latent token 对应的是机翼附近、椅背附近，还是别的位置。

#### `time_steps: (B,)`
这就是 `t`

表示：
**我现在在桥上的哪个时刻**

同样的 latent 状态，在不同时间点，理想的流动方向可能不同。  
所以时间信息必须显式喂进去。

### 2. transport 的整体目标是什么

它最终要输出的是：

`velocity: (B,64,384)`

代码在：
[latent_transport.py#L136](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L136)

```python
velocity = self.velocity_head(self.norm(x, time_cond))
```

这个 `velocity` 的意思就是：

**当前每个 token 应该朝 latent 空间里的哪个方向移动**

所以 transport 不是直接输出 complete latent，  
而是输出“现在该往哪走”。

### 3. 第一步：先把时间 `t` 变成时间 embedding

代码在：
[latent_transport.py#L130](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L130)

```python
time_cond = self.time_embed(time_steps)  # (B, D)
```

具体模块在：
[latent_transport.py#L20](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L20)

```python
class TimeEmbedding(nn.Module):
```

输入是：

`t: (B,)`

输出是：

`time_cond: (B,384)`

### 4. 这个时间 embedding 是怎么变的

代码核心在：
[latent_transport.py#L33](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L33)

```python
t_proj = t[:, None].float() * self.freqs[None, :]
emb = torch.cat([t_proj.sin(), t_proj.cos()], dim=-1)
return self.mlp(emb)
```

先别被公式吓到，直觉上它在做：

- 不直接把一个标量 `t` 喂给网络
- 而是把它展开成一串更丰富的时间特征
- 再过一个小 MLP 变成 384 维

这和 Transformer 里做位置编码很像。  
目的就是让网络更容易区分：

- 早期时刻
- 中期时刻
- 后期时刻

### 5. 为什么不能直接把一个标量 `t` 拼进去

因为一个标量太弱了。

如果你只给网络一个单独数字 `0.37`，它很难把时间的不同阶段编码得足够丰富。  
做成高维 embedding 后，网络更容易学：

“在早期应该怎么流”
“在中期应该怎么流”
“在后期应该怎么收敛到终点”

所以 `time_cond: (B,384)` 本质上就是：

**当前时间的高维描述向量**

### 6. 第二步：对 `centers` 再做一次位置编码

代码在：
[latent_transport.py#L131](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L131)

```python
center_embed = self.center_pos_embed(centers)
```

模块定义在：
[latent_transport.py#L88](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L88)

```python
self.center_pos_embed = nn.Sequential(
    nn.Linear(3, 128),
    nn.GELU(),
    nn.Linear(128, hidden_dim),
)
```

所以：

`centers: (B,64,3) -> center_embed: (B,64,384)`

这和 encoder/decoder 里给 center 做位置编码的逻辑很像。

作用是：

**让 transport 不只看 latent 值，还知道这个 latent token 对应空间中的哪里。**

因为不同位置缺失模式不一样。  
比如飞机机翼附近和机身中段，latent 修正方式通常不同。

### 7. 第三步：构造两路输入 `x` 和 `cond`

代码在：
[latent_transport.py#L132](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L132)
[latent_transport.py#L133](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L133)

```python
x = self.input_proj(state_tokens) + center_embed
cond = self.cond_proj(source_tokens) + center_embed
```

这里有两个非常关键的量：

#### `x`
来自当前状态 `state_tokens`

也就是：
**我现在站在哪**

形状：

`state_tokens: (B,64,384)`
`-> input_proj`
`-> (B,64,384)`
`+ center_embed`
`-> x: (B,64,384)`

#### `cond`
来自起点 `source_tokens`

也就是：
**我是从哪出发的**

形状：

`source_tokens: (B,64,384)`
`-> cond_proj`
`-> (B,64,384)`
`+ center_embed`
`-> cond: (B,64,384)`

### 8. 为什么要同时有 `x` 和 `cond`

这一步特别重要。

如果只给模型 `state_tokens`，那模型只知道：

“我现在在这里”

但它不知道：

“我是从哪个 partial latent 走过来的”

而 bridge/transport 是一个 **条件流动问题**。  
当前状态相似，并不代表起点相同。

所以这里显式保留一条 `cond` 分支，让模型始终知道起点是谁。

你可以把它理解成：

- `x` 是当前状态
- `cond` 是身份证明，告诉你你来自哪个 partial 样本

### 9. 为什么 `x` 和 `cond` 都要加 `center_embed`

因为不管是当前状态还是起点条件，  
它们都对应同一组空间 token。

给两边都加位置编码，就是在强调：

“这个 token 不只是一个 384 维 latent，它还代表空间中的第 i 个局部位置”

所以 transport 看到的不是抽象无位置的 token 序列，  
而是带空间锚点的 token 序列。

### 10. 到这里为止，transport 真正准备好的输入是什么

现在已经有：

`time_cond: (B,384)`
`x: (B,64,384)`
`cond: (B,64,384)`

接下来这三样会进入一串 `LatentTransportBlock`。

代码在：
[latent_transport.py#L134](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L134)

```python
for block in self.blocks:
    x = block(x, cond, time_cond)
```

注意：

- 被不断更新的是 `x`
- `cond` 基本保持不变，作为条件输入
- `time_cond` 也保持不变，作为时间调制信号

这和 decoder 里“query 被更新、context 基本不变”有点像。

### 11. 一句最白话的理解

到 transport block 之前，模型已经把问题整理成了：

- `x`：我现在在桥上的这个位置
- `cond`：我是从这个 partial latent 出发的
- `time_cond`：我现在处在桥上的这个时刻
- `center_embed` 已经加进去了：我还知道当前 token 在物体的哪个空间位置

所以 transport 后面做的事，本质上就是：

**根据“当前位置 + 起点条件 + 时间 + 空间位置”，预测此刻每个 token 该往哪流。**

### 12. 这一小步的完整维度链

输入：

`bridge_state: (B,64,384)`
`src_tokens: (B,64,384)`
`centers: (B,64,3)`
`t: (B,)`

变换后：

`t -> time_embed -> time_cond: (B,384)`

`centers -> center_pos_embed -> center_embed: (B,64,384)`

`bridge_state -> input_proj -> (B,64,384)`
`+ center_embed -> x: (B,64,384)`

`src_tokens -> cond_proj -> (B,64,384)`
`+ center_embed -> cond: (B,64,384)`

然后送进 block：

`block(x, cond, time_cond)`

### 一句总结这一步

transport 真正吃进去的不是单独一个 latent，而是四类信息一起构成的状态：

**当前在哪、从哪出发、现在几点、对应空间哪里。**

---

## 第四步：`LatentTransportBlock` 里具体做什么

代码在：
[latent_transport.py#L58](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L58)

```python
class LatentTransportBlock(nn.Module):
```

它的 forward 在：
[latent_transport.py#L69](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L69)

```python
def forward(self, x, cond, time_cond):
    x = x + self.self_attn(self.self_norm(x, time_cond))
    x = x + self.cross_attn(self.cross_norm_q(x, time_cond), self.cross_norm_ctx(cond))
    x = x + self.mlp(self.mlp_norm(x, time_cond))
    return x
```

你一眼看上去会发现：

它长得很像 decoder block。  
也是三步：

1. self-attention
2. cross-attention
3. MLP

但它和普通 Transformer block 最大的不同是：

**这里不是普通 LayerNorm，而是 `AdaLN`，也就是带条件的自适应归一化。**

### 1. 先看 block 的三个输入

输入是：

`x: (B,64,384)`  
`cond: (B,64,384)`  
`time_cond: (B,384)`

分别表示：

- `x`：当前桥上的状态
- `cond`：起点 partial latent 条件
- `time_cond`：当前时刻的时间 embedding

输出还是：

`x: (B,64,384)`

也就是说：

这个 block 不改变形状，只不断更新“当前状态”。

### 2. 先讲最关键的 `AdaLN`

代码在：
[latent_transport.py#L41](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L41)

```python
class AdaLN(nn.Module):
```

forward 在：
[latent_transport.py#L52](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L52)

```python
def forward(self, x, cond):
    scale, shift = self.proj(cond).unsqueeze(1).chunk(2, dim=-1)
    return self.norm(x) * (1 + scale) + shift
```

这里输入是：

`x: (B,N,D)`  
`cond: (B,D)`

在当前 transport 里就是：

`x: (B,64,384)`  
`cond: time_cond = (B,384)`

### 3. AdaLN 到底在干什么

普通 LayerNorm 是：

先把 `x` 归一化，  
然后用固定的可学习参数做缩放和平移。

但 AdaLN 不一样。

它的缩放和平移，不是固定参数，  
而是 **由条件向量动态生成的**。

这里条件向量就是 `time_cond`。

所以可以理解成：

- 同样一个 `x`
- 如果现在是早期时间 `t`
- 和现在是后期时间 `t`
- 归一化后的调制方式不一样

这就相当于：

**时间在直接控制这个 block 该怎么处理当前 latent。**

### 4. 具体维度怎么变

先做：

`self.proj(cond)`

这里 `cond = time_cond: (B,384)`

`proj` 定义在：
[latent_transport.py#L47](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L47)

```python
self.proj = nn.Sequential(
    nn.SiLU(),
    nn.Linear(dim, dim * 2),
)
```

所以：

`(B,384) -> (B,768)`

然后：

`.unsqueeze(1)`

变成：

`(B,1,768)`

再：

`.chunk(2, dim=-1)`

拆成两半：

`scale: (B,1,384)`  
`shift: (B,1,384)`

然后对 `x` 做：

`norm(x): (B,64,384)`

再广播计算：

`norm(x) * (1 + scale) + shift`

其中：

- `norm(x)` 是 `(B,64,384)`
- `scale` 是 `(B,1,384)`
- `shift` 是 `(B,1,384)`

广播后：

`(B,64,384)`

所以 AdaLN 输入输出形状不变：

`(B,64,384) -> (B,64,384)`

### 5. 这一步语义上像什么

普通 LayerNorm 更像：

“所有时间点都用同一套处理风格”

AdaLN 更像：

“现在是桥的早期、中期、后期，我对 latent 的处理风格要跟着变”

所以时间不是只在网络最前面加一下就算了，  
而是通过 AdaLN 深度参与每一层处理。

你可以把它理解成：

**时间在给整个 transport block 下指令：现在该用什么节奏、什么方式更新 latent。**

### 6. block 的第一步：带时间调制的 self-attention

代码：

```python
x = x + self.self_attn(self.self_norm(x, time_cond))
```

这里：

`self.self_norm` 不是普通 LN，而是 `AdaLN`

也就是：

`x: (B,64,384)`
`time_cond: (B,384)`
`-> AdaLN`
`-> (B,64,384)`
`-> self-attention`
`-> (B,64,384)`
`-> residual add`
`-> x: (B,64,384)`

这一步的作用和 encoder/decoder 里的 self-attention 类似：

让 64 个 token 彼此交流。

但这里多了一层时间控制：

**同样是 token 之间交流，在桥的不同时间阶段，交流方式会不同。**

比如：

- 早期可能更偏向大范围粗调整
- 后期可能更偏向细节收敛

### 7. block 的第二步：cross-attention 读取起点条件 `cond`

代码：

```python
x = x + self.cross_attn(self.cross_norm_q(x, time_cond), self.cross_norm_ctx(cond))
```

这里你要特别注意：

- query 来自当前状态 `x`
- context 来自起点条件 `cond`
- 而 query 这边在进入 cross-attention 前，还会先被 `AdaLN(time_cond)` 调制

也就是说：

当前状态 `x` 会问：
“我作为一个从 partial 出发、现在处在时间 t 的状态，我应该怎么参考起点信息？”

形状上：

`x: (B,64,384)`  
`cond: (B,64,384)`

经过 cross-attention 后仍然是：

`(B,64,384)`

### 8. 为什么这里 cross-attention 看的是 `source_tokens`，不是 `target_tokens`

这个设计非常重要。

因为 inference 的时候你根本没有 `target_tokens`。  
你只有 partial。

所以 transport 必须学成一个：

**只依赖 partial 条件的动态系统**

也就是说，它不能训练时偷偷看 complete 当条件，  
否则测试时就没法用了。

因此这里的条件分支 `cond` 只能来自：

`source_tokens = partial latent`

这表示：

我从 partial 出发，我随时都可以参考 partial 的原始信息，  
但我不能偷看终点。

### 9. block 的第三步：带时间调制的 MLP

代码：

```python
x = x + self.mlp(self.mlp_norm(x, time_cond))
```

这里和前面一样：

- 先用 AdaLN 根据时间调制
- 再过逐 token 的 MLP
- 再 residual 加回去

维度还是：

`(B,64,384) -> (B,64,384)`

语义上是：

前两步做了状态交流和条件读取，  
这一步再把这些更新在每个 token 内部重新整理消化。

### 10. 和普通 Transformer block 相比，它到底多了什么

普通 block 大概是：

`LN -> self-attn -> residual`
`LN -> MLP -> residual`

这里的 transport block 变成了：

`AdaLN(time) -> self-attn -> residual`
`AdaLN(time) + cross-attn(source cond) -> residual`
`AdaLN(time) -> MLP -> residual`

所以它比普通 block 多了两件非常关键的事：

1. **时间调制**
不是静态处理，而是随时间变化

2. **条件输入**
不是无条件更新，而是始终参考 partial latent

所以这个 block 本质上不是普通 Transformer，  
而是一个 **time-conditioned, source-conditioned latent update block**。

### 11. 为什么这种设计很适合 bridge / transport

因为 bridge 的核心本来就是：

- 状态会随时间变化
- 流动方向依赖起点条件
- 同时还要看 token 之间的全局关系

这个 block 刚好把这三件事都放进去了：

- AdaLN 负责时间
- cross-attention 负责起点条件
- self-attention 负责 token 间全局关系

所以结构上是很对路的。

### 12. 一个白话类比

你可以把 `x` 想成一支正在行军的队伍，  
`cond` 是出发时的作战地图，  
`time_cond` 是“现在行军到第几阶段”的命令。

那么这个 block 做的事就是：

1. 先根据当前阶段命令，调整队伍内部协同方式
2. 再根据当前阶段命令，结合出发地图重新判断方向
3. 再把这一轮判断结果在队伍内部整理落实

所以它不是一次性冲刺，  
而是在每个阶段都重新做“内部协同 + 参考起点 + 阶段性调整”。

### 13. 这样的 block 有多少层

代码在：
[latent_transport.py#L95](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L95)

```python
self.blocks = nn.ModuleList([
    LatentTransportBlock(...)
    for _ in range(depth)
])
```

默认：

`depth = 6`

也就是这样的更新 block 会连着做 6 次。

所以：

`x: (B,64,384)`
`-> block 1`
`-> block 2`
`-> ...`
`-> block 6`
`-> 仍然是 (B,64,384)`

只是语义越来越接近“此时刻下正确的流动状态表示”。

### 14. 这一小步的完整维度链

输入：

`x: (B,64,384)`  
`cond: (B,64,384)`  
`time_cond: (B,384)`

#### AdaLN
`time_cond: (B,384)`
`-> proj`
`(B,768)`
`-> split`
`scale: (B,1,384)`
`shift: (B,1,384)`

`x: (B,64,384)`
`-> norm(x)`
`(B,64,384)`
`-> * (1+scale) + shift`
`(B,64,384)`

#### block 内三步
1. `AdaLN(time) -> self-attn -> residual`
2. `AdaLN(time) on x + LN(cond) -> cross-attn -> residual`
3. `AdaLN(time) -> MLP -> residual`

输出：

`x: (B,64,384)`

### 一句白话总结

`LatentTransportBlock` 的本质就是：

**让当前 latent 状态在“时间条件”和“partial 起点条件”的共同控制下，一步一步被更新。**

---

## 第五步：6 个 transport block 之后，怎么输出 `velocity`

经过 6 个 `LatentTransportBlock` 之后，模型终于要把当前状态 `x` 变成一个真正的输出：

`velocity: (B,64,384)`

代码在：
[latent_transport.py#L134](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L134)
[latent_transport.py#L136](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L136)
[latent_transport.py#L137](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L137)

```python
for block in self.blocks:
    x = block(x, cond, time_cond)
velocity = self.velocity_head(self.norm(x, time_cond))
transported_tokens = state_tokens + velocity
return LatentTransportOutput(velocity=velocity, transported_tokens=transported_tokens)
```

这一小步要抓住两个核心：

1. 它输出的是 **速度**，不是直接输出终点
2. 这个速度头被 **零初始化** 了，所以模型一开始近似“恒等映射”

### 1. 先看 6 个 block 后手里有什么

经过 6 层 transport block 之后：

`x: (B,64,384)`

这个 `x` 可以理解成：

“结合了当前状态、起点条件、时间条件、空间位置之后，网络内部整理好的状态表示”

注意，它还不是最终答案。  
它更像是预测速度前的高层特征。

有点像 decoder 里：

- 前面的 block 把 query 变成很强的 feature
- 最后 point head 再把 feature 变成坐标

这里也是一样：

- 前面的 block 把当前状态 `x` 变成很强的 transport feature
- 最后 velocity head 再把 feature 变成速度

### 2. 先再过一次 `AdaLN`

代码：

```python
velocity = self.velocity_head(self.norm(x, time_cond))
```

这里的 `self.norm` 也是 `AdaLN`，定义在：
[latent_transport.py#L99](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L99)

所以：

`x: (B,64,384)`
`time_cond: (B,384)`
`-> AdaLN`
`-> (B,64,384)`

这说明：

**连最终输出速度之前，时间条件还要再调制一次。**

也就是说，模型不是只在中间层考虑时间，  
而是连最后“把特征读成速度”的这一步，也明确受当前时间控制。

### 3. 然后过 `velocity_head`

代码在：
[latent_transport.py#L100](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L100)

```python
self.velocity_head = nn.Linear(hidden_dim, hidden_dim)
```

默认：

`hidden_dim = 384`

所以它做的是：

`(B,64,384) -> (B,64,384)`

输出：

`velocity: (B,64,384)`

这表示：

对每个 token，都预测一个 384 维的 latent 更新方向。

也就是：

- 不是预测一个标量速度
- 不是预测 3D 坐标位移
- 而是在 **latent 空间的 384 维里** 预测更新方向

### 4. 这个 `velocity` 到底是什么意思

它表示的是：

**当前状态下，每个 token 在 latent 空间里应该朝哪个方向移动。**

所以：

`velocity[:, i, :]`

表示第 `i` 个 token 的 384 维更新向量。

你可以把 latent token 想成 384 维空间里的一个点，  
那 `velocity` 就是一根箭头，告诉它下一步该往哪走。

这和 Stage-1 很不一样。  
Stage-1 decoder 最后输出的是点云坐标。  
Stage-2 transport 最后输出的是 latent 里的运动方向。

### 5. 为什么它不直接输出 `pred_tokens`

因为这个模型的设计思路就是：

**学一个动力系统，而不是学一个直接映射。**

如果它直接输出：

`pred_tokens = f(src_tokens)`

那就是普通回归。

但这里想学的是：

`dz/dt = v(z, source, center, t)`

也就是：

“在当前状态 `z`、起点条件 `source`、空间位置 `center`、时间 `t` 下，latent 的变化率是多少”

所以网络 forward 一次输出的应该是 `velocity`，  
而不是一步到位的终点。

### 6. 那 `transported_tokens = state_tokens + velocity` 又是什么

代码：

```python
transported_tokens = state_tokens + velocity
```

所以：

`state_tokens: (B,64,384)`
`velocity: (B,64,384)`
`-> transported_tokens: (B,64,384)`

这一步更像是一个“单步欧拉更新的候选结果”。

也就是说：

如果你现在把 `velocity` 当成一步更新量，  
那新的 token 可以写成：

`new_state = old_state + velocity`

不过在训练主循环里，真正重要的监督是 `out.velocity`；  
而真正用于多步推进的是后面的 Euler integration。

所以这里的 `transported_tokens` 更像是一个顺手给出的“单步更新结果”，不是最终完整 transport 的核心。

### 7. 为什么 `velocity_head` 要零初始化

代码在：
[latent_transport.py#L114](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L114)
[latent_transport.py#L115](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L115)
[latent_transport.py#L116](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L116)

```python
nn.init.zeros_(self.velocity_head.weight)
nn.init.zeros_(self.velocity_head.bias)
```

这非常关键。

零初始化意味着训练刚开始时：

`velocity ≈ 0`

于是：

`transported_tokens ≈ state_tokens`

也就是说，一开始 transport 基本什么都不做，是近似恒等映射。

### 8. 为什么“先什么都不做”是好事

因为 Stage-2 一开始最怕的情况就是：

模型刚初始化时就乱改 latent，  
把 Stage-1 decoder 原本还能勉强处理的 latent 直接搞坏。

如果一开始 `velocity` 很随机，那 Euler 多步积分后 latent 可能会飘得很离谱，  
decoder 输出会非常差，训练也会不稳。

而零初始化以后：

- 一开始 transport 接近 identity
- partial latent 至少不会被乱破坏
- 模型再慢慢学会做“必要的修正”

这就很稳。

你可以把它理解成：

不是一上来就猛打方向盘，  
而是先保持直行，再慢慢学会怎么修正路线。

### 9. 这和 AdaLN 的零初始化是同一个思路吗

是的，思路很一致。

代码里不光 `velocity_head` 零初始化，  
连 AdaLN 里生成 scale/shift 的投影层也做了零初始化：

[latent_transport.py#L117](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L117)
[latent_transport.py#L118](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L118)
[latent_transport.py#L119](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L119)
[latent_transport.py#L120](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L120)

```python
for module in self.modules():
    if isinstance(module, AdaLN):
        nn.init.zeros_(module.proj[-1].weight)
        nn.init.zeros_(module.proj[-1].bias)
```

这意味着训练初期：

- AdaLN 近似退化成普通无仿射 LN
- velocity head 近似输出 0

整个 transport 模型一开始就非常保守。

这和项目说明里那句：

“zero-init transport head so stage-2 starts close to identity”

是完全对应的。

### 10. 这一步和监督信号怎么接上

训练里紧接着就是：

[stage2_train.py#L366](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py#L366)
[stage2_train.py#L367](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py#L367)

```python
out = transport(bridge_state, src_tokens, src.centers, t)
flow_loss = F.mse_loss(out.velocity, velocity_target)
```

也就是：

预测：
`out.velocity: (B,64,384)`

目标：
`velocity_target: (B,64,384)`

然后做 MSE。

所以这个 forward 的直接训练目标非常清晰：

**让 transport 在桥上的任意中间状态，都能预测出正确的 latent 流动方向。**

### 11. 这一步白话类比

可以继续用“行军”的类比。

前面的 6 层 block，相当于参谋系统已经综合了：

- 当前队伍在哪
- 从哪里出发
- 当前行军到第几阶段
- 地图上的空间位置

最后 `velocity_head` 就像参谋长下达一个非常具体的命令：

“现在这一小步，该朝哪个方向走”

零初始化则相当于：

训练一开始，参谋长先默认“不瞎指挥”，  
先保持不动，再逐步学会怎么发出有用指令。

### 12. 这一小步的完整维度链

输入：

`x: (B,64,384)`

`-> AdaLN(time_cond)`
`(B,64,384)`

`-> velocity_head: Linear(384,384)`
`velocity: (B,64,384)`

再得到：

`transported_tokens = state_tokens + velocity`
`-> (B,64,384)`

最终返回：

- `velocity: (B,64,384)`
- `transported_tokens: (B,64,384)`

### 一句白话总结

6 层 transport block 做的是“把当前状态整理成适合预测更新方向的特征”，  
最后的 `velocity_head` 才真正把它读成 latent 空间里的“下一步该往哪走”。  
而零初始化保证了模型一开始几乎不乱动，先从接近恒等映射开始学。

---

## 第六步：为什么既要 `flow_loss`，又要 `recon_loss`

核心代码在：
[stage2_train.py#L366](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py#L366) 到 [stage2_train.py#L372](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py#L372)

```python
out = transport(bridge_state, src_tokens, src.centers, t)
flow_loss = F.mse_loss(out.velocity, velocity_target)
pred_tokens = transport.transport_train(src_tokens, src.centers, num_steps=args.transport_steps)
pred_tokens_dec = normalizer.denormalize(pred_tokens) if normalizer is not None else pred_tokens
pred_points = decoder(pred_tokens_dec, src.centers).coarse_points
recon_loss = chamfer_distance_l1(pred_points, complete_points)
loss = flow_weight * flow_loss + args.recon_weight * recon_loss
```

这一步一定要拆开看，不然很容易觉得“是不是重复监督了”。

其实不是重复，而是两种层次不同的监督。

### 1. 第一条监督：`flow_loss`

先看这两句：

```python
out = transport(bridge_state, src_tokens, src.centers, t)
flow_loss = F.mse_loss(out.velocity, velocity_target)
```

这里：

输入是桥上的中间状态：

`bridge_state: (B,64,384)`  
`src_tokens: (B,64,384)`  
`src.centers: (B,64,3)`  
`t: (B,)`

transport 输出：

`out.velocity: (B,64,384)`

目标是：

`velocity_target: (B,64,384)`

然后做 MSE。

### 2. `flow_loss` 在约束什么

它约束的是：

**在桥上的任意中间状态，transport 预测的局部流动方向，要和目标方向一致。**

也就是：

“你现在站在桥上的这个位置，应该往 complete latent 的方向流动”

所以 `flow_loss` 是一个 **局部、瞬时、微分式** 的监督。

它教的是：

- 此刻怎么动
- 当前方向对不对

而不是最终能不能成功到达终点。

### 3. 为什么只靠 `flow_loss` 还不够

这是一个特别关键的问题。

因为 `flow_loss` 只是在训练：

“局部一步的方向对不对”

但你真正关心的是：

“从 partial latent 出发，连续走很多步之后，最后能不能真的到达一个 decoder 能解码成完整点云的位置”

这两个并不完全等价。

你可以想象：

- 每一步的方向都大致对
- 但多步积累以后还是可能漂
- 或者到的 latent 虽然接近目标方向，但不一定是 decoder 最喜欢的 complete latent 分布

所以如果只优化 `flow_loss`，你学到的 transport 可能是：

“微分意义上还行”

但最终补全质量不一定最好。

### 4. 所以第二条监督来了：`recon_loss`

接下来代码会真正从 partial latent 出发，跑完整个 transport 过程：

```python
pred_tokens = transport.transport_train(src_tokens, src.centers, num_steps=args.transport_steps)
```

默认 `transport_steps = 8`

也就是说：

不是只看桥上的某个中间状态，  
而是从起点 `src_tokens` 出发，按 Euler 方法一步一步积分 8 步，得到最终预测 latent：

`pred_tokens: (B,64,384)`

这一步才是真正模拟 inference 时的 transport 路径。

### 5. `transport_train(...)` 和刚才的 `transport(...)` 有什么不同

这里调用的是：

`transport.transport_train(...)`

代码在：
[latent_transport.py#L160](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L160)

```python
def transport_train(self, source_tokens, centers, num_steps=8):
    return self._euler_integrate(source_tokens, centers, num_steps)
```

它的特点是：

**带梯度**

也就是说，后面的 `recon_loss` 会反向传播 through 整个多步积分过程，  
从而真正训练 transport 在完整 rollout 上表现得更好。

而推理/验证时常用的是：

`transport.transport(...)`

那个版本带 `@torch.no_grad()`，不反传。

### 6. `_euler_integrate(...)` 到底在做什么

代码在：
[latent_transport.py#L140](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L140)

```python
z = source_tokens
dt = 1.0 / float(num_steps)
for step in range(num_steps):
    t = torch.full((source_tokens.shape[0],), step / float(num_steps), ...)
    velocity = self(z, source_tokens, centers, t).velocity
    z = z + dt * velocity
return z
```

这就是一个标准的 Euler 积分。

如果默认 `num_steps = 8`，那：

- `dt = 1/8`
- 第 0 步用 `t=0`
- 第 1 步用 `t=1/8`
- 第 2 步用 `t=2/8`
- ...
- 第 7 步用 `t=7/8`

每一步都做：

`z = z + dt * velocity`

所以最终：

`pred_tokens`

就是从 partial latent 出发，连续走 8 小步得到的结果。

### 7. 为什么训练时一定要真的 roll out 这 8 步

因为 inference 的时候你就是这么用的。

测试时不是说：

- 取一个随机 `bridge_state`
- 只看一小步速度

而是：

**从 partial latent 出发，连续积分很多步，最后得到预测 latent**

所以训练时如果不把这个真实使用过程纳入损失，  
模型就可能在单步上表现不错，但在 rollout 上积累误差。

这和控制系统、序列预测里“teacher forcing 看着还行，free rollout 就漂了”是一个味道。

### 8. 然后为什么要 `denormalize`

代码：

```python
pred_tokens_dec = normalizer.denormalize(pred_tokens) if normalizer is not None else pred_tokens
```

这里：

`pred_tokens: (B,64,384)`

如果 Stage-1 用了 latent normalization，那 decoder 训练时真正熟悉的是：

**未标准化前的 complete latent 空间** 还是 **标准化后的空间**？

这个项目的流程是：

- Stage-1 decoder 在训练时吃的是 normalized latent
- 但 Stage-2 这里在送 decoder 之前显式 `denormalize`

这背后的设计意图是：

**transport 工作在 normalized latent 空间里，但 decoder 最终接回到它在 Stage-1 checkpoint 所定义的解码输入空间。**

从代码角度你要先记住：

`pred_tokens: (B,64,384)`
`-> denormalize`
`pred_tokens_dec: (B,64,384)`

形状不变，只是数值坐标系切换。

### 9. 然后送进 decoder，得到真正的补全点云

代码：

```python
pred_points = decoder(pred_tokens_dec, src.centers).coarse_points
```

所以：

`pred_tokens_dec: (B,64,384)`  
`src.centers: (B,64,3)`  
`-> decoder`
`pred_points: (B,8192,3)`  
或 PCN 下 `(B,16384,3)`

这一步和 Stage-1 decoder 的使用方式一致。

也就是说，Stage-2 最终还是要接受一个最现实的检验：

**你 transport 出来的 latent，经过 decoder 后，生成的点云到底像不像真实 complete。**

### 10. 然后算 `recon_loss`

代码：

```python
recon_loss = chamfer_distance_l1(pred_points, complete_points)
```

所以：

预测：
`pred_points: (B,8192,3)`

真值：
`complete_points: (B,8192,3)`

然后算 Chamfer L1。

这就是一个非常直接的最终任务监督：

**我不管你 latent 看起来多漂亮，最后补出来的点云必须接近真实完整点云。**

### 11. 所以 `flow_loss` 和 `recon_loss` 的分工到底是什么

这是这一步最重要的理解点。

#### `flow_loss`
约束的是局部动态规则

它在问：

“桥上的任意中间状态，当前流动方向对不对？”

这是一个 **局部、瞬时、latent-level** 的监督。

#### `recon_loss`
约束的是完整 rollout 后的最终任务表现

它在问：

“从 partial latent 连续走完全程后，decoder 解码出来的点云好不好？”

这是一个 **全局、终局、point-cloud-level** 的监督。

### 12. 为什么这两个一起用会更好

如果只有 `flow_loss`：

- transport 可能学到看起来合理的局部速度场
- 但最终 decoder 输出未必最好

如果只有 `recon_loss`：

- 模型也许能为了最终 Chamfer 偷学一些奇怪的 latent 路径
- 但 latent 流动会比较不规整，不稳定，也更难学

所以两者结合就形成了：

- `flow_loss` 负责把 latent bridge 的动态结构学正
- `recon_loss` 负责确保最终任务效果真的好

你可以把它理解成：

一个老师盯你“走路姿势对不对”，  
另一个老师盯你“最后有没有走到目的地”。

两边一起管，效果最好。

### 13. 总损失怎么组合

代码：

```python
loss = flow_weight * flow_loss + args.recon_weight * recon_loss
```

也就是：

`total_loss = flow_weight * flow_loss + recon_weight * recon_loss`

默认一般两个都启用。

而且代码里 `flow_weight` 还支持 warmup：

[stage2_train.py#L349](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py#L349)

```python
flow_weight = linear_warmup_value(...)
```

这说明训练时还可以控制：

- 一开始先更看重 reconstruction
- 或者一开始先慢慢把 flow 监督加进来

这也是一种稳定训练的手段。

### 14. 一个白话类比

可以把 Stage-2 想成在训练一个驾驶员。

#### `flow_loss`
像是在训练：
“你当前这个方向盘打得对不对，油门刹车配合得对不对”

也就是盯每一个瞬时动作。

#### `recon_loss`
像是在训练：
“你最后到底有没有把车顺利开到目的地”

如果只盯动作，不盯终点，可能动作看着都挺标准但最后跑偏。  
如果只盯终点，不盯动作，可能学出一套很怪但勉强到终点的开法。

所以两边都要管。

### 15. 这一小步的完整维度链

#### flow 分支
`bridge_state: (B,64,384)`
`+ src_tokens + centers + t`
`-> transport forward`
`out.velocity: (B,64,384)`

和：

`velocity_target: (B,64,384)`

算：

`flow_loss`

#### rollout + recon 分支
`src_tokens: (B,64,384)`
`-> transport_train (8-step Euler)`
`pred_tokens: (B,64,384)`
`-> denormalize`
`pred_tokens_dec: (B,64,384)`
`-> decoder`
`pred_points: (B,8192,3)`
`-> compare with complete_points`
`recon_loss`

### 一句白话总结

Stage-2 不只是要求 transport “每一步方向看起来对”，  
还要求它“从 partial latent 真正走完整条路之后，最后 decoder 解码出来的点云也必须好”。  
所以它同时用了 `flow_loss` 和 `recon_loss`，一个管过程，一个管结果。

---

## 第七步：为什么真正补全时要做 8 步 Euler 积分

核心代码在：
[latent_transport.py#L140](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L140)

```python
def _euler_integrate(self, source_tokens, centers, num_steps=8):
    z = source_tokens
    dt = 1.0 / float(num_steps)
    for step in range(num_steps):
        t = torch.full((source_tokens.shape[0],), step / float(num_steps), ...)
        velocity = self(z, source_tokens, centers, t).velocity
        z = z + dt * velocity
    return z
```

训练时用：
[latent_transport.py#L160](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L160)

```python
transport_train(...)
```

推理/验证时用：
[latent_transport.py#L169](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L169)

```python
transport(...)
```

两者本质上调用的是同一个 `_euler_integrate(...)`，只是一个带梯度，一个不带梯度。

### 1. 先说为什么 `forward` 一次还不够

因为 transport 网络学的不是：

`pred_tokens = f(src_tokens)`

它学的是：

`velocity = v(current_state, source, center, t)`

也就是：

**在当前状态下，当前这一小瞬间，latent 应该朝哪个方向走。**

注意关键词是：

- 当前状态
- 当前时刻
- 当前方向

所以它给的是“速度”或“导数”，  
不是“一步到位的终点”。

### 2. 这和普通回归模型的区别

普通模型可能是：

输入 `x`
直接输出 `y`

比如：

`partial latent -> complete latent`

那一遍 forward 就完了。

但这里不是这样。  
这里更像是解一个动态系统：

`dz/dt = v(z, source, center, t)`

意思是：

latent 状态 `z` 会随着时间变化，  
而变化率由 transport 网络给出。

所以如果你想从 `t=0` 走到 `t=1`，  
就不能只问一次“现在该怎么走”，  
而是要不断问：

- 现在在这里，往哪走？
- 走一点之后到了新位置，再往哪走？
- 再走一点，又该往哪走？

这就是为什么要积分。

### 3. Euler 积分最直白的理解

Euler 积分其实很朴素。

它做的事就是：

1. 先站在起点
2. 问网络：现在该往哪走？
3. 沿那个方向走一小步
4. 到了新位置，再问一次
5. 再走一小步
6. 重复很多次

所以它不是“一步跳到完整 latent”，  
而是“一步一步推过去”。

### 4. `_euler_integrate(...)` 里第一句是什么意思

```python
z = source_tokens
```

所以初始化时：

`z: (B,64,384) = source_tokens`

也就是说，积分开始时的状态就是：

`partial latent`

这很合理，因为桥的起点本来就是 partial。

### 5. 然后定义步长 `dt`

```python
dt = 1.0 / float(num_steps)
```

如果默认：

`num_steps = 8`

那：

`dt = 1/8 = 0.125`

这意味着：

整个从 `t=0` 到 `t=1` 的时间段，被切成了 8 小段。

你可以把它理解成：

不是一大步走完整条桥，  
而是把桥切成 8 个短区间，慢慢走。

### 6. 循环里的 `t` 是怎么取的

```python
t = torch.full((source_tokens.shape[0],), step / float(num_steps), ...)
```

所以每一轮：

- 第 0 步：`t = 0/8 = 0.0`
- 第 1 步：`t = 1/8 = 0.125`
- 第 2 步：`t = 2/8 = 0.25`
- ...
- 第 7 步：`t = 7/8 = 0.875`

因此每一步 transport 网络都明确知道：

“我现在是在桥的第几阶段”

这个非常关键，因为同样的 latent 状态，在不同时间阶段理想的更新方向可能不一样。

### 7. 然后每一步都重新算一次速度

代码：

```python
velocity = self(z, source_tokens, centers, t).velocity
```

这里输入的是：

- `z`：当前状态
- `source_tokens`：起点 partial latent
- `centers`：空间位置
- `t`：当前时间

输出：

`velocity: (B,64,384)`

注意这里和前面训练 `flow_loss` 时很像，  
但现在不是随机桥中间状态，而是 **rollout 过程中真实到达的状态 `z`**。

这意味着：

每走一步，下一步方向都要重新计算。  
不是一开始算好一根总箭头，然后一路不变。

### 8. 然后怎么更新状态

代码：

```python
z = z + dt * velocity
```

这里：

`z: (B,64,384)`  
`velocity: (B,64,384)`  
`dt` 是标量

所以：

`dt * velocity -> (B,64,384)`

再相加：

`z -> (B,64,384)`

形状不变，但状态更新了。

这一步的意义就是：

**沿着当前预测的速度方向，走一个很小的时间步长。**

### 9. 为什么要乘 `dt`，不能直接 `z = z + velocity`

因为 `velocity` 表示的是“变化率”或“方向强度”，  
不是整个时间区间上的完整位移。

如果不乘 `dt`，那每一步都走太猛了。  
切成 8 步以后，每一步只应该走总路程的一小部分。

所以：

- `velocity` 是“每单位时间怎么变”
- `dt` 是“这一步持续多长时间”
- `dt * velocity` 才是“这一步真正走多少”

这和物理里：

`位移 = 速度 × 时间`

是一个直觉。

### 10. 8 步之后得到什么

循环结束后返回：

`z: (B,64,384)`

这个 `z` 就是：

**从 partial latent 出发，按照 transport 学到的速度场，连续走完 8 步之后得到的最终 latent。**

在代码里它就是：

`pred_tokens`

也就是后面送进 decoder 的东西。

### 11. 为什么 inference 也要这样走，而不是训练时才这样走

因为 transport 模型的定义本来就是“预测局部速度”。

既然 forward 只负责告诉你：

“当前该怎么走一小步”

那不管训练还是推理，  
只要你想从起点走到终点，就都得把这一小步过程重复很多次。

所以多步积分不是训练技巧，  
而是这个模型定义的一部分。

### 12. 为什么不直接把步数设得特别大或者特别小

这是个工程折中。

#### 步数太少
比如只走 1 步、2 步

那每一步都太粗糙，  
对连续桥的近似很差，容易学得生硬。

#### 步数太多
比如 50 步、100 步

会更接近连续流动，  
但训练和推理都会更慢，而且梯度传播也更重。

所以默认选 `8` 步，就是一个折中：

- 比一步映射更像连续 transport
- 计算量也还可控

### 13. 你可以把它类比成什么

可以把 latent transport 想成在山路上开车。

网络每次 forward 不会直接说：

“你一脚油门直接到终点吧”

它只会说：

“按你现在这个位置和时刻，方向盘该往哪边打一小下”

于是你就得：

- 打一点
- 车到了新位置
- 再重新看方向
- 再打一点评价

连续这样做，最后才真正到终点。

Euler 积分就是这个“边走边纠偏”的过程。

### 14. 为什么这比“一根总箭头走到底”更合理

因为 latent 空间一般不是线性的简单地形。

从 partial 到 complete 的理想路径，往往不是一条一成不变的直线规则。  
尤其不同时间阶段的调整重点可能不同：

- 前期可能做大结构补全
- 中期可能做整体分布对齐
- 后期可能做细节收敛

所以“每一步都重新根据当前位置和时间决定方向”，  
比“拿一根固定箭头走到底”更灵活，也更符合桥模型的思路。

### 15. 这一小步的完整维度链

初始化：

`z = source_tokens = (B,64,384)`

如果 `num_steps = 8`：

`dt = 1/8`

每一轮：

1. 构造  
`t: (B,)`

2. forward  
`self(z, source_tokens, centers, t)`  
输出  
`velocity: (B,64,384)`

3. 更新  
`z = z + dt * velocity`  
得到  
`z: (B,64,384)`

重复 8 次后：

`pred_tokens = z: (B,64,384)`

### 一句白话总结

transport 网络一次 forward 只会告诉你“现在该往哪走一小步”，  
真正从 partial latent 走到最终 latent，要靠 Euler 积分把这件事重复很多次。  
所以 Stage-2 的补全本质上不是“一跳到终点”，而是“沿着学到的速度场一步一步走过去”。

---

## 第八步：为什么 `pred_tokens` 还要 `denormalize` 再送给 decoder

对应代码在：
[stage2_train.py#L368](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py#L368)
[stage2_train.py#L369](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py#L369)
[stage2_train.py#L370](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py#L370)

```python
pred_tokens = transport.transport_train(src_tokens, src.centers, num_steps=args.transport_steps)
pred_tokens_dec = normalizer.denormalize(pred_tokens) if normalizer is not None else pred_tokens
pred_points = decoder(pred_tokens_dec, src.centers).coarse_points
```

这一小步的核心是：

**transport 工作的 latent 空间，和 decoder 最终吃进去的 latent 空间，要严格对上。**

### 1. 先看这里手里有什么

经过 8 步 Euler 积分之后，得到：

`pred_tokens: (B,64,384)`

这个 `pred_tokens` 是什么？

它是：

**在 Stage-2 的 normalized latent 空间里，transport 从 partial latent 出发走到的最终位置。**

注意这个“normalized latent 空间”四个字很关键。

因为前面 Stage-2 一开始就做了：

`src.tokens -> normalize -> src_tokens`
`tgt.tokens -> normalize -> tgt_tokens`

也就是说，transport 从头到尾工作的 latent，默认是在 normalizer 定义的标准化坐标系里。

### 2. 为什么 transport 要在 normalized latent 空间里工作

因为 Stage-1 已经告诉我们一件事：

encoder 输出的各通道分布不统一，  
直接在原始 latent 空间里学 transport，数值尺度可能很乱。

所以 Stage-2 也沿用了 Stage-1 的 latent normalization。

这样 transport 看到的是一个更规整的空间：

- 各通道尺度更统一
- partial/complete 更容易比较
- `tgt - src` 这种差值更稳定

所以：

`pred_tokens`

此时还是“标准化坐标系中的 latent”。

### 3. 那为什么还要 `denormalize`

因为后面要把它送进 decoder。

而 decoder 是从 Stage-1 继承来的。  
Stage-2 这里非常看重一件事：

**decoder 要接收到和它熟悉的 complete latent 分布一致的东西。**

所以代码会做：

```python
pred_tokens_dec = normalizer.denormalize(pred_tokens)
```

形状不变：

`(B,64,384) -> (B,64,384)`

但数值从“标准化空间”映回“原 latent 坐标系”。

### 4. `denormalize` 数学上做了什么

代码在：
[latent_normalizer.py#L30](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_normalizer.py#L30)

```python
return x * (self.std + self.eps) + self.mean
```

也就是：

如果 normalize 是：

`x_norm = (x - mean) / std`

那么 denormalize 就是：

`x = x_norm * std + mean`

所以它做的是一个标准的逆变换。

形状不变，但把 token 数值重新放回原始 complete latent 的数值范围。

### 5. 为什么不能直接把 normalized `pred_tokens` 扔给 decoder

因为那样会有“坐标系错位”的风险。

你可以把 decoder 想成一个只会读某种格式图纸的工人。

- transport 这边为了方便计算，在“标准化坐标系”里工作
- decoder 这边最后要读的是“它熟悉的 latent 坐标系”

如果你不做 denormalize，  
就等于拿一份不同单位、不同坐标定义的图纸直接丢给 decoder。

那 decoder 很可能读歪。

### 6. 这一步和 Stage-1 的衔接关系是什么

这一步本质上就是在保证：

**Stage-2 transport 的输出，最后要重新落回 Stage-1 decoder 所期待的输入分布。**

这正是两阶段训练设计的核心思想之一：

- Stage-1 先规定“complete latent 应该长什么样，decoder 才能解码好”
- Stage-2 再去学“怎么把 partial latent 运输到这种 latent 形状附近”

所以 Stage-2 不是让 decoder 适应一个完全新的世界，  
而是让 transport 把结果送回 decoder 原本就擅长的世界。

### 7. 这里可以怎么理解得更直白一点

你可以把 normalizer 想成一个翻译坐标系的工具。

- transport 喜欢在“标准化后的内部工作语言”里思考
- decoder 最后要读“原始 complete latent 语言”

所以流程就是：

1. partial/complete 先翻译到 transport 喜欢的语言
2. transport 在这种语言里完成搬运
3. 搬完以后再翻译回 decoder 能读懂的语言

这一步的 `denormalize` 就是最后那次“翻译回去”。

### 8. 然后为什么 decoder 这里还用 `src.centers`

代码：

```python
pred_points = decoder(pred_tokens_dec, src.centers).coarse_points
```

所以 decoder 输入是：

`pred_tokens_dec: (B,64,384)`
`src.centers: (B,64,3)`

这里继续使用 `src.centers`，是因为整个 Stage-2 从 partial 编码开始，就一直围绕这组统一 centers 建立 token 对齐。

所以最后 decode 时，也仍然沿用这组 centers。

这表示：

- 第 `i` 个 token 仍然对应第 `i` 个空间锚点
- decoder 看到的位置结构，和前面 transport 处理时是一致的

这保证了整个链条的空间一致性。

### 9. decoder 在这一步做的事情是什么

这一步和 Stage-1 decoder 的作用完全一致：

`pred_tokens_dec: (B,64,384)`
`+ src.centers: (B,64,3)`
`-> decoder`
`pred_points: (B,8192,3)`

也就是说，decoder 并不关心这些 latent 是怎么来的。  
它只管一件事：

**如果这些 token 看起来像 complete latent，那我就把它们解码成完整点云。**

所以 Stage-2 的成败，其实很大程度上取决于：

transport 有没有把 partial latent 真正送到“像 complete latent”的区域。

### 10. 为什么这一步是 Stage-2 最终任务闭环的关键

因为前面所有 latent-space 的训练，说到底都只是中间手段。  
你最终要的不是一个漂亮的 latent，  
而是一个漂亮的完整点云。

所以只有经过：

`pred_tokens -> denormalize -> decoder -> pred_points`

这条链之后，Stage-2 才真正变成“补全任务”。

否则你只是在 latent 空间里自嗨，  
并没有接受最终几何结果的检验。

### 11. 一个白话类比

可以把 transport 和 decoder 想成两个人接力干活。

- transport 像一个内部数据处理员，他喜欢用标准化表格工作
- decoder 像一个现场施工员，他习惯看原始工程图纸

所以 transport 先在自己的表格体系里把信息整理好，  
但交给 decoder 前，必须把表格重新翻译回施工员看得懂的格式。

`denormalize` 就是这次翻译。

### 12. 这一小步的完整维度链

`pred_tokens: (B,64,384)`
`-> denormalize`
`pred_tokens_dec: (B,64,384)`

然后：

`pred_tokens_dec: (B,64,384)`
`src.centers: (B,64,3)`
`-> decoder`
`pred_points: (B,8192,3)`

再和：

`complete_points: (B,8192,3)`

去算 `recon_loss`。

### 一句白话总结

transport 真正学到的是“在标准化 latent 空间里怎么把 partial latent 推向 complete latent”；  
但最后要交给 decoder 时，必须先把结果映回 decoder 熟悉的 latent 坐标系。  
所以 `denormalize` 的作用，就是把 Stage-2 的内部运输结果，准确接回 Stage-1 的完整形状生成器。

---

## 第九步：Stage-2 里 decoder 到底怎么参与训练

这个问题很关键，因为 Stage-2 虽然重点是 transport，但代码并不是简单地把 decoder 完全锁死。

核心代码在：
[stage2_train.py#L73](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py#L73)
[stage2_train.py#L96](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py#L96)

### 1. Stage-2 一开始怎么加载 decoder

先看这里：

[stage2_train.py#L73](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py#L73)

```python
def load_frozen_decoder(checkpoint_path, device, output_points):
```

里面做的是：

```python
checkpoint = torch.load(checkpoint_path, map_location='cpu')
decoder = QueryCompletionDecoder(...)
decoder.load_state_dict(checkpoint['decoder'])
decoder.eval()
for p in decoder.parameters():
    p.requires_grad = False
```

所以一开始，decoder 的确是：

- 从 Stage-1 checkpoint 里加载
- 先设为 `eval`
- 先把所有参数都 `requires_grad=False`

也就是说：

**初始状态下，decoder 是冻结的。**

这很合理，因为 Stage-2 首先默认：

“Stage-1 已经把 decoder 训练好了。”

### 2. 那为什么后面又说 decoder 不是完全冻结的

因为紧接着代码又会调用：

[stage2_train.py#L277](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py#L277)

```python
decoder_trainable_params = configure_decoder_training(decoder, args.decoder_train_mode, args.decoder_train_last_n)
```

也就是说：

虽然一开始全部冻结，  
但随后会根据配置决定要不要重新打开其中一部分参数训练。

### 3. `configure_decoder_training(...)` 做了什么

代码在：
[stage2_train.py#L96](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py#L96)

它一上来先做：

```python
for p in decoder.parameters():
    p.requires_grad = False
decoder.eval()
```

然后根据 `train_mode` 分三种情况。

### 4. 情况一：`decoder_train_mode = none`

代码：

```python
if train_mode == 'none':
    return []
```

这表示：

**decoder 完全不训练。**

此时 Stage-2 真正训练的只有 transport。

这是一种最纯粹的两阶段方式：

- Stage-1 负责 decoder
- Stage-2 只负责 partial latent -> complete latent transport

优点是职责最清楚。  
缺点是如果 Stage-2 输出 latent 和 Stage-1 看到的 perfect complete latent 还是有一点分布偏差，decoder 完全不适应的话，性能可能受限。

### 5. 情况二：`decoder_train_mode = all`

代码：

```python
if train_mode == 'all':
    decoder.train()
    for p in decoder.parameters():
        p.requires_grad = True
```

这表示：

**整个 decoder 全量参与 Stage-2 训练。**

这样做的好处是适应性最强。  
如果 transport 输出的 latent 和 Stage-1 的理想 latent 有系统偏差，decoder 可以整体一起适配。

但代价也明显：

1. 参数更多
2. 更容易把 Stage-1 已经学好的 complete-shape prior 搞漂
3. 训练更不稳
4. decoder 可能开始“替 transport 擦屁股”，导致职责又混回去了

也就是说，decoder 可能不再只是解码 complete latent，  
而是偷偷学起了一部分补全。

### 6. 情况三：`decoder_train_mode = last_n`，这是默认模式

默认就是这个。

参数在：
[stage2_train.py#L54](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py#L54)
[stage2_train.py#L55](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py#L55)

```python
parser.add_argument('--decoder-train-mode', type=str, default='last_n', ...)
parser.add_argument('--decoder-train-last-n', type=int, default=2)
```

也就是说，默认：

- 模式是 `last_n`
- 只训练 decoder 最后 `2` 个 block

### 7. 代码里具体怎么只打开最后几层

看这里：

[stage2_train.py#L114](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py#L114) 到 [stage2_train.py#L140](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py#L140)

```python
num_blocks = len(decoder.blocks)
block_count = min(max(train_last_n, 1), num_blocks)
trainable_blocks = list(decoder.blocks[num_blocks - block_count:])
for block in trainable_blocks:
    block.train()
    for p in block.parameters():
        p.requires_grad = True

decoder.norm.train()
...
decoder.point_head.train()
...
decoder.query_embed.requires_grad = True
```

这意味着默认 `last_n=2` 时，会训练：

1. decoder 最后 2 个 block
2. 最后的 `norm`
3. 输出头
   - 如果普通模式就是 `point_head`
   - 如果 refine 模式还包括 `seed_head` / `refine_module`
4. `query_embed`

而 decoder 前面的多数 block 仍然冻结。

### 8. 为什么默认只训练最后几层，而不是全部

这是一个非常有工程味的折中。

#### 不全冻的问题
如果 decoder 完全不动，它就只能被动接受 transport 输出。  
哪怕 transport 已经很努力了，只要 latent 分布有点偏，decoder 也可能不够适配。

#### 全开的风险
如果 decoder 全开，它可能改得太多，把 Stage-1 学好的完整形状先验搞松了。  
甚至可能让 decoder 自己承担太多补全责任，削弱 transport 的学习压力。

#### 所以“只动最后几层”是折中
前面的 decoder block 保留 Stage-1 学到的稳定几何先验。  
后面的 block 和输出头做一些适应性调整，去接住 Stage-2 送来的 latent。

可以理解成：

- 前半段 decoder 继续当“完整形状知识库”
- 后半段 decoder 做少量“接口适配”

这个思路通常很稳。

### 9. 为什么 `query_embed` 也要开训练

代码里即使只开最后几层，也会：

```python
decoder.query_embed.requires_grad = True
```

这是因为 query 是 decoder 整个生成过程的起始模板。  
如果 Stage-2 的 latent 分布相对 Stage-1 有轻微变化，  
允许 query 模板一起微调，会比完全锁死更容易适应。

你可以把它理解成：

虽然 decoder 主体不怎么动，  
但允许它稍微调整一下“施工队的初始分工模板”。

这通常是比较划算的。

### 10. 输出头为什么也要开训练

因为输出头离最终点云最近。  
如果 Stage-2 latent 有一点系统偏差，  
最先需要补偿的往往就是 decoder 的最后几层和输出头。

比如：

- query feature 的几何语义整体没问题
- 但坐标回归的细节有一点偏

这时只调整后面的输出部分，成本最小，也最稳。

所以训练最后几层 + 输出头，本质上就是在做一种“低风险适配”。

### 11. 训练时 decoder 的学习率为什么更小

代码在：
[stage2_train.py#L280](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py#L280) 到 [stage2_train.py#L285](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage2_train.py#L285)

```python
optimizer_param_groups = [
    {'params': list(transport.parameters()), 'lr': args.lr},
]
if decoder_trainable_params:
    optimizer_param_groups.append({'params': decoder_trainable_params, 'lr': args.lr * args.decoder_lr_mult})
```

默认：

`decoder_lr_mult = 0.1`

也就是说：

decoder 如果参与训练，学习率默认只有 transport 的十分之一。

这背后的逻辑非常明确：

- transport 是 Stage-2 的主角，应该学得更积极
- decoder 只是微调适配，应该保守一点

这又是一个典型的“别把 Stage-1 学好的东西一下冲坏”的工程策略。

### 12. 这一步和两阶段设计的关系是什么

这一步最能体现两阶段设计的真正味道。

如果完全按最硬的两阶段分工：

- Stage-1 练 decoder
- Stage-2 只练 transport

那就是 `decoder_train_mode = none`

但现实里为了效果，代码默认选了一个更务实的方案：

- 大体上仍然把 decoder 当成 Stage-1 学好的完整形状生成器
- 但允许最后几层稍微适配 Stage-2 的 latent

所以它不是纯理论化的“严格职责隔离”，  
而是工程上的“职责清楚，但允许少量接口微调”。

这很合理。

### 13. 一个白话类比

你可以把 decoder 想成一个已经训练好的老工人。

Stage-2 来了以后，有三种选择：

#### `none`
老工人完全按原流程干活，不做任何改变

#### `all`
让老工人把整个手艺都重新改一遍

#### `last_n`
老工人的核心手艺不变，只调整最后几个工序和出料口

显然第三种最稳妥：

- 不会把原有本事全毁掉
- 又能适应新来的材料

这就是默认 `last_n=2` 的直觉。

### 14. 这一小步的完整结论

Stage-2 中 decoder 的默认角色是：

**以 Stage-1 训练好的 complete-shape decoder 为主，保留其主体能力，只对最后几层和输出端做少量微调，以适应 transport 输出的 latent。**

所以 decoder 在 Stage-2 里不是完全自由发挥，  
也不是完全锁死，  
而是一种很克制的“半冻结适配”状态。

### 一句白话总结

Stage-2 默认不是重新训练 decoder，而是把它当成一个已经会生成完整形状的成熟模块，只允许最后几层和输出头稍微动一动，去适应 transport 输出的 latent 分布；这样既保住 Stage-1 学到的几何先验，又不给 transport 过多甩锅空间。

---

## 第十步：Stage-2 推理时真正会走的主路径

推理时不会再做：

- 随机采样桥中间状态
- 构造 `velocity_target`
- 算 `flow_loss`
- 用 GT complete 对照

因为推理时根本没有 ground truth complete。

所以推理时只走真正有用的主链：

`partial_points -> encoder -> transport rollout -> decoder -> pred_points`

### 1. 推理的第一步：输入 partial 点云

输入是：

`partial_points: (B,2048,3)`

或者单样本时你也可以理解成 batch size 为 1：

`(1,2048,3)`

这一步和 Stage-2 训练开始时一样，  
先进入 encoder。

### 2. 第二步：编码 partial，得到 `src_tokens`

这一步：

`src = encoder(partial_points)`

输出：

`src.centers: (B,64,3)`
`src.tokens: (B,64,384)`

也就是说，partial 点云先被切成 64 个局部 patch，再编码成 64 个 latent token。

这里没有 complete，也没有 target。  
因为推理时你只有 partial。

### 3. 第三步：如果有 normalizer，就先 normalize

这一步虽然你在训练阶段已经见过，但推理时同样需要。

也就是：

`src.tokens -> normalizer.normalize(...) -> src_tokens`

得到：

`src_tokens: (B,64,384)`

为什么推理时也要做？

因为 transport 模型训练时就是在 normalized latent 空间里工作的。  
如果推理时不 normalize，那输入分布就和训练时不一致了。

所以这里必须保持和训练阶段同样的 latent 坐标系。

### 4. 第四步：调用 `transport(...)` 做多步积分

推理时真正调用的是：

[latent_transport.py#L169](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_transport.py#L169)

```python
@torch.no_grad()
def transport(self, source_tokens, centers, num_steps=8):
    return self._euler_integrate(source_tokens, centers, num_steps)
```

也就是：

`pred_tokens = transport.transport(src_tokens, src.centers, num_steps=8)`

注意这里和训练时的 `transport_train(...)` 的区别是：

- 推理版 `transport(...)` 带 `@torch.no_grad()`
- 不保留梯度
- 更省显存

但走的逻辑其实是同一个 `_euler_integrate(...)`。

### 5. 这 8 步积分在推理时具体干了什么

初始化：

`z = src_tokens`

然后每一步都做：

1. 构造当前时间 `t`
2. 调用 transport forward，得到当前速度
3. 用 `z = z + dt * velocity` 更新状态

总共 8 步后，得到：

`pred_tokens: (B,64,384)`

你可以把这理解成：

**把 partial latent 一步一步推成“更像 complete 的 latent”。**

这就是推理时 Stage-2 最核心的 latent 补全过程。

### 6. 这时候 `pred_tokens` 还在哪个空间里

它仍然是在：

**normalized latent 空间**

因为 transport 本身就是在这个空间里工作的。

所以这时候还不能直接喂 decoder。

### 7. 第五步：`denormalize`

接下来做：

`pred_tokens_dec = normalizer.denormalize(pred_tokens)`

输出仍然是：

`(B,64,384)`

但数值从 normalized 空间回到 decoder 熟悉的 latent 坐标系。

这一步的作用你刚刚已经看过：

transport 在自己的工作坐标系里完成搬运，  
最后再把结果翻译回 decoder 看得懂的坐标系。

### 8. 第六步：送进 decoder

然后：

`pred_points = decoder(pred_tokens_dec, src.centers).coarse_points`

所以输入是：

`pred_tokens_dec: (B,64,384)`
`src.centers: (B,64,3)`

输出是：

`pred_points: (B,8192,3)`  
或者在 PCN 默认下是 `(B,16384,3)`

这一步和 Stage-1 的 decoder forward 完全一样。

所以 decoder 在推理时扮演的角色仍然是：

**把一个像 complete latent 的 token 序列，解码成完整点云。**

### 9. 为什么这里还用 `src.centers`

因为整个推理链都围绕 partial 这边选出的 64 个中心点展开：

- partial 编码时靠这组 centers 建立 token
- transport 过程中 token 语义也围绕这组 centers
- 最后 decode 时，自然还要继续沿用这组 centers

所以 `src.centers` 就是整个推理链的空间锚点。

你可以把它想成：

整条补全过程里，token 的“槽位编号”一直没换，  
每个槽位始终对应同一个空间局部区域。

### 10. 推理时没有 complete，那模型到底靠什么补出来

这时候模型靠的是两部分已经学好的能力：

#### encoder + transport
负责把 partial latent 推到 complete latent 分布附近

#### decoder
负责把 complete-like latent 解码成完整点云

也就是说，推理时没有 GT complete，  
但模型已经在训练中学会了：

- 什么样的 latent 是“完整形状 latent”
- 怎么从 partial latent 沿桥走到那附近
- decoder 怎么把那种 latent 变回完整点云

所以测试时它不需要再看 GT。

### 11. 推理时整条链的最简公式

如果只保留真正部署用的主路径，可以写成：

`partial_points`
`-> encoder`
`-> src_tokens`
`-> normalize`
`-> transport rollout (8 steps)`
`-> pred_tokens`
`-> denormalize`
`-> decoder`
`-> pred_complete_points`

这就是 Stage-2 真正的 inference pipeline。

### 12. 你可以把它类比成什么

可以把推理过程想成一个完整流水线：

1. partial 点云先被 encoder 翻译成一种内部语义表示
2. transport 像一个“latent 修复工”，在内部语义空间里一步步补齐缺失
3. decoder 像一个“几何重建工”，把修复后的内部表示重新雕成完整点云

所以推理时不是直接在原坐标里补洞，  
而是：

**先理解，再搬运 latent，再重建几何。**

### 13. 这一整条推理链的维度顺一遍

输入：

`partial_points: (B,2048,3)`

编码：

`-> src.centers: (B,64,3)`
`-> src.tokens: (B,64,384)`

标准化：

`-> src_tokens: (B,64,384)`

多步 transport：

`-> pred_tokens: (B,64,384)`

反标准化：

`-> pred_tokens_dec: (B,64,384)`

decoder：

`-> pred_points: (B,8192,3)`  
或 `(B,16384,3)`

这就是最终输出。

### 14. 一句白话总结

Stage-2 推理时真正走的路径很干净：

**把 partial 点云编码成 latent，然后在 latent 空间里沿着学到的桥一步一步走，最后再交给 Stage-1 训练好的 decoder，把它还原成完整点云。**
