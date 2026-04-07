# 讲解Stage1的文档

这份文档整理的是从“Stage-1 第一步”开始，到 decoder 默认输出、再到 `refine` 模式的完整逐步讲解。内容按原本问答顺序展开，尽量保留原有的比喻、解释节奏和细节。

相关源码主要在：

- [pointmae_encoder.py](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/pointmae_encoder.py)
- [completion_decoder.py](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py)
- [latent_normalizer.py](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_normalizer.py)
- [stage1_train.py](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage1_train.py)

---

## 第一步：输入是 `(B, N, 3)` 然后呢

第一步是 `Group`，也就是把整片点云切成很多局部小块。

代码在：
[pointmae_encoder.py#L29](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/pointmae_encoder.py#L29)

输入是：

`points: (B, N, 3)`

这里：
`B` 是 batch size  
`N` 是点数  
`3` 是每个点的 `(x, y, z)`

默认配置下：
`num_group = 64`
`group_size = 32`
见 [pointmae_encoder.py#L196](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/pointmae_encoder.py#L196)

这一步分两小步。

### 1. 先选 64 个中心点 `centers`

用 FPS，也就是最远点采样：
[pointmae_encoder.py#L24](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/pointmae_encoder.py#L24)

输出：

`centers: (B, 64, 3)`

你可以理解成：
原来一整个点云太密、太乱，先挑 64 个“代表性位置”。

### 2. 再给每个中心点找 32 个邻居

代码在：
[pointmae_encoder.py#L43](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/pointmae_encoder.py#L43)

对每个 center，从原始点云里找最近的 32 个点。

输出先是：

`(B, 64, 32, 3)`

含义是：
每个 batch 里有 64 个局部块  
每个局部块里有 32 个点  
每个点 3 维坐标

但这里还有一个很重要的操作：
它不是直接存原始邻居坐标，而是做了

`neighbor - center`

见：
[pointmae_encoder.py#L54](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/pointmae_encoder.py#L54)

所以最终 `neighborhoods` 还是：

`(B, 64, 32, 3)`

但这个 3 维不再是绝对坐标，而是“相对这个中心点的偏移”。

这是什么意思？

比如某个 center 在飞机机翼上，那它周围 32 个点就会被改写成：

“这个邻居点相对中心点偏左多少、偏上多少、偏前多少”

这样做的好处是：
网络先学局部形状，而不是先被绝对位置干扰。

所以第一步结束后，你可以把结果记成两样东西：

`centers: (B, 64, 3)`
`neighborhoods: (B, 64, 32, 3)`

一句白话总结：

输入一整片点云 `(B, N, 3)`  
先抽 64 个代表位置  
再围着每个位置取 32 个邻居  
于是把整片点云切成了 64 个局部小 patch。

---

## 第二步：`(B, 64, 32, 3)` 进入 `LocalEncoder`

下一步就是把这 `64` 个局部 patch，各自编码成一个 feature token。

代码在：
[pointmae_encoder.py#L66](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/pointmae_encoder.py#L66)

输入是上一步的：

`point_groups: (B, 64, 32, 3)`

这里的意思是：

`B` 个样本  
每个样本 `64` 个局部块  
每个局部块 `32` 个点  
每个点 `3` 维坐标

这一层叫 `LocalEncoder`，它的目标很简单：

把“每个局部块里的 32 个点”压成“一个 384 维向量”。

也就是最后要变成：

`(B, 64, 384)`

先看第一步 reshape。

### 1. 先把 batch 和 group 合并

代码：
[pointmae_encoder.py#L84](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/pointmae_encoder.py#L84)

```python
batch_size, num_group, group_size, _ = point_groups.shape
point_groups = point_groups.reshape(batch_size * num_group, group_size, 3)
```

所以：

`(B, 64, 32, 3) -> (B*64, 32, 3)`

为什么要这么做？

因为接下来网络想“逐个 patch”处理。  
每个 patch 本质上都是一个小点云，大小是 `(32, 3)`。

所以它把原来一个 batch 里的 64 个 patch 全摊开，变成：

“总共有 `B*64` 个小点云，每个小点云有 32 个点，每个点 3 维”

这只是为了方便后面统一送进卷积层，不是语义变化。

### 2. 再转置成 Conv1d 需要的格式

代码：
[pointmae_encoder.py#L86](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/pointmae_encoder.py#L86)

```python
feature = self.first_conv(point_groups.transpose(2, 1))
```

原来是：

`(B*64, 32, 3)`

转置后：

`(B*64, 3, 32)`

为什么要转？

因为 `nn.Conv1d` 的输入格式是：

`(batch, channels, length)`

这里它把：

`3` 当成通道数 channels  
`32` 当成长度 length

也就是把这个局部 patch 看成：

“有 32 个位置，每个位置有 3 维特征”

虽然这里叫卷积，但因为卷积核大小是 `1`，本质上更像对每个点单独做一个共享的 MLP。

### 3. 第一段 1x1 Conv：3 -> 128 -> 256

代码在：
[pointmae_encoder.py#L70](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/pointmae_encoder.py#L70)

```python
self.first_conv = nn.Sequential(
    nn.Conv1d(3, 128, 1),
    nn.BatchNorm1d(128),
    nn.ReLU(inplace=True),
    nn.Conv1d(128, 256, 1),
)
```

输入：
`(B*64, 3, 32)`

经过第一层：
`Conv1d(3, 128, 1)`
变成：

`(B*64, 128, 32)`

意思是：
每个点原来只有 3 维相对坐标，现在被映射成 128 维局部特征。

再经过第二层：
`Conv1d(128, 256, 1)`

变成：

`(B*64, 256, 32)`

到这里，每个局部块里的 32 个点，每个点已经不是单纯 xyz 了，而是一个 256 维的 learned feature。

你可以把它理解成：

原来每个点只知道“我在哪”  
现在每个点开始带有“我可能属于边缘、平面、弯曲结构”的局部语义信息。

### 4. 对 32 个点做 max pooling，抽一个局部全局特征

代码：
[pointmae_encoder.py#L87](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/pointmae_encoder.py#L87)

```python
feature_global = torch.max(feature, dim=2, keepdim=True)[0]
```

输入：
`feature: (B*64, 256, 32)`

对最后那个维度 `32` 做 max，也就是在一个 patch 的 32 个点里，每个通道取最大值。

输出：

`feature_global: (B*64, 256, 1)`

这一步的含义很重要：

它把“这个 patch 整体长什么样”提炼成一个单独向量。

比如一个 patch 是平面边缘、尖角、圆柱面附近，这个全局向量会尽量把这种整体模式提出来。

### 5. 把全局特征复制回每个点，再和点特征拼接

代码：
[pointmae_encoder.py#L88](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/pointmae_encoder.py#L88)

```python
feature = torch.cat([feature_global.expand(-1, -1, group_size), feature], dim=1)
```

先把：
`feature_global: (B*64, 256, 1)`

复制成：
`(B*64, 256, 32)`

然后和原始 `feature: (B*64, 256, 32)` 在通道维拼接：

得到：

`(B*64, 512, 32)`

为什么这么做？

因为现在每个点都同时拿到了两类信息：

1. 自己的局部点特征
2. 整个 patch 的全局摘要

这有点像：
“我不光知道我自己这个点的局部情况，我还知道我所在这整个小块的大概形状”

这样后面再做一层映射，能更好地产生 patch 级表示。

### 6. 第二段 1x1 Conv：512 -> 512 -> 384

代码在：
[pointmae_encoder.py#L76](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/pointmae_encoder.py#L76)

```python
self.second_conv = nn.Sequential(
    nn.Conv1d(512, 512, 1),
    nn.BatchNorm1d(512),
    nn.ReLU(inplace=True),
    nn.Conv1d(512, encoder_channel, 1),
)
```

这里 `encoder_channel = 384`

输入：
`(B*64, 512, 32)`

经过：
`Conv1d(512, 512, 1)`
还是：
`(B*64, 512, 32)`

再经过：
`Conv1d(512, 384, 1)`

变成：

`(B*64, 384, 32)`

现在每个 patch 里的 32 个点，每个点对应 384 维特征。

### 7. 再做一次 max pooling，把整个 patch 压成一个 token

代码：
[pointmae_encoder.py#L90](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/pointmae_encoder.py#L90)

```python
feature_global = torch.max(feature, dim=2, keepdim=False)[0]
```

输入：
`(B*64, 384, 32)`

对 32 个点做 max pooling，输出：

`(B*64, 384)`

这一步就是在说：

“这个局部 patch 的 32 个点，最后合并成一个 384 维向量”

这就是一个 patch token。

### 8. reshape 回 batch 形式

代码：
[pointmae_encoder.py#L91](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/pointmae_encoder.py#L91)

```python
return feature_global.reshape(batch_size, num_group, self.encoder_channel)
```

所以：

`(B*64, 384) -> (B, 64, 384)`

到这里，LocalEncoder 的工作就完成了。

最终输出：

`group_tokens: (B, 64, 384)`

意思是：

每个样本有 64 个局部 patch  
每个 patch 被压成一个 384 维 token

### 你可以把这一步理解成什么

上一步我们只是把大点云切成了 64 个小块。  
这一步是在做：

“把每个小块从一堆点，变成一个向量摘要”

也就是：

`(32 个点的局部几何)`  
压缩成  
`(1 个 384 维局部语义 token)`

所以到这里，模型已经不再直接处理单个点了，而是开始处理“局部块的 token 序列”。

### 这一小步的完整维度链

从上一步输出接过来：

`(B, 64, 32, 3)`
`-> reshape`
`(B*64, 32, 3)`
`-> transpose`
`(B*64, 3, 32)`
`-> first_conv`
`(B*64, 256, 32)`
`-> max pool`
`(B*64, 256, 1)`
`-> expand + concat`
`(B*64, 512, 32)`
`-> second_conv`
`(B*64, 384, 32)`
`-> max pool`
`(B*64, 384)`
`-> reshape back`
`(B, 64, 384)`

一句白话总结：

每个 patch 里原本有 32 个点。  
LocalEncoder 把这 32 个点做特征提取、汇总，再压缩成 1 个 384 维向量。  
于是整片点云就从“很多点”变成了“64 个 patch token”。

---

## 第三步：给 `centers` 做位置编码，再和 `group_tokens` 一起送进 Transformer

下一步是：

已经有了局部 token `group_tokens: (B, 64, 384)`  
还要把每个 token 对应的空间位置 `centers: (B, 64, 3)` 也编码进去，接着送进 Transformer。

代码在：
[pointmae_encoder.py#L159](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/pointmae_encoder.py#L159)

这一步非常重要，因为前面 LocalEncoder 主要提取的是“局部形状长什么样”，但它并不知道“这个局部块在整架飞机的哪里”。

比如两个 patch 都像平面：
一个可能在机翼
一个可能在机尾
如果不给位置，网络很难区分。

### 这一步的输入有两个

1. 局部形状 token
`group_tokens: (B, 64, 384)`

2. 每个 token 的中心坐标
`centers: (B, 64, 3)`

### 先看位置编码怎么做

代码在：
[pointmae_encoder.py#L163](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/pointmae_encoder.py#L163)

```python
self.pos_embed = nn.Sequential(
    nn.Linear(3, 128),
    nn.GELU(),
    nn.Linear(128, trans_dim),
)
```

这里 `trans_dim = 384`

所以它做的是：

`(B, 64, 3)`
`-> Linear(3,128)`
`-> GELU`
`-> Linear(128,384)`
`-> (B, 64, 384)`

也就是说，每个 center 的三维坐标 `(x,y,z)` 会被映射成一个 384 维的位置向量。

输出记作：

`pos: (B, 64, 384)`

### 为什么要把 3 维坐标升到 384 维

因为后面要和 token 一起进入 Transformer。  
而 token 本身的维度是 384，所以位置编码也得变成 384 维，才能相加。

你可以把它理解成：

原来的 token 说的是“这块局部像什么”  
现在的 `pos` 说的是“这块局部在哪里”

两者必须在同一个特征空间里，才能融合。

### 接下来怎么送进 Transformer

代码在：
[pointmae_encoder.py#L186](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/pointmae_encoder.py#L186)

```python
group_tokens = self.encoder(neighborhoods)
pos = self.pos_embed(centers)
x = self.blocks(group_tokens, pos)
return self.norm(x)
```

也就是：

`group_tokens: (B,64,384)`
`pos: (B,64,384)`

一起送进：
`TransformerEncoder`

### TransformerEncoder 是怎么用这个 pos 的

代码在：
[pointmae_encoder.py#L153](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/pointmae_encoder.py#L153)

```python
for block in self.blocks:
    x = block(x + pos)
return x
```

这里意思很直接：

每一层 Transformer block 之前，都先做：

`x + pos`

维度不变：

`(B,64,384) + (B,64,384) = (B,64,384)`

然后再送进 block。

这说明：
位置编码不是只加一次，而是每层都加。  
也就是每一层都反复提醒网络：

“当前这个 token 不光有局部几何特征，它还处在这个空间位置上。”

### 一个 block 里面做什么

代码在：
[pointmae_encoder.py#L132](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/pointmae_encoder.py#L132)

```python
x = x + self.drop_path(self.attn(self.norm1(x)))
x = x + self.drop_path(self.mlp(self.norm2(x)))
```

这是标准 Transformer block，分两段：

1. Self-Attention
2. MLP

维度都不变，始终是：

`(B, 64, 384)`

### 先讲 Self-Attention 这一小步

输入：
`x: (B, 64, 384)`

含义是：
每个样本有 64 个 token  
每个 token 384 维

Self-Attention 的作用是：

让 64 个局部 token 彼此看一眼，互相交换信息。

比如：
机翼附近的 patch 可以参考机身 patch  
椅背 patch 可以参考椅腿 patch

这样每个 token 不再只知道自己这个局部，而开始带有全局上下文。

### Self-Attention 内部维度怎么变

代码在：
[pointmae_encoder.py#L121](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/pointmae_encoder.py#L121)

```python
qkv = self.qkv(x).reshape(batch_size, num_tokens, 3, self.num_heads, dim // self.num_heads).permute(2, 0, 3, 1, 4)
q, k, v = qkv[0], qkv[1], qkv[2]
```

默认：
`num_heads = 6`
`dim = 384`

所以每个 head 的维度是：

`384 / 6 = 64`

于是：

输入：
`x: (B, 64, 384)`

先过一个线性层：
`Linear(384, 384*3)`

得到：
`(B, 64, 1152)`

再 reshape 成：
`(B, 64, 3, 6, 64)`

这里的 `3` 表示 Q、K、V 三份。

再 permute 后拆出来：

`q: (B, 6, 64, 64)`
`k: (B, 6, 64, 64)`
`v: (B, 6, 64, 64)`

这四个 `64` 很容易搞混，我给你拆开：

第一个 `64` 是 token 数量  
也就是 64 个局部块

第二个 `64` 是每个 head 的通道维度

所以可以理解成：

每个样本有 6 个注意力头  
每个头都在 64 个 token 之间做关系计算  
每个 token 在该头里是 64 维表示

接着 attention 会得到：

输出还是：
`(B, 6, 64, 64)`

再拼回去：
`(B, 64, 384)`

最后过一个投影层：
还是 `(B, 64, 384)`

### 这一步到底在语义上发生了什么

这一层之后，每个 token 不再只是“自己那块 patch 的局部编码”。

它变成了：

“自己那块 patch 的局部编码 + 和其他 63 个 patch 交流后的全局上下文编码”

比如原来某个 patch 只知道：
“我像一个平面”

经过 self-attention 后，它可能进一步知道：
“我像一个平面，而且附近还有细长结构和对称部件，所以我大概率处在机翼区域”

### 然后是 MLP

Self-Attention 后，接着做：

`x = x + MLP(LN(x))`

MLP 代码在：
[pointmae_encoder.py#L94](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/pointmae_encoder.py#L94)

默认 `mlp_ratio=4.0`，所以：

`384 -> 1536 -> 384`

也就是：

输入：
`(B,64,384)`

逐 token 地做两层全连接：
第一层升维到 1536  
激活  
再降回 384

输出还是：
`(B,64,384)`

这一步的作用是：
在 token 内部做更强的非线性特征变换。

如果说 attention 更像“token 和 token 交流”  
那 MLP 更像“每个 token 自己消化一下刚才收到的信息”

### 这样的 block 有多少层

代码在：
[pointmae_encoder.py#L148](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/pointmae_encoder.py#L148)

默认：

`depth = 12`

所以会重复 12 次。

整个过程中，形状一直不变：

`(B,64,384) -> (B,64,384) -> ... -> (B,64,384)`

最后再过一层 LayerNorm：
[pointmae_encoder.py#L189](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/pointmae_encoder.py#L189)

输出仍然是：

`tokens: (B,64,384)`

### 这一步结束后，encoder 输出了什么

最终在：
[pointmae_encoder.py#L245](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/pointmae_encoder.py#L245)

返回：

`tokens: (B,64,384)`
`centers: (B,64,3)`
`neighborhoods: (B,64,32,3)`

其中最关键的是：

`tokens: (B,64,384)`

这就是 Stage-1 要送给 decoder 的 latent 表示。

### 一句白话总结这一步

前一步已经把每个局部 patch 压成了一个 token。  
这一步再把每个 token 的空间位置编码进去，并让 64 个 token 通过 Transformer 彼此交流。  
于是输出的 token 不再只是“局部形状摘要”，而是“带有全局上下文的完整物体 latent token”。

### 这一小步的维度链

`group_tokens: (B,64,384)`
`centers: (B,64,3)`
`-> pos_embed`
`pos: (B,64,384)`
`-> x + pos`
`(B,64,384)`
`-> 12 layers transformer`
`(B,64,384)`
`-> LayerNorm`
`tokens: (B,64,384)`

---

## 第四步：对 `(B,64,384)` 做 latent normalization

下一步就是：

encoder 已经输出了  
`tokens: (B, 64, 384)`

Stage-1 不会立刻把它喂给 decoder，很多时候会先做一步 `latent normalization`。

代码在：
[latent_normalizer.py](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_normalizer.py)
训练里调用的位置在：
[stage1_train.py#L323](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage1_train.py#L323)
[stage1_train.py#L324](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage1_train.py#L324)
[stage1_train.py#L325](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage1_train.py#L325)

先说结论：

这一步不改变张量形状。  
输入输出都是：

`(B, 64, 384)`

它改变的是数值分布，不是维度。

### 先讲它为什么存在

你可以把 encoder 输出的 `384` 个通道理解成 384 根“latent 特征尺子”。

问题是，这 384 根尺子的量纲可能很不统一：

有的通道数值范围很大  
有的通道数值范围很小  
有的通道均值偏正  
有的通道均值偏负

如果 decoder 直接吃这种分布很乱的 latent，会有两个问题：

1. 学起来不稳定
2. Stage-2 transport 以后更难对齐到 decoder 喜欢的 latent 分布

所以这里先做一个全局的 channel-wise 标准化，把 latent 空间整理干净。

### 注意：这不是 LayerNorm

很多人会把它和 Transformer 里的 LayerNorm 混掉，但它们完全不是一回事。

这里的 normalizer 是：

对整个训练集里所有 complete latent 统计 384 个通道的全局 mean/std。  
然后训练时每个 token 都用这同一组 mean/std 去标准化。

也就是说，它是“数据级别”的统计，不是每个样本现场算的。

### 它先怎么得到 mean 和 std

代码在：
[latent_normalizer.py#L33](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_normalizer.py#L33)

在 Stage-1 开始前，如果开了 `--latent-normalize`，就会检查 normalizer 有没有统计过；如果没有，就先跑一遍统计流程。见：
[stage1_train.py#L240](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage1_train.py#L240)

统计时做的是：

1. 从 dataloader 里取 `complete_points`
2. 用冻结 encoder 编成 `tokens`
3. `tokens` 形状是 `(B,64,384)`
4. 把它 reshape 成 `(-1,384)`

见：
[latent_normalizer.py#L50](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_normalizer.py#L50)
[latent_normalizer.py#L52](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_normalizer.py#L52)

为什么要 reshape 成 `(-1,384)`？

因为它想把：

所有 batch  
所有样本  
所有 token 位置

全部摊平，统一看成很多个 384 维向量。

比如原来是：

`(B,64,384)`

摊平以后就是：

`(B*64,384)`

再加上很多 batch，就相当于积累成一个巨大的“latent 样本池”。

然后对每个通道分别算：

`mean: (384,)`
`std: (384,)`

对应代码：
[latent_normalizer.py#L60](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_normalizer.py#L60)
[latent_normalizer.py#L61](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_normalizer.py#L61)

所以最后 normalizer 里面保存的是：

`mean.shape = (384,)`
`std.shape = (384,)`

### 训练时真正怎么标准化

代码在：
[latent_normalizer.py#L27](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/latent_normalizer.py#L27)

```python
return (x - self.mean) / (self.std + self.eps)
```

这里输入：
`x: (B,64,384)`

`mean: (384,)`
`std: (384,)`

PyTorch 会自动 broadcast，所以实际做的是：

对最后一个维度 384 个通道逐通道减均值、除标准差。

形状不变：

`(B,64,384) -> (B,64,384)`

但数值分布会变得更规整。

### 可以把它想成什么

假设第 17 个通道原来通常在 `100` 左右波动，第 93 个通道原来通常在 `0.02` 左右波动。

那 decoder 直接学的时候就会觉得：
有的通道特别吵，有的通道几乎没动静。

做完 normalize 后，所有通道都大致被拉到一个更统一的尺度上。  
decoder 学起来就更像在一个“规整坐标系”里工作。

### 在 Stage-1 的训练循环里，这一步怎么接上

代码在：
[stage1_train.py#L321](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage1_train.py#L321)

```python
with torch.no_grad():
    enc = encoder(complete_points)
    tokens = enc.tokens
    if normalizer is not None:
        tokens = normalizer.normalize(tokens)
```

所以这里的数据流是：

`complete_points`
`-> encoder`
`-> tokens: (B,64,384)`
`-> normalize`
`-> normalized_tokens: (B,64,384)`

### 为什么 normalizer 的统计要用 complete 点云，而不是 partial

因为 Stage-1 的 decoder 学的是：

`complete latent -> complete points`

所以它最应该适应的是“完整形状 latent 的分布”。

以后 Stage-2 transport 的目标，也是把 partial latent 推到这个 complete latent 分布附近。

所以这个 normalizer 的 mean/std 必须来自 complete latent，而不是 partial latent。  
这点在仓库说明里也明确写了。

### 这一步会不会改变 token 的语义

不会改变“它表示什么”，但会改变“它怎么表示”。

也就是说：

标准化前后，这 64 个 token 仍然代表那 64 个局部块。  
只是每个通道的数值被平移和缩放到了更统一的尺度。

有点像把一组数据从“厘米、米、毫米混着写”变成统一单位。

### 这一小步的维度链

输入：

`tokens: (B,64,384)`

normalizer 参数：

`mean: (384,)`
`std: (384,)`

输出：

`normalized_tokens: (B,64,384)`

所以：

维度完全没变  
只是数值分布变了

### 一句白话总结

这一步不是提特征，也不是变形状。  
它是在给 encoder 输出的 latent token 做“全局通道标准化”，让 decoder 后面看到的 latent 更稳定、更规整，也方便 Stage-2 以后在同一个 latent 坐标系里做 transport。

---

## 第五步：normalize 之后再加 latent noise

下一步是：

normalize 完之后，Stage-1 训练时通常还会给 latent 加一点高斯噪声。

代码在：
[stage1_train.py#L326](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage1_train.py#L326)
[stage1_train.py#L327](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage1_train.py#L327)

还是先说结论：

这一步也不改变形状。  
输入输出仍然是：

`(B, 64, 384)`

它改变的是数值，让 decoder 不要只会处理“完美 latent”。

### 代码是怎么写的

训练循环里：

```python
if args.latent_noise_std > 0:
    tokens = tokens + args.latent_noise_std * torch.randn_like(tokens)
```

也就是：

如果 `latent_noise_std = 0.1`

那么做的就是：

`tokens_noisy = tokens + 0.1 * noise`

其中：

`noise.shape = (B,64,384)`

所以：

`(B,64,384) + (B,64,384) -> (B,64,384)`

维度完全不变。

### 这一步在干什么

它的意思很简单：

我故意把 encoder 输出的 complete latent 弄脏一点点，  
再让 decoder 依然去重建正确的完整点云。

也就是训练目标从：

“看到完美 latent，就重建完整点云”

变成：

“看到带一点扰动的 latent，也尽量重建完整点云”

这其实是一种 latent-space 的鲁棒性训练。

### 为什么要故意加噪声

这是 Stage-1 里一个非常关键、但很容易被忽略的设计。

原因是：

Stage-2 的 transport 模型输出的 latent，不可能和真实 complete latent 一模一样。

哪怕 transport 学得很好，它输出的也只是：

“接近 complete latent 的 latent”

而不是精确的 ground truth complete latent。

如果 Stage-1 decoder 只在“特别干净、特别标准”的 latent 上训练，  
那 Stage-2 一旦送来一个稍微偏一点的 latent，decoder 可能就会很脆，输出明显变差。

所以加噪声本质上是在提前训练 decoder：

“以后有人给你一个不那么完美的 latent，你也得扛得住。”

### 你可以把它类比成什么

假设 decoder 是一个很会翻译的人。

如果你平时只给他最标准、最工整的句子训练，  
那现实里别人说话一旦有口音、有杂音，他就容易听不懂。

加 latent noise 就像是故意在训练时加一点口音、背景噪音，  
让 decoder 学会容忍输入偏差。

### 噪声是加在哪一步之后

顺序是：

`complete_points`
`-> encoder`
`-> tokens`
`-> normalize`
`-> noise`
`-> decoder`

也就是：

先标准化，再加噪声。

为什么不是先加噪声再标准化？

因为标准化之后，各通道尺度更统一。  
这时候加一个固定标准差的高斯噪声，才更容易控制扰动强度。

否则有些通道本来数值特别大，有些特别小，  
同样的噪声幅度会对不同通道产生很不均匀的影响。

### 噪声的形状怎么理解

如果当前 token 是：

`tokens: (B,64,384)`

那噪声也是：

`noise: (B,64,384)`

也就是说：

每个样本  
每个 token  
每个通道

都会加一个随机扰动。

不是只给整个样本加一个噪声，也不是只给某几个 token 加。  
而是对整个 latent tensor 做逐元素扰动。

### 这会不会把 latent 语义破坏掉

会有一点扰动，但设计上就是想要“轻微破坏”，不是“完全打乱”。

默认噪声比较小，比如 `0.1`，目的不是毁掉语义，而是让 latent 从“理想点”晃到“附近邻域”。

换句话说，它训练的是 decoder 在 complete latent 分布附近的小范围容错能力。

### 这一步和去噪 autoencoder 有点像吗

有点像，但不完全一样。

相似点是：

都在做“给中间表示加噪，再要求恢复正确输出”。

不同点是：

这里不是从原始点云输入端加噪，  
而是在 encoder 之后的 latent 空间加噪。

所以它更像：

latent denoising / latent robustness training

而不是经典输入去噪。

### 训练和验证时这一步一样吗

不一样。

训练时会加噪声：
[stage1_train.py#L326](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage1_train.py#L326)

但验证时没有加：
[stage1_train.py#L160](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage1_train.py#L160) 到 [stage1_train.py#L167](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage1_train.py#L167)

验证流程是：

`complete -> encoder -> normalize -> decoder`

没有额外 noise。

这是合理的，因为验证时我们想测的是：
当输入是真实 complete latent 时，decoder 的标准重建能力。

而训练时加噪声，是为了增强鲁棒性。

### 这一步结束后，接下来送给 decoder 的是什么

就是：

`tokens_noisy: (B,64,384)`

再加上 encoder 之前就给出的：

`enc.centers: (B,64,3)`

然后进入 decoder：

`decoder(tokens_noisy, enc.centers)`

对应：
[stage1_train.py#L331](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage1_train.py#L331)

### 一句白话总结

这一步就是：

给标准化后的 latent token 轻轻“晃一下”，  
让 decoder 不要只会吃完美输入，而要学会在 latent 有偏差时依然稳定重建完整点云。

### 这一小步的维度链

`tokens: (B,64,384)`
`-> normalize`
`(B,64,384)`
`-> + gaussian noise`
`(B,64,384)`

维度没变，数值变得稍微更“脏”一点。

---

## 第六步：decoder 先造出一组 `query: (B,256,384)`

下一步正式进入 decoder。

现在 decoder 拿到两样东西：

`encoder_tokens: (B, 64, 384)`  
`encoder_centers: (B, 64, 3)`

对应 forward 在：
[completion_decoder.py#L177](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L177)

```python
batch_size = encoder_tokens.shape[0]
query = self.query_embed.expand(batch_size, -1, -1)
context = encoder_tokens + self.center_pos_embed(encoder_centers)
```

这一步做了两件事：

1. 造出一组 decoder query
2. 把 encoder token 和 center 位置编码融合成 context

先只讲第 1 件事。

### 1. query 是什么

decoder 里有一个可学习参数：

[completion_decoder.py#L141](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L141)

```python
self.query_embed = nn.Parameter(torch.zeros(1, num_queries, hidden_dim))
```

默认：
`num_queries = 256`
`hidden_dim = 384`

所以它的形状是：

`query_embed: (1, 256, 384)`

注意，这不是从输入点云算出来的。  
它是模型自己学出来的一组参数。

你可以把它理解成：

decoder 预先准备好的 256 个“生成槽位”  
或者 256 个“要去读取 latent 信息的探针”

### 2. 为什么是 `(1, 256, 384)`

这里每个维度含义是：

`1`：先只存一份模板  
`256`：一共有 256 个 query  
`384`：每个 query 是 384 维向量

也就是说，模型里常驻着 256 个可学习向量。

它们一开始是随机初始化的，训练过程中慢慢学成：

“第 1 个 query 更擅长负责某类结构”
“第 57 个 query 更擅长负责另一类结构”
……

当然不是严格一一对应某个语义部件，但直觉上可以这么想。

### 3. 为什么要 expand 到 batch

代码：
[completion_decoder.py#L179](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L179)

```python
query = self.query_embed.expand(batch_size, -1, -1)
```

于是：

`(1, 256, 384) -> (B, 256, 384)`

这不是复制出新的可训练参数，而是把同一套 query 模板应用到 batch 里的每个样本。

也就是说：

对于 batch 里的每个物体，decoder 都从同一套 256 个 query 出发。  
然后这些 query 再根据当前样本的 encoder context，变成这个样本专属的 query feature。

### 4. 这组 query 一开始有没有物体信息

没有。

刚 expand 出来的：

`query: (B,256,384)`

本质上只是 256 个可学习模板向量，  
它们还没看 encoder token，也还没看当前物体。

所以这一刻的 query 更像一组“空白但有偏好的槽位”。

真正和当前样本绑定，要等后面的 self-attention / cross-attention。

### 5. 为什么 decoder 不直接拿 64 个 encoder token 输出点，而要再造 256 个 query

这是 query-based decoder 的关键设计。

encoder token 的任务是“表示输入 latent 信息”  
decoder query 的任务是“组织输出点云”

这两个角色其实不一样。

64 个 encoder token 更像是：
对输入物体做了 64 个局部摘要

而 256 个 decoder query 更像是：
我要用 256 个生成槽位去构造输出形状

为什么要分开？

因为输出点云往往比输入 token 更密。  
比如这里：

encoder token 是 64 个  
decoder query 是 256 个  
最终输出点是 8192 个

如果只用 64 个 encoder token 直接出点，表达会比较受限。  
增加 query 数，相当于给 decoder 更细的生成分工。

### 6. query 最后和点数是什么关系

默认：

`num_queries = 256`
`output_points = 8192`

所以：

`points_per_query = 8192 / 256 = 32`

见：
[completion_decoder.py#L138](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L138)

这意味着：

最后每个 query 大致负责生成 32 个点。

所以从直觉上，你可以把 query 想成：

“最终点云的 256 个生成小单元”

每个小单元从 encoder latent 里取信息，再各自产生一小片点，最后拼起来形成完整点云。

### 7. 现在这一刻的 query 维度变化

就这一小步而言，非常简单：

模型参数里存着：
`query_embed: (1,256,384)`

进入当前 batch 后：
`query = (B,256,384)`

语义上：
从“全局共享的一套 query 模板”
变成“这个 batch 中每个样本都拿到一套相同初始 query”

### 8. 白话理解这一步

你可以把 decoder 想成一个施工队。

encoder 给的是：
“这栋建筑的设计信息，压缩成了 64 个摘要 token”

decoder 不会直接让这 64 个摘要自己去盖楼，  
而是先派出 256 个施工小组，也就是 256 个 query。

每个小组先拿着自己的初始工具包出发。  
接下来它们会去读 encoder 提供的信息，然后各自负责生成一部分结构。

所以 query 的作用不是“表示输入”，而是“承载输出生成过程”。

### 这一小步的维度链

`query_embed: (1,256,384)`
`-> expand by batch`
`query: (B,256,384)`

---

## 第七步：构造 `context = encoder_tokens + center_pos_embed(encoder_centers)`

下一步讲 decoder 里的 `context`。

刚才讲的是：

`query = (B,256,384)`

现在看另一行：

[completion_decoder.py#L180](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L180)

```python
context = encoder_tokens + self.center_pos_embed(encoder_centers)
```

这里输入有两个：

`encoder_tokens: (B,64,384)`  
`encoder_centers: (B,64,3)`

目标是构造出 decoder 读取的信息源 `context`。

### 1. 为什么 decoder 还要再处理一次 centers

你可能会想：

encoder 里不是已经给 centers 加过位置编码了吗，为什么 decoder 这里还要再来一次？

原因是：

encoder 的位置编码是为了帮助 encoder 自己在编码阶段理解 token 的空间关系。  
而 decoder 现在是另一个模块，它也需要明确知道：

“这 64 个 token 分别处在物体的哪里”

虽然 `encoder_tokens` 里已经隐含了一些位置信息，但这里再显式加一次 position embedding，会让 decoder 的 cross-attention 更容易利用空间结构。

你可以理解成：

encoder 阶段的位置编码，是给 encoder 自己看的。  
decoder 阶段的这次位置编码，是给 decoder 自己看的。

### 2. center_pos_embed 怎么做

代码在：
[completion_decoder.py#L142](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L142)

```python
self.center_pos_embed = nn.Sequential(
    nn.Linear(3, 128),
    nn.GELU(),
    nn.Linear(128, hidden_dim),
)
```

默认 `hidden_dim = 384`

所以：

`encoder_centers: (B,64,3)`
`-> Linear(3,128)`
`-> GELU`
`-> Linear(128,384)`
`-> (B,64,384)`

记作：

`center_pos: (B,64,384)`

含义是：
把每个中心点的 xyz 位置，变成一个 384 维的位置向量。

### 3. 然后和 encoder_tokens 相加

`encoder_tokens: (B,64,384)`  
`center_pos: (B,64,384)`

相加后：

`context: (B,64,384)`

维度完全不变。

但语义变成了：

每个 context token 同时包含：

1. 这个局部 patch 的几何语义
2. 这个 patch 所在的空间位置

### 4. 为什么是加法，不是拼接

这里用的是：

`token + pos`

而不是：

`concat(token, pos)`

主要因为：

1. 保持维度固定在 384，方便后面 attention
2. 这是 Transformer 里很常见的位置融合方式
3. 语义上它是在说：“位置是这个 token 的一部分属性”，而不是另一条独立分支

如果拼接，就会变成 `(B,64,768)`，后面整个 decoder 维度都要改。

### 5. 这里的 context 到底是什么

可以把 `context` 理解成：

decoder 之后 cross-attention 要去读取的“记忆库”。

也就是：
encoder 把完整形状压成了 64 条记忆，每条记忆 384 维。  
现在又把每条记忆对应的位置加进去，形成真正给 decoder 使用的上下文表示。

所以：

`context: (B,64,384)`

它不是输出点，也不是 query。  
它更像是 decoder 可以反复查阅的 latent memory。

### 6. query 和 context 的角色区别

到这里，decoder 里已经有两组东西了：

`query: (B,256,384)`  
`context: (B,64,384)`

两者分工很清楚：

`query`
负责生成，代表 256 个输出槽位

`context`
负责提供信息，代表 encoder 压缩出来的 64 条 latent 记忆

你可以把它们想成：

query 是提问的人  
context 是资料库

后面 cross-attention 就是在做：

“256 个 query 分别去 64 条 context 里查自己需要的信息”

### 7. 这一刻 decoder 里已经准备好了什么

现在 decoder 真正进入 block 之前，手里有：

`query: (B,256,384)`  
`context: (B,64,384)`

接下来每个 `QueryDecoderBlock` 就会对这两者做：

1. query 内部 self-attention
2. query 对 context 的 cross-attention
3. query 自己再过 MLP

重点注意：

在 decoder block 里，变化的主要是 `query`。  
`context` 通常保持不变，作为固定信息源被读取。

### 8. 白话理解这一步

如果把 decoder 想成一群施工小组：

刚才的 256 个 query 是 256 个施工小组。  
现在的 `context` 就像一份带坐标标注的施工资料库。

资料库里有 64 条信息，每条都写着：

“这个局部结构长什么样”
“它在整体物体的哪个位置”

接下来每个施工小组就会去翻这份资料库，决定自己该生成什么。

### 这一小步的维度链

`encoder_centers: (B,64,3)`
`-> center_pos_embed`
`(B,64,384)`

`encoder_tokens: (B,64,384)`
`+ center_pos`
`= context: (B,64,384)`

所以到这一步为止，decoder 里准备好的两组核心张量是：

`query: (B,256,384)`  
`context: (B,64,384)`

---

## 第八步：decoder block 里先做 query 的 self-attention

下一步是 decoder block 里的第一小步：

先让 `query` 自己内部做一次 `self-attention`。

代码在：
[completion_decoder.py#L91](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L91)
[completion_decoder.py#L92](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L92)

```python
query = query + self.self_attn(self.self_norm(query))
```

我们现在手里有：

`query: (B,256,384)`  
`context: (B,64,384)`

但这一步还不看 `context`，只处理 `query` 自己。

### 1. 为什么 query 先自己交流，而不是直接去看 context

因为 256 个 query 不是彼此独立的。

最终它们要一起生成一个完整点云。  
如果每个 query 都各干各的，不互相协调，就很容易出现：

1. 有些区域重复生成
2. 有些区域没人负责
3. 整体结构不协调

所以在去读取 encoder 信息之前，先让 query 之间内部沟通一下：

“我们彼此大概怎么分工”
“谁更关注哪类区域”
“整体输出结构如何协调”

这就是 self-attention 的作用。

### 2. 先做 LayerNorm

代码里先做：

`self.self_norm(query)`

这里是标准 `LayerNorm(384)`，定义在：
[completion_decoder.py#L83](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L83)

输入输出形状都不变：

`(B,256,384) -> (B,256,384)`

它只是把每个 query token 的特征做归一化，方便后面的 attention 更稳定。

这一步你可以暂时简单理解为：
“先把每个 query 的数值整理一下，再开始交流”

### 3. Self-Attention 的输入是什么

输入：

`x: (B,256,384)`

表示：
每个样本里有 256 个 query token  
每个 query token 是 384 维

代码在：
[completion_decoder.py#L54](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L54)

### 4. Self-Attention 内部第一步：线性映射成 Q、K、V

代码：
[completion_decoder.py#L56](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L56)

```python
qkv = self.qkv(x).reshape(bsz, ntok, 3, self.num_heads, dim // self.num_heads).permute(2, 0, 3, 1, 4)
q, k, v = qkv[0], qkv[1], qkv[2]
```

默认：
`num_heads = 6`
`dim = 384`

所以每个 head 的维度：

`384 / 6 = 64`

先过线性层：
`Linear(384, 384*3)`

于是：

`(B,256,384) -> (B,256,1152)`

再 reshape 成：

`(B,256,3,6,64)`

这里：

`3` 表示 Q/K/V  
`6` 表示 6 个头  
`64` 表示每个头的维度

再 permute 后得到：

`q: (B,6,256,64)`  
`k: (B,6,256,64)`  
`v: (B,6,256,64)`

你可以这样理解：

每个样本里有 6 个注意力头  
每个头都在 256 个 query 之间做关系计算

### 5. 接下来 attention 在算什么

attention 本质上是在问：

“第 i 个 query 应该多关注第 j 个 query？”

也就是说，256 个 query 两两之间会有一个相关性。

对每个 head 来说，最终会形成一个大小大致为：

`(256,256)`

的注意力关系矩阵。

加上 batch 和 heads 后，可以粗略理解为：

`(B,6,256,256)`

虽然代码里没有显式把这个矩阵存出来，而是直接用：
[completion_decoder.py#L59](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L59)

```python
x = F.scaled_dot_product_attention(q, k, v, dropout_p=drop_p)
```

但你脑子里可以把它想成：

每个 query 都在看其余 255 个 query，决定自己该吸收谁的信息。

### 6. 输出维度怎么回来

attention 输出先是：

`(B,6,256,64)`

然后代码：
[completion_decoder.py#L60](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L60)

```python
x = x.transpose(1, 2).reshape(bsz, ntok, dim)
```

于是：

`(B,6,256,64) -> (B,256,6,64) -> (B,256,384)`

再经过一个线性投影层：
还是

`(B,256,384)`

所以 self-attention 的输入输出形状都是：

`(B,256,384) -> (B,256,384)`

### 7. 再加上残差

回到 block 代码：

```python
query = query + self.self_attn(self.self_norm(query))
```

意思是：

原来的 `query`
加上
“self-attention 处理后的 query 增量”

所以输出仍然是：

`(B,256,384)`

这个残差连接的意义是：
不是把原 query 全替换掉，而是在原来的基础上做修正。

### 8. 这一步语义上发生了什么

这一层之后，每个 query 不再只是一个初始模板。  
它已经带上了其他 query 的信息。

比如：

某个 query 本来偏向负责“细长结构”  
经过 self-attention 后，它可能感知到：
“已经有别的 query 在关注附近区域了，那我应该稍微偏向另一个子区域”

所以 self-attention 做的是 query 之间的协调和重分工。

### 9. 为什么这一层在 point cloud decoder 里重要

因为点云不是一串固定顺序的词。  
生成点云时最怕的是：

1. 点堆在一起
2. 某些区域漏掉
3. 局部之间缺少全局协调

query self-attention 就是在提前建立：

“这 256 个生成槽位之间的协作关系”

所以它对整体几何布局很重要。

### 10. 白话类比

还用刚才“施工小组”的类比。

现在有 256 个施工小组，每组先拿到一个初始任务模板。  
但它们还没看施工资料库之前，先开个内部协调会：

“谁去做屋顶”
“谁去管墙面”
“谁去负责边缘细节”
“哪些工作容易重叠，先协调一下”

这就是 self-attention。

它不是在看外部资料，而是在内部统一分工。

### 11. 这一小步的维度链

输入：
`query: (B,256,384)`

`-> LayerNorm`
`(B,256,384)`

`-> qkv linear`
`(B,256,1152)`

`-> reshape`
`(B,256,3,6,64)`

`-> split`
`q,k,v: (B,6,256,64)`

`-> self-attention`
`(B,6,256,64)`

`-> merge heads`
`(B,256,384)`

`-> projection`
`(B,256,384)`

`-> residual add`
`query: (B,256,384)`

---

## 第九步：cross-attention，query 去读取 context

下一步是 decoder block 里最核心的一步：

`cross-attention`

代码在：
[completion_decoder.py#L93](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L93)

```python
query = query + self.cross_attn(self.cross_norm_q(query), self.cross_norm_ctx(context))
```

这一步开始，`query` 终于去读取 `context` 了。

我们现在有：

`query: (B,256,384)`  
`context: (B,64,384)`

你可以把这一步理解成：

256 个 decoder query 去 encoder 提供的 64 条 latent memory 里查资料，  
把当前物体的完整形状信息读出来。

### 1. 为什么叫 cross-attention

因为这次不是“自己看自己”了。

上一小步 self-attention 是：

`query <- query`

这次 cross-attention 是：

`query <- context`

也就是：

query 作为提问者  
context 作为被查询的记忆库

所以叫 cross-attention。

### 2. 先做 LayerNorm

代码里先做：

`self.cross_norm_q(query)`  
`self.cross_norm_ctx(context)`

定义在：
[completion_decoder.py#L85](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L85)
[completion_decoder.py#L86](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L86)

所以：

`query: (B,256,384) -> (B,256,384)`  
`context: (B,64,384) -> (B,64,384)`

这一步只是规范数值，形状不变。

### 3. Cross-Attention 的输入是谁

看 `CrossAttention.forward`：
[completion_decoder.py#L31](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L31)

```python
def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
```

这里很关键：

Q 来自 `query`  
K、V 来自 `context`

这和 self-attention 不一样。

具体来说：

当前输入是：

`query: (B,256,384)`  
`context: (B,64,384)`

也就是：

有 256 个 query token  
有 64 个 context token

这代表：
每个 query 都会去看 64 个 context token，决定自己该从哪些 encoder latent 里取信息。

### 4. Q、K、V 怎么变维度

代码在：
[completion_decoder.py#L34](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L34)
[completion_decoder.py#L35](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L35)
[completion_decoder.py#L36](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L36)

```python
q = self.q(query).reshape(bsz, nq, self.num_heads, dim // self.num_heads).permute(0, 2, 1, 3)
k = self.k(context).reshape(bsz, nk, self.num_heads, dim // self.num_heads).permute(0, 2, 1, 3)
v = self.v(context).reshape(bsz, nk, self.num_heads, dim // self.num_heads).permute(0, 2, 1, 3)
```

默认：
`num_heads = 6`
`dim = 384`
所以每个 head 还是 `64` 维。

于是：

`query: (B,256,384)`
经过 `Linear(384,384)` 变成：
`(B,256,384)`
再 reshape + permute：

`q: (B,6,256,64)`

`context: (B,64,384)`
经过 `Linear(384,384)` 后：
`(B,64,384)`
再 reshape + permute：

`k: (B,6,64,64)`  
`v: (B,6,64,64)`

这里你要特别注意两个不同的 “64”：

`context` 里那个 `64` 是 token 数量  
每个 head 最后的 `64` 是 head 维度

所以：

`q` 里有 256 个 query 位置  
`k/v` 里有 64 个 context 位置

### 5. attention 在算什么关系

每个 query 都会对 64 个 context token 打分。

所以对每个头来说，注意力矩阵大致可以理解成：

`(256,64)`

加上 batch 和 head 后，可以想成：

`(B,6,256,64)`

意思是：

对于每个 query，模型在问：
“这 64 个 encoder token 里，我应该重点参考哪几个？”

比如某个 query 最终负责机翼边缘附近的点，  
那它可能会对和机翼区域相关的 context token 给更高权重。

### 6. 输出怎么回来

attention 输出先是：

`(B,6,256,64)`

然后合并 head：
[completion_decoder.py#L39](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L39)

```python
x = x.transpose(1, 2).reshape(bsz, nq, dim)
```

得到：

`(B,256,384)`

再过投影层：
仍然：

`(B,256,384)`

所以 cross-attention 整体输入输出是：

`(B,256,384) -> (B,256,384)`

### 7. 再做残差相加

回到 block 里：

```python
query = query + self.cross_attn(...)
```

所以最后还是：

`query: (B,256,384)`

只不过这时 query 已经不再是“只经过内部协调的 query”了，  
而是“已经从 encoder context 里读到当前物体信息的 query”。

### 8. 这一小步语义上到底发生了什么

这是 decoder 真正“读取输入 latent”的地方。

之前的 query 只是模板，self-attention 只是模板之间协调。  
到 cross-attention 这里，它们才真正变成“针对当前这个样本”的 query。

也就是说：

同样一套初始 query 模板，  
面对不同物体时，会因为读取到不同的 context，而变成不同的 query feature。

所以 cross-attention 是 decoder 把“通用生成模板”变成“当前样本专属生成特征”的关键步骤。

### 9. 为什么是 query 去读 context，而不是 context 去读 query

因为 decoder 的目标是更新 query，让 query 最终去生成点。  
context 只是信息源，不负责生成。

所以逻辑是：

`query` 主动问  
`context` 被动提供信息

更新后的对象当然是 query，不是 context。

这也是为什么 block 里每层都主要在改 `query`，而 `context` 基本不变。

### 10. 白话类比

继续用“施工队”的类比。

前一步 self-attention，相当于 256 个施工小组先内部协调分工。  
这一步 cross-attention，相当于每个施工小组拿着自己的任务，去翻工程资料库。

资料库里有 64 份压缩过的设计信息。  
每个小组会关注自己最相关的那几份资料，读出来后更新自己的施工方案。

所以这一步结束后，每个 query 已经不再是“泛泛的小组”，而是“知道当前物体该怎么干活的小组”。

### 11. 这一小步的维度链

输入：

`query: (B,256,384)`  
`context: (B,64,384)`

`-> LayerNorm`
`query: (B,256,384)`
`context: (B,64,384)`

`-> q from query`
`q: (B,6,256,64)`

`-> k,v from context`
`k: (B,6,64,64)`
`v: (B,6,64,64)`

`-> cross-attention`
输出：
`(B,6,256,64)`

`-> merge heads`
`(B,256,384)`

`-> projection`
`(B,256,384)`

`-> residual add`
最终：
`query: (B,256,384)`

---

## 关于这一步：batch 里的 `(256,384)` 一样吗

不是。

更准确地说，要分“这一步之前”和“这一步之后”来看。

### 1. 刚 `expand` 出来时，是一样的

在这行：
[completion_decoder.py#L179](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L179)

```python
query = self.query_embed.expand(batch_size, -1, -1)
```

此时每个 batch 样本拿到的初始 `query` 都来自同一个参数 `query_embed: (1,256,384)`。

所以刚开始时：

`query[0] == query[1] == ... == query[B-1]`

也就是每个样本的 `(256,384)` 初始值是一样的。

### 2. 经过 self-attention 后，仍然一样

因为 self-attention 这一步只看 `query` 自己：
[completion_decoder.py#L92](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L92)

```python
query = query + self.self_attn(self.self_norm(query))
```

如果不同 batch 的初始 `query` 一样，而这一步又没有读入样本相关的 `context`，那么输出也还是一样。

所以在进入 cross-attention 之前，batch 内各样本的 `query` 仍然相同。

### 3. 到了 cross-attention 之后，就不一样了

关键在这行：
[completion_decoder.py#L93](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L93)

```python
query = query + self.cross_attn(self.cross_norm_q(query), self.cross_norm_ctx(context))
```

这里 `context` 是从每个样本自己的 encoder 输出得到的：

`context: (B,64,384)`

不同样本的 `context` 一般不同，因为它们对应不同的点云。

所以虽然 `query` 起点一样，但在 cross-attention 时，每个样本的 query 去读的是自己那份 context，于是更新结果就会不同。

也就是说：

- 初始 query 一样
- self-attention 后还一样
- 一旦做了 cross-attention，就开始因样本不同而不同

### 一句话总结

`query` 一开始是 batch 内共享的同一套模板；  
真正让它变成“每个样本专属”的，是 cross-attention 读取各自不同的 `context` 这一步。

---

## 第十步：block 最后再过一个 MLP

下一步是 block 的最后一步：

`MLP`

代码在：
[completion_decoder.py#L94](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L94)

```python
query = query + self.mlp(self.mlp_norm(query))
```

现在这一步发生时，`query` 已经经过了：

1. self-attention
2. cross-attention

所以这时候的 `query: (B,256,384)` 已经带上了当前样本的 context 信息。

### 1. 这一步先做什么

先做 `LayerNorm`：

`query: (B,256,384) -> (B,256,384)`

定义在：
[completion_decoder.py#L88](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L88)

这一步还是不改形状，只是规范数值。

### 2. 然后进入 MLP

MLP 定义在：
[completion_decoder.py#L64](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L64)

```python
hidden = int(dim * mlp_ratio)
self.net = nn.Sequential(
    nn.Linear(dim, hidden),
    nn.GELU(),
    nn.Dropout(drop),
    nn.Linear(hidden, dim),
    nn.Dropout(drop),
)
```

默认：
`dim = 384`
`mlp_ratio = 4.0`

所以 hidden dim 是：

`384 * 4 = 1536`

于是这一步对每个 query token 做的是：

`384 -> 1536 -> 384`

输入输出形状是：

`(B,256,384) -> (B,256,1536) -> (B,256,384)`

### 3. 注意：MLP 不会让不同 query 互相交流

这一点很重要。

self-attention 和 cross-attention 都会在 token 之间建立关系。  
但 MLP 不是这样。

MLP 是“逐 token 独立”处理的。

也就是说，它对第 1 个 query 做一个 384 维到 1536 再回 384 的非线性变换；  
对第 2 个 query 也一样；  
但第 1 个 query 不会在这一步看第 2 个 query。

所以它做的事情不是“交流”，而是“消化”。

### 4. 这一步在语义上是干什么

你可以这样理解：

前两步：

1. self-attention 让 query 之间协调
2. cross-attention 让 query 从 context 里取信息

那 MLP 这一步就是：

“每个 query 把刚刚拿到的信息，在自己内部做一次更强的非线性加工”

如果说 cross-attention 是“读资料”  
那 MLP 更像是“把读到的资料整理成自己最终可用的表示”

所以这一步很像“特征提纯”和“语义重整”。

### 5. 为什么不能只靠 attention，不要 MLP

因为 attention 更擅长“信息路由”和“信息聚合”：

- 谁看谁
- 从哪读多少信息
- token 之间如何交互

但 attention 本身不太擅长做强的逐点非线性特征变换。

MLP 正好补这个缺口。  
它能在每个 token 内部做更丰富的 feature mixing 和非线性映射。

Transformer 里几乎都会有这两类模块配对出现：

1. attention 负责交流
2. MLP 负责消化

### 6. 输出后再加残差

代码里是：

```python
query = query + self.mlp(self.mlp_norm(query))
```

所以最后还是残差连接。

输入：
`(B,256,384)`

输出：
`(B,256,384)`

含义是：
不是把原 query 丢掉，而是在原 query 基础上加一层非线性修正。

### 7. 一个完整 QueryDecoderBlock 到这里就结束了

所以一整个 block 的顺序是：

1. self-attention
2. cross-attention
3. MLP

从形状上看，全程都是：

`query: (B,256,384) -> (B,256,384)`

但是语义在不断增强：

- 一开始只是共享模板 query
- self-attention 后，query 之间完成内部协调
- cross-attention 后，query 读到了当前样本的 encoder 信息
- MLP 后，这些信息被进一步整理成更适合生成点云的 query feature

### 8. 这样的 block 会重复多少次

在 decoder 里有：

[completion_decoder.py#L147](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L147)

```python
self.blocks = nn.ModuleList([
    QueryDecoderBlock(...)
    for _ in range(depth)
])
```

默认：
`depth = 6`

所以这一整套流程会重复 6 次。

代码在 forward 里：
[completion_decoder.py#L181](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L181)

```python
for block in self.blocks:
    query = block(query, context)
```

也就是：

`(B,256,384)`
经过第 1 个 block
还是 `(B,256,384)`

再经过第 2 个 block
还是 `(B,256,384)`

……

共 6 次。

所以 6 层结束后，`query` 的形状没变，但语义已经非常丰富。

### 9. 然后还有一层 norm

6 个 block 后，代码又做了：

[completion_decoder.py#L183](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L183)

```python
query = self.norm(query)
```

输出仍然是：

`query: (B,256,384)`

这个就是 decoder 最终用于出点的 query feature。

### 10. 白话总结这一步

这一步 MLP 不负责“看别人”，只负责“整理自己”。

前面 cross-attention 相当于每个 query 从 encoder 资料库里读了一堆信息；  
MLP 则把这些信息在每个 query 内部重新加工一遍，变成更适合最后生成点云的表示。

### 这一小步的维度链

输入：
`query: (B,256,384)`

`-> LayerNorm`
`(B,256,384)`

`-> Linear(384,1536)`
`(B,256,1536)`

`-> GELU`
`(B,256,1536)`

`-> Linear(1536,384)`
`(B,256,384)`

`-> residual add`
最终：
`query: (B,256,384)`

---

## 第十一步：最后一步，`query: (B,256,384)` 怎么变成 `(B,8192,3)`

最后一步就是：

把 decoder 处理好的

`query: (B,256,384)`

真正变成点云

`pred_points: (B,8192,3)`

代码在：
[completion_decoder.py#L183](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L183) 到 [completion_decoder.py#L189](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L189)

先讲默认更常用的 `refine=False` 路径。

### 1. 先过最后一层 LayerNorm

代码：
[completion_decoder.py#L183](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L183)

```python
query = self.norm(query)
```

形状不变：

`(B,256,384) -> (B,256,384)`

这一步就是在最终输出前，再把 query feature 规范一下。

### 2. 默认路径：point_head

如果 `refine=False`，走这里：
[completion_decoder.py#L156](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L156)

```python
self.point_head = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),
    nn.GELU(),
    nn.Linear(hidden_dim, self.points_per_query * 3),
)
```

默认：
`hidden_dim = 384`

如果是 ShapeNet 默认：
`output_points = 8192`
`num_queries = 256`

所以：

`points_per_query = 8192 / 256 = 32`

这个值定义在：
[completion_decoder.py#L138](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L138)

于是最后一层的输出维度是：

`32 * 3 = 96`

所以 point head 做的是：

`384 -> 384 -> 96`

### 3. 具体维度怎么变

输入：
`query: (B,256,384)`

经过第一层：
`Linear(384,384)`

还是：

`(B,256,384)`

经过 GELU：
还是：

`(B,256,384)`

经过第二层：
`Linear(384,96)`

得到：

`(B,256,96)`

这时候的 `96` 是什么意思？

因为每个 query 负责输出 32 个点，  
每个点 3 维坐标，

所以：

`96 = 32 * 3`

也就是说：

现在每个 query 已经不再是 feature 了，  
而是直接吐出 32 个点的坐标参数。

### 4. reshape 成真正的点云

代码：
[completion_decoder.py#L188](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L188)

```python
coarse_points = self.point_head(query).reshape(batch_size, self.output_points, 3)
```

所以：

`(B,256,96)`

reshape 成：

`(B,8192,3)`

为什么可以这么 reshape？

因为：

`256 * 96 = 256 * (32*3) = 8192 * 3`

所以等价于：

每个 query 贡献 32 个点  
256 个 query 一共贡献：

`256 * 32 = 8192` 个点

最终就是一个完整点云。

### 5. 这一步在语义上是什么意思

你可以把这一步理解成：

前面的 6 层 decoder block，已经把每个 query 变成了一个“知道该生成什么局部结构”的高级表示。  
最后的 point head 只是把这种高级表示翻译成具体坐标。

所以 point head 本身不负责复杂推理，  
复杂推理前面已经做完了。  
它更像一个最终的“坐标回归头”。

### 6. 为什么每个 query 负责 32 个点，而不是 1 个点

如果每个 query 只出 1 个点，那 256 个 query 只能出 256 个点，太稀了。  
但如果 query 数太多，attention 计算又会更贵。

所以这里用了一个折中：

query 数量保持在 256，方便 Transformer 处理；  
每个 query 再输出一小片点，最终达到 8192 个输出点。

这相当于：

query 是比较高层的生成单元  
不是一 query 对应一坐标点，而是一 query 对应一个小局部 patch

### 7. 输出对象是什么

最终返回的是：
[completion_decoder.py#L189](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L189)

```python
return CompletionDecoderOutput(query_tokens=query, coarse_points=coarse_points)
```

也就是：

`query_tokens: (B,256,384)`  
`coarse_points: (B,8192,3)`

Stage-1 训练时真正拿去算 loss 的，就是这个：

`coarse_points`

### 8. 然后 Stage-1 怎么算损失

训练代码在：
[stage1_train.py#L331](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage1_train.py#L331)
[stage1_train.py#L332](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage1_train.py#L332)

```python
dec = decoder(tokens, enc.centers)
cd_loss = chamfer_distance_l1(dec.coarse_points, complete_points)
```

所以：

预测：
`dec.coarse_points: (B,8192,3)`

真值：
`complete_points: (B,8192,3)`

然后算 Chamfer Distance。

也就是在问：

“预测出来的这些点，整体上离真实完整点云近不近？”

### 9. 如果开 refine，会发生什么

如果 `refine=True`，最后一步不是直接 `384 -> 96`。

而是：

1. 先预测 seed point：
`(B,256,384) -> (B,256,3)`
见 [completion_decoder.py#L185](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L185)

2. 再用 `PointRefinement` 给每个 seed 展开成一个局部 patch：
见 [completion_decoder.py#L111](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L111)

最终还是：
`(B,8192,3)`

但你现在先抓默认路径就够了。

### 10. 用一句最直白的话总结最后一步

前面 decoder 学到的是 256 个“该怎么生成局部结构”的 query feature。  
最后一步就是把每个 query feature 直接翻译成 32 个三维点，  
然后把 256 份小局部拼起来，得到完整点云。

### 最后一步的完整维度链

`query: (B,256,384)`
`-> LayerNorm`
`(B,256,384)`
`-> Linear(384,384)`
`(B,256,384)`
`-> GELU`
`(B,256,384)`
`-> Linear(384,96)`
`(B,256,96)`
`-> reshape`
`pred_points: (B,8192,3)`

### 如果把整个 Stage-1 一口气串起来

`complete_points: (B,8192,3)`
`-> Group`
`(B,64,32,3), centers=(B,64,3)`
`-> LocalEncoder`
`(B,64,384)`
`-> Transformer encoder`
`tokens=(B,64,384)`
`-> normalize/noise`
`(B,64,384)`
`-> decoder query init`
`query=(B,256,384)`
`-> 6 layers decoder blocks`
`(B,256,384)`
`-> point head`
`pred_points=(B,8192,3)`
`-> Chamfer loss with GT`

---

## refine 打开后会发生什么，以及它有什么作用

开 `refine` 以后，decoder 的最后一段“出点方式”会变掉。

不开 `refine` 时，逻辑是：

`query: (B,256,384) -> point_head -> (B,256,96) -> reshape -> (B,8192,3)`

也就是每个 query 直接一次性回归 `32` 个点的坐标。

开了 `refine` 以后，逻辑变成两步：

1. 先出一个 `seed point`
2. 再围着这个 seed 展开出一小片局部 patch

对应代码在：
[completion_decoder.py#L152](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L152)
[completion_decoder.py#L184](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L184)

### 先看结构怎么变

开 `refine=True` 时，decoder 不再创建 `point_head`，而是创建：

`seed_head`
`refine_module`

见：
[completion_decoder.py#L152](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L152)

```python
self.seed_head = nn.Linear(hidden_dim, 3)
self.refine_module = PointRefinement(hidden_dim, self.points_per_query)
```

默认还是：
`num_queries = 256`
`output_points = 8192`
所以：

`points_per_query = 32`

### 第一步：先预测 seed 点

代码：
[completion_decoder.py#L185](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L185)

```python
seed_points = self.seed_head(query)
```

输入：
`query: (B,256,384)`

输出：
`seed_points: (B,256,3)`

也就是说，每个 query 不再直接出 32 个点，而是先出 1 个三维中心点。

你可以把它理解成：

每个 query 先决定“我负责的那一小块，大致中心落在哪”。

### 第二步：围着 seed 展开局部 patch

代码在：
[completion_decoder.py#L98](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L98)

`PointRefinement` 里有一个可学习网格：

```python
self.grid = nn.Parameter(torch.randn(1, points_per_query, grid_dim) * 0.1)
```

默认：
`points_per_query = 32`
`grid_dim = 2`

所以这个 grid 的形状是：

`grid: (1,32,2)`

它表示一个可学习的二维小模板。  
注意这里是二维，不是三维。

然后 forward 时：

[completion_decoder.py#L111](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L111)

```python
B, Q, D = query_features.shape
P = self.points_per_query
feat = query_features.unsqueeze(2).expand(B, Q, P, D)
grid = self.grid.expand(B, Q, -1, -1)
offsets = self.mlp(torch.cat([feat, grid], dim=-1))
seeds = seed_points.unsqueeze(2).expand(B, Q, P, 3)
return (seeds + offsets).reshape(B, Q * P, 3)
```

我们逐步看维度。

输入：

`query_features: (B,256,384)`
`seed_points: (B,256,3)`

先把每个 query feature 复制 32 份：

`feat: (B,256,32,384)`

把 learnable grid 也扩到每个 query：

`grid: (B,256,32,2)`

然后拼起来：

`cat([feat, grid], dim=-1) -> (B,256,32,386)`

再过一个 MLP：
[completion_decoder.py#L105](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L105)

```python
nn.Linear(hidden_dim + grid_dim, hidden_dim),
n.GELU(),
n.Linear(hidden_dim, 3),
```

也就是：

`386 -> 384 -> 3`

输出：

`offsets: (B,256,32,3)`

这 32 个 offset 表示：

对于某个 query 的 seed 点，我要在它周围长出 32 个三维偏移点。

再把 seed 扩展成：

`seeds: (B,256,32,3)`

最后相加：

`seeds + offsets -> (B,256,32,3)`

再 reshape：

`(B,256,32,3) -> (B,8192,3)`

这就是最终输出点云。

### 所以 refine 的核心思想是什么

不开 refine：

每个 query 直接“凭空”回归 32 个点坐标。

开 refine：

每个 query 先找一个局部中心 seed，
再在这个中心附近生成一个结构化的小 patch。

所以 refine 更像是：

“先粗定位，再局部展开”

而不是“一步到位直接吐很多点”。

### 为什么这会更有用

主要有两个作用。

#### 1. 更容易学局部连续表面

直接输出 32 个点时，这 32 个点之间没有显式结构关系。  
网络只能自己隐式学：

“这 32 个点应该围成一小片面”

这很难，容易出现：

- 点扎堆
- 点散乱
- 局部表面不连续

而 refine 模式下，这 32 个点共享同一个 seed，并且共享一个二维小网格模板。  
这相当于给了网络一个先验：

“这 32 个点应该是围绕某个局部中心展开的一小片局部表面”

所以更容易形成连续 patch。

#### 2. 更利于细长结构和薄结构

像飞机尾翼、机翼边缘、椅子腿这种结构，常见问题是：

直接回归很多点时，网络容易把点生成到大块主体上，细节部分照顾不到。

refine 模式下，每个 query 先要决定 seed 落点。  
这会迫使它先学“关键局部位置”在哪里。  
一旦 seed 放对了，再在周围展开 patch，就更容易覆盖这些细结构。

这也是仓库备注里说的：
`refine` 有助于补细节、薄结构。

### 为什么还要加 seed loss

Stage-1 训练里，如果 `dec.seed_points is not None`，会额外对 seed 和 GT 算一个 Chamfer loss：

[stage1_train.py#L334](/root/autodl-tmp/projects/BridgeRAE/bridgerae/training/stage1_train.py#L334)

```python
seed_loss = chamfer_distance_l1(dec.seed_points, complete_points)
loss = loss + args.seed_loss_weight * seed_loss
```

这很重要。

因为如果没有 seed loss，网络可能会偷懒：

seed 放得不太合理，全靠后面的 offsets 硬拉开。

加了 seed loss 以后，训练会明确要求：

“这 256 个 seed 本身就要尽量覆盖真实完整点云的重要区域”

于是形成一种 coarse-to-fine 的过程：

1. seed 先学会粗覆盖
2. refine_module 再围着 seed 生成稠密 patch

### 为什么 `refine_module` 最后一层是零初始化

代码在：
[completion_decoder.py#L173](/root/autodl-tmp/projects/BridgeRAE/bridgerae/models/completion_decoder.py#L173)

```python
nn.init.zeros_(self.refine_module.mlp[-1].weight)
n.init.zeros_(self.refine_module.mlp[-1].bias)
```

这表示训练一开始：

`offsets ≈ 0`

于是初始输出大致就是：

`coarse_points ≈ seed_points`

也就是先从“只会输出 seed”开始学。  
然后训练过程中，模型再慢慢学会如何在 seed 周围长出局部 patch。

这个初始化很稳，因为它避免了一开始 patch 乱飞。

### 一句话对比

不开 `refine`：

每个 query 直接回归一坨点，简单，但更容易无结构、扎堆。

开 `refine`：

每个 query 先放一个 seed，再围着 seed 展开局部 patch，更有局部几何先验，通常更利于细节和表面连续性。

### 完整维度链

开 `refine=True` 时，最后几步是：

`query: (B,256,384)`
`-> seed_head`
`seed_points: (B,256,3)`

`query -> expand`
`(B,256,32,384)`

`grid -> expand`
`(B,256,32,2)`

`concat`
`(B,256,32,386)`

`-> refine MLP`
`offsets: (B,256,32,3)`

`seed expand`
`(B,256,32,3)`

`seed + offsets`
`(B,256,32,3)`

`-> reshape`
`coarse_points: (B,8192,3)`

如果你把默认路径和 refine 路径对比着看，就会很清楚：

默认路径是直接回归点；  
refine 路径是先放 seed，再围 seed 长 patch。
