// =====================================
// 全局设置
// =====================================
#set page(

  margin: 2.5cm
)

#set text(
  font: "Times New Roman",
  size: 12pt,
)

#set par(
  justify: true,
  leading: 1.4em,
)

// =====================================
// 封面
// =====================================
= 强化学习实验报告

学生姓名：  
学号：  
课程名称：  
实验名称：  
日期：  

#pagebreak()

// =====================================
// 0. Torch 配置
// =====================================
== 0. Torch 配置

=== 0.1 运行环境
- 操作系统：
- Python 版本：
- PyTorch 版本：

=== 0.2 硬件配置
- CPU：
- GPU：
- CUDA / MPS：

=== 0.3 环境搭建说明
（如 Conda / venv / pip 安装方式）

#pagebreak()

// =====================================
// 1. Warm-up（经典强化学习）
// =====================================
== 1. Warm-up（经典强化学习）

=== 1.1 Monte Carlo（MC）

==== 1.1.1 First-Visit MC 核心思想
（原理说明）

==== 1.1.2 更新对象与策略改进方式
（Value / Policy 更新）

==== 1.1.3 Blackjack 实验结果
image("blackjack.png", width: 80%)

==== 1.1.4 结果分析
（实验现象分析）

---

=== 1.2 Temporal Difference（TD）

==== 1.2.1 SARSA 与 Q-learning 的更新差异
（On-policy 与 Off-policy）

==== 1.2.2 CliffWalking 实验对比
image("cliffwalking.png", width: 80%)

==== 1.2.3 现象解释
（风险偏好、收敛路径分析）

---

=== 1.3（选做）算法扩展讨论

==== 1.3.1 Every-Visit MC
（与 First-Visit 对比）

==== 1.3.2 Double Q-learning
#image("episode_stats_double_q_learning_length.png",width:35%)
#image("episode_stats_double_q_learning_reward.png",width:35%)

#image("episode_stats_q_learning_length.png",width:35%)
#image("episode_stats_q_learning_reward.png",width:35%)

共同点

两者在前期 episode length 都非常大（400–600 步），说明：

初期策略接近随机

agent 经常在环境中“乱走”或反复试探

随着训练推进，episode length 都迅速下降并趋于稳定（≈15–30 步）

差异

Double Q-learning

收敛过程更平滑

后期 episode length 波动略小

Q-learning

虽然也能快速下降

后期仍可看到更多“尖峰”（偶尔走冤枉路）

Episode Reward（平滑后）

Q-learning 的 reward 曲线

前期 reward 从 ≈ -300 快速上升

中后期：

reward 大致稳定在 -40 到 -60

但存在 频繁而明显的下跌

有时突然掉到 -100 附近

波动幅度较大

Double Q-learning 的 reward 曲线

上升趋势与 Q-learning 相似

平均 reward 更高（更不负）

大幅下跌显著减少

曲线整体更加平滑

原因:

标准 Q-learning 的更新公式

Q(s,a)←Q(s,a)+α(r+γa′max​Q(s′,a′)−Q(s,a))

同一个 Q 表 ，既负责选动作，又负责评估动作

容易选中被高估的动作，而不是真实最优动作

E[max(Q+ϵ)]>max(E[Q])

Q 值系统性偏大

agent 过度自信

实际执行时，发现动作没那么好 → reward 突然变差

Double Q-learning

a∗​=argamax​QA​(s′,a)

QB(s,a)←QB(s,a)+α(r+γa′max​QA(s′,a′)−QB(s,a))

QA选动作

QB估价值

QA的噪声不等于QB的噪声，两者不太可能同时高估一个动作​

#pagebreak()

// =====================================
// 2. 实验原理（主任务）
// =====================================
== 2. 实验原理（主任务）

=== 2.1 二元零和马尔可夫博弈
（状态、动作、奖励、转移）

二元：只有两个玩家

零和：一方的收益直等于另一方的损失

马尔科夫；状态转移只取决于当前状态与双方动作

=== 2.2 Naive Self Play
（自博弈训练思想）

Naive：没有任何温度化机制

用当前策略同时cos自己和对手，play完后同时更新策略，再用新的策略继续play
=== 2.3 Actor-Critic 方法
（Actor / Critic 分工）

Actor根据当前状态得出每个动作的概率，然后执行动作，更新状态

Critic给出价值评估，指导Actor更新。
#pagebreak()

// =====================================
// 3. 模型设计
// =====================================
== 3. 模型设计



=== 3.1 __init__ 方法

 SelfAttention
 
 self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)
 
self.ln = nn.LayerNorm(channels)

Actor

self.start_block = Conv + BN + ReLU

self.backbone = ResBlock + SelfAttention + ResBlock ×2

self.policy_head = Conv1x1 → Flatten → Linear

Critic

self.value_head = Linear → ReLU → Linear(board_size  board_size)

GobangModel

self.actor = Actor(...)

self.critic = Critic(...)

=== 3.2 forward 方法
 SelfAttention
 
 flat = x.flatten(2).permute(0, 2, 1)

 attn_out,  = self.mha(flat, flat, flat)

 out = self.ln(flat + attn_out)

 return out.permute(0, 2, 1).reshape(B, C, H, W)

Actor

if isinstance(x, np.ndarray):

x = torch.tensor(x)

...

board_flat = x.view(x.size(0), -1)

legal_mask = (board_flat == 0).float()

Critic

qmap = self.value_head(out)

indices = [position_to_index(...)]

output = q_map[batch_indices, indices]

GobangModel

return self.actor(x), self.critic(x, action)

=== 3.3 设置动机
将二维棋盘视为 
𝑁*2个 token

每个格子可以关注任意其他格子

用CNN进行局部建模
 
连子、活三、活四

用Attention规划全局战略

长距离威胁、双线进攻
#pagebreak()

// =====================================
// 4. 约束策略分析
// =====================================
== 4. 约束策略分析

=== 4.1 置 0 约束（非法动作）
（Mask 机制）

legal_mask = (board_flat == 0).float()

logits = torch.where(

  legal_mask == 1,
  
  logits,
  
  torch.tensor(-1e9).to(device)

)

对所有 非法落子位置 logiti​←−∞

在 softmax 后π(ai​∣s)=0,防止选择非法动作
=== 4.2 归一化约束
（概率分布约束）

probs = F.softmax(logits, dim=1)

Softmax 保证：∑​π(a∣s)=1  合法动作在剩余概率空间中重新归一


#pagebreak()

// =====================================
// 5. Bug 修复
// =====================================
== 5. Bug 修复

=== Bug 1

问题现象：

 Target 参与反向传播

targets = rewards + gamma*next_qs

原因分析：

如果不 detach()

梯度会沿着critic → next_qs → targets → critic_loss反向流回 target 自身

修复方案：

targets = rewards + gamma*next_qs.detach()

detach() 明确表达target 是常数，不参与梯度计算
=== Bug 2

问题现象：

Actor 改了 Critic

actor_loss = - torch.mean(
   
  torch.log(aimed_policy + eps）qs

)

原因分析：

这里的 qs被直接乘进 actor_loss

Actor loss 的梯度会流入 Critic

违背了 Actor的角色分工

修复方案：

actor_loss = -torch.mean(
 
  torch.log(aimed_policy + eps) qs.detach()

)
=== Bug 3

问题现象：

忘记 .step()

self.critic.optimizer.zero_grad()

critic_loss.backward()

没有 step()


self.actor.optimizer.zero_grad()

actor_loss.backward()

没有 step()

原因分析：

没有 .step()相当于参数不动

修复方案：

self.critic.optimizer.step()

self.actor.optimizer.step()

#pagebreak()

// =====================================
// 6. 其他问题记录
// =====================================
== 6. 其他问题记录



#pagebreak()

// =====================================
// 7. 疑难点思考
// =====================================
== 7. 疑难点思考

=== 7.1 初始困惑
（不理解的地方）

=== 7.2 后续理解
（理解过程）

=== 7.3 当前认识
（总结）

#pagebreak()

// =====================================
// 8. 曲线分析（主任务）
// =====================================
== 8. 曲线分析（主任务）
#image("loss_tracker.png", width: 45%)
=== 8.1 Loss 曲线分析

Actor Loss

呈现出剧烈的波动，且随着 Episode 的增加，负向波峰越来越大（从最初的 0 附近扩大到 -80 左右）

分析：

在代码中，actor_loss = -torch.mean(torch.log(aimed_policy + 
eps) qs.detach())

Loss 是负的，说明 $log(\pi) \cdot Q$ 的值很大。

五子棋是稀疏奖励，当后期获胜得到巨大的正向Q奖励，loss会出现大幅度负向跳变

Critic Loss

始终处于高频震荡状态，基准线维持在 0-40 之间，并在后期出现了一些极端离群点（Spikes），最高接近 160

分析

我的model比较傻，基本上就是莫名其妙的赢或输了，（导致局势瞬息万变？）

导致 MSE 很难收敛到极小值，

但 Critic Loss 没有出现发散，说明他勉强跟得上Actor的变化
=== 8.2 Entropy 曲线分析
熵值从最初的 4.5 以上 迅速下降，并在 5000 episode 后进入 0.5 到 2.0 之间的震荡区间，后期甚至出现了接近 0 的情况

早期（0-5000）： 熵值高说明模型在进行广泛的探索，动作比较随机。

中期（5000-15000）： 策略开始收敛，模型学会了一些基本的堵截和连线技巧，收敛到了某些特定套路。

后期（20000+）： 熵值多次触底，说明模型变得非常确定（但据我观察20000+以后model非常傻，是蠢到了一定程度并且固定了）
=== 8.3 综合分析


#pagebreak()

// =====================================
// 9. 模型优化（选做）
// =====================================
== 9. 模型优化

=== 9.1 超参数调整


=== 9.2 网络结构调整
（深度、宽度）

=== 9.3 架构创新
模块 使用的技术 目的

ResBlock, CNN + 残差连接 + BatchNorm,提取深层局部空间特征，保持训练稳定。

SelfAttention, 多头注意力 (MHA) + LayerNorm,捕捉全局关联，突破 CNN 感受野限制。

Actor,  CNN + ResNet + Attention + Softmax,输出每个合法落子位置的概率分布。

Critic, CNN + ResNet + Attention + MLP,评估当前局面的价值 (Q-value)。

这里我简单说一下我的submission的改进的过程

最初就是完成了所有填空，训练以后发现什么都没学到，大概是因为有bug

我首先尝试的是resnet，训练过很多次，有几千轮的和几万轮的，整体的感觉就是很蠢

后来又加入了cnn，结果更蠢了，

最后我尝试在加入attention，训了大概三万轮，才发现在10000~20000的模型有点开智的迹象




#pagebreak()

// =====================================
// 10. 思考题（选做）
// =====================================
== 10. 思考题（选做）

=== 问题描述
（是否达到纳什均衡）

=== 分析与讨论
（博弈论视角）

#pagebreak()

// =====================================
// 11. 课程反馈（必做）
// =====================================
== 11. 课程反馈

=== 11.1 学习收获
（个人总结）

=== 11.2 课程建议
（改进建议）
