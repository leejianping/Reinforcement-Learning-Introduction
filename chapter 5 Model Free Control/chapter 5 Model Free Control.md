# 第五讲 无模型控制(Model-Free Control)

上一讲中介绍了无模型预测问题，即在环境动力学模型未知的情况下，估计给定策略的价值函数。在本讲中，将介绍无模型控制问题，即在环境动力学模型未知的情况下，优化给定策略的价值函数。所谓优化价值函数是指智能体通过学习，尽可能多的获得奖励。

## 5.1 简介

现实中有很多使用无模型控制的例子，比如控制一艘船的航行；控制直升机的飞行，机器人的行走，下围棋游戏等等。对于大部分这些问题，要么我们对其模型运行机制未知，但是我们可以去经历、去采样；要么是虽然问题模型是已知的，但问题的规模太大以至于计算机无法计算，除非使用采样的办法。无模型控制可以解决这些问题。

**同策略(On-Policy)与异策略(Off-Policy)学习的定义**

**行为策略（Behavior Policy）：** 智能体在与环境互动的过程中采用的策略，即在交互过程中用来做决策，选取动作，然后产生数据（也称为采样、经历）
**目标策略（Target Policy）：** 智能体想要优化的策略（又称评估的策略），利用行为策略产生的数据来学习、优化目标策略。

**同策略(On-Policy)学习：** 指的是执行行为策略 $\pi$ 采样得到一些经验，然后从这些经验中来学习目标策略  $\pi$ 。通俗地说，智能体的行为策略与目标策略是同一个策略。其基本思想是“从工作中学习”。
**异策略Off-Policy)学习：**  指的是执行行为策略 $\mu$ 采样得到一些经验，然后从这些经验中来学习目标策略  $\pi$ 。通俗地说，智能体的行为策略与目标策略不是同一个策略。其基本思想是，从一个策略来学习另一个策略，是“站在巨人的肩膀上”的一种实现。


## 5.2 同策略蒙特卡洛控制(On-Policy Monte-Carlo Control)
 
**5.2.1 通用策略迭代（Generalised Policy Iteration）**

先来回顾下动态规划是如何进行策略迭代。通用策略迭代（Generalised Policy Iteration）回顾

![](./images/1%20GPI.PNG ":size=350")

通用策略迭代由两个交替的过程组成：一个是策略评估，另一个是改善策略。如上图左边所示，从策略π和价值函数V开始，每一次箭头向上代表着利用当前策略进行价值函数的估计，每一次箭头向下代表着根据更新的价值函数贪婪地改进策略。所谓贪婪，是指每次都采取使得所有可能的状态价值函数最大的动作。最终将收敛至最优策略和最优价值函数。

**蒙特卡洛评估的通用策略迭代(GPI With Monte-Carlo Evaluation)**

上图的通用策略迭代中，策略评估能否用蒙特卡洛评估呢，即 $V =V_{\pi}$ ？答案是否定的，这是因为根据价值函数V(s)贪婪地改进策略时需要知道MDP模型，即,

$$
\pi^{\prime}(s) = \argmax_{a \in A} R_{s}^{a} + P_{ss'}^{a} V(s')
$$

与之相反的是，根据动作价值函数Q(s，a)贪婪地改进策略时不需要知道环境模型，

$$
\pi^{\prime}(s) = \argmax_{a \in A} Q(s,a)
$$

因为，蒙特卡洛评估只能用于模型未知时，而用Q值贪婪地改善策略不需要知道模型，因此，可以在蒙特卡洛评估的通用策略迭代中使用Q值。


**动作价值函数的通用策略迭代(GPI with Action-Value Function)**

如下图所示，从初始的策略 $\pi$ 和初始的动作价值函数Q开始, 使用蒙特卡洛策略评估来更新q值，然后根据贪婪算法改进策略。

![](./images/2%20qGPI.PNG ":size=350")

**5.2.2 探索(Exploration)**

虽然解决了模型未知的问题，但仍存在问题，我们在做蒙特卡洛策略评估时，是通过采样来评估q值的，当根据q值使用贪婪算法来改善策略的时候，很可能因为没有足够的采样而导致产生一个并不是最优的策略。所以，我们需要不时的尝试一些新的动作，这就是探索（Exploration），使用下面示例来解释。

例5-1 贪婪动作选择

![](./images/3%20exampGreedy.PNG ":size=350")

如图，现有两扇门，考虑如下的动作、奖励并使用贪婪算法改善策略：
* 打开左侧门得到即时奖励为0， V(left) = 0
* 打开右侧门得到即时奖励为+1，V(right) = +1
* 打开右侧门得到即时奖励为+3，V(right) = +2
* 打开右侧门得到即时奖励为+2，V(right) = +2
...
根据q值使用贪婪算法时，接下来你将会继续打开右侧的门，而不会尝试打开左侧门。

这种情况下，你确定打开右侧门是否就是最好的选择呢？显而易见，答案是否定的。因此，使用完全贪婪算法改善策略通常不能得到最优策略。为了解决这一问题，需要引入一个随机探索机制，以一定的概率选择当前最好的策略，同时以一定的概率随机选择其它可能的动作，这就是Ɛ-贪婪探索。

**Ɛ-贪婪探索（Ɛ-Greedy Exploration）**

Ɛ-贪婪探索的目标是使得某一状态s下所有可能的m个动作都以一定的非零概率被选中执行，是保证持续的探索的最简单的思想。以1-Ɛ的概率选择当前认为最好的行为（贪婪策略），以Ɛ的概率在所有可能的m个动作中随机选择一个（也包括那个当前最好的动作）。数学表达式如下：

$$
\mathrm\pi(a|s)= \begin{cases}
\epsilon/m +1 - \epsilon {\quad} if {\;} a^* = \underset{a \in A}{\operatorname{\argmax}} {\;} Q_(s,a) \\
\epsilon/m {\quad}  {\quad} {\quad}  {\quad} otherwise
\end{cases}
$$

选择的动作分为两类，一类是贪婪选取的动作（在某一状态下只有一个）， 另一类是随机选取的动作。首先贪婪选取的动作的概率为 1- Ɛ， 随机选取的动作的概率为Ɛ，Ɛ/m就是随机选择每一个动作的概率。既然随机选择时也包含了贪婪选择的那个动作，那么贪婪选择的动作的概率分为两部分：一是以1-Ɛ概率选择；二是，在随机探索时也可能选中这个动作，概率为Ɛ/m，因此就是1-Ɛ+Ɛ/m。如果算总概率的话，贪婪选的动作 a* 只有一个，因此为1-Ɛ+Ɛ/m；而随机选择的m动作个动作中，除去 a* 还剩m-1个，因此，概率为(m-1)Ɛ/m。两部分相加刚好为1。

在CS285课上，Ɛ-贪婪探索的公式略微不同，以Ɛ的概率在所有可能的m-1动作中随机选择一个（除去了贪婪选取的那个动作）

$$
\mathrm\pi(a|s)= \begin{cases}
1 - \epsilon {\quad} {\quad} if {\;} a^* = \underset{a \in A}{\operatorname{\argmax}} {\;} Q_(s,a) \\
\epsilon/(m-1) {\quad}   otherwise
\end{cases}
$$

上述两者唯一区别就是，随机探索时是否包含贪婪选取的那个动作，感觉后者更好理解一些。

**Ɛ-贪婪策略改进（Ɛ-Greedy Policy Improvement）**

定理：对于任意Ɛ-贪婪策略 $\pi$ ，根据 $q_{\pi}$ 使用Ɛ-贪婪策略得到的策略 $\pi'$ 是策略 $\pi$ 的改进，即  $v_{\pi'}(s) \geq v_{\pi}(s)$  。
证明如下，

$$ \begin {aligned}
q_{\pi}(s,\pi'(s)) &= \sum_{a \in A} \pi'(a|s) q_{\pi}(s, a)\\
&= {\epsilon}/m \sum_{a \in A}  q_{\pi}(s, a)  + (1-\epsilon) \max_{a \in A} q_{\pi}(s, a)\\
&\geq {\epsilon}/m \sum_{a \in A}  q_{\pi}(s, a)  + (1-\epsilon) \sum_{a \in A} \frac{\pi(a|s) - {\epsilon}/m}{1-\epsilon} q_{\pi}(s, a)\\
&=\sum_{a \in A} \pi(a|s) q_{\pi}(s, a) = v_{\pi} (s)
\end {aligned}
$$

因此，从策略改进定理有，  $v_{\pi'}(s) \geq v_{\pi}(s)$  

**蒙特卡洛策略迭代**

上面解决了无模型评估(Q值)和局部最优(Ɛ-贪婪)这两个问题，终于看到蒙特卡洛策略迭代的全貌：使用Ｑ函数进行策略评估，使用Ɛ-贪婪探索来改善策略。如下图，

![](./images/4%20MCPI.PNG ":size=350")



蒙特卡洛控制如下图，

![](./images/5%20MCcontrol.PNG ":size=350")

上图的每一个箭头都对应着多个Episode。也就是说一般在经历了多个Episode之后才依次进行Ｑ函数更新或策略改善。实际上我们也可以在每经历一个Episode之后就更新Ｑ函数或改善策略。但不管使用那种方式，在Ɛ-贪婪探索算下我们始终只能得到基于某一策略下的近似Ｑ函数，且该算法没有一个终止条件，因为它一直在进行探索。因此我们必须关注以下两个方面：一方面我们不想丢掉任何更好信息和状态，另一方面随着我们策略的改善我们最终希望能终止于某一个最优策略，因为事实上最优策略不应该包括一些随机行为选择。

**5.2.3 无限探索下的有限贪婪GLIE（Greedy in the Limit with Infinite Exploration）**

定义：GLIE在有限的时间内进行无限可能的探索。具体分为两个方面：
* 所有状态动作对（state-action pair）会被无限次探索，

$$
\lim_{k \rightarrow \infty} N_{k}(s,a) = \infty
$$

* 策略收敛于一个贪婪策略，

$$
\lim_{k \rightarrow \infty} \pi_{k}(a|s) = 1(a= \argmax_{a' \in A} Q_{k}(s,a') )
$$


随着探索的无限延伸，贪婪算法中Ɛ值趋向于0。例如,我们取Ɛ = 1/k（k为探索的Episode数目），那么该Ɛ贪婪蒙特卡洛控制就具备GLIE特性。

**GLIE的蒙特卡洛控制(GLIE Monte-Carlo Control)**

基于GLIE的蒙特卡洛控制流程如下：
1. 用策略 $\pi$ 来采样第k个回合：$\lbrace S_{1}, A_{1},R_{2},\dots,S_{T} \rbrace  \sim  \pi$
2. 对于该回合中出现的每一个状态动作对 $S_{t}$ 和 $A_{t}$ ,更新其计数和Q函数，

$$ \begin {aligned}
&N(S_{t},A_{t}) \leftarrow N(S_{t},A_{t}) + 1\\
&Q(S_{t},A_{t}) \leftarrow Q(S_{t},A_{t}) + \frac{1}{N(S_{t},A_{t})} (G_{t} - Q(S_{t},A_{t}))
\end {aligned}
$$

3. 基于新的动作价值行数改善策略，
   
$$ \begin {aligned}
&\epsilon \leftarrow 1/k\\
&\pi \leftarrow \epsilon-greedy(Q)
\end {aligned}
$$

定理：GLIE蒙特卡洛控制能收敛至最优的状态动作价值函数，$Q(s,a) \rightarrow q_{*}(s,a)$

例5-2 之前的21点游戏（BlackJack）最优策略

下图最终给出了二十一点比赛时的最优策略（对于使用该策略进行赌博导致的输赢，本文不负任何责任）

![](./images/6%20blackjackexamp.PNG ":size=350")

最优策略：
当你手上有可用A时，大多数情况下当你的牌面和达到17或18时停止要牌，如果庄家可见的牌面在2-9之间，你选择17，其它条件选择18；
当你手上没有A时，大多数情况下牌面和达到16就要停止叫牌，当庄家可见的牌面在2-7时，这一数字更小至13甚至12。这种极端情况下，宁愿停止叫牌等待让庄家的牌爆掉。

## 5.3 同策略时序差分学习(On-Policy Temporal-Difference Learning)

**MC和TD的控制对比(MC vs. TD Control)**

与MC对比，TD学习有许多优点：低方差，在线学习，从不完整序列学习。因此，很自然的想法：在我们的控制回路中，用TD来替代MC，
* 把TD应用到Q(S,A)
* 用Ɛ-贪婪来改进策略
* 每一个时间步都进行更新

**5.3.1Sarsa( $\lambda$ )**

在这一小节中，介绍同策略TD控制问题。首先TD控制学习的是动作价值函数而非状态价值函数。特别地，在一个同策略方法中，对于当前的行为策略 $\pi$ 及所有的状态s和动作a，我们必须估计 $q_{\pi}(s,a)$ 。因此，这里我们考虑的是状态-动作对到状态-动作对之间的转移，学习状态-动作对的价值。

**用Sarsa来更新动作价值函数**

SARSA的名称来源于四元组 $(S_{t},A_{t},R_{t+1},S_{t+1},A_{t+1})$ ,这个四元组构成了从一个状态-动作对转移到下一个状态-动作对。其备份图如下图所示，

![](./images/7%20SarsaBP.PNG.PNG ":size=350")

上图直观解释：一个智能体处于当前状态S，在这个状态下它可尝试各种不同的动作，当遵循当前行为策略（Ɛ-贪婪）时，会根据当前行为策略选择一个动作A，智能体执行这个动作，与环境发生实际交互，环境会根据其动作给出即时奖励R，并且进入下一个状态S'，在这个后续状态S'，再次遵循当前策略，产生一个动作A'，此时，智能体并不执行该动作，而是通过自身当前的动作价值函数得到该S'A'状态-动作对的价值，利用该价值同时结合智能体S状态下采取动作A所获得的即时奖励来更新智能体在S状态下采取动作A的动作价值。SARSA更新如下，

$$
Q(S,A) \leftarrow Q(S,A) + \alpha(R+\gamma Q(S',A') -Q(S,A))
$$


与蒙特卡洛控制不同的是，Sarsa在每一个时间步都要更新Q值，同样使用Ɛ-贪婪探索的来改善策略。同策略SARSA控制如下图

![](./images/7%20sarsa.PNG ":size=350")

同策略Sarsa的控制算法如下，

![](./images/8%20sarsaalg.PNG ":size=350")

>注：
1.算法中的Q(s,a)是以一张大表存储的，这不适用于解决规模很大的问题；
2.对于每一个回合，在S状态时根据当前策略采取的动作A，同时该动作也是实际回合发生的动作，在更新SA状态动作对的价值循环里，智能体并不实际执行在S'下的A'动作，而是将动作A'留到下一个循环执行。

定理：满足如下两个条件时，Sarsa算法将收敛至最优动作价值函数， $Q(s,a) \rightarrow q_{*}(s,a)$ ，

1. 策略 $\pi_{t}(a|s)$ 的序列符合GLIE特性
2. Robbins-Monro 序列的步长 $\alpha_{t}$ 满足，
   
$$
\sum_{t=1}^{\infty} \alpha_{t} = \infty \\
\;\\
\sum_{t=1}^{\infty} \alpha_{t}^{2} \lt \infty 
$$  

上述公式中，第一个约束用来保证执行次数足够多，最终克服任何初始条件或者随机的起伏；而第二个约束用来保证最终收敛所用的次数足够少。

例5-3 有风的格子世界
如下图所示，环境是一个长方形（10*7）的格子世界，同时有一个起始位置S和一个终止目标位置G，格子下方的数字表示对应的列的风的强度。如，当风的强度等于1时，智能体进入该列的某个格子时，会按图中箭头所示的方向被风移动一格，同理，风的强度为2时，智能体在该列顺风移动2格，以此类推模拟风的作用。任何试图离开格子世界的动作都会使得智能体停留在移动前的位置。对于智能体来说，它不清楚整个格子世界的构造，即它不知道格子是长方形的，也不知道边界在哪里，也不清楚起始位置、终止目标位置的具体为止。对于它来说，每一个格子就相当于一个封闭的房间，在没推开门离开当前房间之前它无法知道会进入哪个房间。智能体具备记住曾经去过的格子的能力。智能体可以执行的动作是朝上、下、左、右移动一步。在到达目标之前，每一步的奖励为-1，无折扣。

![](./images/9%20windygrid.PNGG ":size=350")

求解：智能体如何才能找到一条从起始位置S到终止目标位置G的最短路线？

解答：首先将这个问题用强化学习常用的语言重新描述。这是一个不基于模型的控制问题，即智能体在不清楚模型动力学的条件下寻找最优策略的问题。环境信息包括格子世界的所有格子：10*7的长方形；起始和终止格子的位置，可以用二维或一维的坐标描述，同时还包括智能体在任何时候所在的格子位置。风的设置是环境动力学的一部分，它与长方形的边界及智能体的行为共同决定了智能体下一步的状态。智能体从环境观测不到自身位置、起始位置以及终止位置信息的坐标描述，智能体在与环境进行交互的过程中学习到自身及其它格子的位置关系。智能体的行为空间是离散的四个方向。智能体每行走一步获得即时奖励为-1，达终止目标位置的即时奖励为0，目标是找到最优策略，即最短路径。折扣系数λ可设为1。

最短路线如下图所示：

![](./images/10%20windygrid.PNG ":size=350")

智能体通过学习找到下面的动作序列（共15步）能够获得最大的奖励: -14
R,R,R,R,R,R,R,R,R,D,D,D,D,L,L

在智能体学习的早期，由于智能体对环境一无所知，SARSA算法需要探索许多不同的动作，因此在一开始的2000多步里，智能体只能完成少数几个完整的回合，但随着智能体找到一条从起点到终点的路径，其快速优化策略的能力就体现得很明显了，因为它不需要走完一个回合才能更新动作价值，而是每走一步就根据下一个状态能够得到的最好动作价值来更新当前状态的动作价值。

动手编程实践会理解更深，后续将给出编程实现链接。

**n-步Sarsa**
前面介绍的Sarsa称为1-步Sarsa或者Sarsa(0)。这里将扩展到n-步Sarsa。回顾上一讲中，n-步TD，定义了n步回报。把n-步TD回报中的状态价值用动作价值代替，也就是这里定义n-步Q回报，如下表

|步数n|更新名称|收获|
|----|----|----|
|n=1|Sarsa or Sarsa(0)| $q_{t}^{(1)} = R_{t+1} +\gamma Q(S_{t+1})$ |
|n=2||$q_{t}^{(2)} = R_{t+1} +\gamma R_{t+2} + \gamma^{2} Q(S_{t+2})$|
|...||...|
|n= $\infty$ |MC|$q_{t}^{(\infty)} = R_{t+1} +\gamma R_{t+2} + \dots + \gamma^{T-1} R_{T}$|

表中的n=1时的回报可由两部分组成：一部分是离开状态的即时奖励，另一部分是下一个状态动作对的动作价值函数 $Q(S_{t+1})$ 。

定义：n-步Q回报(Q-return)，

$$q_{t}^{(n)} = R_{t+1} +\gamma R_{t+2} + \dots + \gamma^{n-1} R_{t+n} + \gamma^{n} Q(S_{t+n})
$$

>注：1. 这里的n步回报用 $q_{t}^{(n)}$ 表示，但在Sutton的第二版的书中用的是 $G_{t:t+n}$ 表示；个人觉得这里是为了突出是Q值的回报，区别于V值的回报，但还是用 $G_{t:t+n}$ 表示好理解，因为我们一直用G来表示回报，这里用 $q_{t}^{(n)}$ 容易与Q值混淆。2. 这里的动作函数用 $Q(S_{t+n})$ 表示，而在Sutton的第二版的书中用的是$Q(S_{t+n},A_{t+n})$ 表示。

n-步Sarsa朝着n-步Q-回报方向更新 $Q(s,a)$ ,

$$
Q(S_{t}，A_{t}) \leftarrow Q(S_{t}，A_{t}) + \alpha(q_{t}^{(n)} - Q(S_{t}，A_{t}))
$$

>注：同样这里的回报用q来表示，第二版书中用G表示。

**Sarsa( $\lambda$ )前向视角（forward View）**

类似于上一讲中的TD( $\lambda$ )，对所有的n-步Q-收获按权重 $(1-\lambda)\lambda^{n-1}$ 求和，得到 $q^{\lambda}$ 回报，

$$
q_{t}^{\lambda} = (1-\lambda) \sum_{n=1}^{\infty} \lambda^{n-1}q_{t}^{(n)}
$$


前向Sarsa( $\lambda$ )朝 $q^{\lambda}$ 方向更新，

$$
Q(S_{t}，A_{t}) \leftarrow Q(S_{t}，A_{t}) + \alpha(q_{t}^{\lambda} - Q(S_{t}，A_{t}))
$$

备份图如下：

![](./images/11%20FVsarsa.PNG ":size=350")

从1-步Sarsa 到n-步Sarsa，再到Sarsa( $\lambda$ )，这些概念我们似乎很熟悉，其实与上一讲中：从1-步TD 到n-步TD，再到TD( $\lambda$ )是一样的，只是在Sarsa中把TD中用V值替换成Q值，其他都是一样的。

**Sarsa( $\lambda$ )反向视角（Backward View）**

同上一讲对于TD(λ)的反向视角一样，在在线算法中引入资格迹（Eligibility Trace）概念，只是在Sarsa( $\lambda$ )反向视角中的E值针对的不是一个状态，而是一个状态动作对：

$$ \begin{aligned}
E_{0}(s, a) &= 0 \\
E_{t}(s, a) &= \gamma \lambda  E_{t-1}(s,a) + 1 (S_{t} = s,A_{t} = a)
\end{aligned}
$$

E值对某一个状态动作对的发生频率和发生时间距离进行了综合度量，某个状态动作对发生越频繁，其E值就越大，某个状态动作对发生的时间越近，其E值就越大。

对于每一个状态动作对（s,a) 都更新 $Q(s, a)$ ，$Q(s, a)$ 正比于TD-error $\delta_{t}$ 和资格迹 $E_{t}(s, a)$ ，更新如下，

$$
\delta_{t} = R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_{t},A_{t}) \\
Q(s,a) \leftarrow Q(s,a)  + \alpha \delta_{t} E_{t}(s,a)
$$

Sarsa( $\lambda$ )引入ET概念可以更有效的在线学习，因为不必要学习完整的回合，数据用完即可丢弃。ET通常较多地应用于在线学习算法中(online algorithm)。

Sarsa( $\lambda$ )的算法实现如下：

![](./images/12%20alg.PNG ":size=350")

>注：E(s,a)在每访问完一个回合后需要重新置0，这体现了ET仅在一个Episode中发挥作用；其次，算法更新Q和E的时候针对的不是某个回合里的Q或E，而是针对智能体掌握的整个状态空间和动作空间产生的Q和E。算法为什么这么做呢？个人认为，这是因为我们是通过采样回合来学习整个状态空间和动作空间上的最优策略。

将来给出此算法的代码实践讲解，先挖个坑。

例 5-4 格子世界中的迹（Traces in gridworld）

下图用格子世界的例子来阐述Sarsa和Sarsa(λ)算法的区别。假定最左侧图描述的路线是智能体得到的一个完整回合的路径(采取两种算法中的任一算法)。对格子世界基本描述：（1）认定每一步的即时奖励为0，直到终点处即时奖励为1；（2）根据算法，除了终点以外的任何状态动作对的Q值可以是任意的，但所有的Q值均初始化为0；（3）该路线是第一次找到终点的路线。

![](./images/13%20lambdaexamp.PNG ":size=350")

下面分析有些繁琐，也可以直接看分析的小结。

**Sarsa(0)算法：**

由于是同策略学习，一开始智能体对环境一无所知，即Q值均为0，它将随机选取移步动作。在到达终点前的每一个位置S，依据当前策略，选取一个移步动作，执行该动作，环境会将其放置到一个新位置S'，同时给出即时奖励为0，在新的位置S'上，根据当前的策略它会选择新位置下的一个动作，智能体不执行该动作，仅在表中查找新状态下新动作的Q'值，由于Q=0，依据更新公式，它将把刚才离开的位置以及对应的动作的状态动作对价值Q(S,A)更新为0。如此重复直到智能体最到达终点位置 $S_{G}$ ，获得一个即时奖励1，此时智能体会更新其到达终点位置前所在那个位置（图中终点位置正下方的格子，用 $S_{G-1}$ 来表示）时采取向上移动的那个状态动作对价值Q( $S_{G-1},A_{up}$ )，此值不再是0，这是智能体在这个回合中唯一一次用非0数值来更新Q值。这样完成一个回合，此时智能体已经并只进行了一次有意义的动作价值函数的更新；同时根据新的动作价值函数产生了新的策略。这个策略绝大多数时与之前的相同，只当智能体处在 $S_{G-1}$ 位置时将会有一个确定的动作：向上移动。这里不要误认为Sarsa算法只在经历一个完整的回合之后才更新，它每走一步都会更新，只是在这个例子中，由于我们的设定，多数时候更新的数据和原来一样罢了。

此时如果要求智能体继续学习，则环境将其放入起点。智能体的第二次寻路过程一开始与首次一样都是盲目随机的，直到其进入终点位置下方的位置 $S_{G-1}$ ，智能体的策略要求其选择向上的动作直接进入终点位置。

经过第二回合的迭代，智能体知道到达终点下方的位置 $S_{G-1}$ 的动作价值比较大，因此智能体会将其通过移动一步可以到达 $S_{G-1}$ 的其它位置的Q值更新为非0。如此重复，若采用贪婪策略更新，智能体最终将得到一条到达终点的路径，这条路径的倒数第二步永远是在终点位置的下方。若采用Ɛ-greedy策略更新，那么智能体还会尝试到终点位置的左上右等其它方向的相邻位置价值也比较大，此时智能体每次完成的路径可能都不一样。通过重复多次搜索，这种Q值的实质性的更新将覆盖越来越多的状态动作对，智能体在早期采取的随机动作的步数将越来越少，直至最终实质性的更新覆盖到起始位置。此时智能体将能直接给出一条确定的从起点到终点的路径，即收敛于最优策略。

**Sarsa(λ)算法：**

该算法同时还针对每一个回合维护一个关于状态动作对(S,A)的E表，初始时E表值均为0。当智能体首次在起点 $S_{0}$ 决定移动一步 $A_{0}$ (向右)时，达到新位置为 $S_{1}$ 。这个过程中：首先智能体会做一个标记，使E的值增加1，表明智能体刚刚经历过这个事件( $S_{0},A_{0}$ )；其次它要估计这个事件的对于解决整个问题的价值，也就是估计TD误差，此时依据公式结果为0。随后智能体将要更新该回合中所有已经经历的Q值，由于存在E值，那些在( $S_{0},A_{0}$ )之前近期发生或频繁发生的(S,A)的Q值将改变得比其他Q值明显些，此外智能体还要更新其E值。对于刚从起点出发的智能体，这次更新没有使得任何Q值发生变化，E值有实质的更新。随后的过程类似，智能体有意义的发现就是对路径有一个记忆，体现在E值里，具体的Q值没发生实质变化。直到智能体到达终点位置时发生改变。此时智能体得到了一个即时奖励1，它会发现这一次变化（从 $S_{G-1}$ 采取向上动作）明显，计算这个TD误差为1，同时告诉整个经历过程中所有的(s,a)，根据其与($S_{G-1},A_{up}$) 的密切关系更新这些状态动作对的Q（上图右所示），智能体在这个回合中经历的所有状态动作对的Q值都将得到一个非0的更新，但是那些与 $S_{G-1}$ 离得近及发生频繁的状态动作对的价值提升得更加明显，E值的作用。

在图示的例子中没有显示某一状态动作频发的情况，如果智能体在寻路的过程中绕过一些弯，多次到达同一个位置，并在该位置采取的相同的动作，最终智能体到达终止状态时，就产生了多次发生的状态动作对，这时的状态动作对的价值也会得到明显提升。智能体每得到一个即时奖励，同时会对所有历史事件的价值进行依次更新，那些与当前事件关系紧密的事件的Q值改变得较为明显。这里的事件指的就是状态动作对。在同一状态采取不同动作是不同的事件。

当智能体重新从起点第二次出发时，它会发现起点处向右走的价值不再是0。如果采用greedy策略更新，智能体将根据上次经验得到的新策略直接选择右走，并且一直按照原路找到终点。如果采用Ɛ-greedy策略更新，那么智能体还会尝试新的路线。

由于为了解释方便，做了一些约定，这会导致问题并不要求智能体找到最短一条路径，如果需要找最短路径，需要在每一次状态转移时给智能体一个负的奖励。

**分析小结**
上图中右边两个格子世界的箭头表示了当智能体到达终点后，状态价值函数会被提升的格子，以及提升多少。可以看出，一步Sarsa方法只会提升最后一个动作价值函数（一个箭头），然而Sarsa( $\lambda$ )会提升从最后一个动作到回合开头的所有的动作价值函数，而且每个格子提升的幅度也不一样，以离终点的近期程度衰减（即离终点的时间越远，衰减越大）。

与一步Sarsa相比，Sarsa( $\lambda$ )通过一次尝试，智能体就能获得更多关于如何到达目标的信息，（在这里不必是最好的路径）。因此，Sarsa( $\lambda$ )可以大大加速学习速度。


## 5.4 异策略学习(Off-Policy Learning)

同策略学习的特点就是当前遵循的策略就是个体学习改善的策略。异策略学习指的是在遵循一个策略 $\mu(a|s)$ 的同时评估另一个策略 $\pi(a|s)$ ，即 遵循行为策略$\mu(a|s)$ 采样 $\lbrace S_{1},A_{1},R_{2},...,S_{T} \rbrace$ ,计算 $V{\pi}(s)$ 或 $q_{\pi}(s，a)$  来评估目标策略$\pi(a|s)$  。
为什么异策略学习很重要呢？
* 可以较容易的从人类经验或其他智能体的经验中学习
* 可以从一些旧的策略中学习，可以比较两个策略的优劣
* 其中可能也是最主要的原因就是遵循一个探索策略时，学习最优策略
* 可以在遵循一个策略，同时学习多个策略
  
同样根据是否从完整的回合中学习，可以将其分为基于蒙特卡洛的和基于TD的。基于蒙特卡洛的异策略学习仅有理论上的研究价值，在实际中毫无用处。在解释这一结论时引入了“重要性采样（importance sampling）”这个概念，

### 5.4.1 重要采样(importance sampling)

估计不同分布的期望

$$\begin {aligned}
\mathbb E_{x \sim P}[f(X)] &= \sum P[X]f[X]\\
&=\sum Q[X] {\;} \frac{P[X]}{Q[X]} {\;} f[X]\\
&= \mathbb E_{x \sim Q}\left[ \frac{P[X]}{Q[X]} f(X)\right]
\end {aligned}
$$

重要采样的目的是，求P[X]分布下的期望转换成求分布Q[X]下的期望。

**重要采样用于异策略蒙特卡洛(Importance Sampling for Off-Policy Monte-Carlo)**

用从 $\mu$ 产生的回报来评估 $\pi$ ，根据两个策略之间的相似性赋予回报 $G_{t}$ 权重。重要采样相乘的校正沿着整个回合，

$$
G_{t}^{\pi / \mu} = \frac{\pi(A_{t}|S_{t}) \pi(A_{t+1}|S_{t+1})}{\mu(A_{t}|S_{t}) \mu(A_{t+1}|S_{t+1})} \dots \frac{\pi(A_{T}|S_{T})}{\mu(A_{T}|S_{T}) } G_{t}
$$

朝着校正的回报更新价值，

$$
V(S_{t}) \larr V(S_{t}) + \alpha(G_{t}^{\pi / \mu}  - V(S_{t}))
$$

当 $\pi$ 非零 $\mu$ 为零时不可用；重要采样可以大大增加方差

**重要采样用于异策略TD(Importance Sampling for Off-Policy TD)**

用从 $\mu$ 产生的TD目标来评估 $\pi$ ，用重要采样来赋予TD目标 $R + \gamma V(S')$ 权重，只需要一次简单的重要采样来校正，

$$
V(S_{t}) \larr V(S_{t}) + \alpha \left(\frac{\pi(A_{t}|S_{t}) }{\mu(A_{t}|S_{t}) } (R_{t+1} + \gamma V(S_{t+1})) - V(S_{t})\right)
$$

比MC重要采样更低的方差，策略只需要简单的一步相似即可。
TD采样更新解释：智能体处在状态 $S_{t}$ 中，基于行为策略 $\mu$ 选取一个动作$A_{t}$ ，执行该动作后进入新的状态 $S_{t+1}$  ，在当前策略下如何根据新状态的价值调整原来状态的价值呢？异线策略的方法就是，在状态 $S_{t}$ 时比较分别依据目标策略 $\pi$ 和当前遵循的策略 $\mu$ 产生动作 $A_{t}$ 的概率大小，如果策略 $\pi$ 得到的概率值与遵循当前策略 $\mu$ 得到的概率值接近，说明根据状态 $S_{t+1}$  的价值来更新 $S_{t}$  的价值同时得到两个策略的支持，这一更新操作比较有说服力。同时也说明在状态 $S_{t}$ 时，两个策略有接近的概率选择动作 $A_{t}$ 。假如这一概率比值很小，则表明如果依照被评估的策略，选择动作 $A_{t}$ 的概率很小，这时候我们在更新 $S_{t}$ 价值的时候就不能过多的考虑基于当前策略得到的状态 $S_{t+1}$ 的价值。同样概率比值大于1时的道理也类似。这就相当于借鉴被评估策略的经验来更新我们自己的策略。所以概率比值也称为重要采样校正。

### 5.4.2 Q-学习(Q-Learning)

重要采样异策略TD这种思想应用得最好的算法是基于TD(0)的Q-学习。Q-学习的基本方法是，更新一个状态的Q值时，采用的不是当前遵循策略的下一个状态的Q价值，而是采用的待评估策略产生的下一个状态的Q价值。公式如下，

$$
Q(S_{t}，A_{t}) \leftarrow Q(S_{t}，A_{t}) + \alpha(R_{t+1} + \gamma Q(S_{t+1},A') - Q(S_{t}，A_{t}))
$$

式中，TD目标 $R_{t+1} + \gamma Q(S_{t+1},A')$ 是根据另一个评估策略 $\pi$ 选择的动作 $A'$ 得到下一个状态的Q值 （而不是根据行为策略 $\mu$ 选择的 $A_{t+1}$ 得到下一个状态的Q值）， $Q(S_{t}，A_{t})$ 朝着动作 $A'$ 方向(即TD目标)更新。

**异策略Q-学习的控制** 

智能体遵循的行为策略 $\mu$ 是对当前动作价值函数 $Q(s,a)$ 的一个Ɛ-greedy策略，而目标策略 $\pi$ 是对当前状态动作价值函数 $Q(s,a)$ 不包含Ɛ的单纯greedy策略：

$$
\pi (S_{t+1}) = \argmax_{a'} Q(S_{t+1},a')
$$

这样Q-学习的TD目标可以简化为，

$$\begin {aligned}
&R_{t+1} + \gamma Q(S_{t+1},A')\\
&=R_{t+1} + \gamma Q(S_{t+1},\argmax_{a'} Q(S_{t+1},a'))\\
&=R_{t+1} + \max_{a'} \gamma Q(S_{t+1},a')
\end {aligned}
$$

在状态 $S_{t}$ 根据行为策略Ɛ-greedy采取动作 $A_{t}$ 的Q价值将朝着 $S_{t+1}$ 状态所具有的最大Q价值的方向做一定比例的更新。这种算法能够使greedy策略 $\pi$ 最终收敛到最佳策略。由于智能体实际与环境交互的时候遵循的是Ɛ-greedy策略，它能保证经历足够丰富的新状态。

**Q-学习控制算法**

Q-学习的备份图和更新的公式如下，

![](./images/14%20Qlearning.PNG ":size=350")

上图中，在状态S时，根据行为策略 $\mu$ (Ɛ-greedy)采取动作A，得到即时奖励R，进入下一个状态S'。在状态S'根据另一个评估策略 $\pi$  (greedy)采取所有动作中使得Q(S',a)值最大的a作为A'（若是Sarsa算法，在状态S'继续根据行为策略 $\mu$ (Ɛ-greedy)采取动作A'，这是同策略与异策略的主要区别）。Q-学习的更新公式如下，

$$
Q(S，A) \leftarrow Q(S，A) + \alpha(R_{t+1} + \gamma \max_{a'} Q(S',a') - Q(S，A))
$$


定理：Q学习控制将收敛至最优状态行为价值函数：$Q(s,a) \rightarrow q_{*}(s,a) $ 。

用于异策略控制的Q-学习算法如下，

![](./images/15%20QLalg.PNG ":size=350")

例5-5 悬崖行走例子，比较Sarsa与Q-Learning，即同策略与异策略的区别。
如下图所示，图中灰色的长方形表示悬崖，在其两端有一个起点S和一个目标终点G。除了进入悬崖，其他所有的状态转移的奖励都为-1，进入悬崖的奖励为-100，且把智能体重新送回起点。动作空间为上下左右。这是一个无折扣的回合性任务的例子。可以看出最优路线是贴着悬崖上方行走。

![](./images/16%20examp1.PNG ":size=350")

下图给出了Sarsa与Q-Learning方法的性能对比，两种方法的行为策略采用Ɛ-greedy，Ɛ=0.1。可以看出，短暂的初始化后，Q-学习学到了最优策略的价值，即沿着悬崖上方往右走。不幸的是，这个结果偶尔会掉入悬崖，因为采用了Ɛ-greedy行为策略来选择动作。另一方面，Sarsa方法考虑了动作的选取，学到了更长但是更安全的路径，即沿着格子的上沿往右走。虽然Q-学习确实学到了最优策略的价值，其在线性能比Sarsa差，Sarsa学到的是迂回的策略。当然，若Ɛ逐渐减小，两种方法都会逐渐地收敛于最优策略。

![](./images/16%20examp2.PNG ":size=350")

## 5.5 小结

下面两张表概括了各种的DP算法和TD算法，同时也概况了各种不同算法之间的区别和联系。总的来说TD是采样+有数据引导(bootstrap)，DP是全宽度+实际数据。从贝尔曼期望方程角度分为：
* 针对状态价值V的贝尔曼期望方程：迭代法策略评估（DP）和TD学习
* 针对动作价值函数q的贝尔曼期望方程：Q-策略迭代（DP）和SARSA
* 针对 $q_{*}$ 的贝尔曼最优方程：Q-价值迭代(DP)和Q-学习


![](./images/17%20relation.PNG ":size=350")

下表总结各种算法的更新公式

![](./images/18%20relation2.PNG ":size=350")

至此，David Silver强化学习公开课的第一部分就结束了（1-5讲），第一部分主要介绍强化学习的基础理论和强化学习算法的核心思想，这些算法的价值函数用表格来存储，因此也称为表格解方法（Tabular Solution methods），这些方法中可以找到准确的解，也称为准确的最优价值函数和最优策略。对应Sutton第二版书中的第一部分（chapter2-chapter8）。这些算法只能求解规模比较小的问题。在接下来的第二部分，将聚焦于各种价值函数、策略函数的近似表示；用于求解大规模问题，这部分称为近似解方法（Approximation solution methods）。


参考
1.David Silver第5课
2.Richard Sutton 《Reinforcement Learning  A Introduction》chapter(6，7，12)
3.叶强《David Silver强化学习公开课中文讲解及实践》第五讲
4.搬砖的旺财《David Silver 增强学习——Lecture 5 不基于模型的控制（三）》