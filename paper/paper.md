## 题目：EIHMR: Collaborative Human-Camera estimation for  for 4D Human Capture

## 摘要
Recovering global 3D human motion from monocular video captured by a moving camera is a fundamental yet challenging problem, as camera ego-motion and human body motion are tightly entangled in the image observations. The prevailing two-stage paradigm treats camera estimation and motion reconstruction as isolated processes, causing errors on both sides to be further amplified when combined in the world coordinate system. To address this, we draw inspiration from the human inner visual simulation mechanism and propose EIHMR, Embodied Imagination Human Mesh Recovery, a collaborative human-camera co-estimation framework. EIHMR comprises two complementary stages that form a decouple-then-collaborate loop: Scene-Aware Local Human Motion Reconstruction, SA-LHMR, reprojects motion sequences into a frozen keyframe viewpoint and applies metric depth and kinematic constraints to produce geometrically consistent local motion; Motion-aware SLAM, MO-SLAM, then re-renders the refined motion as static meshes into the original frames, converting dynamic human regions into structured matching cues that substantially strengthen camera estimation. On the EMDB dataset, EIHMR reduces global trajectory error by approximately 11.5\%, demonstrating strong capability in long-range scene-aligned human motion reconstruction.

## 导师意见：
#### 姜老师：
1.如果范式就和之前的方法有比较大的差别，可以在intro清楚地点出，可以配一张简单的图，大的范式区别是我们是把相机轨迹和人体轨迹进行了collaborate合作估计。在局部SA-LHMR用了的是相机运动和人体运动的解耦decouple。
2.整体的写法要有一个核心点，并且不断地重复强化。不能用提出两个问题的写法。整体的结构要和下面的Method一致。
3.Method的第一部分有些杂糅，把pipeline的意义和具体的流程融在了一起，可以参考yuzhi的方法来写。
4.协作式人体相机联合估计，这个关键词要在描述前序工作缺点的时候，或者描述common 缺点的时候点出来，进行观点的强化，突出协作式，协作式也可以体现在标题中。
5.这里提到了decouple，可以将此作为关键点体现在标题中？最好是前后说法一致。最好有一两个点，有一些fancy的理论支撑，而不是像随意设计的一样。需要突出难点，比如说以前的方法都耦合->受到人类启发可以解耦->解耦怎么怎么难->我们克服难点。
6.这一段太长了，Why在intro里应该介绍，方法细节应该在后面的小节介绍。这一节只要简单讲一下pipeline就行。应该采用总分的形式。如果要简单介绍的话可以写个前言。需要梳理一下整体的逻辑。
#### 王老师：题目要再精简，留下最为核心的思想
#### 誉之: 
1.感觉上面两个question的内容说的有点太繁杂了，可以尝试抓住一些关键词highlight或者斜体来论述，让读者直接抓住重点
2.感觉摘要的方法可以合并一波，然后下面的however跟现有方法的问题合成一句
3.摘要这两个点可以用连接词连接一下，不大容易看出来这是你做的两个贡献点
### GVHMR的Abstract:
We present a novel method for recovering world-grounded human motion from monocular video. The main challenge lies in the ambiguity of defining the world coordinate system, which varies between sequences. Previous approaches attempt to alleviate this issue by predicting relative motion in an autoregressive manner, but are prone to accumulating errors. Instead, we propose estimating human poses in a novel Gravity-View (GV) coordinate system, which is defined by the world gravity and the camera view direction. The proposed GV system is naturally gravity-aligned and uniquely defined for each video frame, largely reducing the ambiguity of learning image-pose mapping. The estimated poses can be transformed back to the world coordinate system using camera rotations, forming a global motion sequence. Additionally, the per-frame estimation avoids error accumulation in the autoregressive methods. Experiments on in-the-wild benchmarks demonstrate that our method recovers more realistic motion in both the camera space and world-grounded settings, outperforming state-of-the-art methods in both accuracy and speed. The code is available at https://zju3dv.github.io/gvhmr.

### GVHMR的intro示范：
World-Grounded Human Motion Recovery (HMR) aims to reconstruct continuous 3D human motion within a gravity-aware world coordinate system. Unlike conventional motion captured in the camera frame [Kanazawa et al. 2018], world-grounded motion is inherently suitable as foundational data for generative and physical models, such as text-to-motion generation [Guo et al. 2022; Tevet et al. 2023] and humanoid robot imitation learning [He et al. 2024]. In these applications, motion sequences must be high-quality and consistent in a gravity-aware world coordinate system. 

Most existing HMR methods can recover promising camera-space human motion from videos [Kocabas et al. 2020; Shen et al. 2023; Wei et al. 2022]. To recover the global motion, a straightforward approach is to use camera poses [Teed et al. 2024] to transform camera-space motion to world-space. However, the results are not guaranteed to be gravity-aligned, and errors in translations and poses can accumulate over time, resulting in implausible global motion. Recent work, WHAM [Shin et al. 2024], attempts to recover global motion by autoregressively predicting relative global poses with RNN. While this method achieves significant improvements, it requires a good initialization and suffers from accumulated errors over long sequences, making it challenging to maintain consistency in the gravity direction. We believe the inherent challenge stems from the ambiguity in defining the world coordinate system. Given the world coordinate axes, any rotation around the gravity axis defines a valid gravity-aware world coordinate system. 

In this work, we propose GVHMR to estimate gravity-aware human poses for each frame and then compose them with gravity constraints to avoid accumulated errors in the gravity direction. This design is motivated by the observation that, for a person in any image, we humans are able to easily infer the gravity-aware human pose, as shown in Fig. 2. Additionally, given two consecutive frames, it is easier to estimate the 1-degree-of-freedom rotation around the gravity direction, compared to the full 3-degree-of-freedom rotation. Therefore, we propose a novel Gravity-View (GV) coordinate system, defined by the gravity and camera view directions. Using the GV system, we develop a network that predicts the gravity-aware human orientation. We also propose a recovery algorithm to estimate the relative rotation between GV systems, enabling us to align all frames into a consistent gravity-aware world coordinate system. 

Thanks to the GV coordinates, we can process human rotations in parallel over time. We propose a transformer [Vaswani et al. 2017] model enhanced with Rotary Positional Embedding (RoPE) [Su et al. 2024] to directly regress the entire motion sequence. Compared to the commonly used absolute position encoding, RoPE better captures the relative relationships between video frames and handles long sequences more effectively. During inference, we introduce a mask to limit each frame's receptive field, avoiding the complex sliding windows and enabling parallel inference for infinitely long sequences. Additionally, we predict stationary labels for hands and feet, which are used to refine foot sliding and global trajectories. In summary, our contributions are threefold: 1. We propose a novel Gravity-View coordinate system and the global orientation recovery method to reduce the cumulative errors in the gravity direction. 2. We develop a Transformer model enhanced by RoPE to generalize to long sequences and improve motion estimation. 3. We demonstrate the effectiveness of our approach through extensive experiments, showing that it outperforms previous methods in both in-camera and world-grounded accuracy.

## Introduction
Recovering global 3D human motion from monocular video constitutes a fundamental problem in computer vision. Beyond estimating kinematic body poses, this task requires recovering complete human trajectories within a world coordinate system to support downstream applications such as human-scene interaction understanding, embodied intelligence, and digital human animation. The challenge is especially acute when the input is captured by a moving camera, as camera ego-motion and human body motion are tightly entangled in the image observations, rendering their separation inherently ambiguous.

The prevailing paradigm addresses this challenge through a two-stage pipeline: camera trajectories are first estimated via visual SLAM, after which human body poses are independently reconstructed and transformed into world coordinates. However, this paradigm treats camera estimation and motion reconstruction as isolated processes, and this isolation introduces two critical limitations. On the camera side, SLAM must mask out dynamic human regions to satisfy the static-scene assumption; when the subject dominates the field of view, the available visual features are drastically reduced, severely degrading camera estimation quality. On the motion side, body poses estimated without access to scene geometry lack physically grounded spatial constraints, resulting in non-physical artifacts such as foot sliding, ground penetration, and floating limbs. These two sources of error are further amplified when combined in the world coordinate system.

We posit that overcoming these limitations necessitates a collaborative human-camera estimation paradigm. Our approach draws inspiration from the human cognitive process of perceiving others' motion, hence the name Embodied Imagination HMR, EIHMR. When observing a moving person, humans do not passively register visual impressions; rather, they mentally anchor a stable viewpoint, leverage scene cues such as foot-ground contact to reconstruct a scale-consistent local motion sequence, and subsequently employ their kinematic understanding of the human body to localize themselves and reconstruct the surrounding environment. This decouple-then-collaborate cognitive loop motivates the design of our framework: camera information provides scene anchors for the estimated motion sequences, facilitating scene-aware motion refinement, and the refined motion, in turn, enhances camera localization.

Concretely, we propose EIHMR, Embodied Imagination Human Mesh Recovery, a collaborative framework that emulates the human ability to mentally simulate observed motion in an imagined stable reference frame. EIHMR comprises two complementary stages: Scene-Aware Local Human Motion Reconstruction, SA-LHMR, and Motion-aware SLAM, MO-SLAM. SA-LHMR reprojects motion sequences into a frozen keyframe viewpoint, thereby eliminating camera ego-motion interference, and applies metric depth and kinematic constraints via sliding-window optimization to produce geometrically consistent local motion free of non-physical artifacts. MO-SLAM re-renders the refined motion as geometrically consistent static meshes and injects them into the original frames, converting the human body from an excluded dynamic region into a source of dense, kinematically-aware matching cues that substantially strengthen camera estimation, particularly in human-dominated and texture-scarce scenarios. Together, the two stages realize a decouple-then-collaborate loop that supersedes the isolated, unidirectional information flow of prior work. Our contributions are summarized as follows:

\begin{enumerate}
\item We introduce EIHMR, a collaborative human-camera co-estimation framework that first decouples human and camera motion in a frozen-camera reference frame for scene-aware refinement, and then feeds the refined motion back to augment camera estimation, establishing a bidirectional information flow absent in existing methods.

\item We propose SA-LHMR, which reprojects motion into a unified keyframe viewpoint and applies a two-stage refinement: root trajectory correction via forward-kinematics stationarity analysis, followed by contact-aware inverse kinematics guided by metric depth, effectively eliminating foot sliding and scene penetration artifacts.

\item We propose MO-SLAM, which re-renders refined human motion as geometrically consistent static meshes and integrates them into the SLAM pipeline through forced keyframe injection, adaptive image replacement, and human-aware correlation weighting, transforming the human body from a source of interference into structured features that provide dense matching cues and markedly improve camera estimation robustness.

\item We validate EIHMR through comprehensive experiments, demonstrating that the proposed collaborative strategy yields consistent gains across three complementary evaluation dimensions: in-camera body pose estimation, camera trajectory recovery in dynamic human-dominated scenes, and world-coordinate global human trajectory reconstruction.
\end{enumerate}


### Pipeline

我们的目标是从野外移动相机拍摄的单目视频 $\mathcal{V} = \{I_t\}_{t=0}^{T}$ 中恢复完整的三维人体运动。我们将全局运动分解为世界坐标系下的 $SE(3)$ 根轨迹 $\{H_t\}_{t=0}^{T}$ 和相机坐标系下以SMPL姿态序列表示的运动学身体运动 $\{\Theta_t\}_{t=0}^{T}$。全局人体运动可通过估计相机轨迹 $\{C_t\}_{t=0}^{T}$ 及各时刻人体相对于相机的位置 $\{T_t\}_{t=0}^{T}$ 来获取：

$$H_t = C_t \cdot T_t.$$

#### EIHMR Pipeline

现有两阶段范式首先通过掩膜排除动态人体区域以满足SLAM的静态场景假设来估计相机轨迹：

$$\{C_t\} = \text{Mask-SLAM}(\{I_t\}, \{M_t\}),$$

然后基于恢复的相机参数条件重建人体运动：

$$\{\Theta_t, H_t\} = \mathcal{M}(\mathcal{V}, \{C_t\}).$$

这种单向流水线将相机估计与运动重建视为孤立过程：SLAM无法从估计的人体运动中获益，而人体姿态估计器则继承了所有相机误差且没有任何校正机制。

EIHMR用解耦-协作循环替代了这种孤立范式（Fig. overview）。SA-LHMR将姿态序列投影到冻结的关键帧视角下，通过运动学约束和度量深度进行场景感知的局部精修：

$$\{\Theta_t', H_t'\} = \text{SA-LHMR}\left(\mathcal{V}, \{C_t\}, \{\Theta_t\}, \{H_t\}\right).$$

精修后的运动随后被渲染为几何一致的静态网格叠加到原始帧上，生成运动增强图像 $\{\tilde{I}_t\}$。MO-SLAM利用这些结构化视觉线索进行鲁棒的相机估计：

$$\{C_t'\} = \text{MO-SLAM}\left(\{\tilde{I}_t\}, \{M_t\}\right).$$

SA-LHMR和MO-SLAM的详细公式分别在Sec 3.2和Sec 3.3中介绍。

### SA-LHMR

我们从视频中选取关键帧集合 $\mathcal{K} = \{k_i\}$，以每个关键帧的视角作为局部参考坐标系。对于关键帧 $k_i$，将滑动窗口 $\mathcal{W}_i$ 内邻帧的人体运动通过相对相机变换投影到该视角。具体地，对于距关键帧偏移为 $j$ 的帧，其相机坐标系下的根位置 $T_{ij}$ 被变换到关键帧坐标系：

$$H_{ij}^{k} = C_{k_i}^{-1} \cdot C_{ij} \cdot T_{ij},$$

其中 $C_{k_i}$ 和 $C_{ij}$ 分别为关键帧和邻帧的相机到世界变换矩阵。身体姿态 $\{\Theta_{ij}\}$ 因定义在局部身体坐标系下而保持不变。由于所有帧共享统一视角，相机自运动被彻底消除，许多运动估计中的非物理问题便清晰可辨：人体模型与场景地面缺乏合理的接触关系，本应静止的足部关节产生了违反运动学的滑步位移，肢体穿透楼梯、地面等场景表面。这些非物理运动伪影的产生源于两方面：其一，初始SLAM提供的相机参数存在估计误差；其二，具备时序感知能力的HPE网络在错误相机信息的条件下，为维持运动的时序流畅性而牺牲了与图像观测的几何一致性。值得注意的是，这些误差可以通过前向运动学和场景深度的方式被显式地度量与修正，但却难以被端到端的深度网络有效学习。

受人类运动认知过程的启发——人在估计他人运动轨迹时，并非完全依赖条件反射式的直觉，而是会在想象空间中尝试重建运动序列，并重点关注人体与场景交互的关键部位（如足部与地面的接触位置）来实现对运动的精确定位——我们提出了两阶段的动作精修策略。第一阶段，我们基于前向运动学的静态关节分析修正运动序列的根轨迹；第二阶段，我们利用度量深度图检测场景穿透的接触点，并通过逆运动学优化身体姿态以消除穿透。关键的是，由于统一视角下所有帧共享同一深度图参考，我们能够以单帧深度估计同时约束整个窗口内多帧的多个接触点，形成强大的时空一致性约束。

**阶段一：基于前向运动学的根轨迹修正。** 我们以滑动窗口方式遍历关键帧 $k_i$ 下的局部序列。对于以中心帧 $c$ 为锚点、窗口大小为 $2W+1$ 的窗口 $\mathcal{W}_c = [c-W, c+W]$，首先通过前向运动学（FK）计算窗口内各帧的世界坐标系关节位置：

$$J_{ij} = \text{FK}(\Theta_{ij}, \boldsymbol{\beta}, H_{ij}^{k}), \quad j \in \mathcal{W}_c$$

HPE网络同时输出逐帧静态置信度 $\mathbf{s}_t \in [0,1]^6$（分别对应左右脚踝、脚趾和手腕六个末端关节），用以识别应与场景保持静态接触的关节。对于被识别为静态的关节，我们计算其帧间位移并将其作为根轨迹的修正信号：静态关节在世界坐标系中本不应发生位移，因此将其位移的逆作为平移修正量反向传播到根轨迹 $H_{ij}^{k}$ 上，以消除由相机估计误差和HPE时序平滑引入的运动轨迹偏差。修正过程以窗口中心帧为锚点，保持中心帧的根位置不变，仅调整邻帧的相对位移。

对于重叠窗口区域的帧，各窗口贡献的修正后根轨迹通过基于帧到中心帧距离的高斯权重进行归一化合并：

$$H_{j}^{k'} = \frac{\sum_{i} w_{ij} \cdot H_{ij}^{k'}}{\sum_{i} w_{ij}}, \quad w_{ij} = \exp\left(-\frac{(j - c_i)^2}{2\sigma^2}\right)$$

其中 $c_i$ 为第 $i$ 个窗口的中心帧索引，$\sigma$ 控制高斯衰减的带宽，确保重叠区域的平滑过渡。

**阶段二：基于场景深度的接触感知逆运动学。** 根轨迹修正后，运动序列中仍可能存在肢体穿透场景表面的非物理现象——例如脚部穿过楼梯台阶或地面。我们利用关键帧视角下的度量深度图 $D_c = \text{Metric3D}(I_{k_i})$ 来检测并纠正这些穿透。

首先，我们进行窗口局部的尺度对齐。由于SLAM估计的相机坐标系与度量深度图之间可能存在尺度偏差，我们将关键帧相机坐标系下的3D关节 $J_j \in \mathbb{R}^3$ 投影到图像平面以获取像素坐标和深度：

$$z_j^{\text{smpl}} \begin{pmatrix} u_j \\ v_j \\ 1 \end{pmatrix} = K \cdot J_j,$$

其中 $K$ 为相机内参矩阵，$(u_j, v_j)$ 为投影像素坐标，$z_j^{\text{smpl}}$ 为关节深度。对应的场景深度 $z_j^{\text{scene}} = D_c(u_j, v_j)$ 从度量深度图中在 $(u_j, v_j)$ 处采样。以中心帧全部22个关节为采样点，计算尺度因子：

$$\alpha = \text{median}\left(\frac{z_j^{\text{scene}}}{z_j^{\text{smpl}}}\right), \quad j = 1, \ldots, 22$$

随后，对窗口内每帧的每个静态足部关节，我们检测其是否穿透了场景表面：将关节投影到中心帧相机坐标系，比较尺度对齐后的关节深度 $z_j^{\text{smpl}} \cdot \alpha$ 与深度图值 $z_j^{\text{scene}}$，若 $z_j^{\text{smpl}} \cdot \alpha > z_j^{\text{scene}} + \epsilon$（$\epsilon$ 为穿透容差），则判定为穿透。对于每个穿透关节，我们利用深度图反投影获取其在世界坐标系下的目标位置：

$$\mathbf{p}_j^{\text{target}} = R_{cw}^{\top} \left(\frac{z_j^{\text{scene}}}{\alpha} \cdot K^{-1} \begin{pmatrix} u_j \\ v_j \\ 1 \end{pmatrix} - \mathbf{t}_{cw}\right)$$

其中 $(u_j, v_j)$ 为关节的像素坐标。最终，我们优化窗口内所有帧腿部运动链（髋→膝→踝→脚趾）上的姿态增量 $\delta\boldsymbol{\theta}$，最小化接触关节到目标位置的距离：

$$\mathcal{L}_{\text{contact}} = \sum_{j \in \mathcal{S}} \rho\left(\|\text{FK}(\Theta + \delta\boldsymbol{\theta})_j - \mathbf{p}_j^{\text{target}}\|_2\right)$$

其中 $\mathcal{S}$ 为所有穿透的静态关节集合，$\rho(\cdot)$ 为Geman-McClure鲁棒核函数。优化同时施加姿态正则化 $\mathcal{L}_{\text{reg}} = \|\delta\boldsymbol{\theta}\|_2^2$ 和时序平滑约束 $\mathcal{L}_{\text{smooth}}$，以确保修正后的姿态在物理上合理且时序连贯。与根轨迹修正类似，各窗口优化后的身体姿态 $\Theta_{ij}'$ 同样通过高斯距离加权在重叠区域进行归一化合并，生成全局一致的姿态序列。


### MO-SLAM