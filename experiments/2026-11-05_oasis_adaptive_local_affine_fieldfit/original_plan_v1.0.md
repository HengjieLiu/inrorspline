# Codex 执行任务：OASIS 真实变形场的自适应局部 affine 表示能力研究

**版本：v1.0 · 2026-11-04（洛杉矶时间）**

**研究对象：OASIS 3D 测试集既定顺序中的前 10 个 pair，来自四种已完成配准的方法。**

**交付目标：实现、运行并分析一个统一的 field-fitting 框架，比较直接 DVF、受可逆性条件约束的 DVF、SVF 三条路径。**

> 本文件可以整体交给能访问现有代码、图像和变形场的 Codex。它是执行规范与实验解释，不代表实验已经完成，也不预设哪种表示会胜出。目录和命令中标明“建议”的内容需要 Codex 创建；真实数据路径、现有配置和坐标约定必须从项目中核实。

---

## 0. 给执行 Codex 的任务与工作方式

你正在参与 DIR-papers 项目。请在当前研究代码环境中完成以下工作，而不是只返回另一份建议：

1. 阅读适用的项目说明，审计已有 OASIS 测试列表、四种方法的输出及 deformation utilities。
2. 建立不可变的数据 manifest 和坐标转换适配层。
3. 实现共享的局部 affine 向量场表示、固定/自适应权重，以及三个 transform renderer。
4. 先完成数学与坐标验证，再进行小规模真实场 pilot。
5. 冻结设置后，完成本文件定义的核心实验；扩展实验按明确条件执行。
6. 输出逐个 target 的原始结果、汇总图、可复现命令和中文解释。

本轮是 **已有参考场的表示/压缩与拟合实验**，不是从图像相似性出发重新配准。

- 不重新训练 SITReg，不重跑四种来源方法的图像配准。
- 原始参考场和已有结果只读；适配后的副本、拟合场和结果另存。
- 可以读取图像、标签和原始 warped image 做坐标核对与结果评价；标签不用于选择模型、控制点或优化损失。
- 不根据目标结果好坏换掉原定的前 10 个 pair。
- 未找到路径时，先检查当前仓库、配置、日志和当前项目已有文档。只有仍然无法定位必要输入时，列出具体缺失项；继续完成其他不依赖该输入的工作。
- 若遇到缺失场、不可解析方向或数值失败，保留该条目和原因，不以另一条“好做”的结果替代。
- 对资源开销先做 pilot，避免无计划的全因子搜索。不要在发现单个配置失败后中止整个研究。

建议实验目录：

```text
experiments/oasis_polyaffine_fieldfit_v1/
```

如果仓库已有实验目录规范，遵循现有规范，并在报告中提供实际路径。

---

## 1. 研究问题：这次实验到底要回答什么

### 1.1 四类参考场

使用同一组 OASIS pair 的以下已有结果：

| 来源标识 | 用户描述 | 必须核实的内容 |
|---|---|---|
| SITReg | 已训练方法的配准输出 | 最终复合变换、方向、坐标、是否包含 affine |
| MR-BSpline-Unbounded | 多分辨率无界 B-spline | 多层是系数细化、相加还是变换复合 |
| MR-BSpline-Bounded-2Comp | 多分辨率有界 B-spline，两个场复合 | 两个场的具体含义、顺序、界限、最终输出 |
| MR-BSpline-SVF | 多分辨率 B-spline SVF | 是单个最终 SVF，还是多个 SVF 指数的复合 |

四种来源是 **四类待拟合目标**，不是这次要重新排名的注册算法。所有目标都是已有方法估计的参考变换，不是解剖 ground truth。

### 1.2 主问题与证据

| 问题 | 必须使用的证据 |
|---|---|
| 局部 affine 比局部 translation 更有表示效率吗？ | 相同名义参数预算下的误差曲线；另列相同控制点数 |
| 自适应空间分配是否有效？ | 同一权重族下 uniform vs adaptive；相同参数与相同总时间 |
| 精度来自更好的函数空间，还是更容易优化？ | 固定几何下 DVF 最小二乘参考解与迭代优化差距 |
| 用 affine 后是否仍需要 SVF？ | 直接 DVF、可逆性约束 DVF、SVF 的拟合误差、拓扑及成本 |
| SVF 拟合不好意味着局部 affine 不够强吗？ | dense-SVF 或原始 SVF 诊断，积分收敛与多初值检查 |
| 函数保持的 split 是否有实际价值？ | split 前后场/变换变化，以及相同最终结构的 warm/cold start |
| 结论是否依赖参考场来源？ | 四种来源分面展示，禁止只汇总为一个平均数 |

这是一个小规模、逐 pair 的机制研究。10 个 pair 不能支持普遍优越性，也不能代替最终真实图像配准实验。

### 1.3 与之前 synthetic 实验的关系

之前的归档原型包含两个独立部分：

1. **固定字典：**cubic B-spline、normalized Gaussian translation、Gaussian local affine，用线性最小二乘隔离表示误差。
2. **层次模型：**smooth partition-of-unity affine tree，用 logistic gate 二分权重；split 时保持当前向量场，之后重新拟合。

第二部分不是“把 flat Gaussian 随便拆成两个小 Gaussian”。之前 adaptive 用已知目标场的残差和候选重拟合做选择，计算量高于 uniform；它在某些人工目标上失败。

本次延续这两组思想，但：

- 改为 3D 真实参考场；
- 增加导数、Jacobian、重采样和成本评价；
- 明确区分 DVF 和 SVF；
- 保留 adaptive 失败例；
- 不把既有原型的高精度当作真实配准有效性的证据。

如果原型脚本在当前环境不可用，本文件给出了足够的模型定义，不应仅因缺少旧 zip 而停止。

---

## 2. 核心解释：用了 local affine，为什么还要讨论 SVF？

### 2.1 必须分清三个层次

**局部运动模型、空间混合方式、最终变换构造，是三个不同的设计选择。**

定义物理坐标 \(x\in\mathbb R^3\)，局部中心 \(c_i\)，固定尺度矩阵 \(D_i\)，以及非负、归一化权重 \(w_i(x)\)：

\[
z_i(x)=D_i^{-1}(x-c_i),\qquad
f_i(x)=B_i z_i(x)+t_i,\qquad
\sum_i w_i(x)=1.
\]

\[
a_\theta(x)=\sum_i w_i(x)f_i(x).
\]

每个局部 affine 向量场有 9 个斜率参数和 3 个平移参数。这里 \(B_i\) 作用于归一化的局部坐标；真正的物理坐标导数是 \(B_iD_i^{-1}\)。

同一个 \(a_\theta\) 可以用于两种不同的模型：

\[
\phi_{\mathrm{DVF}}(x)=x+a_\theta(x),
\]

\[
\phi_{\mathrm{SVF}}(x)=\operatorname{Exp}_{\mathrm{flow}}(a_\theta)(x).
\]

前者把 \(a_\theta\) 解释为位移，后者把它解释为单位时间内的平稳速度。它们可以共用表示与代码，但相同系数通常不会产生相同的最终变换。

还要避免一个看似合理的误解：给定同一有限维场空间 \(S\)，\(\{\mathrm{Id}+a:a\in S\}\) 与 \(\{\operatorname{Exp}(a):a\in S\}\) 一般没有简单的包含关系。积分可能用少量速度参数产生复杂的最终位移，也可能限制某些目标的可达性。因此，D0 的 LS reference 只是 D0 自己的参考，不是同参数 SVF 的理论精度上限。

**因此，不是“有 affine 就不用 SVF”，也不是“任何可逆变换都必须用 SVF”。Affine 决定局部函数形式；SVF 是构造整体可逆变换的一条路线。**

### 2.2 每个局部 affine 都可逆，混合后仍可能不可逆

如果整个变换只有一个 global affine，\(T(x)=Ax+b\)，那么 \(\det A\ne0\) 就保证可逆；若还要求保持方向，则取 \(\det A>0\)。这种情况下可以直接约束 \(A\)，完全不必使用 SVF。困难来自多个局部 affine 的空间混合。

先看一个不涉及空间变化权重的反例：

\[
A_1=I,\qquad
A_2=R_z(\pi)=
\begin{pmatrix}
-1&0&0\\
0&-1&0\\
0&0&1
\end{pmatrix}.
\]

两者都是 determinant 为 \(+1\) 的可逆旋转。但是：

\[
\tfrac12 A_1+\tfrac12 A_2=
\operatorname{diag}(0,0,1),
\]

混合后的映射把两个维度压扁，不可逆。**正 determinant 的矩阵集合不是凸集。**

对空间变化的 blending，问题更多。直接 DVF 的 Jacobian 是：

\[
D\phi(x)=I+\sum_i w_i(x)B_iD_i^{-1}
              +\sum_i f_i(x)\nabla w_i(x)^\top.
\]

最后一项来自权重随位置变化。局部 affine 的 determinant 约束没有约束这一项；权重越尖锐、相邻局部运动差异越大，它可能越显著。

因此以下做法都不能单独作为完整的可逆性保证：

- 只让每个局部 affine 的 determinant 为正；
- 只限制局部旋转、尺度或 affine 矩阵的范数；
- 只保证 Gaussian/logistic blending 很平滑；
- 只在体素中心观察到正 Jacobian；
- 只加入 folding penalty；
- 对每个位置的混合矩阵做 matrix exponential。

最后一种构造 \(\exp(M(x))\tilde x\) 中，矩阵自身可逆不等于坐标映射可逆，因为 \(M\) 依赖 \(x\)。它也不是空间变化速度场的 ODE 积分。本实验禁止把它标记为 SVF。

上述 averaging 问题也是经典 log-Euclidean polyaffine 工作使用速度场与流积分的动机之一；见参考资料 [R1]。这里的反例与后面的充分条件是本计划的数学说明。

### 2.3 不用 SVF，也可以获得可逆性保证

对直接位移变换：

\[
\phi(x)=x+u(x),
\]

在一个凸域 \(\Omega\) 上，若对某个一致的向量范数及其诱导矩阵范数：

\[
\sup_{x\in\Omega}\|Du(x)\|\le \rho<1,
\]

则对任意 \(x,y\in\Omega\)：

\[
(1-\rho)\|x-y\|
\le \|\phi(x)-\phi(y)\|
\le (1+\rho)\|x-y\|.
\]

这给出：

- injectivity：两个不同点不会映到同一点；
- bi-Lipschitz 性：逆映射在像集上连续且稳定；
- 若 \(u\) 足够光滑，则 \(\phi\) 是到其像集的微分同胚；
- 对光滑情形，\(I+tDu\) 在 \(t\in[0,1]\) 上均非奇异，因此 determinant 与 identity 同号。

**这是充分条件，不是必要条件。**某个直接 DVF 未通过这个界，不能据此断言它有折叠。大角度旋转就可能可逆但不满足这种近 identity 的条件。纯平移可以很大，因为它的导数为零。

边界也必须说清楚：该条件保证到 \(\phi(\Omega)\) 的可逆性，不自动保证 \(\phi(\Omega)=\Omega\)。要讨论整个 \(\mathbb R^3\) 上的全局逆，或保证 fixed-point inverse 迭代始终有效，还需要明确域外延拓和相应条件。

SITReg 提供了不依赖 SVF 的实际先例：它针对特定 B-spline 表示推导控制点界，并结合可逆的小更新与求逆机制；论文还区分连续表示和采样后插值的界。[R2] **不能把那里的 clipping 数值直接搬到 Gaussian 或 affine tree。**

### 2.4 小的 bounded DVF 复合可以表达较大变形

若每个：

\[
\psi_k(x)=x+u_k(x)
\]

均满足可逆性的充分条件，那么连续映射意义下：

\[
\phi=\psi_m\circ\cdots\circ\psi_1
\]

也是可逆的。这样可以通过多次较小变化表达单步 contraction 条件不允许的变形。

但是：

- \(m\) 个独立场有更多系数；比较时总参数预算必须合并计算。
- 复合顺序必须固定；DVF 相加不是复合。
- 每一步的可逆性条件必须覆盖它实际接收到的坐标域；后续场不能只在最初的 box 内有界，却接收已经移出该 box 的点。
- 每次复合后重新栅格化、插值，可能改变连续映射的性质；最终导出的 dense field 需要重新检查。
- 它与单一 SVF 不等价。多个不同 SVF 的指数复合一般也不等于简单把这些速度相加后积分。

本轮把两段 bounded-DVF composition 作为明确的扩展诊断，不混进单场 DVF 的结果中。

### 2.5 SVF 提供什么，又不自动解决什么

SVF 定义：

\[
\frac{d\phi_t(x)}{dt}=v(\phi_t(x)),\qquad \phi_0(x)=x.
\]

在满足光滑性、流存在等条件的连续情形下：

\[
\det D\phi_1(x)
=\exp\left(\int_0^1\operatorname{div}v(\phi_t(x))\,dt\right)>0,
\qquad
\phi_1^{-1}=\operatorname{Exp}_{\mathrm{flow}}(-v).
\]

它不要求最终位移 \(u=\phi_1-\mathrm{Id}\) 满足单步 \(\|Du\|<1\)，因此能自然表达较大平滑变化。

但本实验仍须核实：

1. 有限步 scaling-and-squaring、有限网格与插值会引入误差；理论可逆不代表导出结果必然零 folding。
2. DVF endpoint fitting 的线性问题，在 SVF 分支会变成经过积分器的非线性问题。
3. 给定任意已有 DVF，不能把它直接当作 SVF，也不能对它逐点取矩阵 log 来获得 SVF。
4. 单个平稳流并不涵盖所有可逆变换；不能假定多步复合的参考场都能被一个 SVF 无误差表达。
5. 如果参考场本身有折叠，要求一个可逆模型精确复现其拓扑是不合适的。小 endpoint error 与导数/拓扑一致性也不是同一件事。

### 2.6 本轮采用的三个主要版本

| 版本 | 从共享向量场到变换 | 目的 | 需要怎样描述可逆性 |
|---|---|---|---|
| D0：direct DVF | \(\mathrm{Id}+a_\theta\) | 测量无约束表示容量 | 不保证；实际测量 |
| D1：bounded DVF | \(\mathrm{Id}+a_\theta^{\mathrm{bounded}}\) | 测量不用积分的可逆性约束代价 | 明确 certificate 类型、适用域与裕度 |
| V：SVF | \(\operatorname{Exp}_{\mathrm{flow}}(a_\theta)\) | 测量积分路线的容量、优化与数值成本 | 连续理论与离散检查分别报告 |

**推荐研究顺序：先 D0 隔离表示能力，再用 D1 与 V 比较可逆性路线。**

---

## 3. 实现统一性：共享什么，允许哪里不同

### 3.1 必须共用的部分

三分支必须共用：

- 参考变换、物理坐标、图像几何与 ROI；
- 向量场 basis、局部坐标归一化、权重族；
- uniform layout、split 定义、参数计数与 checkpoint 格式；
- endpoint fitting loss 与空间采样；
- interpolation / boundary policy 的记录方式；
- deformation composition utility；
- 指标实现、图表、时间统计与结果 schema。

只允许在 renderer、约束处理和必要的求解器上有差异。DVF 可以利用线性结构；SVF 必须通过积分器反向传播。**统一实现不意味着强行使用同一个学习率，或故意不用 DVF 的最小二乘解。**

### 3.2 建议接口

以下为待实现的接口示意，允许适配现有代码风格：

```python
class FieldModel:
    def evaluate(self, points_mm): ...
    def spatial_jacobian(self, points_mm): ...
    def linear_operator(self, points_mm, weights=None): ...
    def refine_preserving_field(self, split_spec): ...
    def count_parameters(self): ...
    def export_geometry(self): ...

class TransformRenderer:
    def render(self, model, grid_geometry, mode, settings): ...
    # mode: direct_dvf | bounded_dvf | svf

def endpoint_loss(predicted_map_mm, target_map_mm, samples, weights): ...
def compose_maps(outer_map, inner_map, geometry): ...
def evaluate_result(prediction, reference, masks, metadata): ...
```

固定几何的 linear operator 与普通 evaluate 必须表示同一个场；不能一份用于最小二乘、另一份用于最终评价。

### 3.3 两种比较口径同时保留

1. **相同结构比较：**固定完全相同的几何和树结构，只改变 renderer。它更接近“积分/约束机制的差异”。
2. **完整流程比较：**D0、D1、V 各自按同一种 adaptive 规则发展结构。这比较整个算法，不能把全部差异只归因于积分。

uniform 树和固定 B-spline grid 天然适合第一种比较。

对 adaptive 树，在 pilot 的同一批目标上做双向 topology replay：把 D0 选出的树和 V 选出的树分别冻结，让 D0、D1、V 都拟合它们。禁止只把 D0 的树称为“完全中立结构”；它由 D0 的拟合过程选择。

### 3.4 模型输出与实际 rasterized map

同时区分：

- analytic field：直接 evaluate basis 得到的连续场；
- dense rendered field：采样到网格后由 trilinear interpolation 定义的场；
- serialized/reloaded field：真正保存、加载并用于 image warping 的结果。

核心 DVF/SVF 对比使用同一评价网格和同一最终重采样路径。D0 的 analytic fitting 结果可以单列，但不能拿它与 V 的低分辨率重采样结果直接声称积分劣势。

---

## 4. P0：数据、pair 顺序与坐标审计

### 4.1 锁定前 10 个 pair

从现有测试 manifest / dataloader / 运行日志中确定“前 10 个”的实际顺序。

- 如果项目既定测试顺序是 shuffle 后的 ranks 0–9，使用该既定顺序并记录 seed。
- 如果是原始列表顺序，使用原顺序。
- 不自行引入新的 shuffle，不根据历史提示强加某个 seed，不按文件名排序替换 dataloader 顺序。
- 记录 fixed ID、moving ID、方向、pair index，并保存列表 hash。
- 四种来源必须对应相同 pair。计划有 40 个 targets；缺失时明确报告每类实际数量与共同可用子集。
- 检查是否有重复 pair 或多个 pair 共享同一个 subject；这会影响后续统计解释。

### 4.2 每个 target 的 manifest 字段

至少记录：

```text
pair_id, pair_rank, fixed_id, moving_id, source_method
source_field_path, source_field_hash, source_config_path
fixed_image_path, moving_image_path, fixed_mask_path
source_shape, tensor_layout, vector_channel_order, dtype
field_kind: displacement | coordinate_map | velocity | transform_chain
mapping_direction, pullback_or_pushforward
coordinate_units: mm | voxel | normalized_grid
fixed_affine, moving_affine, spacing, orientation
align_corners, interpolation, padding_mode
includes_global_affine, composition_order
native_velocity_available, native_control_points_available
canonical_shape, canonical_grid_affine
conversion_code_revision, conversion_status, warnings
```

保存原始配置中对 composition、SVF integration 和 bounds 的实际定义，不根据文件夹名称推断。

### 4.3 canonical convention

统一为定义在 fixed/reference 网格上的 pullback coordinate map：

\[
\phi^\star:x_{\mathrm{fixed,mm}}\longmapsto y_{\mathrm{moving,mm}},
\qquad
I_{\mathrm{warped}}(x)=I_M(\phi^\star(x)).
\]

在明确共同坐标框架后，定义：

\[
u^\star(x)=\phi^\star(x)-x.
\]

如果项目使用预处理后的共同 atlas 坐标，而不是原始扫描物理坐标，明确记录这套坐标，不混用两者。

对一般 voxel coordinate map：

\[
x=H_F\tilde i,\qquad
y=H_M\widetilde{j(i)},\qquad
u^\star(x)=y-x.
\]

不能在两幅图像的 affine、spacing 或 orientation 不同时，直接把原始三个通道乘一个 spacing 就当作完成转换。

normalized sampling grid 到 voxel center coordinate 的公式必须与现有 renderer 一致：

\[
j=(g+1)(n-1)/2 \quad(\texttt{align\_corners=True}),
\]

\[
j=((g+1)n-1)/2 \quad(\texttt{align\_corners=False}).
\]

PyTorch 的 5D tensor 轴顺序与 grid 最后一维的 \(x,y,z\) 不是同一回事；见 [R3]。不要依赖“看起来大致对齐”的图像来猜轴顺序。

### 4.4 复合与方向

若 \(\phi_A(x)=x+u_A(x)\)、\(\phi_B(x)=x+u_B(x)\)，采用右边先执行：

\[
(\phi_A\circ\phi_B)(x)
=x+u_B(x)+u_A(x+u_B(x)).
\]

要求用非交换的简单解析变换验证实现，不能只用平移测试。

- 两个 bounded field 的最终目标必须由真实顺序复合获得。
- 多层速度求和后积分与逐层积分后复合需要分开记录。
- forward/pushforward field 不能通过负号变成 inverse/pullback field。
- 若必须求逆，先核实存在性和数值残差；无法可靠求逆的目标不得悄悄纳入统一方向。

### 4.5 审计验收

对每种来源至少一个目标：

1. 用原始 warp utility 重现原结果。
2. 用 canonical map 经适配后重现同一 warped image / label。
3. 比较 coordinate-map round trip、image difference 和现有配准指标。
4. 若差异超过 float/interpolation 可以解释的水平，先修复适配，不进入 fitting。

记录逐个 target 的 displacement RMS / P95 / max、非有限值、原始 folding 指标、原始 image/label 指标（若可用），作为 reference baseline。

不得先平滑、裁剪或缩放参考场来“方便拟合”。如果需要研究平滑目标，作为另外的带标签实验保留。

---

## 5. 共享向量场模型与参数预算

### 5.1 共同全局 affine 能力

固定字典比较需要避免“某模型能再现 global affine，另一个不能”成为主要优势。

root 与局部 affine 都用固定的物理中心/尺度归一化输入坐标；所有误差和导数仍换算回 mm。数值归一化不改变函数空间，改变坐标时必须同步变换 coefficients。

所有核心模型应包含全局 affine 向量场的表示能力：

- cubic B-spline：在正确的 basis、边界与坐标约定下验证 affine reproduction；
- Gaussian translation：加入共同 global affine，并消除明显的常数冗余；
- Gaussian local affine：自身已有 affine reproduction；
- tree translation：root 使用 global affine，后续节点增加局部 translation contrast；
- tree affine：root 与后续 contrast 都使用 affine。

如果现有 B-spline 边界实现不能再现 affine，修复或给所有相关比较加入相同 global component，并报告额外冗余。不能未经测试假定边界性质成立。

### 5.2 固定 Gaussian 字典

\[
\tilde w_i(x)=
\exp\left[-\tfrac12\sum_{j=1}^3
\left(\frac{x_j-c_{ij}}{\sigma_{ij}}\right)^2\right],
\qquad
w_i=\frac{\tilde w_i}{\sum_k\tilde w_k}.
\]

首轮：

- 固定规则中心，不学习 center/covariance；
- 用物理距离确定各轴尺度；
- Gaussian translation 与 Gaussian affine 共用权重定义；
- 默认 \(\sigma/h=0.85\)；pilot 可比较 \(\{0.55,0.85,1.25\}\)，随后冻结主设置；
- 如展示每个目标的 best-\(\sigma\) envelope，单列为额外空间验证选优结果，并计算搜索成本；
- 使用数值稳定的 normalization；
- 不为了速度直接采用 hard top-k / nearest-neighbor support truncation。若使用平滑截断，视为另一个 basis 并重新验证导数和 reproduction。

### 5.3 Cubic B-spline baseline

优先复用并核对项目已有的 cubic B-spline basis，但本实验拟合的是最终参考场，而不是自动复用原配准损失。

要求：

- 明确 knots、degree、padding、grid spacing 与物理范围；
- 在相同参数预算下选择合理的近等物理间距格点；
- 同样使用线性最小二乘参考求解；
- 若做 knot refinement，使用正确 knot insertion，不用普通 resize 冒充函数保持细化；
- 若参考场正好由同一 B-spline 空间生成，可做 native-coefficient reconstruction 验证数值下限；这不是一般目标上的优越性证据。

### 5.4 smooth partition-of-unity tree

从覆盖模型域的一个 root 开始，root 权重为 1。

对某个 parent 的权重：

\[
r(x)=\operatorname{sigmoid}\left(\frac{x_a-m}{\tau}\right),
\qquad
w_L=w_P(1-r),\quad w_R=w_Pr.
\]

默认 \(\tau=0.12\) 倍 parent 对应物理边长，冻结后不逐 target 调整。边界 box 用于定义层次结构；logistic 权重并不是严格紧支撑。

由 \(w_L+w_R=w_P\)，若两个 child 初始表示同一个 parent affine 函数，split 时场严格保持。

parent 写为：

\[
f_P(x)=B_PD_P^{-1}(x-c_P)+t_P.
\]

child 的正确继承是：

\[
B_C=B_PD_P^{-1}D_C,\qquad
t_C=t_P+B_PD_P^{-1}(c_C-c_P).
\]

只复制中心化参数而不处理 center/scale 的变化，是错误实现。

### 5.5 推荐用等价的 hierarchical contrast 参数化

为了避免重复 global affine 参数，也避免 split 后更换全部列，可使用：

\[
a(x)=f_0(x)+\sum_{n\in\mathcal I}q_n(x)\,\Delta f_n(x),
\]

其中 \(\mathcal I\) 是 internal split nodes，\(q_n\) 为到该 node 的 path weight 乘其 right gate，\(\Delta f_n\) 为该 split 引入的 affine 或 translation contrast。

对于全 affine tree，这与 leaf-weighted affine 表示具有相同函数空间。用小规模数值测试验证转换。

- 新 split 增加一个 contrast，初始化为零。
- 旧 basis、旧 coefficients 不变，split 瞬间完整场保持。
- 每个 contrast 的 center/scale 在创建时固定，避免重新归一化旧列。
- tree translation 使用 affine root 加 translation contrasts。
- tree affine 使用 affine root 加 affine contrasts。

这个形式还便于构建 split 保持的解析导数界。不要把 mixed contrast 模型误称为每个 leaf 独立拥有相同自由度的模型。

### 5.6 参数数要按实际实现计算

在上述非冗余基础约定下：

| 模型 | 名义 motion 参数 |
|---|---:|
| affine root only | 12 |
| cubic B-spline translation，\(K\) 个系数点 | \(3K\)，前提是没有额外重复 global 项 |
| Gaussian translation + global affine | \(3K+9\)，去除 3 个常数冗余后 |
| Gaussian local affine | \(12K\) |
| affine-root translation tree，\(K\) leaves | \(12+3(K-1)=3K+9\) |
| affine tree，\(K\) leaves | \(12K\) |
| mixed contrast tree | \(12+3N_T+12N_A\) |
| 两段独立场复合 | 两段实际参数之和 |

此外报告：

- 控制点数/leaf 数；
- nominal coefficient count；
- 固定几何与可学习几何分别多少参数；
- 小规模设计矩阵 rank / retained condition estimate；
- checkpoint 序列化字节数，包含 tree structure、center/scale 等元数据。

离散自适应选择也有模型复杂度与存储成本，不能因为不是 gradient-trained variable 就称为“完全免费”。

### 5.7 默认预算与格点匹配

核心 motion 参数 cap 建议为：

\[
P\in\{768,\;1536,\;3072\}.
\]

对应 affine tree 的 leaves 为 \(64,128,256\)；translation tree 根据实际公式取不超过 cap 的 \(K\)。

固定 lattice 枚举合理的三维格点数，优先匹配物理间距与 FOV aspect ratio，并在不超过 cap 的情况下接近预算。不要为了严格整数匹配使用非常扁的格子。

- 记录实际 \(P\)，曲线横轴必须用实际值。
- 两个方法参数差在 5% 内时，可标记 approximate matched budget，并列出差值。
- 超过 5% 时保留曲线，不能称为严格 equal-parameter；可补充附近预算。
- same-\(K\) 与 same-\(P\) 是两套分析，不相互替代。
- 在同一条 refinement 路径上保存所需 checkpoints，避免重复运行相同配置。

如果最高预算仍明显欠拟合且 pilot 表明资源可接受，扩展到 \(6144\)，必要时再到 \(12288\)。这些扩展单独记录，不能只给一个方法扩容。

---

## 6. 拟合目标、采样与求解器

### 6.1 共同 endpoint loss

所有分支以最终 coordinate map 为目标：

\[
\mathcal L_{\mathrm{endpoint}}
=\frac{\sum_{x\in S_{\mathrm{fit}}}\omega(x)
\|\hat\phi_\theta(x)-\phi^\star(x)\|_2^2}
{\sum_{x\in S_{\mathrm{fit}}}\omega(x)}.
\]

同一坐标系下，它等价于比较最终 displacement。单位为 \(\mathrm{mm}^2\)。

核心容量实验不加入 bending、Jacobian penalty、coefficient L2 或图像相似性项。D1 的可逆性条件是明确的模型约束；V 的积分是明确的模型构造。

如果研究正则化，单列 accuracy–regularity 曲线，尽可能在最终场上定义相同物理量。不直接比较不同参数向量的相同 L2 系数。

### 6.2 ROI 与空间采样

优先复用 fixed image 的已有 brain mask；无 mask 时，用统一、目标方法无关的规则生成，检查后冻结。

- primary fitting / evaluation ROI：固定 brain mask 内、参考值有效的点；
- secondary ROI：brain 周边固定 10 mm ring，以及完整有效 FOV；
- derivative ROI：按数值 stencil 从共同 ROI 适当腐蚀，不能每个模型各自选择；
- 可逆性 certificate：覆盖声明的完整凸 model/evaluation box，不能只在不规则 brain mask 里检查。

四种来源共享同一个 pair 的 mask 与采样点。图像评价的 in-bounds mask 以参考变换和固定规则定义；预测越界单独计数，不通过丢弃这些点改善结果。

mask 与 sample IDs 是 pair-level 固定对象。若某来源在这些位置出现 NaN/Inf，默认把该 source target 标记为失败并报告位置/比例；如改用四来源共同有限域，必须预先声明、对所有来源一致应用并报告删去比例。不得各来源悄悄换 mask 后仍称为共同采样。

field fitting 的有效性由 fixed-domain 场是否定义决定。即使参考 map 指向 moving image FOV 外，该位置的 coordinate-map / EPE 仍可有效；不能据此删除。moving-FOV validity 只影响图像/标签重采样诊断，并与 field validity 分开记录。

建议起点：

- fit：最多 65,536 个分层空间采样点；
- spatial validation：最多 16,384 个与 fit 不重合的点；
- quick evaluation：最多 65,536 个另外固定的点；
- 最终 primary metrics：完整 native-resolution ROI，分块计算。

如 ROI 较小，按固定比例划分；保存 index/hash。使用空间分层与固定随机 seed，避免纯规则下采样造成 aliasing。

核心 fit/validation/quick-eval points 都是 canonical native grid 的 voxel centers。三分支的 loss 取共享最终 renderer 在这些位置的值；使用粗积分 grid 的优化阶段也必须正确映射回这些物理位置。若另加任意连续位置的 off-grid test，则 D0、D1、V 都使用最终 trilinear interpolant 评价；D0 analytic off-grid evaluation 单独列出，不混成同一个离散目标。

这里的 validation 是同一个已知场的空间检查，不是新的受试者泛化测试。目标场残差用于 adaptive 选择是允许的，但这属于 **reference-field-driven adaptation**，不能当作无监督图像配准的可用自适应信号。

若只有这 10 个 pair 可用，pilot 用 ranks 0–1 做数值与资源校准，并把整个研究称为 development diagnostic。另列 ranks 2–9 的冻结配置结果；不把调过参数的全部 10 个说成独立 held-out test。

### 6.3 D0：固定几何下的线性 least-squares reference

固定 centers、widths、tree 和 global basis 后：

\[
u_\theta=S\theta,\qquad
\min_\theta\|\sqrt W(S\theta-u^\star)\|_2^2.
\]

使用：

- 小规模：float64、列缩放后的 SVD/QR reference；
- 大规模：matrix-free LSQR/LSMR，提供 forward 与 adjoint；
- chunking、tensor separability 或递归树计算，避免分配完整 voxels × parameters dense matrix；
- 相同 fitting samples、相同 weights 与同一停止规则。

LSQR/LSMR 是迭代求解器。记录 residual、normal residual、iteration count、condition estimate 和 termination reason；见 [R4]。不能因为调用了 LSQR，就把有限精度结果称为已证明的全局最优。

建议：

- 小型 SVD 保留相对阈值 \(10^{-10}\)，同时做 rank-sensitive 诊断；
- large solve 从 \(10^{-6}\) 相对求解容限开始；
- 用延长求解与小型 reference 检查是否已足够收敛；
- 宽 Gaussian 若严重病态，报告这一点，不静默加大 ridge；
- 若必须使用 damping，单列 damping 值及其对拟合精度的影响。

这个 reference 只针对 **固定几何、指定样本与 loss**，不是所有自适应几何的最优解，也不是全部 native voxels 上的严格误差下界。

### 6.4 优化 gap

在同一个固定 basis 上比较迭代拟合与 D0 least-squares reference：

\[
\Delta E=E_{\mathrm{iterative}}-E_{\mathrm{LS\ reference}}.
\]

先在训练目标上计算 gap，再分别报告 native-ROI error。不要把不同 ROI、不同样本或不同正则条件的差异称为 optimization gap。

使用参数尺度归一化/列预条件；不要只要求各模型使用相同 Adam learning rate。

pilot 中比较：

- normalized Adam；
- normalized Adam 后接 L-BFGS；
- D0 LS reference。

最终挑选稳定且足够收敛的共同优化流程，记录 family-specific stable learning rate；公平性由收敛结果与时间曲线共同判断。

非线性求解的建议初始规则：每个完整 fitting job 总 Adam 上限 1,000 steps，必要时再接不超过 100 次 L-BFGS objective evaluations；多分辨率阶段共享这个总预算，不在每层重新增加一套免费预算。对 tree path，job 指某一 target/family/renderer/initialization 从 root 到最大预算的整条路径，所有 intermediate refits 都计入；每次 split 不得重置上限。为最终结构、最终 native-grid 阶段预留至少 300 个 Adam steps 的可用额度，可按冻结的收敛标准提前停止。若某些固定 lattice budgets 独立运行，明确标记 independent jobs，并按其实际独立成本比较。

每 25 steps 在固定 spatial-validation set 上记录一次；最终固定结构/网格至少优化 150 steps 后，若连续 5 次检查的相对改善均小于 \(10^{-4}\)，可以停止。早期结构的 split/refit 调度另由 pilot 冻结，不能用早停跳过尚未运行的预算。pilot 可因明确未收敛证据统一调整这些起点，并在主实验前冻结。记录实际 function/gradient evaluations，不把 L-BFGS 一次外层调用当作一次普通梯度步。

该规则只是计算预算与停止起点，不证明收敛。若主结论依赖一个明显未收敛的配置，按预定诊断规则增加预算并保留原结果；不只对预期获胜的方法追加计算。

### 6.5 V：SVF 的 endpoint fitting

同一 \(a_\theta\) 现在解释为速度：

\[
\hat\phi_\theta=\operatorname{Exp}_{\mathrm{flow}}(a_\theta),
\qquad
\min_\theta\mathcal L_{\mathrm{endpoint}}.
\]

要求：

- through-integrator differentiation；
- 与 D0 相同的目标、mask、样本和参数预算；
- 相同的局部坐标归一化；
- 同一套 image-independent field-fitting runner；
- 不把 \(\|a_\theta-u^\star\|^2\) 当成 SVF 最终 loss。

最少比较两个确定性初始值：

1. zero / coarse-to-fine inherited velocity；
2. 用 DVF LS coefficients 作为小变形近似的初始化。

第二种只是一种 warm start，不表示它是正确的 velocity。记录获得该初始值的时间；不在 equal-time 比较中把它当免费输入。

pilot 若两初值收敛差异很小，主实验可以固定一个；否则主实验保留两初值，并同时报告 best-observed accuracy 与总搜索时间。

### 6.6 SVF 数值积分

采用项目已验证的 scaling-and-squaring，或实现统一版本：

\[
\phi_{1/2^n}\approx \mathrm{Id}+v/2^n,\qquad
\phi_{2t}=\phi_t\circ\phi_t.
\]

- composition 的坐标、插值和边界与其他模块一致；
- 记录 integration grid、padding/extension、initial step 与 squaring steps；
- pilot 比较 \(n=6,7,8,9\) 中必要的相邻设置；
- 如不同 \(n\) 下最终图像/场/折叠发生实质改变，尚不能把误差归因于表示能力；
- 必要时在稀疏点用高精度 ODE integration 检查 flow，作为积分诊断，不作为另一个主方法；
- 优化可用 coarse-to-fine integration grid，但最终 refinement 和评价必须覆盖 native grid；不得只给 SVF 更粗的最终网格。

对速度场使用 \(n\) 与 \(n+1\) 的变化应相对于报告的 fitting error 足够小；阈值在 pilot 中预先冻结。不存在跨数据通用的“7 步一定够”。

解析 affine-flow 验证应使用完整轨迹位于积分网格和有效 padding 内的点，或提供与解析 affine 一致的域外延拓。border-padding bias 不一定随 squaring steps 增加而消失；不能把它误归因为积分精度或表示能力。

### 6.7 如果存在原生 SVF，增加一个有辨识力的辅助实验

仅当 source 确实保存了单个 \(v^\star\)，且使用原始积分器可重现 \(\phi^\star\) 时：

1. 直接最小二乘拟合 \(v^\star\)；
2. 积分拟合的速度后评价 endpoint error；
3. 与直接 endpoint optimization 比较；
4. 从原始 \(v^\star\) 启动 dense-SVF check，测量本实验积分器/坐标转换的数值下限。

这能帮助区分 velocity representation、非线性优化和积分误差。

若来源是多个 \(\operatorname{Exp}(v_k)\) 的复合，不能把 \(\sum_kv_k\) 当作同一个原生 SVF。

---

## 7. D1：一个能真正检查条件的 bounded-DVF 分支

### 7.1 保证对象必须明确

对以下两种对象分别给出标志：

```text
analytic_field_certificate
rendered_trilinear_field_certificate
certificate_domain
bound_value, bound_norm, rho, safety_margin
```

连续 smooth affine/Gaussian/tree 模型通过解析界，可以讨论严格的光滑微分同胚。

dense trilinear DVF 是连续、分片光滑的；若通过覆盖所有 cells 的 Lipschitz certificate，可保证相应域上的 bi-Lipschitz homeomorphism。它在 cell 边界未必是 \(C^1\)，不能不加说明地称为严格 \(C^1\) diffeomorphism。

### 7.2 可实现的 dense trilinear certificate

对每个 interpolation cell \(C\)，计算 trilinear displacement 在该 cell 的 8 个角点的 **cell-sided Jacobian**：

\[
J^{(C)}_b,\qquad b\in\{0,1\}^3.
\]

这里不是体素中心的 central-difference Jacobian。三线性插值的 Jacobian 在 cell 内可以由这些角点 Jacobian 作凸组合表示，因此对于凸矩阵范数：

\[
\sup_{x\in C}\|Du(x)\|_\infty
\le \max_b\|J^{(C)}_b\|_\infty.
\]

于是：

\[
B_{\mathrm{grid}}(u)=
\max_{C,b}\|J^{(C)}_b\|_\infty
\]

是一个覆盖声明域内所有 trilinear cells 的上界。矩阵 \(\infty\)-norm 是最大绝对行和，不是最大元素绝对值。

实现可直接由 cell 的八个 world-mm displacement values \(U_{ijk}\) 构造。角点 \((a,b,c)\) 处，对局部 index coordinates 的导数为：

\[
J^{\mathrm{index}}_{abc}
=\left[
U_{1bc}-U_{0bc},\
U_{a1c}-U_{a0c},\
U_{ab1}-U_{ab0}
\right].
\]

若 index-to-world 的线性部分为 \(H\)，则 \(J^{\mathrm{world}}_{abc}=J^{\mathrm{index}}_{abc}H^{-1}\)。即使两个相邻 cells 共享一个 corner，它们在该点的单侧导数也可能不同，两者都要检查。

实现要求：

- 按完整 grid-to-physical linear transform 换算导数；
- 分块遍历所有相关 cells 和八个 cell-sided corners；
- 使用足够浮点精度与明确的安全余量；
- 小型随机场上与 cell 内密集采样交叉检查；
- certificate 覆盖整个凸 box，而不是只遍历 brain-mask voxels；
- 明确边界和域外延拓。若只证明 box 内性质，不声称 box 外也有保证。

这是基于明确插值函数结构的充分界，与“多采一些点检查 Jacobian”不同。若执行环境不能正确实现，必须把结果标记为 empirical bound，不能伪称 certified。

### 7.3 解析 tree bound：作为连续模型的补充

对 hierarchical contrast 形式：

\[
a(x)=A_0(x-c_0)+t_0+
\sum_n q_n(x)\left[A_n(x-c_n)+t_n\right],
\]

令：

\[
M_n=\sup_{x\in\Omega}
\|A_n(x-c_n)+t_n\|_\infty,\qquad
G_n\ge\sup_{x\in\Omega}\|\nabla q_n(x)\|_1.
\]

一个保守上界是：

\[
B_{\mathrm{analytic}}
=\|A_0\|_\infty+
\sum_n\left(\|A_n\|_\infty+M_nG_n\right).
\]

box 上的 \(M_n\) 可以按 affine function 的顶点/绝对值界计算。logistic 单 gate 的最大导数为 \(1/(4\tau)\)；对 path gate product 可用逐项 product-rule 上界构造 \(G_n\)。

这一界包含 blending-gradient 项，新增零 contrast 不会改变界；但它可能很保守。必须同时报告 bound 与实际 observed derivative 的差距。

**若这个界让拟合明显变差，结论应是“该充分界/约束实现过于保守”，不能直接说所有非 SVF 方法表达能力差。**

### 7.4 约束归一化与训练

默认 \(\rho=0.9\)，作为预先指定的可行性设置，不宣称最优。

对一个有效上界 \(B\)，可定义：

\[
\alpha=\min\left(1,\frac{\rho}{B+\epsilon}\right).
\]

保留一个常量平移 \(b\)，缩放非恒定部分：

\[
a_{\mathrm{safe}}(x)=b+\alpha(a_\theta(x)-b).
\]

因为常量平移的导数为零，这比把整个大平移一起缩小更合理。shared coefficient representation 应能精确实现这一变换。

注意：

- 这是可行性 radial rescaling，不是一般意义的 Euclidean projection，也不是约束问题的最优解。
- D1 的 endpoint loss 必须计算在约束后的场上，再进行训练；不能只在 D0 结束后缩小一次就声称完成了 constrained fitting。
- D0 → post-hoc rescaling 可以单列，作为约束代价诊断。
- 若 loss 通过 \(\alpha\) 反向传播，记录其实现；若采用 projected optimizer，则记录完整操作与 optimizer-state policy。
- 每个被记录/导出的 D1 结果都必须通过完整 certificate；不得用上一次的界、抽样的界或局部 affine clipping 代替。
- 报告约束激活频率、\(\alpha\)、bound 计算时间，以及最终误差。

核心 D1 可首先使用 grid certificate；analytic certificate 是补充验证。不得混淆二者的保证对象。如果要求一个结果同时满足二者，必须同时检查。

#### 固定结构下，原始 D1 约束问题可以是凸的

当 geometry 冻结且 \(u_\theta=S\theta\) 时，endpoint loss 是凸二次函数，约束 \(B_{\mathrm{grid}}(S\theta)\le\rho\) 也是凸约束。对每个 cell-corner Jacobian 的一行 \(j_r(\theta)\in\mathbb R^3\)：

\[
\|j_r(\theta)\|_1\le\rho
\iff
s^\top j_r(\theta)\le\rho
\quad\text{对所有 }s\in\{-1,+1\}^3.
\]

因此在小型问题上可以得到 quadratic programming reference。不要在大图像上显式构建所有 cell 的巨大不等式矩阵；使用小型 pilot 或有明确 separation check 的 active-constraint / cutting-plane 实现。

当 D1 精度差异影响主结论时，至少在一个小型、固定 geometry/grid/samples 的问题上，对比：

1. D0 unconstrained LS；
2. D1 原始凸约束问题的收敛 reference；
3. 当前 radial-rescaling 训练实现。

这样才能区分“可行集合限制的代价”和“我们选择的约束求解方式没有找到足够好的可行解”。radial reparameterization 的优化困难不代表原始固定结构 D1 问题本身有非凸局部最优；自适应结构选择则仍是另一个问题。

### 7.5 两段 bounded DVF 扩展

仅当单段 D1 明显受 bound 限制时，在 pilot targets 上运行：

\[
\phi=(\mathrm{Id}+u_2)\circ(\mathrm{Id}+u_1).
\]

- 比较同总参数预算：每段分得的参数加总等于单段预算。
- 再单列同每段预算：总参数翻倍，明确标注。
- 两段均需完整 bound；实际 chain 与栅格化后的结果分别检查。
- 在解释中区分“更多可逆小更新”与“更多参数”。

不要未经诊断就把多段方案作为默认，以免失去对单 DVF 与单 SVF 的清晰比较。

---

## 8. 自适应策略与函数保持 split

### 8.1 核心四组 tree 实验

| 局部模型 | Uniform | Adaptive |
|---|---|---|
| translation contrasts + affine root | T-U | T-A |
| affine contrasts + affine root | A-U | A-A |

四组使用相同 gate、初始 domain、采样与最终 budget。它们用于分开局部模型自由度和空间分配的作用。

### 8.2 第一版 adaptive 选择应简单、可审计

对当前 endpoint residual \(r(x)=\hat\phi(x)-\phi^\star(x)\)，定义：

\[
E_i=\sum_{x\in S_{\mathrm{fit}}}
\omega(x)w_i(x)\|r(x)\|^2.
\]

- uniform：优先较大几何体积的 leaf，沿最长物理轴 split；
- adaptive：优先 \(E_i\) 大的 leaf，第一版同样沿最长物理轴 split；
- ties 使用固定 node ID 顺序；
- 设置固定最小 box 尺度和最大深度，避免退化；
- 每轮可批量 split 少量 leaves，之后统一 refit；
- 首轮建议每次增长不超过当前 leaves 的 25%，在预算 checkpoints 处精确停止。

这先只改变“在哪里增加容量”，避免把复杂 axis search 和算力增加同时混入。

原型中的“试三个 axis、每个候选完整 refit、选最优”可作为 oracle-trial 版本，仅在小规模 pilot 上比较。候选试验时间全部计入。

对 SVF，endpoint residual 按起始位置分配到局部 velocity supports 是一个启发式，并非正确误差估计器的证明。若它效果不好，可在扩展中比较 velocity-gradient / sensitivity score，但不得静默换成更有利的规则。

### 8.3 split 保持的验收

每次 split 在优化之前记录：

- analytic field max absolute difference；
- dense rendered field difference；
- D0 / D1 / V 最终 coordinate-map difference；
- 如 D1 bound 改变，检查是否是 representation-dependent conservative bound 引起；
- split 前后参数数、树深度、时间。

double-precision 小场测试应接近数值舍入水平；3D float32 的容限通过实际稳定性检查固定。不得把一个明显可见的场跳变描述为“差不多保持”。

对 V，若 field 在完整 integration grid 上相同，固定积分器的结果也应保持；仅在少量 fit points 相同不够。

对真正保持场且旧函数空间嵌套于新空间的 split，在相同训练样本、相同 loss 下，理论上的最小 training loss 不应上升。实际 refit 后明显上升，应检查求解容限、rank truncation、约束实现或初始化转移；它不是“更多容量导致表示能力变差”的证据。空间验证误差和最终 native-ROI error 则可能上升，这两种现象要分开。

### 8.4 warm start 不能只凭 split 保持就宣布有效

在完全相同的最终结构上比较：

1. inherited coefficients + refit；
2. cold start + refit；
3. 可选普通 clone/recenter without correct inheritance，仅作错误/不保持方案对照。

比较达到同一 error threshold 的时间与目标函数轨迹。

对 D0，在 solver 充分收敛且函数空间相同的条件下，warm 与 cold 的最优 fitting loss 应一致；差异主要应体现为求解成本或未收敛情况。V 的 warm start 可能影响非凸优化达到的解。固定结构的 D1 原始凸约束问题也不应因初值改变最优目标值；若 radial-rescaling 实现表现出初值敏感性，应解释为求解/参数化问题，并参考 §7.4 的受约束 reference。

optimizer state 默认在结构变化时重建；如继承 moments，需给出与参数变换一致的规则，并单列分析。

### 8.5 mixed translation/affine enrichment 扩展

核心结果完成后，在 pilot targets 上探索：

- 增加一个 translation split；
- 把现有 translation contrast 升级成 affine，新增 9 个 slope 参数且初值为零；
- 增加一个 affine split。

候选按实际 \(\Delta P\) 以及预估/实际计算时间比较，而不是只看 residual reduction。

这项研究才回答“何时增加局部模型复杂度、何时增加空间分辨率”。第一版先不把它与所有 datasets、renderers 和超参数做全因子组合。

---

## 9. 分阶段实验矩阵与优先级

### 9.1 P0：inventory 与转换

覆盖全部 10 pairs × 4 sources。

交付：manifest、coordinate audit、reference statistics、缺失/失败列表。通过后才运行模型 fitting。

### 9.2 P1：数学验证与 2-pair pilot

使用 ranks 0–1 的四类目标，最多 8 个 fields。

完成：

- 三维 identity、translation、affine reproduction；
- coordinate/compose/inverse sanity checks；
- linear-operator adjoint check；
- tree split inheritance；
- D0 小型 SVD 与 matrix-free solver 一致性；
- D1 bound 的 cell/analytic 验证；
- V 的 affine exponential 和 integration convergence；
- 中间预算的真实场拟合与 runtime/memory profiling；
- 最高预算、最大 native grid 上的短时 forward/backward、matrix-free operator 与完整 cell-bound profiling，覆盖 V、D1 和最重的 tree/字典配置；
- 必要的 Gaussian width 与优化器校准。

pilot 结束写出并冻结 `frozen_protocol.yaml`，列出实际预算、样本数、积分设置、停止规则和预计总成本。

成本估算至少承认：P2 有 840 个 target/family/budget 终点配置，P3 的 V 有 360 个终点配置，D1 的中预算有 80 个配置；tree 路径可复用 checkpoints，但中间 refits、候选试验和多个初值仍要另外计入。不能只测一个小预算，就假定最高预算或 native-grid 的成本线性增长。以 profile 决定 chunk size、缓存与分批运行方式，保存完整待运行矩阵。

chunking 不自动保证反向传播内存低：autograd 可能保留全部 chunks 的计算图。profile 必须实测 forward 与 backward 的峰值，必要时使用 checkpointing、重新计算或有验证的 operator 实现。

### 9.3 P2：D0 表示能力，全部参考场

核心模型：

1. B-spline translation uniform；
2. Gaussian translation uniform；
3. Gaussian local affine uniform；
4. tree T-U；
5. tree T-A；
6. tree A-U；
7. tree A-A。

每个 target 运行三个 parameter caps，保存 refinement 曲线。固定几何用 LS reference；自适应结构在每个 checkpoint 冻结后也做收敛检查。

同点数对照尽量复用已有 checkpoints；缺少重叠 \(K\) 时只补必要配置。

这一部分用于回答 representation / adaptation，不包含 SVF 的优化复杂性。

### 9.4 P3：D0 / D1 / V 的主要比较

采用能够清楚解释机制的三个 family：

1. uniform B-spline translation；
2. uniform affine tree；
3. adaptive affine tree。

**D0 与 V：**计划覆盖全部 40 targets 和三个 budgets，复用 P2 的 D0 结果。

**D1：**

- 首先在全部 40 targets 的中间预算完成 affine tree 的 uniform/adaptive 对照；
- 在 pilot targets 上完成三个 budgets 与 grid/analytic certificate 的区别；
- 若主要结论依赖 D1 budget，则扩展到全部 targets 的另外两个 budgets；
- 预算受限时，不把仅有中间预算的 D1 描述成完整 capacity curve。

执行顺序：

1. 全 targets 的中间预算，确保四类来源和两条主分支都有完整结果；
2. 全 targets 的较小预算；
3. 全 targets 的较大预算；
4. bounded / numerical diagnosis 与必要扩展。

如果 pilot 显示资源限制使完整矩阵不现实，先输出明确的预计成本与已完成覆盖，采用可恢复 job manifest 分批推进。不能静默删掉某来源、失败案例或 SVF 分支后报告“全部完成”。

### 9.5 P4：机制诊断，避免全因子膨胀

固定使用 pilot targets；必要时按预先定义的最大异常 gap 规则再加入少量诊断目标，并标记为事后诊断。

包括：

- D0 optimizer vs LS gap；
- D0/V 双向 topology replay，D1 同结构重用；
- warm/cold split；
- SVF integration \(n\) 与 grid sensitivity；
- 约束训练 vs D0 post-hoc bound scaling；
- 如可用，native-SVF fitting；
- 必要的 dense-SVF / two-component bounded-DVF check。

仅在小型样本上量化 seed sensitivity，默认 seeds 为 0、1、2；最终主设置的 seed 预先冻结。

### 9.6 P5：按证据决定是否值得做的扩展

- mixed h/p adaptation；
- 额外高参数预算；
- 可学习 Gaussian geometry；
- field-space regularization trade-off；
- 有现成可靠实现时加入 adaptive B-spline / THB-spline baseline；
- 标签初始化或图像驱动 adaptation。

这些不属于第一轮核心结论的前提。没有 adaptive B-spline 对照时，结论限定为“优于本次 uniform B-spline 或 tree controls”，不宣称优于所有自适应参数化。

---

## 10. 评价指标：先 fidelity，再几何与使用代价

### 10.1 最终变换拟合误差

\[
\mathrm{EPE}_{\mathrm{mean}}
=\frac1N\sum_x\|\hat\phi(x)-\phi^\star(x)\|_2,
\]

\[
\mathrm{RMSE}
=\sqrt{\frac1N\sum_x\|\hat\phi(x)-\phi^\star(x)\|_2^2},
\]

\[
\mathrm{relative\ RMSE}
=\sqrt{
\frac{\sum_x\|\hat\phi(x)-\phi^\star(x)\|_2^2}
{\sum_x\|\phi^\star(x)-x\|_2^2}
}.
\]

报告 mean EPE、RMSE、P95 EPE、最大误差，单位 mm；relative RMSE 以百分数显示。目标位移接近零时，不使用不稳定的相对误差作主指标。

用 residual after best global affine 的 relative error 作为补充，判断误差是否被易拟合的全局运动掩盖；分母过小时标记 undefined。它不替代总场 error。

### 10.2 导数与 Jacobian fidelity

统一在物理坐标计算：

\[
E_{\nabla}=
\sqrt{\frac1N\sum_x
\|D\hat\phi(x)-D\phi^\star(x)\|_F^2}.
\]

至少报告：

- gradient / Jacobian-matrix RMSE；
- determinant absolute difference；
- 各自 determinant 的 minimum、P01、median、P99；
- sampled folding fraction：\(\det D\phi\le0\)；
- 目标和预测的 folding-location agreement，可用作描述；
- 对两者均为正 determinant 的共同区域，报告 log-Jacobian error；
- 各自 positive region 上的 std(log J)，同时给出该区域覆盖率。

不得使用 \(\log|\det J|\) 把 folding 隐藏掉；不得把 central-difference 的零 folding 当作全连续域的证明。

folding 与 bound 分别在 primary brain ROI 和完整声明 box 内汇总。对于 D1，certificate 是整个 box 的条件；不能仅展示 brain 内较好的统计而省略 box 的失败。

DVF fidelity 与几何“更好”分开解释：拟合器可能通过平滑减少参考场折叠，同时变得更不忠实。

### 10.3 重采样效果

使用相同 image warp utility：

- 比较 \(I_M\circ\hat\phi\) 与 \(I_M\circ\phi^\star\) 的 normalized RMSE / MAE；
- 比较 source-warped label 与 fit-warped label 的一致性；
- 若有 fixed segmentation，同时报告 Dice/HD95 的保留程度；
- 报告预测 out-of-bounds fraction 与 reference out-of-bounds fraction。

这衡量压缩后的变换是否保留原配准效果，不是新的 ground-truth deformation accuracy。不要用 labels 挑模型或 stopping checkpoint。

### 10.4 inverse 与 consistency

- V：使用 \(\operatorname{Exp}(-v)\)，测量两个方向的 composition residual；
- D1：在已证明条件与域外处理允许时使用 fixed-point inverse，否则数值 inverse 只报告实际收敛情况；
- D0：仅在可可靠求逆时报告 inverse residual；不能强制把不可逆结果“求出一个 inverse”。

inverse residual 与“交换输入重新配准后的 symmetry”不是同一指标。本轮没有重新运行双向 image registration，不宣称验证了算法 input-swap symmetry。

inverse-composition 指标记录有效积分/采样域及覆盖率。不得静默删除大量越界轨迹后声称 inverse consistency 很好，也不要把已知 padding bias 与求逆误差混为一谈。

### 10.5 参数、时间、内存与存储

至少拆分：

```text
basis_setup_time
fit_solver_time
adaptive_scoring_time
candidate_trial_time
split_transfer_time
bound_time
integration_time
final_render_time
evaluation_time
peak_cpu_memory, peak_gpu_memory
model_serialized_bytes, dense_field_bytes
```

端到端 fitting time 包含结构选择、失败候选、warm-start 构造、约束和积分；不包含四种既有来源方法的原始注册时间。

记录 GPU synchronization、warm-up 与统一硬件。参考 LS 的 CPU 时间和 GPU Adam 时间分别报告，不能把 solver 差异误当纯 basis 差异。

equal-time 曲线要来自真实 elapsed time；equal-iteration 只作辅助，因为各方法每步成本不同。

pilot 冻结要记录的 time checkpoints，保存实际 solver/structure checkpoint。只有终点的 LS solve 只显示一个终点，不连接成虚构的 anytime curve。V 使用 P2 的 LS 结果作 warm start 时，虽然可以复用文件，其独立端到端时间比较仍需计入生成该初始化所需的成本。

---

## 11. 统计、图表与结果解释

### 11.1 分析单位

主要分析单位是 image pair；每个来源分别有最多 10 个 pair。

- 先逐 pair 计算指标，再等权汇总。
- 不把数百万 voxels 当作独立样本。
- 不把同一 pair 的四类参考场当作 40 个独立受试者。
- 如果 pair 共享 subject，说明依赖结构；小规模描述性结果优先，bootstrap 区间也只是探索性参考。

报告 median/IQR、mean/SD、逐 pair paired differences、win/tie/loss。

优先给效果大小与散点。若做检验，预先指定少数对照，例如 A-A vs A-U、A-A vs T-A、D0 vs V，并对检验族进行 Holm 校正。禁止对所有指标、预算与方法随意搜索显著性。

### 11.2 必须输出的图

1. **D0 error vs actual parameters：**四种来源各一幅/一个 panel。
2. **相同 \(K\) 与相同 \(P\) 的分开比较：**明确 local affine 的额外自由度。
3. **tree 2×2：**T/A × uniform/adaptive，配对连线或差值图。
4. **D0/D1/V 对照：**endpoint fidelity、folding、derivative fidelity、time。
5. **error vs wall time：**包含 adaptive 与 integration 开销。
6. **optimization gap：**D0 迭代结果与 LS reference。
7. **split event：**场变化、loss 变化、warm/cold 的轨迹。
8. **3D qualitative：**同一固定切片位置的 displacement error、Jacobian、变形网格、tree 分布。

所有方法对同一 target 使用相同色标、slice index 和 ROI。预先选择一个代表案例；另外按最大误差/失败规则展示 worst case，不只挑漂亮案例。

### 11.3 解释模板

| 观察 | 支持的解释 | 仍不能推出 |
|---|---|---|
| affine 只在同 \(K\) 更好，同 \(P\) 不更好 | 更多局部参数起主要作用 | 参数效率更高 |
| D0 LS 优势存在，但 Adam 优势消失 | 存在可达到性/condition 问题 | 函数空间没有优势 |
| adaptive 同 \(P\) 更好，同时间更差 | 参数分配有收益，决策成本高 | 实际更快 |
| D1 比 D0 差，且 bound 很保守 | 当前充分界或约束求解器可能限制结果；用小型受约束 reference 区分 | 所有 DVF 约束都不如 SVF |
| V 比 D0 差，dense-SVF 也差 | 可能涉及目标类别、积分或优化 | 已证明不存在可拟合的 SVF |
| V 比 D0 差，native velocity 很容易拟合 | endpoint optimization / integration 值得检查 | local affine 表示必然不够 |
| D0 EPE 小但 derivative error / folding 大 | 点位置拟合不能替代几何 fidelity | 变换质量已经足够 |
| 某来源特别难拟合 | 其场的结构/高频/复合方式可能不同 | 该来源方法解剖上更差 |
| 函数保持 split 但 warm start 不更快 | 保持性成立，实际加速未获支持 | 继承公式无效 |

报告先说明证据，再提出假说。对非凸 SVF 优化的失败，不给出没有证据的存在性结论。

---

## 12. 必要的验证与失败处理

这些是验证本研究数学与坐标正确性的必要检查，不是一般软件覆盖率工程。

### 12.1 最小验证集合

1. Identity 和大平移：坐标、单位与 bound 不误罚常量导数。
2. 任意小型 affine DVF：可解析的值和 Jacobian。
3. 单一 affine velocity：与 4×4 homogeneous matrix exponential 的流结果比较。
4. 两个不交换 affine 的 composition：确定执行顺序。
5. 同时可逆但混合后奇异的反例：确保没有错误的 blending 保证。
6. 加权 basis 的 forward/adjoint inner-product test。
7. analytic derivative 与 finite difference / autograd 的小样本比较。
8. split 前后 evaluate、renderer 与 certificate 的保持情况。
9. cell-sided bound 覆盖 cell 内密集采样。
10. serialized/reloaded output 重算指标的一致性。

### 12.2 有限失败策略

- coordinate audit 失败：隔离该来源/条目，修复后再拟合，不污染主表。
- LS 未收敛：允许一次有记录的延长/预条件改进；仍失败则标记 unresolved，不把当前误差当最优容量。
- SVF 两初值明显分歧：标记 optimizer-sensitive，做指定诊断，不无上限随机重启。
- D1 bound 过于保守：保留该结果，比较 observed derivative / analytic / grid bounds。
- OOM：优先 chunking、matrix-free、减少缓存；不能只给某方法降低最终空间分辨率。
- 单个高级扩展失败：保留核心结果，记录限制，不阻塞已授权的其他可运行实验。

---

## 13. 建议输出结构、配置与可恢复执行

### 13.1 输出结构

```text
oasis_polyaffine_fieldfit_v1/
  README.md
  protocol.md
  frozen_protocol.yaml
  manifest/
    pair_order.csv
    targets.jsonl
    coordinate_audit.md
    reference_metrics.csv
  configs/
    pilot.yaml
    direct_capacity.yaml
    dvf_svf_comparison.yaml
    bounded_dvf.yaml
    diagnostics.yaml
  src/
    field_io.py
    basis_models.py
    adaptive_tree.py
    renderers.py
    bounds.py
    solvers.py
    metrics.py
    run.py
  results/
    jobs.jsonl
    metrics_per_target.csv
    solver_traces/
    split_events/
    checkpoints/
    failures.jsonl
  figures/
  reports/
    pilot_report.md
    final_report.md
```

这是建议结构，不要求重复现有仓库已经有的通用模块。

### 13.2 配置起点

```yaml
experiment:
  name: oasis_polyaffine_fieldfit_v1
  objective: fit_existing_reference_transforms
  pair_selection: existing_test_order_first_10
  source_methods:
    - sitreg
    - mr_bspline_unbounded
    - mr_bspline_bounded_2comp
    - mr_bspline_svf

geometry:
  canonical_map: fixed_mm_to_moving_mm_pullback
  source_conventions: audit_required
  final_evaluation_grid: native_common_fixed_grid
  model_domain: explicit_convex_box

sampling:
  seed: 0
  max_fit_points: 65536
  max_spatial_val_points: 16384
  max_quick_eval_points: 65536
  final_eval: full_native_roi

representation:
  motion_parameter_caps: [768, 1536, 3072]
  gaussian_sigma_over_h: 0.85
  tree_gate_width_fraction: 0.12
  affine_root: true
  train_geometry: false
  split_init: exact_field_preserving

renderers:
  modes: [direct_dvf, bounded_dvf, svf]
  bounded_dvf:
    rho: 0.9
    primary_certificate: full_cell_trilinear_lipschitz
    analytic_certificate: diagnostic
    preserve_constant_translation: true
  svf:
    integration: scaling_and_squaring
    squaring_steps: determine_in_pilot
    native_grid_refinement: required
    padding_and_extension: audit_and_freeze

loss:
  type: physical_endpoint_mse
  deformation_regularizer: none
  image_similarity_term: none

solver:
  direct_reference: matrix_free_lsqr_or_lsmr
  small_reference: float64_svd
  nonlinear: normalized_adam_then_lbfgs_if_helpful
  stopping_rules: freeze_after_pilot

report:
  language: Chinese
  keep_failures: true
  per_source_analysis: true
  pair_level_summary: true
```

不要把 YAML 中的语义占位值直接传给数值 API；在冻结配置中替换为经过验证的实际值。

### 13.3 Job manifest

每个 job 至少含：

```text
job_id, pair_id, source_method, family, renderer, geometry_mode
budget_cap, actual_motion_params, actual_geometry_params
seed, initialization, solver, certificate_type
config_hash, input_hash, code_revision
status, start_time, end_time, checkpoint_path, failure_reason
```

重复执行要能跳过 hash 一致且完整的结果；输入、代码或配置改变时不能错误复用旧结果。

---

## 14. 最终中文报告必须回答的问题

请先用一页以内的核心结论回答：

1. 在各类真实参考场上，local affine 在同参数预算下是否优于 translation？
2. adaptive 是否优于 uniform？在同时间下还成立吗？
3. 哪一部分差异是表示容量，哪一部分是 optimizer 没达到 reference？
4. D0、D1、V 的精度、拓扑与计算成本有什么实际差别？
5. 不用 SVF 的 bounded-DVF 路线是否可行？限制来自 bound 保守性、预算，还是实现/优化？
6. SVF 的失败是否经过积分、初值、原生 velocity / dense control 的必要检查？
7. 对下一步图像驱动 registration，最值得保留什么？目前还缺少什么证据？

随后提供：

- 数据与坐标审计结果；
- 模型公式和实际参数计数；
- 配置、运行环境、完整覆盖表；
- 主图与逐 pair 原始数字；
- 数值/统计限制；
- adaptive 和非线性优化失败案例；
- 精确复现命令；
- 只针对已观察问题提出下一步，不默认开展网络训练或大规模 benchmark。

禁止写：

- “拟合某方法的场更好，所以临床配准更准确”；
- “每个 affine 可逆，所以 blend 后一定可逆”；
- “观察到零 folding，所以数学上已证明 diffeomorphic”；
- “SVF 拟合失败，所以该目标一定不属于任何 stationary flow”；
- “同控制点数领先，所以参数效率更高”；
- “split 保持，所以 warm start 一定更快”；
- “所有 40 个 fields 是独立的 40 个样本”。

---

## 15. 完成条件

- [ ] 前 10 pairs 的顺序和四种参考场身份已经锁定。
- [ ] 坐标、单位、方向、composition 与 interpolation 已通过审计。
- [ ] D0、D1、V 使用共享 field model 与 endpoint evaluation。
- [ ] 固定字典与 adaptive tree 的关系清楚，没有把两者当成同一方法。
- [ ] 参数数、结构数、rank 与存储开销分别记录。
- [ ] D0 least-squares reference 的收敛状态可审计。
- [ ] 核心 D0 和 SVF 结果覆盖指定 targets 与 budgets，或明确逐项列出未完成原因及可恢复 jobs。
- [ ] D1 的保证对象、域、bound 与裕度明确；未通过的结果没有被标为 certified。
- [ ] 同点数、同参数、同总时间三种比较分开。
- [ ] native-grid fidelity、导数、folding、重采样效果与成本都已报告。
- [ ] 失败案例保留，来源分别汇总，最终结论不超出证据。

---

## 参考资料与出处说明

本文件的预算、采样数、分阶段矩阵和停止策略是本次研究的建议设置，不是引用文献的既定最优值。数学反例、tree contrast/bound 和 trilinear certificate 的说明用于指导本次实现，应通过规定的小型验证核实代码。

- **[R1] Arsigny et al.** *A Fast and Log-Euclidean Polyaffine Framework for Locally Linear Registration.* JMIV, 2009. [作者稿 PDF](https://www-sop.inria.fr/asclepios/Publications/Xavier.Pennec/Arsigny_JMIV_polyaffine.pdf)。用于核实直接 affine blending 与速度流构造的区别。
- **[R2] Honkamaa and Marttinen.** *SITReg: Multi-resolution architecture for symmetric, inverse consistent, and topology preserving image registration.* MELBA, 2024. [全文](https://www.melba-journal.org/papers/2024:026.html)，尤其 §3.3–3.4 与 Appendix D。[代码](https://github.com/honkamj/SITReg)。用于核实非 SVF 的 bounded deformation 路线与连续/采样表示的区分。
- **[R3] PyTorch 官方文档。** [grid_sample](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html)。用于核实 grid 坐标、align_corners、三维插值与边界规则；具体行为以实验环境安装版本为准。
- **[R4] SciPy 官方文档。** [LSQR](https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html)。用于核实 matrix-free least squares、停止条件与缩放；本实验的收敛判断仍需实际记录。
- **原 synthetic 实验。** [adaptive_polyaffine_study.zip](https://chatgpt.com/library?file_id=libfile_709a461fafa48191a0132d5065857680&account_id=personal)。本计划已核对其中 `protocol.txt`、`numerical_summary.txt` 与 `run_experiment.py`；它提供二维原型，不提供这次 OASIS 的真实运行结果。

**最后的研究判断原则：先回答“在同等预算下能拟合多好”，再回答“能否稳定地保持几何性质”，最后回答“达到这一结果需要多少计算”。这三个问题分别报告，并用统一实现把它们连接起来。**
