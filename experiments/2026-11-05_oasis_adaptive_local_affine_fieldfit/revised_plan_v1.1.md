# OASIS 真实变形场的自适应局部 affine 表示能力研究

版本：v1.1 · 2026-11-05（America/Los_Angeles）

状态：本地审计后的执行计划，未运行拟合实验。用户已确认来源设置和 D0 优先范围。
本文件优先于 [原始计划](original_plan_v1.0.md) 中冲突的队列、来源、ROI、路径和完成条件；
原件用于保留完整数学推导及任务来源。本轮交付是实验建档与计划修订。

## 1. 研究问题和此次调整

主问题：在同一批真实参考变形场上，相同参数和实际计算预算下，局部 affine 是否比
translation 更有效；自适应空间分配是否比 uniform 更有效；优势是否能由充分收敛的
固定几何 least-squares（LS）结果支持。

这是一项已有场的表示/压缩研究，不重新配准、不训练 SITReg，不用标签选结构、
超参数或 checkpoint。参考场是已有算法输出，不是解剖真值。

| 原计划 | v1.1 决策及原因 |
| --- | --- |
| 从运行环境推断 first10，可能使用原始顺序 | 明确锁定仓库 seed-3 shuffled ranks 0–9；用户已确认 |
| 四种来源但未锁定深度、λ 和 SVF 类型 | 8→4→2、λ=1；原始 SITReg、prolonged DVF、每层两步 bounded、refined single SVF/velocity penalty |
| D0 与 SVF 都覆盖完整大矩阵 | D0 为主实验；D1/V 为固定 pilot targets 的较小后续比较 |
| 从极端 affine 混合反例展开 | 从真实场局部变化、拟合残差和导数保真展开；反例只检查错误保证 |
| 优先已有 brain mask | 必须使用 fixed raw image > 0；不使用 seg > 0 作为拟合域 |
| 结果可放实验目录 | 实验目录只放源文件；所有生成物写 ARTIFACT_ROOT |
| 通用容器/旧绝对路径 | 使用已运行的默认容器及其配置用户；来源从报告字段解析 |

## 2. 平滑场假说与直接 blending 的地位

局部模型保持原式：

\[
f_i(x)=B_iD_i^{-1}(x-c_i)+t_i,\quad
a(x)=\sum_iw_i(x)f_i(x),\quad \sum_iw_i(x)=1.
\]

D0 直接使用 \(\phi(x)=x+a(x)\)。不要求每个局部 affine 单独为可逆矩阵，
也不预先施加 SVF 或 Jacobian penalty；先测其在真实场上的容量和几何行为。
本计划的“局部 affine”指 affine 向量场表示，不冒称复现经典 log-Euclidean 算法。

对二阶光滑的目标位移，局部一阶 Taylor 近似的余项可随局部 Hessian 和邻域尺度平方
减小。这是 affine 适合平滑变形的动机，不是相同参数预算下必然胜出的证明：
每个 affine 有 12 个运动参数，而 translation 只有 3 个。

\[
Da=\sum_iw_i B_iD_i^{-1}+\sum_i(f_i-f_{\rm ref})\nabla w_i^T.
\]

第二式利用 \(\sum_i\nabla w_i=0\)，\(f_{\rm ref}\) 可取任一共同参考值。
它说明 blending 项与局部运动分歧、gate 梯度相关；公共大平移本身不会造成这一项。
平滑权重不自动限制其幅度，场平滑也不自动意味着近 identity 或无折叠。

180° 反例仍是有效的反例，但不代表 OASIS 常出现这种运动；将其降为小型单元验证。
经典 polyaffine 文献讨论了直接位移融合缺乏一般可逆保证，并提出流构造，
这不要求本研究先采用 SVF。[Arsigny 等作者稿，§2.1](https://www-sop.inria.fr/asclepios/Publications/Xavier.Pennec/Arsigny_JMIV_polyaffine.pdf)

P0 对各参考场按 ROI/full box 测量位移、\(Du\)、Jacobian determinant 和局部导数变化；
固定物理窗口的一阶 Taylor residual 用作描述性诊断。窗口建议 8/16/32 mm，
pilot 后固定，不按某个模型的优势选择。D0 拟合后另外记录两个导数项的范数分布、
相邻局部 affine 分歧和 gate width，检查残差/新生折叠是否集中在 transition 区。
这些诊断不进入主损失，不凭视觉“看起来光滑”确认假说。

现有来源报告提示 unbounded λ=1 场存在明显 sampled folding，其他来源也不严格为零。
因此“逼近参考场”与“获得更好的几何性质”分别报告，不能要求一个可逆模型精确保留
已折叠参考场的拓扑，也不能把较低 folding 自动说成较好的场拟合。

## 3. 固定 cohort 与来源选择

Split：`configs/splits/oasis/pair_test_200_shuffled_seed000003.txt`。
第一列 fixed，第二列 moving；registration 的 moving-to-fixed 标签对应 fixed-to-moving
pull sampling。禁止再 shuffle 或改用排序列表前十行。

| rank | fixed | moving | 原排序 index |
| --- | --- | --- | --- |
| 0 | 0333 | 0115 | 158 |
| 1 | 0113 | 0017 | 62 |
| 2 | 0037 | 0394 | 14 |
| 3 | 0117 | 0322 | 69 |
| 4 | 0084 | 0108 | 47 |
| 5 | 0288 | 0024 | 137 |
| 6 | 0194 | 0184 | 110 |
| 7 | 0014 | 0149 | 6 |
| 8 | 0215 | 0326 | 116 |
| 9 | 0173 | 0136 | 88 |

完整 split SHA-256：`40ff80b23407c37ed8a20092f018fd9588e5df3eb95628069ec42da698aac3a0`。
前十条规范化为 UTF-8 `fixed,moving\n`、无表头时 SHA-256：
`606600ea66c63406a8328a46b37e01475fe15de59a3ea6a3edb751a5046673c6`。
这十对含 20 个不同 subject；四来源共享同十对，不是 40 个独立样本。
first10 已多次参与历史探索；ranks 2–9 只能称本轮冻结配置的非 pilot 子集，不能称 untouched test。

以下 key 均相对 `${ARTIFACT_ROOT}/repo_experiments/`。每个来源必须恰好十条，
按 rank 和 fixed/moving ID 联结，不依赖 JSON 行顺序或模糊文件 glob。

| handle | report key / 选择器 | 最终目标与本地证据 |
| --- | --- | --- |
| SITReg | `oasis_seed3_first10_mrdbscp_dl_ncc9_diffusion_6way_v1/results.json`，`case_metrics` 中 `representation=SITReg` | 历史原始 λ=1 best-Dice checkpoint 的导出场；使用 `source_path/source_sha256`，不是新 sweep 的最优测试点 |
| MR-BSpline-Unbounded | 同一 report 的 `candidates`，`spec.representation=842,resolution=pyramid,alpha=1` | 单个经过 knot prolongation 的最终 DVF；`dvf_path/dvf_sha256`，不是 dense refinement 或各层场之和 |
| MR-BSpline-Bounded-2Comp | `oasis_seed3_first10_bspline_bounded2_842_ncc9_lambda_sweep_v1/results.json`，`candidates` 中 `alpha=1` | 每层两个独立 bounded increment，三层共六个 retained maps；使用最终组合 `dvf_path/dvf_sha256` |
| MR-BSpline-SVF | `oasis_seed3_first10_bspline_svf_refine_842_8421_ncc9_lambda_sweep_v1/continuation_v2/results.json`，`candidates` 中 `spec.schedule=842,regularize=velocity,alpha=1` | prolong 一个 total SVF；最终 native-grid 七次 squaring 导出场；不是各层指数的复合，也不是 displacement penalty |

四类各 10 个文件存在且 shape 与既有约定一致。这里只确认文件/metadata 可读，
不能把这些检查当成完成的 40-target 数值坐标审计。
来源 λ=1 的 regularization 意义不同；这里固定一个可追溯 operating point，
不宣称四方法正则强度、原配准算力或拓扑相同。

保留上述确切历史来源；不在 P43/P70/P76/P79 的同名独立结果之间静默替换。
接受标志、report hash、输入 hash、最终 field hash 及来源代码 revision 均进入未来 manifest。
若某条失效，保留缺失/失败记录，不用别的 rank 或 λ 补齐。

## 4. 环境与坐标适配

数据定位：当前容器的 `OASIS_ROOT` / `INRORSPLINE_OASIS_ROOT`，文件规则
`test/img<ID>.nii.gz`、`test/seg<ID>.nii.gz`。实际主机路径与 UID/GID 只保留在
ignored `private/data_sources.md`；不要复制旧 launcher 中的路径到新公开配置。

已运行容器：`py310sitreg_v1_artifact_store`。使用其 configured user；当前环境与
AGENTS 中历史主机的数字用户不同，不能照抄旧 `--user`。repo bind 只读，raw/shared
bind 只读，ARTIFACT_ROOT bind 可写。主机负责源文件编辑；容器负责数据 Python。
已验证 import：Python 环境中的 PyTorch 2.6.0+cu118、NumPy 2.2.3、SciPy 1.15.2、
NiBabel 5.3.2。实际 launch 再记录 Python、image digest、GPU 和 backend。

不重新创建容器，不要求旧训练仓库可导入才开始场拟合。只复用经核对的纯数值工具；
原生 SVF checkpoint 重放若依赖缺失代码，单列为诊断依赖，不运行历史训练入口。

### 4.1 三套轴约定必须分开

- 原始标量图像代表例：shape `(160,192,224)`，LIA，1 mm isotropic。
- `common.reorient_lia_to_ras`：transpose `[0,2,1]`，再 flip axes 0、2；
  得到既有 scalar array `(160,224,192)`。必须同步计算新的 index-to-world affine。
- PyTorch 将该数组视为 D,H,W；grid vector 的 x,y,z 指 W,H,D，不能直接视为 world R,A,S。
- SITReg NIfTI：`(3,160,224,192)`，既有 loader 约定是 zyx voxel displacement，
  已在 repo array orientation；不再次按 raw scalar 图像重排。
- 三个 spline 输出：`(160,224,192,3)`，normalized-grid xyz displacement，
  `align_corners=True`。各通道先乘 `(W-1,H-1,D-1)/2=(95.5,111.5,79.5)`，
  再变换到 array-index 顺序与物理坐标。

SITReg 的 channel-first NIfTI header 不能未经验证当作三维矢量场物理 affine。
从对应图像的重定向几何和保存代码共同构建映射。不要仅因 1 mm spacing 就跳过轴/符号审计。
PyTorch grid 的 axis/normalization/boundary 语义参照官方文档，实际以安装版本验证。
[grid_sample](https://docs.pytorch.org/docs/2.6/generated/torch.nn.functional.grid_sample.html)

Canonical 输出为 fixed image 预处理空间中定义的 physical pullback map：

\[
x=H_F\tilde i,\quad \phi^\star(x)=H_M\widetilde{j(i)},\quad
u^\star(x)=\phi^\star(x)-x,\quad I_W(x)=I_M(\phi^\star(x)).
\]

这里是预处理 OASIS 空间，不擅自解释成原扫描空间。每例核对 H_F/H_M，不能只用 rank0
的 affine 代替所有 cases。组合固定为右边先执行：
\(\phi_A\circ\phi_B=x+u_B(x)+u_A(x+u_B(x))\)。bounded 来源按真实六段顺序解释，
SVF 来源按一个最终 velocity 解释。主目标使用已保存最终 dense map，不重算替换其字节。

### 4.2 P0 数值验收仍需完成

建立不可变的外部 manifest：原计划 §4.2 字段全部保留，并加 report selector、
接受记录 hash、source regularization、reference import status、拟合 renderer 与来源 renderer。
每条记录含实际路径和 hash；公开文档只写 symbolic root。

对所有 40 targets 做 hash/finite/ID/shape/geometry 检查和 coordinate round trip；
每来源至少 rank0 重现原 warp/label metric，再核对其余例的转换诊断。
从原 utility 明确 image、label、field、integration 各自 padding，不能把它们混成一个默认值。
旧 `common.warp_full` 未指定 padding（默认 zero），SVF composition 使用 border：
两者用途不同，必须分开冻结。主新图像保真评价建议 image=trilinear/zero、
label=nearest/zero；来源重现使用原各自路径，差异显式记录。

参考 target 不先平滑、裁剪或重新缩放。出 moving FOV 的映射仍参与 field error，
图像指标另用固定 reference validity 域并报告覆盖率与预测越界比例。

## 5. 表示、采样和求解

核心七族：uniform cubic B-spline translation、uniform normalized-Gaussian translation、
uniform Gaussian local affine、tree T-U/T-A/A-U/A-A。
同一权重族内比较 uniform/adaptive；Gaussian 与 tree 分开解释。

所有族具备 global affine 再现能力。Gaussian translation 加 global affine 并消除三个
常数冗余；translation tree 使用 affine root。B-spline 必须验证含边界的 affine reproduction，
若补 global term，修正实际参数计数。不要把 representation 名称当成 reproduction 证明。

| 模型 | 运动参数数 |
| --- | --- |
| global affine root | 12 |
| B-spline，K 个系数点 | 3K（无重复 global 项时） |
| Gaussian translation + affine root | 3K+9（明确消除常数冗余后） |
| Gaussian affine | 12K |
| affine-root translation tree，K leaves | 3K+9 |
| affine tree，K leaves | 12K |

主预算 caps `[768,1536,3072]`；affine tree 为 `[64,128,256]` leaves，
translation tree 最大 `[253,509,1021]` leaves。Gaussian/spline lattice 按物理 FOV
aspect ratio 枚举，记录含 padding 的实际系数数，不用扁格子强行凑数。
actual P 相差≤5% 才称 approximately matched；same-K 和 same-P 分开画。
另记录 rank/condition、geometry/结构元数据和 serialized bytes。
这些预算相对原始百万级 spline 参数是很强的压缩，不预设能保持原注册精度。

显式保存 translation tree 的 K=64/128/256 checkpoints（P=201/393/777），
用于与 affine tree 的 same-K 对照；它们沿通往最大预算的同一路径产生。
每个 target 另求一个12参数 global-affine anchor（共40个），用于总场/残余场评价。
Gaussian same-K 对照尽量复用已有 lattice；需要额外 lattice 时在 pilot 中列出并计入
补充配置，不把不同中心布局默认为同结构。这些 anchors、额外 checkpoints/refits
不包含在840个主 endpoints 的计数内，但其时间、存储和覆盖必须计入总成本。

Gaussian 默认 sigma/h=0.85；仅 pilot 可比较 0.55/0.85/1.25。
tree 使用 logistic gate，tau=0.12×parent 对应物理边长；几何不梯度学习。
uniform 按最大 box volume 选 leaf、最长物理轴中点二分；adaptive 按
\(E_i=\sum_{S_{fit}}w_i(x)\|\hat\phi-\phi^\star\|^2\) 选 leaf，仍用最长轴中点。
ties 固定 node ID；每批增长≤25%，至少允许一次 split，预算处精确停止。
最大深度/最小物理边长在 pilot 冻结，触及限制时记录实际容量而不伪称达到 cap。

tree 首选原计划 §5.5 的 hierarchical contrast 形式；新 contrast=0 时旧函数保持。
若用 leaf form，必须按 §5.4 转换 center/scale，不能简单复制 centered coefficients。
每次 split 检查 analytic/dense/renderer 变化，优化前保存证据；solver state 默认重建。
固定最终结构做 warm/cold 对照；保持函数并不保证更快。

primary ROI=`fixed_raw_image > 0`，在 minmax normalization 之前生成并一起重定向。
secondary 为固定 10 mm ring 和 full valid FOV；derivative ROI 用统一 stencil erosion。
fit/validation/quick-eval 分别最多 65536/16384/65536 个互不重叠的空间分层 voxel centers，
sampling seed=0（区别于 pair shuffle seed=3），每 pair 四来源共用 index/hash。
小 ROI 按 4:1:4 分配并保持各自上限。最终评价完整 native ROI，分块处理。
非有限参考值默认隔离 target，不能各来源偷偷换 mask。空间 validation 不是新 subject 验证集。

损失为 physical endpoint MSE（向量平方和的点均值，mm²），无 image similarity、
BE、Jacobian penalty、ridge 或 coefficient L2。D0 固定几何使用 matrix-free LSQR/LSMR，
forward/adjoint 与 evaluate 同一函数；小型 float64 SVD/QR 核对，列缩放/预条件可用。
初始 atol/btol=1e-6，小型 rank cutoff=1e-10；记录残差、normal residual、iteration、
condition 和终止原因，延长一次检查稳定性，未收敛不标“容量最优”。
[SciPy LSQR](https://docs.scipy.org/doc/scipy-1.15.2/reference/generated/scipy.sparse.linalg.lsqr.html)

同固定几何比较 normalized Adam、Adam+L-BFGS 与 LS，training gap 与 native-ROI gap 分开。
pilot 非线性起点：每完整路径总 Adam≤1000 steps、L-BFGS≤100 objective evaluations，
每25步记录；最终 native/固定结构预留≥300步额度，至少150步后可按连续5次相对改善<1e-4
早停。结构变化不能重置总预算；D0 LS iterations 单列，不强行等同 Adam steps。
冻结时给出真实时间 checkpoints 和收敛规则；不从只有终点的 LS 构造虚假 anytime 曲线。

## 6. 分阶段矩阵（替代原计划 §9）

| 阶段 | 范围 | 规模/完成条件 |
| --- | --- | --- |
| P0 inventory/coordinate/regime audit | 十对×四来源 | 40 targets；hash、finite、坐标、warp gate 后才准拟合 |
| P1 D0 数学与资源 pilot | ranks0–1×四来源，七族，中预算起步 | 56 个主要 pilot endpoints；另做最大预算 native operator/render profile，不冒充完整主表 |
| P2 D0 主实验 | 40 targets×七族×三预算 | 840 endpoint configurations；树路径 checkpoint 可复用，实际 refit 次数/成本另外记 |
| P3 小型 D1/V 后续 | ranks0–1×四来源；BS-U、A-U、冻结 D0 结构的 A-A；P=1536 cap | 每分支24个 endpoints，共48个新 endpoints，复用24个 D0 endpoints |
| P4 机制诊断 | 同八个 pilot targets；按以下规则限额 | warm/cold、optimizer gap、双向结构 replay、积分和 certificate 检查；逐项记覆盖 |

P2 按中/低/高预算分批实现完整覆盖，先交付 D0 主问题证据。P3 是预先指定的后续工作，
不因 D0 表现好就删掉，也不阻塞 D0 阶段报告。P3 中固定 D0 adaptive 结构有选择偏向，
只能解释为条件于该结构的 renderer 比较；不能声称是中立结构或 SVF 自适应能力结论。

P4 在八个 pilot targets 上为 A-A/中预算运行 V 自己的同规则 adaptive 路径，
再让 D0/D1/V 拟合 V 选出的冻结结构，与 D0 选出结构配对；这是额外24 endpoints，
不能算进上面的48，也不能漏计结构搜索时间。D0 warm/cold 和 optimizer-gap 同样仅用
pilot/中预算固定结构，不扩展为所有 target 的全因子搜索。
V 自选结构的最终 V 结果若配置/hash 一致可复用；24是覆盖的 endpoint 数，
不是额外 solve 次数。所有结构搜索、中间 refit、双初值和重拟合分别记录成本。

P1 冻结实际 lattice、solver、容限、时间 checkpoints、chunk/cache、RAM/GPU 峰值、
存储/时间预估和失败策略。P3 另冻结 D1/V 数值设置；始终保留 P1 的 D0 配置，
若必须改动则新 protocol revision 并清楚区分结果，不能混合复用。
配置来源一致且 hash 完全匹配才允许将 pilot endpoints 计入主矩阵。

不是默认范围：全部40-target SVF三预算、D1全预算、6144/12288扩容、混合T/A升级、
多段 D1 或密集 SVF。高预算仍欠拟合时先报告曲线与预计成本，再提出对各族一致的扩容。
若 P3 的结论需要跨十对证据，提出全40-target中预算扩展，不从八个 pilot fields 外推。

## 7. D1 和 SVF 的边界

D1：\(Id+u\)，rho=0.9。主 certificate 是完整凸 box 内所有 trilinear cells 的
八个 cell-sided **位移 Jacobian Du** 的 induced infinity norm 上界，
不是变换 Jacobian \(D\phi=I+Du\)、中心差分或最大元素值；identity 的该界必须为0。
使用 physical Du（包含 index-to-world 逆矩阵），分块 float64 和明确安全余量。
原计划 §7 的公式及 cell 内凸组合论证保留，但必须由实现验证，未验证只能叫 empirical。
有限 box 的 bi-Lipschitz-to-image 保证不等于映回同一 box；trilinear map 是分片光滑，
不能冒称全局 C1 diffeomorphism。inverse 还需域外延拓/迭代域条件。

训练用 \(b+\alpha(a-b)\)，\(\alpha=\min(1,\rho/(B+\epsilon))\)，保留常量平移。
loss 在约束后计算；post-hoc scaling 单列，不能当作 constrained optimization。
analytic tree bound 为辅助，记录其保守程度。固定几何原约束问题是凸二次目标+凸约束；
若 radial training 的劣势影响结论，用一个小型 QP reference 分离求解器和可行域限制。
SITReg 的 B-spline cap 不可直接用于 Gaussian/tree。
[SITReg 论文](https://www.melba-journal.org/papers/2024%3A026.html)

V：同一 affine 向量场解释为 velocity，through-integrator endpoint optimization，
不是把 DVF 当 velocity 的 LS loss。最终 native grid，scaling-and-squaring；
pilot 核对7/8及必要时9步（来源目标仍为已保存的7步输出，不随新积分设置改变）。
使用 zero 与 D0-LS 两个确定性初始化，分别记算力；若差异明显保留双初值并报搜索时间。
P3 的24 endpoints 是结构/预算配置数，不是保证只做24次非线性 solve。
保持固定 grid/padding 的 affine-flow matrix-exponential 验证，区分边界偏差与积分误差。

native SVF 可从已记录 checkpoint 恢复时，先复现其7步最终场，再做 velocity LS 诊断；
checkpoint/依赖是否可读及可重放仍待验，不凭文件名认定已取得 velocity。
若非线性失败影响结论，只对预先指定 rank0 的 SVF 来源和最大异常 gap target
各做一个 dense-SVF 诊断（若为同一 target 则一次）；保留失败，不断言不存在 stationary flow。
单段 D1 明显受界限限制时，提出同总参数两段 D1 诊断；不是自动启用的新主模型。

## 8. 评价、资源与结论范围

- 场保真：mean EPE、vector RMSE、P95/max EPE（mm），relative RMSE；补充 best-global-affine
  residual-normalized error，分母近零标 undefined，不用大平移掩盖局部误差。
- 导数/几何：physical Jacobian-matrix RMSE、det error、min/P01/median/P99、folding fraction；
  正 determinant 共同区域 log-J error 并报覆盖率。区分原有 folding、预测新增 folding、
  原折叠被平滑消除；绝不使用 log(abs(J))。
- image/label：同一路径的 source-warp vs fit-warp MAE/NRMSE、label agreement、
  对 fixed labels 的 Dice/HD95/ASD 保留程度；labels 仅评价，不挑 checkpoint。
- native rasterized/reloaded map 是主评价对象；analytic off-grid 单列，不能与 V 的插值场混比。
- inverse residual 仅在条件和有效域明确时报告，双向 residual/覆盖率一起给；
  不声称测了 input-swap symmetry。D1 certificate 覆盖 full box，不用 foreground 掩盖失败。
- 参数、几何元数据、model/dense bytes，setup/solver/scoring/refit/bound/integration/render/eval
  分项时间和总时间、CPU/GPU peak memory；CUDA 同步，记录并发负载，不修改其他 jobs。

按 pair 等权、按来源分面：逐对 paired differences、mean/SD、median/IQR 和 win/tie/loss。
主要为描述性研究，不以 voxel 数增加样本量，不把四来源 pooled 成40个独立样本。
本轮不默认做显著性检验；若以后加入需预先定义少数对照及多重检验策略。

图：D0 error-vs-actual-P、same-K、tree2×2、error-vs-wall-time、optimization gap、
split/warm-cold、P3/P4 renderer fidelity/topology，以及固定 rank0 和规则化 worst-case
的切片/网格/tree 可视化。同行同 target 用相同切片/色标，失败也展示。
最终中文报告分别回答容量、适应性、求解器、几何和成本问题；P3/P4 仅八个 pilot targets
的覆盖限制必须写在图标题/表注和结论中。

## 9. 验证与可恢复执行

最小科学验证沿用原计划 §12：identity、大平移、小 affine、单 affine velocity 的矩阵指数、
非交换 composition、奇异 blending 反例、operator adjoint、导数 finite-difference/autograd、
split preservation、cell-bound dense probes、serialized round trip。
新增 raw/reoriented/world/grid 轴对照和共同 foreground/source-selector 检查。
小型验证通过后才做真实场 pilot，D1/V 验证在该后续阶段前完成。

配置/job manifest 保存 input/config/code hashes、环境、pair/source/budget/geometry、
初始化、求解器、约束域、状态、失败原因和路径。输入/配置/代码改变不复用旧结果。
rank/来源失败隔离，覆盖表完整保留，不替换 case。
LS 允许一次已记录的延长或预条件修正；SVF 先用指定双初值与积分诊断，不无限重启。
OOM 优先 chunk/recompute/matrix-free，不能只降低某分支最终分辨率。
多 chunk autograd 可能保留全部 graph，profile 必须实测 backward peak。

所有新 GPU 工作设置并记录 requested/effective `cudnn.benchmark=False`、
`cudnn.deterministic=True`，TF32 off；不声称 grid_sample backward bitwise deterministic。
先串行一个 worker，按 pilot 峰值和其他 jobs 负载拟定队列；历史八槽队列不是本任务默认资源授权。
阶段内 checkpoint 用独立 run ID，禁止覆盖 shared store 的旧结果或跨主机同时续跑同一 ID。

## 10. 文件与完成条件

repo：本目录的原件、修订计划、执行规范、技术日志；未来只增加 source configs、代码、
必要测试及 output-free notebook。运行目录为
`${ARTIFACT_ROOT}/runs/oasis_seed3_first10_adaptive_local_affine_fieldfit_v1/<unique-run-id>/`；
review 目录为同 key 的 `${ARTIFACT_ROOT}/repo_experiments/` 子树，包含 manifest、
coordinate audit、frozen config snapshot、metrics、figures、中文报告和 executed notebook。
不得复制原计划 §13 的 results/figures/generated reports 结构到 Git worktree。

本轮建档完成：原件保真、选择已确认、修订计划/执行规范/技术日志和 registry 相互一致。
未来阶段分别验收：P0坐标通过；P1配置/成本冻结；P2全部840 endpoints或逐项失败解释；
P3全部48新 endpoints；P4规定的结构replay/诊断覆盖。阶段部分完成就明确写 partial。
“D0 主研究完成”不等于整个含后续机制比较的研究完成；论文结论必须受实际覆盖约束。
