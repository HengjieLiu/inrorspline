# 公开发布实施进度

用户已授权开始实施。范围以 `README.md` 和 `release_spec.json` 的最终选择为准。
本记录留在私有仓库，与 `progress.json` 同步；每个里程碑记录产物、验证和限制。

当前发布状态（2026-09-30）：**SINR 作者许可待确认，正式公开发布暂缓**。
这是用户明确确认的状态，不是误改。历史技术验收不代表发布许可，后续收尾见
[PUBLICATION_READINESS.md](PUBLICATION_READINESS.md)。

追加任务已完成：提供三数据集完整实验重跑代码（不实际重跑），详见
[EXPERIMENT_REPRODUCTION.md](EXPERIMENT_REPRODUCTION.md)。下文 M1–M5 是首轮
已完成记录；新增 M6–M8 的独立验收也已完成，见文末和追加报告。

| 里程碑 | 内容 | 状态 |
| --- | --- | --- |
| M1 | 冻结图表数据、OASIS 测试对、COPD 来源和结果协议 | 已完成 |
| M2 | 可移植数据准备、显式 crop 版本、坐标与旧数据一致性验证 | 已完成（未重做 R231 推理） |
| M3 | 核心方法、三个固定版本 submodule/wrapper、配置与评估 | 已完成（未重跑优化） |
| M4 | 图表生成、Docker、README、poster 与运行文档 | 已完成 |
| M5 | 全新环境验收、公开内容审计和交付 | 已完成，待维护者审阅/提交 |

## 当前约束

- 不重跑论文优化或 sweep；使用现有冻结结果。
- 更新表使用 9 月 16 日五种方法的配对计时 N=10，历史 OASIS100 精度另列。
- COPD 是论文后扩展；Lung 加速和诊断实验代码暂不发布。
- 保留现有数据、结果、私有改动和公开仓库历史。
- 提交、推送和镜像发布不在本次实施中自动执行。
- 用户已选择 MIT，自有代码使用 MIT；第三方授权保持独立。

## 实施记录

### 开始实施

- 已检查两个工作区。私有实验代码存在用户改动，保持不动。
- 公开仓库已有三张图和未跟踪的 poster；保留这些资产。
- 已确认现有研究容器可用，后续数据检查在容器内进行。
- 当前优先制作小型、可核验、无私有路径的冻结图表输入。

### M1 已完成：冻结数值与协议

- 公开 `results/manifest.json` 记录 16 个输入/资产的校验值；保留原始 CSV 的来源与 SHA-256。
- Figure 2 包含 OASIS 41 行、4DCT 39 行；已在断网、无原始数据/GPU 的环境重绘。
- 重绘 PNG 与公开 `figures/fig2.png` 字节级相同：`eb0db83bbd1f96d34c3afc9c25576e56c7944c07cc04ce74ece6f615d23e3e16`。
- OASIS seed-3 前 10/100 对已导出；9 月 16 日全部 110 条计时记录逐条匹配 fixed/moving ID。
- 更新计时表、历史 OASIS100 精度、已发表 Table 1 分开保存；未混成一次新实验。
- COPD 选择 P0/P1、MR-D-BSCP P2e、pTVReg，共 40 条病例记录；pTVReg 参考使用 MATLAB crop，其他三行是 legacy crop。
- 新发现并显式注明：肺部 Dual-INR PL1A 的作者 blur 配置是 **3000 steps**，不将其错误改称 2500。

### M2–M4 实施中

- 已抽取公开算法内核、建立 15 份显式配置与三个上游内容校验 wrapper；目前只验证导入/构造，尚未完成数值验收。
- 已实现版本化 corrected/legacy 数据准备和原始 R231 mask 接口，拒绝覆盖已存在输出。
- 已修正私有 TRE 标准中“4DCT/COPD 都是 corrected”的错误描述；没有修改历史数据或实验代码。
- 首次 Docker build 通过，使用已缓存的历史镜像不可变 digest，新增依赖约数百 MB；未清理任何既有镜像。
- 公开注册命令默认仅展示配置，必须显式 `--execute` 才优化。本次不会执行该选项。
- `git submodule add` 自动暂存了 `.gitmodules` 与三个 gitlink；未提交或推送，最终交付前会说明 index 状态。

不能把导入/配置验证当成完整优化复现。剩余验收按里程碑继续记录。

### M2 已完成：数据与 crop 修复

- 对 4DCT/COPD 共 20 例，核验原始 landmark → crop → registration 坐标及逆变换；每相 300 点均在域内。
- 与历史 `preprocess_audit.json` 的 bounds、crop shape、registration shape 完全一致。
- 4DCT/COPD 各 case01/02 的 fixed/moving 共 8 组预处理图像与 mask 数组逐元素相同。
- COPD case01 的裁剪边界/109 与 110 层差异加入合成单元测试；新旧协议目录隔离且禁止覆盖。
- OASIS 实际 rank0 读入成功，形状 `(160,224,192)`；完整原始排序 200 对与公开 split 一致。
- 已提供 R231 本地权重接口并确认依赖/API；本次复用历史 mask 验证几何，没有重新做 R231 推理。

### M3 已完成：内核与 wrapper 合同

- 8 项迁移数值检查通过：OASIS/Lung 的 IDIR/SINR 初始化、输出、坐标梯度，以及 cps2/cps4 spline 解码、梯度、解析 BE 与私有版本精确一致。
- 三个固定版本的上游源码导入与内容 hash 验证通过；Dual 配置确认 P0/P1=2500，PL1A=3000。
- 公开单元测试 31 项通过（测试中禁止 Adam/AdamW 的 step），涵盖 knot insertion、TRE、预处理输出、配置和冻结表格。
- 单病例注册/独立评估入口已建立；没有通过真实优化验证完整训练轨迹，文档明确此限制。
- Docker 可见 RTX 6000 Ada，torch 2.6.0+cu118 / CUDA 11.8 / cuDNN 90100；仅做标量 CUDA 操作。

### M4 环境修正

- 基础镜像 `pip check` 暴露无关旧包问题：`itk` 元包 WHEEL tag 非法、`PyGObject` 缺少 pycairo。
- 新 Dockerfile 仅在派生镜像移除两个未使用包，随后执行依赖检查；不更改原镜像或研究容器。

### M4 已完成：Docker 与公开文档

- Docker 重新构建成功，`pip check` 通过；另移除依赖 itk 的未使用 `icon_registration`，共三个派生镜像内旧包。
- 匿名 registry HEAD 请求返回 200，确认基础镜像 digest 不依赖作者登录凭据。
- README 显示 Figure 1/2、poster；补全 MIT、第三方说明、引用、数据/方法/运行/验证文档。
- 新增接口分支验收，48 项测试通过；15 个注册配置只用合成引擎输出验证调度和序列化，没有执行优化。
- Python 文件已格式化并通过未定义名称检查。
- 最终验收采用显式 allowlist 的全新源文件快照，不含 `.git`、私有挂载或研究代码；这不是声称已经提交后的 Git clone 验收。

### M5 首轮干净快照验收及修正

- 独立镜像内冻结图表生成通过，Figure 2 PNG 再次字节级相同。
- 首轮独立镜像测试 46 通过、2 失败；上游校验发现 `.dockerignore` 的 `third_party/**/data` 误排除了 SINR 的 `conf/data` 配置，而非算法问题。
- 已移除这条过宽规则及同类 `results` 排除规则；数据、权重和缓存仍有根目录/类型级排除，公开快照仍使用源码 allowlist。
- 将重建修正后的全新快照并重新执行全部验收，不将首次失败计为完成。

### M5 已完成：最终验收与交付

- 修正后从第二份干净快照构建成功。镜像 `rightpriordir:release-local`，ID `sha256:026b2b13bc0a68b263bc632b9c0f5d87fbf61d178099de4bdf9eb37c996c4381`；约 14.4 GB，未发布。
- 镜像内断网运行，无私有代码/数据挂载：48 项测试全部通过（6.24 s）；三个上游校验、15 份配置、16 项冻结输入、标量 CUDA 检查通过。
- 151 个清单文件及 151 个已安装包版本全部匹配；加上清单自身共 152 个发布文件、约 13.6 MB；镜像源码内无 Git 元数据。
- 冻结 Figure 2 PNG 再次与原图字节级相同；四份表格生成成功。PDF 生成时间元数据可能改变，不声称其字节级可重复。
- 用最终镜像和只读研究/数据挂载再次核验：8 项内核数值检查、20 例坐标几何、8 组图像/mask、一对实际 OASIS 数据及完整 200 对顺序均通过。
- 公开路径/链接扫描、Git diff 格式检查通过；三个第三方工作树无改动。公开历史 HEAD 仍为 `9185bab`，没有新提交。
- 生成的验收证据保存在 `${ARTIFACT_ROOT}/repo_experiments/2026-09-29_public_release_acceptance_v1`，包括 JUnit、迁移检查 JSON、图表、渲染清单和发布文件清单。摘要及 SHA-256 写入 `progress.json`。
- `.gitmodules` 和三个 gitlink 是添加 submodule 时自动暂存的；其他公开改动仍未暂存/未跟踪。未 commit、push、tag 或发布镜像。私有原有实验改动保持不动。

## 交付范围与明确保留事项

第一版实现和 M1–M5 验收完成，不代表重新运行了论文实验。

- 已提供：MIT、自有方法与三个 submodule/wrapper、15 份配置、独立评估、版本化肺部预处理、冻结图表、Docker、README 两图及 poster、详细运行文档。
- 未执行：注册优化、R231 mask 推理、corrected COPD 新结果、Lung 加速补跑、外部参考方法环境和诊断实验复现。
- 所有旧输入、历史 COPD legacy 结果和发表论文数值保持原样；pTVReg 的不同 crop 与 PL1A 的 3000 steps 已公开说明。
- 下一步是维护者审阅；提交与发布需另行授权。版本文件发生修改后应重新生成公开清单并重复快照验收。

私有工具脚本用途：`export_frozen_inputs.py` 是冻结结果整理器；`export_source.py`
是一次性源抽取器（**不要直接重跑覆盖已手工整理的公开源码**）；
`verify_migration.py` 执行不含优化的迁移检查；`seal_snapshot.py` 生成带校验值的
允许清单快照。生成快照须选用不存在的新输出目录。

## M6–M8 已完成：两层复现的追加实现

- 审计结论：首版单病例入口不足以支持完整实验重跑，现补齐三数据集实验矩阵、
  批量优化/评估入口、校验后恢复、失败尝试保留、分片和新结果汇总/绘图。
- 提供 OASIS paper 3600、4DCT paper 370、COPD extension 30 个病例×参数组合；
  另有 OASIS base/fast accuracy 1000 次，以及独立 N=10 cold/warm timing 110 次。
- 修正 Dual 历史 seed 为 42；明确 PL1A 3000 steps 和 sample SD 例外。
- Docker 干净快照中 74 项测试通过（8.41 s），16 项冻结输入和 162 个文件校验
  通过；Figure 2 PNG 及四个冻结表格与首版相同。无真实 optimizer step。
- 当前公开说明为 `docs/experiments.md`，README 已明确两层入口。
- 人读审计报告：`EXPERIMENT_REPRODUCTION.md`；机器报告：
  `experiment_reproduction.json`；主 `progress.json` / `release_spec.json` 已同步。
- 仍未执行任何真实实验、R231、commit、push 或镜像发布。外部参考环境、
  2400-step legacy 上下文、诊断实验和 Lung 加速继续不在重跑范围。

## 文档补充：Acknowledgement（2026-09-29）

- 按用户要求，在公开 README 增加致谢：ARC 论文用于 alignment–regularity
  评估思路的启发；IDIR、SINR、Dual-INR 用于开源代码贡献，不照搬示例中的其他项目。
- 已核对 ARC 论文标题与 arXiv 链接：<https://arxiv.org/abs/2503.07185>；
  三个代码链接与已有 submodule 来源一致，第三方条款说明保持不变。
- 此次仅更新 README 与发布清单、私有记录；不修改代码、配置或冻结结果。
  M8 的 74 项测试与镜像验收记录仍指向原快照，本次不重跑测试或重建镜像。
- 发布清单重新生成，162 个文件的 hash 与大小核验通过；与 M8 快照相比，
  仅 README 内容变化。相对链接、私有路径扫描和 diff 格式检查通过。
  当前清单 SHA-256：`7c58cb9f4cf98c4aba5b600bd24de7bc149ea7bc64a2c34d33fc10cf57168a69`。

## M9–M10 已完成：2026-09-30 发布收尾，许可仍待确认

- 用户明确确认 SINR permission pending；公开 README/第三方说明与所有活动私有记录
  已同步，不再把历史“已获许可”当作当前状态。正式发布仍暂缓。
- OASIS 下载和解压命令补齐，增加 `prepare-oasis`、84 行显式 subject 映射及源文件
  校验值。0333/0115 是顺序 ID，分别映射原始 0370/0123；168 个文件与历史输入完全相同。
- 85 项禁止优化的测试通过；真实 archive 的只读 hash preflight 通过，不写新医学数据。
- 用一次性候选 Git 提交做全新递归 clone，再从 clone 构建 Docker；最终断网环境
  85 项测试（10.932 s）、166 个文件 hash、151 个依赖版本和所有配置/suite 检查通过。
- Git 打包暴露 CRLF 检查问题，新增 `.gitattributes` 保留精确字节；没有改动冻结
  CSV。Figure 2 PNG 与四张冻结表格的校验值不变。
- 公开 91 个变更已完整暂存，95 个 index 条目与通过验收的候选 tree 相同；私有规划
  未进入公开 index。正式 HEAD 仍是 `9185bab`，没有正式 commit/push/tag 或镜像发布。
- 新镜像：`rightpriordir:prepublish-20260930`；当前清单 SHA-256：
  `09c1e2ddb51e20771ab6de438cfdb056c5c5be70faf39d92c71c9a4cffc4b27e`。
- 详细人读/机器报告：`PUBLICATION_READINESS.md` / `publication_readiness.json`；
  外部证据：`${ARTIFACT_ROOT}/repo_experiments/2026-09-30_public_release_prepublication_v1`。
- 剩余条件：SINR 许可确认（或另行批准不同范围），更新状态/清单后，用户明确授权
  正式提交和推送。技术验收不替代许可，未补跑论文实验或 R231 推理。
