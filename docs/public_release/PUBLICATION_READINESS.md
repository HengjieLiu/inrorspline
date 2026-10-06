# 公开发布收尾：2026-09-30

## 当前决定

- 用户确认 `Author permission pending` 是真实意图；早先“已获 SINR 许可”的
  记录已被此决定取代。保留三个 submodule，不擅自移除 SINR 或改变论文范围。
- 技术收尾继续；包含 SINR 的正式仓库/镜像发布暂缓，等待许可确认或另行批准范围。
- 不运行真实优化、不改历史数据、不推送、不改正式分支历史。
- 用一次性候选 Git 仓库/提交进行全新递归 clone 验收；这不是正式仓库的 release commit。

## 里程碑

| ID | 内容 | 状态 |
| --- | --- | --- |
| M9 | 许可状态同步、OASIS 精确映射与整理入口、忽略规则加固 | 已完成，85 项禁止优化的测试通过 |
| M10 | 清单更新、候选提交的递归 clone/Docker 验收、完整暂存范围 | 已完成，正式发布仍待许可 |

## OASIS 关键发现

历史 organizer 按 Neurite v1.0 `subjects.txt` 的一基顺序编号，不能根据目录名里的
原始 subject 数字映射。0333 → OASIS_OAS1_0370_MR1，0115 → OASIS_OAS1_0123_MR1。
84 位 test-pool subjects 的 168 份图像/标签压缩文件与历史 organized 输入 SHA-256
全部一致。之前同数字原始 subject 的检查不一致，正是这个编号差异，不是数据损坏。

公开 `splits/oasis_subject_map.csv` 固定映射与原文件校验值；`prepare-oasis` 只做
按映射复制，不重采样、不改方向、不改标签。默认仅预览，`--check-data` 校验源内容，
`--execute` 才复制；拒绝已有输出、错误 subject 顺序或内容，失败保留部分输出。
公开说明补齐发布方链接、3D archive MD5、下载解压、Docker mount 与完整命令。

来源核对：

- 发布方：<https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md>
- 历史 organizer：`wavereg/server_share/dataset_overview_and_split/brainMR_oasis_v1/step1_organize.ipynb`。
- 原始 `subjects.txt` SHA-256：`52a8a9cae5992837140464f8d28a69135b096484786f3cb85c50834958eee4ca`。

最终验收结果见下方；机器记录见 `publication_readiness.json`。

候选 Git 暂存验收发现冻结 CSV 保留 CRLF，默认 `git diff --check` 将回车标成
trailing whitespace。没有重写冻结输入；新增 `.gitattributes` 保留精确字节，
仅对 CSV 允许 CRLF，继续检查真实行尾空格。此问题说明文件快照验收不能替代
真实 Git 打包检查；修正后将使用新目录重新构造候选提交。

下载入口 HEAD 返回 200；只读取 tar 的首个 header 发现 archive 没有外层目录，
已修正公开解压命令，显式创建 `neurite-oasis.v1.0` 容器目录。未下载整个 archive。

## M10 最终验收与交付

- 临时候选提交 `b9fd9c0416021b0281de9d072476934c92148aa3` 经全新递归 clone，
  三个 upstream 从公开 HTTPS 地址拉取，固定版本及全部文件校验通过。
- 从该 clone 构建 `rightpriordir:prepublish-20260930`，image ID
  `sha256:99cf5c78616f21355dcc56e007750e8ea94fe2fdafa93b53de04a5a889dfafc8`。
  环境层复用已验收缓存；没有发布镜像，也没有清理旧镜像或研究容器。
- 无网络、无私有源码或数据挂载的最终镜像内 **85 tests passed (10.932 s)**；
  166 个发布清单文件、151 个安装包版本、16 个冻结输入、15 个配置及全部 suite 校验通过。
- Figure 2 PNG 与四个冻结表格保持原校验值。PDF 创建时间仍不作逐字节保证。
- 新 organizer 对真实 archive 的只读 `--check-data` 验证 168 个文件通过，没有写准备数据。
- 按用户要求修复不完整暂存：公开仓库 index 共 95 个条目，本次 91 个变更全部暂存；
  staged tree `af6603a9fc57c01a1f2466cf109a6a1e057f382c` 与验收候选提交完全一致。
  私有计划未进入公开仓库；无额外未暂存/未跟踪发布文件，三个 submodule 工作树保持干净。
- 正式 HEAD 仍为 `9185bab`；没有在正式仓库 commit/push/tag，也未运行真实优化/R231。
- 当前发布清单 SHA-256：`09c1e2ddb51e20771ab6de438cfdb056c5c5be70faf39d92c71c9a4cffc4b27e`。
- 验收证据：`${ARTIFACT_ROOT}/repo_experiments/2026-09-30_public_release_prepublication_v1`；
  含技术验收、候选 clone、完整暂存、路径/凭据模式扫描与冻结图表。

## 后续操作边界

技术收尾完成不等于允许公开。唯一未解决的外部发布条件是 SINR 许可；记录 pending
本身不能消除它。得到许可后需更新 pending/hold 文案、刷新清单、重新核对暂存和验收，
再由用户明确授权正式 commit/push。若希望改为不含 SINR 的版本，须另行确定范围。

当前仅建议后续 commit message：`Prepare reproducible RightPriorDIR release candidate`。
完整精确暂存路径见外部 `candidate_acceptance.json` 的 `explicit_staging_paths`；
无需 `git add .`，不要将此私有规划目录拷进公开仓库。
