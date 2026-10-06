# 第二层：从数据重跑实验

发布许可补充（2026-09-30）：SINR 作者许可待确认，完整候选包及镜像的公开发布
暂缓。下文是技术实现/测试记录，不代表许可已解决。

状态：代码、说明与隔离验收已完成。用户明确要求提供 OASIS、DIR-LAB 4DCT、COPD 的重跑代码和说明，
**本次不执行真实优化**。这是对首轮 M1–M5 范围的扩展，不把原来的单病例
入口或冻结图表验收等同于完整实验复现。

## 审计结论与实施合同

- 层 1：冻结 CSV → 论文 Figure 2/表格，保持不变；原始结果不能被新运行覆盖。
- 层 2：数据准备 → 明确病例/方法/权重矩阵 → 优化与逐例评估 → 完整性核验 →
  汇总 CSV/表格/新结果图。默认仅预览，显式 `--execute` 才启动优化。
- OASIS：100 对、6 种方法的论文曲线网格；另提供 base/fast 质量重跑和
  N=10 配对 cold/warm 计时，不能用普通单病例 wall time 冒充计时表端点。
- 4DCT：10 例，corrected crop；论文内发布方法的完整权重网格，Dual PL1A
  保留 3000-step 例外。2400-step legacy 上下文行与外部 pTVReg 仍仅冻结展示。
- COPD：10 例，legacy crop；P0/P1 与 MR-D-BSCP P2e 固定设置，论文后扩展。
- 外部 DL/Greedy/pTVReg、诊断实验、Lung 加速继续不纳入重跑环境。
- 恢复只能跳过校验通过的完整结果；失败重试保留旧尝试，数据/源码/配置漂移
  必须拒绝混入同一次实验。部分病例汇总必须明确标记，不能冒称 N=100/N=10。
- 新发现：原公开 Dual 配置 seed=1，而研究 controlled/PL1A 入口默认 seed=42。
  本轮将核对并修正公开历史复现设置，同时记录变更，不改动冻结数值。

## 里程碑

| ID | 内容 | 状态 |
| --- | --- | --- |
| M6 | 实验网格、协议、范围与来源审计 | 已完成 |
| M7 | 批量重跑、恢复、配对计时、汇总和新结果绘图 | 已完成 |
| M8 | 三数据集说明、离线/模拟测试、Docker 快照验收与报告同步 | 已完成 |

验收允许解析全部实验计划、合成/模拟运行、校验聚合数学和失败恢复；禁止
Adam/AdamW optimizer step，不运行真实训练、不补跑缺失论文结果。

## 已实现与首轮验证

- `experiment.py`：五套明确矩阵；输出源码、依赖与数据 hash；默认预览，显式
  执行；逐例子进程、互斥锁、分片、验证后恢复、旧失败尝试保留。
- `timing.py`：每候选独立进程，在进程内 cold rank0 + warm rank0–9；DVF-only，
  禁止在观测间插入精度评估；单 GPU 顺序执行，记录 requested/effective backend。
- `summarize.py`：严格覆盖/校验，三数据集 CSV、表格、新 ARC 图及冻结均值差异；
  部分结果显式标记，不修改旧论文数值。
- 精确矩阵：OASIS paper 3600，4DCT paper 370，COPD 30，OASIS accuracy 1000，
  OASIS timing 10 个持久候选进程、110 次观测。
- Dual seed=42 已由原 COPD `resolved_job_manifest.json` 和 PL1A 入口核验；
  原公开 seed=1 修正。另核实 Dual PL1A 原汇总使用 sample SD，4DCT 自有方法
  使用 population SD；新汇总为每行保留相应 ddof，而不是一刀切。
- 首轮 69 项离线/模拟测试通过（6.07 s），没有执行真实 optimizer step。
- README、数据准备连接、三数据集完整命令及计时/恢复/汇总说明已加入公开
  `docs/experiments.md`。

## M8 最终验收

- 从不含 Git 元数据/私有数据的 allowlist 快照构建 Docker 成功，依赖不变；
  镜像 ID `sha256:4eab64551bfc11d0007a392d83147d8737a32652fc683798369b3e31507c8e0c`。
- 断网、无私有源码/数据挂载：**74 项测试全部通过，8.41 s**；包括五套计划、
  精确网格、三数据集模拟批量运行、断点恢复、保留失败重试、输入变更拒绝、
  完成清单完整性、部分汇总、绘图、计时先后顺序及不同 GPU 身份拒绝。
- 一个测试曾因过宽 subprocess mock 拦截字体探测而失败；已缩小 mock 边界，
  最终测试在新容器字体缓存环境下通过。这不是实际实验失败。
- `pip check`、162 个清单文件及 151 个包版本通过；加清单自身共 163 文件，
  约 13.7 MB。Markdown 本地链接和私有路径扫描通过，第三方工作树无改动。
- 16 个冻结输入校验通过；Figure 2 PNG 字节级一致，四个冻结表格 SHA-256
  与首版一致。只更新了计时协议说明/校验值，没有改动冻结测量数值。
- 验收产物保存于 `${ARTIFACT_ROOT}/repo_experiments/2026-09-29_public_release_layer2_acceptance_v1`；
  清单、镜像及报告 SHA-256 见 `experiment_reproduction.json`。
- 未执行真实注册、R231 推理或 Lung 加速；没有 commit、push、tag 或镜像发布。
  新代码提供完整运行路径，但不宣称未经执行的训练轨迹或计时已等同历史结果。

交付入口：公开 `README.md`、`docs/experiments.md`、五份 `experiments/*.json`；
实际重跑必须由用户显式加 `--execute`。提交/发布仍需另行授权。
