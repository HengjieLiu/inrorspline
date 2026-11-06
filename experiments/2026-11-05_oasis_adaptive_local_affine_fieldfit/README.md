# OASIS 自适应局部 affine 真实场拟合

最新状态：2026-11-05 08:50:14 PST队列全部结束，1680/1680 jobs完成，
2640/2640 endpoint records齐全，无运行失败、无OOM重试。总耗时约7小时32分。
首次结果复核已写入外部`execution_20261105_v3/reports/completed_review_v1/`，
复核源为[review_completed_run.py](review_completed_run.py)。科学分析尚未全部完成：
575/840个D0主结果未满足LS收敛标准，全部1680个SVF主结果最终阶段达到迭代预算。
因此当前误差表是已实现拟合效果，不是已证明的表示能力最优值。以下为历史启动/调度记录。

最新授权调度：[多GPU续跑规范](scheduler_v1.md)，GPU0/1最多各16个worker，
GPU2/3最多各4个；共享队列按各GPU可用内存分配，每60秒轮询，CUDA OOM自动回队。
已于02:26PST启用；94个测试及四GPU native烟测通过，93个完成job与八个在跑worker保留。
数值protocolv1.3保持不变；以下GPU0启动记录为此次调度续跑前的历史快照。

状态：v3于2026-11-05 01:18:25 PST（09:18:25Z）启动GPU0完整队列。
初次确认controller及8个worker存活，1672等待、0失败、无warning；source hashes匹配。
76个CPU测试、17个native资源检查及3个worker烟测（7个endpoints）通过。
独立多步review通过既有foreground积分准则；raw full-box敏感性失败及ROI外大误差保留。
v2已因conditioning失败停止并保留输出。队列运行不表示研究或全box精度验收完成。
日期：2026-11-05，America/Los_Angeles。登记号：P81。

- [原始计划 v1.0](original_plan_v1.0.md)：用户附件的逐字节副本，不作修订。
- [当前计划 v1.3](revised_plan_v1.3.md)及[配置](protocol_v1.3.json)：共享full-box列conditioning与真实多步验收；D0/SVF完整矩阵不变。
- [历史计划 v1.2](revised_plan_v1.2.md)及[配置](protocol_v1.2.json)：保留首次完整队列的协议与失败溯源，不复用为v3结果。
- [此前修订 v1.1](revised_plan_v1.1.md)：保留此前缩小SVF范围的审阅记录，已被v1.2覆盖。
- [执行规范](codex_execution_spec.md)：实现边界、预检顺序、未来代码位置和输出路径。
- [技术日志](technical_log.md)：审计、实现、v2启动/失败及v3修正证据与限制。
- [分析完成清单](analysis_completion_ledger.md)：完整拟合矩阵与仍待实现的机制分析、最终统计/图表/notebook分别列出。

用户已确认：沿用 seed-3 shuffled first10；四类参考来源采用 8→4→2、λ=1
（SITReg 为原始历史 λ=1 checkpoint 的输出）；最新要求D0/SVF各840完整配置，
D1保留较小机制比较。所有GPU工作仅GPU0，最多八个并发jobs，按实测内存调度。
v1.3的SVF七族与replay统一保留zero/global_affine_ls双初值；global-affine位移LS
直接作为初始velocity而非矩阵log。D0/D1 local LS不变；多步验收限定于已记录的foreground准则。
这里的 λ 是参考配准的设置，
新 field-fitting 主实验没有 deformation regularization。

“OASIS 场较平滑，局部 affine 可能有效”是待检验假说，不预设其成立。
180° 混合反例仅保留为数学/实现检查，不作为要求采用 SVF 的实验依据。

原件 SHA-256：
`8559292adc68b0fbd416e7476863b0f46f6154f9d546a84ba023e5e6c346706b`。
原件保留其原始换行和链接；其中的 ChatGPT Library 链接仅作出处档案，
不构成本地可用依赖，也不应直接进入匿名发布包。

本目录只保存手写计划、规范和日志。未来 manifest、转换场、拟合结果、图表、
自动报告及执行后 notebook 均存于 `${ARTIFACT_ROOT}`。
