#!/usr/bin/env python
"""Coverage-first external P81 reports; failed/missing cells never disappear."""
from __future__ import annotations
import argparse
from collections import defaultdict,Counter
import csv
from pathlib import Path
import statistics
from runtime import read_json,atomic_json,now,KEY,assert_external


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    root=assert_external(a.root);manifest=read_json(root/'launch_manifest.json')
    rows=[];failures=[]
    state=read_json(root/'queue_state.json') if (root/'queue_state.json').exists() else {'jobs':[],'status':'not_started'}
    review=root/'reports';review.mkdir(exist_ok=True)
    for job in manifest['jobs']:
        for cap in job['budgets']:
            path=root/'endpoints'/job['id']/f'cap{cap}.json'
            row={k:job[k] for k in ('id','target_id','mode','family','initialization','kind')}
            row.update(budget_cap=cap,status='missing')
            if path.exists():
                try:
                    result=read_json(path);fg=result['metrics']['field_error']['foreground']
                    row.update(status=result['status'],scientific_status=result['scientific_status'],
                               actual_motion_params=result['actual_motion_params'],
                               source_method=result['source_method'],pair_rank=result['pair_rank'],
                               fit_time_sec=result['elapsed_fit_and_structure_sec'],
                               **{k:v for k,v in fg.items() if isinstance(v,(int,float))})
                except Exception as exc:row.update(status='report_error',error=str(exc))
            rows.append(row)
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (review/'per_endpoint.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    groups=defaultdict(list)
    for row in rows:
        if row['status']=='complete' and row['kind']=='main':
            groups[tuple(row.get(k) for k in ('source_method','mode','family','budget_cap','initialization'))].append(row)
    summary=[]
    for group,vals in groups.items():
        item=dict(zip(('source_method','mode','family','budget_cap','initialization'),group))
        item.update(n=len(vals),complete_cohort=len(vals)==10,
                    scientific_status_counts=dict(Counter(v.get('scientific_status','unknown') for v in vals)),
                    all_solvers_resolved=all(v.get('scientific_status')=='evaluated' for v in vals),metrics={})
        for metric in ('rmse_mm','mean_epe_mm','p95_epe_mm','fit_time_sec'):
            nums=[v[metric] for v in vals if isinstance(v.get(metric),(float,int))]
            if nums:item['metrics'][metric]={'mean':statistics.mean(nums),'sd':statistics.stdev(nums) if len(nums)>1 else None}
        summary.append(item)
    outcome={'at':now(),'queue_status':state['status'],'jobs':dict(Counter(j['status'] for j in state['jobs'])),
             'endpoints':dict(Counter(r['status'] for r in rows)),'planned':manifest['coverage'],
             'summary':summary,'complete_endpoint_matrix':all(r['status']=='complete' and r.get('scientific_status')=='evaluated' for r in rows),
             'study_complete':False,'analysis_status':'coverage_report_only; mechanism diagnostics and final scientific review separate'}
    atomic_json(review/'summary.json',outcome)
    lines=['# P81 实验覆盖与结果','',f'更新：{now()}','',f"队列状态：{state['status']}",
           f"Job 状态：{outcome['jobs']}",f"Endpoint 状态：{outcome['endpoints']}",'',
           'D0与SVF各840个主配置；SVF保留zero/global-affine-LS两个初值。每个来源独立汇总。',
           '未完成/失败项保留在per_endpoint.csv；summary.json对不足十例的组明确标记不完整。',
           '均值描述已序列化的结果；未收敛D0仍计入描述统计并单独计数，不能当作已求得的LS最优值。',
           'SVF的LS初始化、结构搜索及双初值总成本需计入完整流程比较；并行时间不是独占硬件速度。','',
           '当前报告是队列覆盖报告。未完整覆盖、未收敛或尚缺诊断时，不宣称科学研究全部完成。']
    (review/'report_zh.md').write_text('\n'.join(lines)+'\n')
    print(outcome['jobs'],outcome['endpoints'],flush=True)


if __name__=='__main__':main()
