"""Curate public numerical inputs; never load images or run an optimizer."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--artifact-root', type=Path, required=True)
    p.add_argument('--research-root', type=Path, required=True)
    p.add_argument('--release-root', type=Path, required=True)
    a = p.parse_args()
    out = a.release_root.resolve()
    if not (out / '.git').is_dir() or out == a.research_root.resolve():
        raise ValueError('Require the distinct public Git checkout')
    store = a.artifact_root / 'repo_experiments'
    old = store / 'migration_2026-09-02/experiments'
    records = []

    def export(source, target, select=None, keep=None):
        rows = list(csv.DictReader(source.open()))
        if select:
            rows = [r for r in rows if select(r)]
        if not rows:
            raise ValueError(f'Empty input {source.name}')
        keys = [k for k in rows[0] if keep is None or keep(k)]
        path = out / target
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            w.writerows({k:r[k] for k in keys} for r in rows)
        text = path.read_text()
        if any(x in text for x in ['/home/', '/mnt/', '/database/', '/data/hengjie']):
            raise ValueError(f'Private path in {target}')
        records.append({'path': target, 'sha256': digest(path), 'row_count':len(rows),
                        'source_id': source.relative_to(store).as_posix(),
                        'source_sha256':digest(source), 'columns':keys})
        return rows

    fig = old / 'paper_artifacts/pareto_figure/v013_final_dirlab4dct_axes_labels'
    oasis = export(fig/'oasis100_pareto_source_rows.csv', 'results/paper/oasis100_arc.csv',
        keep=lambda k: k in {'method_key','method_id','method_label','display_method','dataset','n_cases','alpha','lr'}
        or k.startswith(('log_jdet_std_fg_', 'jdet_le0_pct_fg_', 'dice_mean_', 'hd95_mean_')))
    assert len(oasis)==41 and all(int(r['n_cases'])==100 for r in oasis)
    assert all(float(r['log_jdet_std_fg_mean']) >= 0 for r in oasis)
    lung = export(fig/'source_rows.csv', 'results/paper/dirlab4dct_arc.csv',
        keep=lambda k: k not in {'source_path','source_paths','elapsed_sec_mean','elapsed_sec_std','peak_mem_gb_mean','peak_mem_gb_std'})
    assert len(lung)==39
    accuracy = export(old/'2026-08-13_oasis_selected_fast_first100/fast_vs_2500_baselines.csv',
        'results/paper/oasis100_accuracy.csv', keep=lambda k: k in {
            'candidate','method_label','baseline_schedule','cases','baseline_cases','fast_epochs_mean','baseline_epochs',
            'fast_dice','baseline_dice','fast_hd95','baseline_hd95','fast_asd','baseline_asd',
            'fast_first10_reused_count','fast_new_rank10_99_count'})
    assert len(accuracy)==5 and all(int(r['cases'])==100 for r in accuracy)
    timing = export(store/'2026-09-16_oasis_table1_base_fast_ft_first10_gpu3/timing_summary.csv',
        'results/updated_oasis/timing_first10.csv', keep=lambda k: k in {
            'candidate','method','method_label','config','cudnn_benchmark','cudnn_deterministic',
            'cold_count','warm_count','cold_endpoint_sec','warm_endpoint_mean_sec','warm_endpoint_sd_sec',
            'cold_actual_epochs','warm_actual_epochs_mean','cold_training_peak_mem_gb','warm_training_peak_mem_gb_mean'})
    assert len(timing)==10 and all(int(r['warm_count'])==10 and int(r['cold_count'])==1 for r in timing)
    copd_dir = old/'2026-08-03_dual_inr_copd_six_arm'
    allowed = {'P0','P1','MR-D-BSCP P2e','pTVReg official'}
    population = export(copd_dir/'population_summary.csv','results/copd_extension/population.csv',select=lambda r:r['method'] in allowed)
    cases = export(copd_dir/'casewise_tre_and_regularity.csv','results/copd_extension/cases.csv',select=lambda r:r['method'] in allowed)
    assert len(population)==4 and len(cases)==40
    for m in allowed:
        assert {int(r['case_id']) for r in cases if r['method']==m} == set(range(1,11))
    pairs_source = a.research_root/'configs/splits/oasis/pair_test_200_shuffled_seed000003.txt'
    pairs = re.findall(r'\((\d+),\s*(\d+)\)', pairs_source.read_text())
    assert len(pairs)==200
    for n in [10,100]:
        path=out/f'splits/oasis{n}.csv'; path.parent.mkdir(exist_ok=True)
        with path.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['rank','fixed_id','moving_id'])
            w.writerows((i,*pair) for i,pair in enumerate(pairs[:n]))
        records.append({'path':path.relative_to(out).as_posix(),'sha256':digest(path),'row_count':n,
            'source_id':pairs_source.relative_to(a.research_root).as_posix(),'source_sha256':digest(pairs_source)})
    # The original loader also validates membership in the sorted 200-pair list.
    for name,items in [('pair_test_200_shuffled_seed000003.txt',pairs),('pair_test_200.txt',sorted(pairs))]:
        path=out/'splits'/name
        path.write_text('# fixed ID, moving ID\n'+''.join(f'({x}, {y})\n' for x,y in items))
        records.append({'path':path.relative_to(out).as_posix(),'sha256':digest(path),'row_count':200,
            'source_id':pairs_source.relative_to(a.research_root).as_posix(),'source_sha256':digest(pairs_source)})
    timing_cases=export(store/'2026-09-16_oasis_table1_base_fast_ft_first10_gpu3/timing_case_rows.csv',
        'results/updated_oasis/timing_cases.csv',keep=lambda k:k in {
            'candidate','case_rank','fixed_id','moving_id','original_case_index','timing_condition',
            'cudnn_benchmark','cudnn_deterministic','loaded_images_to_dvf_file_sec','actual_epochs','status','done'})
    assert len(timing_cases)==110
    for row in timing_cases:
        rank=int(row['case_rank'])
        assert 0 <= rank < 10
        assert tuple(row[k].zfill(4) for k in ['fixed_id','moving_id']) == tuple(x.zfill(4) for x in pairs[rank])
    sorted_indices=[sorted(pairs).index(pair) for pair in pairs[:10]]
    assert sorted_indices==[158,62,14,69,47,137,110,6,116,88]
    for target in ['results/paper/table1_published.csv','results/updated_oasis/protocol.json',
                   'figures/fig1.png','figures/fig2.png','poster/offgrid_poster_v3.pdf']:
        path=out/target
        item={'path':target,'sha256':digest(path),'source_id':'author-provided static asset or manually documented protocol'}
        if target.endswith('table1_published.csv'):
            pdf=a.research_root/'docs/offgrid_submission/submitted_files/camera_ready_off-grid_30.pdf'
            item.update(source_id='camera_ready_off-grid_30.pdf, Table 1 (manual transcription of rounded entries)',source_sha256=digest(pdf),row_count=5)
        records.append(item)
    manifest={'schema_version':1,'sources_are_frozen':True,'optimization_rerun':False,
        'paper_figure2_source_version':'v013_final_dirlab4dct_axes_labels',
        'updated_table':'2026-09-16 paired timing N=10; historical OASIS100 accuracy is separate',
        'copd_selection':{'methods':sorted(allowed),'role':'post-paper extension','missing_method_rows':'not rerun'},
        'inputs':records}
    (out/'results/manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'exported_inputs':len(records),'rows':{r['path']:r.get('row_count') for r in records},
        'timing_pairs_verified':110,'first10_original_indices':sorted_indices},indent=2))


if __name__=='__main__':
    main()
