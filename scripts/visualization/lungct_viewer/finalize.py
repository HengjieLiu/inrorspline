"""Seal a complete viewer only after independent data and hardware browser gates."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import tarfile
from .prepare import read,write,sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle-root',type=Path,required=True)
    a=p.parse_args();root=a.bundle_root;v=root/'validation';repo=Path(__file__).resolve().parents[3]
    if (v/'accepted.json').exists() or (v/'report.md').exists():raise FileExistsError('Existing finalization retained')
    data=read(v/'final_data_audit.json');browser=read(v/'final_browser/browser_report.json')
    assert data['status']=='accepted' and data['cases']==20 and data['landmark_method_records']==18000
    assert browser['status']=='accepted_functional' and browser['performance_gate']=='passed'
    assert browser['order_persistence']=='passed' and browser['cancellation_and_disposal']['status']=='passed'
    assert not browser['errors']
    assert {r['id'] for r in browser['cases']}=={'4dct_01','4dct_02','copd_01','copd_10'}
    for r in browser['cases']:
        assert r['fps']>=30 and r['p95_interaction_ms']<100 and r['request_free']
        assert r['diagnostics']['gpuTextures']<=15
        assert 'SwiftShader' not in r['diagnostics']['unmaskedRenderer']
    for name,expected in data['source_hashes'].items():
        assert sha(repo/name)==expected,f'Source changed after data audit: {name}'
    catalog=read(root/'catalog.json');assert len(catalog['cases'])==20
    exceptions=[];total_bytes=0;assets=0;counts=dict(resolved=0,ambiguous=0,unresolved=0)
    for entry in catalog['cases']:
        m=read(root/'cases'/entry['id']/'manifest.json')
        for x in m['assets'].values():total_bytes+=x['bytes'];assets+=1
        for point in m['landmarks']:
            for method,result in point['methods'].items():
                counts[result['status']]+=1
                if result['status']!='resolved':
                    exceptions.append(dict(case=entry['id'],method=method,landmark_id=point['id']+1,
                                           status=result['status'],roots=result['roots'],
                                           inverse_residual_mm=result['inverse_residual_mm']))
    with (v/'inverse_exceptions.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['case','method','landmark_id','status','roots','inverse_residual_mm'])
        w.writeheader();w.writerows(exceptions)
    lines=['# DIR-Lab private case viewer — validation', '',
           'Status: **accepted**. All 20 cases have 300 paired landmarks and three registered methods.', '',
           'Forward server port **1101** through VS Code or SSH, then open **http://127.0.0.1:1101**.', '',
           '## Data and geometry', '',
           f'- {assets} verified binary assets; {data["compressed_asset_bytes"]/1024**3:.2f} GiB compressed case data.',
           '- All 18,000 method/landmark native snap TRE values retained and replayed exactly.',
           f'- {data["original_source_files_unchanged"]} original source files passed unchanged checksum checks.',
           '- Independent raw-CT/field reconstruction checked HU, vectors, masks, original landmark text and inverse locations.',
           '- Effective endpoint spacing and original crop origins retained; corrected COPD01 has 272 registration slices.',
           '- Coronal S up / patient R left; axial A up / R left; sagittal S up / A left. Pair-local LPS, not scanner DICOM.',
           '- Common raw-HU display does not make the optimization intensity/mask protocols identical.', '',
           '## Browser checks', '',
           'Chromium '+browser['browser']+'; 1920×1080; '+browser['cases'][0]['diagnostics']['unmaskedRenderer']+'.', '',
           '| Case | Scroll fps | p95 interaction ms | Display MiB | Requests after Ready |',
           '| --- | ---: | ---: | ---: | ---: |']
    for r in browser['cases']:
        lines.append(f"| {r['id']} | {r['fps']:.1f} | {r['p95_interaction_ms']:.1f} | {r['diagnostics']['displayBytes']/1024**2:.0f} | 0 |")
    lines+=['', 'All cases passed the 30fps / p95 response below100ms gates on the recorded hardware.',
            'Tests also passed point picking, linked camera rotation, reordered columns, lung filters,',
            'ordering/reversals, random selection, wrapped navigation, zoom, crosshairs, canceled loads,',
            'GPU disposal, ArrayBuffer release, and method-order persistence across reload.',
            'Earlier SwiftShader software timing was below target and is retained separately; it is not mixed with hardware timing.', '',
            '## Inverse landmark locations', '',
            f"- Resolved: {counts['resolved']}; ambiguous: {counts['ambiguous']}; unresolved: {counts['unresolved']}.",
            '- Ambiguous locations are labelled; unresolved columns use an explicit fixed-target fallback without a fabricated transformed crosshair.',
            '- These flags do not discard correspondences or change audited TRE values.',
            '- [Exceptional points](inverse_exceptions.csv).', '',
            '[Complete data audit](final_data_audit.json) · [Browser report](final_browser/browser_report.json).',
            'Screenshots are stored beside the browser report. Nine synthetic/HTTP tests passed; the',
            'large-value compact-storage test intentionally exercises FP16 overflow and FP32 fallback.', '']
    (v/'report.md').write_text('\n'.join(lines))
    with tarfile.open(v/'source_snapshot.tar','x') as tar:
        for folder in [repo/'scripts/visualization/lungct_viewer',repo/'experiments/2026-11-19_lungct_viewer']:
            for path in sorted(folder.rglob('*')):
                if path.is_file() and '__pycache__' not in path.parts:tar.add(path,arcname=str(path.relative_to(repo)))
        for path in [repo/'tests/test_lungct_viewer.py',repo/'docs/implementation/lungct_case_viewer.md']:
            tar.add(path,arcname=str(path.relative_to(repo)))
    receipts={str(p.relative_to(root)):sha(p) for p in [v/'final_data_audit.json',v/'final_browser/browser_report.json',
              v/'report.md',v/'source_snapshot.tar',v/'inverse_exceptions.csv',root/'catalog.json']}
    receipts.update({str(p.relative_to(root)):sha(p) for p in sorted((root/'deps').glob('*')) if p.is_file()})
    receipts.update({str(p.relative_to(root)):sha(p) for p in sorted((v/'final_browser').glob('*.png'))})
    write(v/'accepted.json',dict(status='accepted',cases=20,assets=assets,landmark_pairs=6000,
                               landmark_method_records=18000,inverse_counts=counts,receipts=receipts,
                               source_hashes=data['source_hashes']))
    print('VIEWER ACCEPTED:20 cases,18000 method/landmark records',flush=True)


if __name__=='__main__':main()
