"""Verify immutable bundle receipts and assets for the comparison-only viewer."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Refusing to overwrite an audit')
    catalog=json.loads((a.bundle_root/'catalog.json').read_text())
    expected={f'{d}_{i:02}' for d in ('4dct','copd') for i in range(1,11)}
    assert {c['id'] for c in catalog['cases']}==expected
    report=dict(status='accepted',cases=[],assets=0,landmarks=0,bytes=0)
    for entry in catalog['cases']:
        case=a.bundle_root/'cases'/entry['id']
        receipt=json.loads((case/'accepted.json').read_text())
        assert receipt['status']=='accepted'
        assert digest(case/'manifest.json')==receipt['manifest_sha256']
        m=json.loads((case/'manifest.json').read_text());g=m['geometry']
        assert len(m['landmarks'])==300 and len(m['assets'])==13
        assert g['step_lps_mm'][0]>0 and g['step_lps_mm'][1]>0 and g['step_lps_mm'][2]<0
        if entry['id']=='copd_01':assert g['shape_zyx'][0]==272
        guarded=any('forest' in v['variant'] for v in m['methods'])
        assert guarded==(entry['id'] in ('4dct_04','4dct_05','4dct_06'))
        for asset in m['assets'].values():
            path=case/asset['file']
            assert path.stat().st_size==asset['compressed_bytes']
            assert digest(path)==asset['sha256'],path.name
            report['assets']+=1;report['bytes']+=path.stat().st_size
        report['landmarks']+=300
        report['cases'].append(dict(id=entry['id'],shape_zyx=g['shape_zyx'],guarded=guarded,
                                  manifest_sha256=receipt['manifest_sha256']))
    assert report['assets']==260
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='cases'}),flush=True)


if __name__=='__main__':main()
