"""Create an allowlisted public source snapshot, never copy private Git history."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import shutil
import sys

TOP_FILES=['.gitattributes','.gitignore','.gitmodules','.dockerignore','README.md','LICENSE','THIRD_PARTY.md','CITATION.cff','Dockerfile','requirements.txt']
DIRECTORIES=['configs','docs','experiments','figures','poster','results','rightpriordir','splits','tests']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--public',type=Path,required=True);p.add_argument('--snapshot',type=Path,required=True);a=p.parse_args()
    root=a.public.resolve()
    if a.snapshot.exists():raise FileExistsError('Snapshot must be a new directory')
    extraction_path=root/'docs/source_extraction.json';extraction=json.loads(extraction_path.read_text())
    for record in extraction['files']:record['curated_sha256']=sha(root/record['path'])
    extraction['post_extraction_changes']=['explicit_public_crop_naming','strict_OASIS_input_validation','unused_import_cleanup','formatting_only']
    extraction_path.write_text(json.dumps(extraction,indent=2)+'\n')
    env={'python':sys.version.split()[0],
         'base_digest':'sha256:8a159ba5e7f438cf36ea10d74a4a45fac1b0250d7d70f0dba114eefcbe3416da',
         'requirements_sha256':sha(root/'requirements.txt'),
         'packages':{d.metadata['Name']:d.version for d in sorted(importlib.metadata.distributions(),key=lambda d:d.metadata['Name'].lower())}}
    (root/'docs/environment.json').write_text(json.dumps(env,indent=2)+'\n')
    files=[root/p for p in TOP_FILES]
    for directory in DIRECTORIES:
        files.extend(p for p in (root/directory).rglob('*') if p.is_file() and not {'__pycache__','.pytest_cache','.ruff_cache'} & set(p.parts) and p.suffix not in {'.pyc','.pyo'})
    lock=json.loads((root/'third_party/lock.json').read_text())
    files.append(root/'third_party/lock.json')
    for name,record in lock.items():files.extend(root/'third_party'/name/p for p in record['files'])
    files=sorted(set(files))
    for path in files:
        relative=path.relative_to(root)
        if '.git' in relative.parts or path.name in {'AGENTS.md','Pasted text.txt','.env'}:raise ValueError(relative)
        if relative.parts[0]!='third_party':
            if path.suffix in {'.npy','.npz','.nii','.img','.pt','.pth','.ckpt'}:raise ValueError(f'Medical/runtime artifact: {relative}')
            if path.suffix in {'.py','.md','.json','.csv','.txt','.cff'} or path.name in {'Dockerfile','.gitmodules'}:
                text=path.read_text()
                forbidden=['/home/hengjie','/homebase/','/database/','/mnt/shengdata','/data/hengjie','AGENTS.md','docs/public_release','-----BEGIN PRIVATE KEY-----']
                if any(token in text for token in forbidden):raise ValueError(f'Private content: {relative}')
        if path.suffix=='.md' and relative.parts[0]!='third_party':
            for link in re.findall(r'\]\(([^)]+)\)',path.read_text()):
                if re.match(r'^[a-z]+:|^#',link):continue
                target=link.split('#')[0]
                if target and not (path.parent/target).is_file():raise ValueError(f'Broken local link in {relative}: {target}')
    payload={'schema_version':1,'scope':'public source + frozen numeric inputs; no patient images or optimization outputs',
        'submodule_commits':{name:r['commit'] for name,r in lock.items()},
        'files':{p.relative_to(root).as_posix():{'sha256':sha(p),'bytes':p.stat().st_size} for p in files}}
    (root/'release_manifest.json').write_text(json.dumps(payload,indent=2)+'\n')
    files.append(root/'release_manifest.json')
    for path in files:
        target=a.snapshot/path.relative_to(root);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
    print(json.dumps({'file_count':len(files),'bytes':sum(p.stat().st_size for p in files),
        'snapshot_git_metadata':False,'public_manifest_sha256':sha(root/'release_manifest.json'),'local_links_valid':True,'private_path_scan_pass':True},indent=2))


if __name__=='__main__':main()
