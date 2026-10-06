"""One-time mechanical extraction of audited research kernels into the release.

This is private migration tooling, not a public runtime dependency. Optimizers
are never invoked. All manually designed public interfaces use normal patches.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path

HEADER = '''from __future__ import annotations
import argparse
import csv
import json
import math
import os
import re
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
'''


class ReleaseTransform(ast.NodeTransformer):
    """Disable deferred instrumentation, not the optimization math."""
    diagnostic_fields = {'sample_checkpoints': False, 'finest_loss_diagnostic': False,
        'global_ncc_diagnostic_interval': 0, 'image_sim_eval_interval': 0,
        'prolongation_diagnostics': False, 'prolongation_preserve_check': False,
        'save_intermediate_dense': False}

    def visit_Attribute(self, node):
        if isinstance(node.value, ast.Name) and node.value.id == 'args' and node.attr in self.diagnostic_fields:
            return ast.copy_location(ast.Constant(self.diagnostic_fields[node.attr]),node)
        return self.generic_visit(node)

    def visit_If(self,node):
        original=ast.unparse(node.test)
        if 'sampler is not None' in original or original.startswith(('should_global_ncc_diagnostic(', 'should_eval_image_similarity(')):
            return [self.visit(x) for x in node.orelse]
        node=self.generic_visit(node)
        if isinstance(node.test, ast.Constant):
            return node.body if node.test.value else node.orelse
        # Literal comparisons such as 0 > 0 introduced by disabled switches.
        try:
            if not any(isinstance(n,(ast.Name,ast.Attribute,ast.Call)) for n in ast.walk(node.test)):
                return node.body if ast.literal_eval(ast.unparse(node.test)) else node.orelse
        except (ValueError,SyntaxError):
            pass
        return node

    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Attribute) and node.func.attr == 'build_siren':
            if any(isinstance(n,ast.Attribute) and n.attr=='spline_omega' for a in node.args for n in ast.walk(a)):
                node.keywords.append(ast.keyword(arg='upstream',value=ast.Constant('SINR')))
        return node


class FoldConstants(ast.NodeTransformer):
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if isinstance(node.func,ast.Name) and node.func.id in {'bool','int','float'} and len(node.args)==1 and isinstance(node.args[0],ast.Constant) and not node.keywords:
            return ast.Constant({'bool':bool,'int':int,'float':float}[node.func.id](node.args[0].value))
        return node
    def visit_Compare(self,node):
        node=self.generic_visit(node)
        if all(isinstance(n,ast.Constant) for n in [node.left,*node.comparators]):
            return ast.Constant(eval(compile(ast.Expression(ast.fix_missing_locations(node)),'<constants>','eval'),{'__builtins__':{}}))
        return node
    def visit_If(self,node):
        node=self.generic_visit(node)
        if isinstance(node.test,ast.Constant):return node.body if node.test.value else node.orelse
        return node
    def visit_IfExp(self,node):
        node=self.generic_visit(node)
        if isinstance(node.test,ast.Constant):return node.body if node.test.value else node.orelse
        return node


def extract(source, roots, *, transform=False, overrides=None):
    tree=ast.parse(source.read_text())
    if transform:
        tree=ast.fix_missing_locations(ReleaseTransform().visit(tree))
        tree=ast.fix_missing_locations(FoldConstants().visit(tree))
    definitions={}
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)):
            definitions[node.name]=node
        elif isinstance(node,(ast.Assign,ast.AnnAssign)):
            targets=node.targets if isinstance(node,ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target,ast.Name): definitions[target.id]=node
    for name,value in (overrides or {}).items():
        definitions[name]=ast.parse(value).body[0]
    needed=set(roots); pending=list(needed)
    while pending:
        key=pending.pop()
        if key not in definitions: continue
        for node in ast.walk(definitions[key]):
            if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Load) and node.id in definitions and node.id not in needed:
                needed.add(node.id);pending.append(node.id)
    emitted=[];seen=set()
    # Preserve original definition order; place replacement constants in situ.
    for name,node in definitions.items():
        if name in needed and id(node) not in seen:
            emitted.append(node);seen.add(id(node))
    return '\n\n'.join(ast.unparse(n) for n in emitted)+'\n'


def defaults(path):
    result={}
    tree=ast.parse(path.read_text())
    parse=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='parse_args')
    for node in ast.walk(parse):
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Attribute) or node.func.attr!='add_argument':continue
        name=ast.literal_eval(node.args[0]).removeprefix('--').replace('-','_')
        kw={k.arg:k.value for k in node.keywords}
        if 'default' in kw:
            try:result[name]=ast.literal_eval(kw['default'])
            except (ValueError,TypeError):pass
        elif 'action' in kw:
            try:action=ast.literal_eval(kw['action'])
            except ValueError:continue
            if action in ('store_true','store_false'):result[name]=action=='store_false'
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--research-root',type=Path,required=True);p.add_argument('--release-root',type=Path,required=True);a=p.parse_args()
    r=a.research_root;o=a.release_root;records=[]
    def emit(source,target,roots,header=HEADER,transform=False,overrides=None):
        src=r/source;body=extract(src,roots,transform=transform,overrides=overrides)
        path=o/target;path.parent.mkdir(parents=True,exist_ok=True)
        text='"""Audited registration/data kernels; see docs/methods.md for protocol scope."""\n'+header+'\n'+body
        for forbidden in ['/home/', '/database/', '/data/hengjie', 'AGENTS.md']:
            if forbidden in text:raise ValueError(f'{forbidden} in {target}')
        path.write_text(text)
        records.append({'path':target,'source':source,'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
            'extracted_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'transform':'portable imports; frozen defaults; deferred instrumentation disabled' if transform else 'dependency-closed definitions'})
        return body
    for dataset,folder,roots in [
        ('oasis','oasis_alpha_sweep',['train_idir','train_spline','train_multistage_direct_cp','default_lr','is_multistage_method']),
        ('lung','lungct_no_sweep_repro',['train_idir','train_single_stage_spline','train_multistage','default_lr','default_epochs'])]:
        base=f'scripts/experiments/{folder}'
        body=emit(f'{base}/run_one.py',f'rightpriordir/{dataset}_engine.py',roots,
            HEADER+f'from . import {dataset} as c\n',transform=True,
            overrides={'CheckpointSampler':'CheckpointSampler = object'})
        c_roots={node.attr for node in ast.walk(ast.parse(body)) if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=='c'}
        if dataset=='oasis':
            c_roots.update(['load_oasis_case','set_cuda_device','regularity_metrics','save_dense_disp','accuracy_row'])
            extra='import nibabel as nib\nimport scipy.ndimage\ntry:\n    from surface_distance import metrics as surface_distance_metrics\nexcept ImportError:\n    surface_distance_metrics = None\n'
            overrides={'DEFAULT_OASIS_ROOT':'DEFAULT_OASIS_ROOT = Path(os.environ.get("OASIS_ROOT", "data/oasis"))',
                'DEFAULT_PAIR_FILE':'DEFAULT_PAIR_FILE = Path(__file__).resolve().parents[1] / "splits/pair_test_200_shuffled_seed000003.txt"'}
        else:
            c_roots.update(['load_case','loader_audit','set_cuda_device','make_loss_volume','make_sampled_lcc_offsets',
                'sampled_ptv_local_cc_loss','idir_sampled_tv_physical','idir_bending_energy','final_metrics_from_dense','dense_norm_xyz_to_vox_zyx',
                'landmark_tre_from_dense_disp','resize_reg1mm_disp_to_native_crop_vox','displacement_regularities'])
            extra='';overrides={}
        overrides['build_siren']='def build_siren(hidden_dim, hidden_layers, omega, input_dim=3, *, upstream="IDIR"):\n    from .upstream import siren\n    return siren(upstream, [int(input_dim)] + [int(hidden_dim)] * int(hidden_layers) + [3], float(omega))'
        emit(f'{base}/common.py',f'rightpriordir/{dataset}.py',c_roots,HEADER+extra,overrides=overrides)
        d=defaults(r/f'{base}/run_one.py');d.update(ReleaseTransform.diagnostic_fields);d['epochs']=2500
        (o/f'rightpriordir/{dataset}_defaults.json').write_text(json.dumps(d,indent=2)+'\n')
    raw='scripts/data/common/lungct.py'
    raw_roots=['resolve_4dct_case','resolve_copd_case','load_raw_4dct_image','load_raw_copd_image','load_landmarks_xyz','mask_root_case_dir','landmark_mask_fraction']
    emit(raw,'rightpriordir/raw_lung.py',raw_roots,HEADER)
    emit('scripts/data/preprocessing/lungct/generate_4dct_ptvprep.py','rightpriordir/geometry.py',[
        'img_thr','landmarks_xyz_one_based_to_zyx_zero_based','ptvreg_crop_bounds_zyx_from_landmarks_xyz',
        'task4_legacy_crop_bounds_zyx_from_landmarks_xyz','crop_zyx','resize_image_trilinear_zyx','resize_mask_nearest_zyx',
        'transform_landmarks_native0_to_crop','transform_landmarks_crop_to_registration','points_inside_shape'],HEADER)
    # Pure author helper subset; the release wrapper loads the pinned submodule.
    emit('scripts/experiments/lungct_dual_inr_author/adapter.py','rightpriordir/dual_helpers.py',
        ['AUTHOR_ADAMW_WEIGHT_DECAY','resolve_device','seed_everything'],HEADER)
    emit('scripts/experiments/lungct_dual_inr_controlled/adapter.py','rightpriordir/dual_adapter.py',
        ['build_controlled_config','train_controlled','variant_protocol'],
        HEADER+'import copy\nimport hashlib\nimport torch.nn.functional as torch_f\nfrom .dual_helpers import AUTHOR_ADAMW_WEIGHT_DECAY, resolve_device, seed_everything\nfrom .upstream import load_external_api, AUTHOR_EXTERNAL_COMMIT\nfrom . import lung as mr_common\n')
    for name,folder,imports in [
        ('dual_data','lungct_dual_inr_controlled','from . import raw_lung as lungct_io\nfrom . import lung as common\n'),
        ('dual_copd_data','lungct_dual_inr_controlled_copd','from . import raw_lung as lungct_io\nfrom . import lung as common\nfrom . import dual_data as base\n')]:
        src=r/f'scripts/experiments/{folder}/data.py'
        funcs=[n.name for n in ast.parse(src.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
        emit(str(src.relative_to(r)),f'rightpriordir/{name}.py',funcs,HEADER+imports)
    plot_header='''from __future__ import annotations
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D
'''
    emit('scripts/experiments/oasis_alpha_sweep/plot_first10_first20_alpha5_saved_baselines.py',
        'rightpriordir/_plot_oasis.py',['draw_panel'],plot_header)
    emit('experiments/paper_artifacts/build_standardized_pareto.py','rightpriordir/_plot_style.py',
        ['_draw_composite_oasis_panel'],plot_header)
    emit('experiments/paper_artifacts/pareto_figure/build_final_dirlab4dct_axes_labels.py',
        'rightpriordir/_plot_lung.py',['draw_lung_panel','ALL_METHODS','SELECTED_METHODS'],plot_header,
        overrides={k:f'{k} = {v!r}' for k,v in {'X_COL':'fg_logj_std_mean','Y_COL':'original_snap_tre_mean_mm_mean',
            'X_STD_COL':'fg_logj_std_std','Y_STD_COL':'original_snap_tre_mean_mm_std'}.items()})
    (o/'docs/source_extraction.json').write_text(json.dumps({'schema_version':1,'purpose':'source provenance, not a runtime dependency','files':records},indent=2)+'\n')
    print(json.dumps({'extracted_files':len(records),'files':[x['path'] for x in records]},indent=2))


if __name__=='__main__':main()
