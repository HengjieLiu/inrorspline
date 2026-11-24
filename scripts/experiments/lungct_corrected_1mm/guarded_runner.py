"""Explicitly approved forest support around the unchanged CorrField runner."""
import os
from pathlib import Path
import sys


def main():
    os.environ['CUDA_VISIBLE_DEVICES']=sys.argv[sys.argv.index('--gpu')+1]
    source=(Path(sys.argv[sys.argv.index('--corrfield-source')+1])
            if '--corrfield-source' in sys.argv else Path('/opt/corrfield-source'))
    sys.path.insert(0,str(source))
    import importlib
    upstream=importlib.import_module('corrfield.corrfield')
    import core
    from forest_guard import install
    import forest_guard
    trace=[]
    install(upstream,trace)
    evaluate_original=core.evaluate

    def evaluate_guarded(field_path,data_dir,out_dir,identity):
        if len(trace)!=4:
            raise ValueError(f'Expected two stages and two directions; got {len(trace)} graph calls')
        if any(row['added_edges']!=0 or row['parameter_changes'] for row in trace):
            raise ValueError('Forest guard changed graph edges or parameters')
        tagged=dict(identity,implementation_variant='corrfield_componentwise_forest_v1',
                    graph_guard_trace=trace,guarded_graph_calls=sum(row['components']>1 for row in trace),
                    guard_source_sha256=core.sha(forest_guard.__file__),
                    guarded_runner_sha256=core.sha(__file__))
        core.write_json(Path(out_dir)/'graph_guard_trace.json',trace)
        return evaluate_original(field_path,data_dir,out_dir,tagged)

    core.evaluate=evaluate_guarded
    import runner
    runner.main()


if __name__=='__main__':
    main()
