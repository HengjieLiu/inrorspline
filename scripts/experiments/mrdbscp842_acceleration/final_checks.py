"""CPU-only source/contract checks, with external machine-readable evidence."""
import argparse
import ast
import io
import json
from pathlib import Path
import subprocess
import unittest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output",type=Path,required=True)
    a = p.parse_args()
    import torch
    torch.set_num_threads(6)
    from . import test_contracts,test_components
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream,verbosity=2).run(unittest.TestSuite([
        unittest.defaultTestLoader.loadTestsFromModule(test_contracts),
        unittest.defaultTestLoader.loadTestsFromModule(test_components)]))
    sources = list(Path("scripts/experiments/mrdbscp842_acceleration").glob("*.py"))
    sources.append(Path("scripts/experiments/oasis_alpha_sweep/run_one.py"))
    syntax = []
    for source in sources:
        try:
            ast.parse(source.read_text(),filename=str(source))
            syntax.append(dict(path=str(source),passed=True))
        except SyntaxError as exc:
            syntax.append(dict(path=str(source),passed=False,error=str(exc)))
    commands = []
    for argv in (["git","diff","--check"],
                 ["python","-B","scripts/experiments/check_tracked_notebook_outputs.py"]):
        run = subprocess.run(argv,text=True,capture_output=True)
        commands.append(dict(argv=argv,returncode=run.returncode,stdout=run.stdout,stderr=run.stderr))
    payload = dict(complete=result.wasSuccessful() and all(v["passed"] for v in syntax)
                   and all(v["returncode"] == 0 for v in commands),
                   tests_run=result.testsRun,tests_passed=result.wasSuccessful(),test_log=stream.getvalue(),
                   syntax=syntax,commands=commands,cuda_initialized=torch.cuda.is_initialized())
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps(payload,indent=2),flush=True)
    if not payload["complete"] or payload["cuda_initialized"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
