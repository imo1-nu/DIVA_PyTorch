"""Run the preregistered generation, final CNN training and held-out evaluation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--workers',type=int,default=18)
    args=p.parse_args()
    directory=args.data_dir.resolve()
    logs=directory/'logs'
    logs.mkdir(exist_ok=True)
    stages=[
        ('preflight',[str(ROOT/'experiments/transfer_peaks_diffwave/verify.py'),'--data-dir',str(directory)]),
        ('generation',[str(ROOT/'experiments/transfer_peaks_diffwave/generate.py'),'--data-dir',str(directory),'--workers',str(args.workers)]),
        ('training',[str(ROOT/'experiments/diffwave/train.py'),'--data-dir',str(directory),'--condition','transfer_peaks','--output',str(directory/'runs/transfer_peaks')]),
        ('evaluation',[str(ROOT/'experiments/diffwave/evaluate.py'),'--data-dir',str(directory),'--conditions','transfer_peaks','--output',str(directory/'transfer_evaluation')]),
    ]
    env=os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1',MPLCONFIGDIR='/tmp/transfer-peaks-diffwave-mpl')
    started=time.time()
    history=[]
    for stage,command in stages:
        record=dict(stage=stage,state='running',started_unix=time.time(),command=[sys.executable,*command])
        history.append(record)
        (directory/'state.json').write_text(json.dumps(dict(started_unix=started,stages=history),indent=2)+'\n')
        print(f'Start {stage}',flush=True)
        stage_env=env.copy()
        if stage in ('preflight','generation'):stage_env['CUDA_VISIBLE_DEVICES']=''
        with (logs/f'{stage}.log').open('w') as stream:
            result=subprocess.run([sys.executable,*command],cwd=ROOT,env=stage_env,stdout=stream,stderr=subprocess.STDOUT)
        record.update(state='complete' if result.returncode==0 else 'failed',exit_code=result.returncode,seconds=time.time()-record['started_unix'])
        (directory/'state.json').write_text(json.dumps(dict(started_unix=started,stages=history),indent=2)+'\n')
        print(f'{stage}: {record["state"]}, {record["seconds"]:.1f}s',flush=True)
        if result.returncode: raise SystemExit(result.returncode)
    print('All generation/training/evaluation stages complete.',flush=True)


if __name__=='__main__':main()
