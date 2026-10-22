"""Resume the two-condition study with durable state and per-phase logs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent


def run(args):
    directory=args.data_dir.resolve()
    logs=directory/'logs'; logs.mkdir(exist_ok=True)
    state_path=directory/'study-state.json'
    state=dict(started=datetime.now(timezone.utc).isoformat(),data_dir=str(directory),phases=[])
    def save(phase,status,**extra):
        state.update(phase=phase,status=status,updated=datetime.now(timezone.utc).isoformat(),**extra)
        state_path.write_text(json.dumps(state,indent=2)+'\n')
    if args.wait_for_pairs:
        save('generation','waiting')
        while True:
            path=directory/'paired-generation.json'
            if path.exists():
                generation=json.loads(path.read_text())
                if len(generation['samples'])+len(generation['failures'])==984:
                    if generation['failures']:
                        save('generation','failed',failures=generation['failures'])
                        raise RuntimeError('Generation failed; inspect recorded failures')
                    break
            print('Waiting for all 984 pairs; progress is in pairs-progress.json.',flush=True)
            time.sleep(30)
    else:
        command=[sys.executable,str(HERE/'data.py'),'--data-dir',str(directory),'--workers','12']
        subprocess.run(command,check=True)
    def phase(name,command):
        print('START '+name,flush=True)
        save(name,'running',command=command)
        start=time.perf_counter()
        with (logs/(name+'.log')).open('a') as handle:
            process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            for line in process.stdout:
                handle.write(line);handle.flush();print(line,end='',flush=True)
            code=process.wait()
        state['phases'].append(dict(name=name,command=command,exit_code=code,seconds=time.perf_counter()-start))
        if code:
            save(name,'failed');raise RuntimeError(f'{name} failed with exit {code}')
        save(name,'completed')
    for condition in ('author','corrected'):
        output=directory/'runs'/condition
        command=[sys.executable,str(HERE/'train.py'),'--data-dir',str(directory),
                 '--condition',condition,'--output',str(output)]
        if (output/'last.pt').exists(): command.append('--resume')
        phase('train-'+condition,command)
    phase('evaluate',[sys.executable,str(HERE/'evaluate.py'),'--data-dir',str(directory),
                      '--output',str(directory/'evaluation')])
    phase('collect',[sys.executable,str(HERE/'collect.py'),'--data-dir',str(directory)])
    save('study','completed')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--wait-for-pairs',action='store_true')
    run(parser.parse_args())
