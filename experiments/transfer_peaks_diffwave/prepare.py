"""Snapshot baseline inputs and pinned H-observation source in a separate dataset."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT=Path(__file__).resolve().parents[2]
REVISION='b89a625'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    old,out=args.baseline.resolve(),args.output.resolve()
    out.mkdir(parents=True,exist_ok=False)
    files={}
    for relative in ('human','assets','conditions/corrected','runs/corrected','sources/corrected','evaluation'):
        shutil.copytree(old/relative,out/relative)
        for p in (old/relative).rglob('*'):
            if p.is_file():
                new=out/p.relative_to(old)
                if sha(new)!=sha(p): raise ValueError('Baseline copy mismatch')
                files[str(p.relative_to(old))]=sha(p)
    shutil.copy2(old/'manifest.json',out/'manifest.json')
    files['manifest.json']=sha(old/'manifest.json')
    srcdir=out/'sources/transfer_peaks'
    srcdir.mkdir(parents=True)
    archive=subprocess.check_output(['git','-C',str(ROOT),'archive','--format=tar',REVISION,'src'])
    with tarfile.open(fileobj=io.BytesIO(archive)) as tf:
        tf.extractall(srcdir,filter='data')
    path=srcdir/'src/diva_targets.py'
    text=path.read_text()
    before='                x0 = base_x0  # + wrapper\n'
    assert text.count(before)==1
    path.write_text(text.replace(before,before+'                x1 = production_info[min_label]\n                x2 = production_info[max_label]\n'))
    hashes={str(p.relative_to(srcdir/'src')):sha(p) for p in sorted((srcdir/'src').rglob('*')) if p.is_file()}
    provenance=dict(revision=subprocess.check_output(['git','-C',str(ROOT),'rev-parse',REVISION],text=True).strip(),
        source_files_sha256=hashes,source_fingerprint=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest(),
        adapters=['Initialize float target min/max exactly as baseline conditions.py; CPU generation hides CUDA'],
        prepare_code_sha256=sha(Path(__file__)))
    (srcdir/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    (out/'baseline_snapshot.json').write_text(json.dumps(dict(source=str(old),files_sha256=files),indent=2)+'\n')
    print(f'Copied and hash-checked {len(files)} baseline files; source {provenance["source_fingerprint"]}',flush=True)


if __name__=='__main__': main()
