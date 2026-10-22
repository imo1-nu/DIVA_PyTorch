"""Evaluate every published test file, retaining unbounded/clipped composites."""
import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import sys
import time
import types

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
from pesq import pesq
from scipy.signal import resample_poly
from scipy.stats import ttest_rel
import soundfile as sf
import torch

from conditions import sha256
from data import load_pair
from model import ConditionerUpsampler, MelSpectrogram, SR, load_vocoder, synthesize

METRICS = ['pesq','csig_raw','cbak_raw','covl_raw','segsnr','csig','cbak','covl']


def quality_module():
    # pysepm.__init__ imports optional SRMR dependencies unused here.
    # Import the unchanged pinned qualityMeasures module without that initializer.
    name = '_torchdiva_pysepm'
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(importlib.metadata.distribution('pysepm').locate_file('pysepm'))]
        sys.modules[name] = package
    return importlib.import_module(name+'.qualityMeasures')


def quality(clean, processed):
    q = quality_module()
    divisor = math.gcd(SR, 16000)
    clean = resample_poly(clean.astype(np.float64), 16000//divisor, SR//divisor)
    processed = resample_poly(processed.astype(np.float64), 16000//divisor, SR//divisor)
    p = float(pesq(16000, clean, processed, 'wb'))
    seg = float(q.SNRseg(clean, processed, 16000))
    llr = float(q.llr(clean, processed, 16000, used_for_composite=True))
    wss = float(q.wss(clean, processed, 16000))
    raw = dict(csig_raw=3.093-1.029*llr+0.603*p-0.009*wss,
               cbak_raw=1.634+0.478*p-0.007*wss+0.063*seg,
               covl_raw=1.594+0.805*p-0.512*llr-0.007*wss)
    result = dict(pesq=p, segsnr=seg, llr=llr, wss=wss, **raw)
    result.update({key.removesuffix('_raw'):float(np.clip(value, 1, 5)) for key,value in raw.items()})
    if not all(np.isfinite(value) for value in result.values()):
        raise ValueError('Nonfinite quality metric; no sample is silently excluded')
    return result


def aggregate(rows):
    result = {}
    groups = sorted({(r['condition'],r['method']) for r in rows})
    for condition, method in groups:
        subset = [r for r in rows if (r['condition'],r['method'])==(condition,method)]
        result[f'{condition}/{method}'] = dict(n=len(subset), **{
            key:dict(mean=float(np.mean([r[key] for r in subset])),
                     sd=float(np.std([r[key] for r in subset],ddof=1))) for key in METRICS})
    comparisons = {}
    for condition in sorted({r['condition'] for r in rows}):
        baseline = {r['name']:r for r in rows if r['condition']==condition and r['method']=='diva'}
        for method in ('vanilla','enhanced'):
            treatment = {r['name']:r for r in rows if r['condition']==condition and r['method']==method}
            if not treatment:
                continue
            if set(treatment)!=set(baseline):
                raise ValueError('Paired comparison requires all matching test files')
            item = {}
            for key in METRICS:
                before = np.array([baseline[n][key] for n in sorted(baseline)])
                after = np.array([treatment[n][key] for n in sorted(baseline)])
                delta = after-before
                test = ttest_rel(after,before)
                # Identical constant vectors have undefined t; preserve as null.
                item[key] = dict(mean_change=float(delta.mean()),
                    t=float(test.statistic) if np.isfinite(test.statistic) else None,
                    p_two_sided=float(test.pvalue) if np.isfinite(test.pvalue) else None)
            comparisons[f'{condition}/{method}-diva'] = item
    for method in ('diva','vanilla','enhanced'):
        author={r['name']:r for r in rows if r['condition']=='author' and r['method']==method}
        corrected={r['name']:r for r in rows if r['condition']=='corrected' and r['method']==method}
        if not author or not corrected: continue
        if set(author)!=set(corrected): raise ValueError('Cross-condition file mismatch')
        item={}
        for key in METRICS:
            before=np.array([author[n][key] for n in sorted(author)])
            after=np.array([corrected[n][key] for n in sorted(author)])
            test=ttest_rel(after,before)
            item[key]=dict(mean_change=float((after-before).mean()),
                t=float(test.statistic) if np.isfinite(test.statistic) else None,
                p_two_sided=float(test.pvalue) if np.isfinite(test.pvalue) else None)
        comparisons['corrected-author/'+method]=item
    return dict(groups=result, paired_tests=comparisons,
                multiplicity='Exploratory unadjusted paired two-sided t tests; n=10 files, not independent speakers')


def evaluate(args):
    directory, output = args.data_dir.resolve(), args.output.resolve()
    output.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    manifest = json.loads((directory/'manifest.json').read_text())
    if not manifest.get('official_metadata_sha256'):
        raise ValueError('Official data mapping validation is required')
    samples = [s for s in manifest['samples'] if s['split']=='test']
    if len(samples)!=10:
        raise ValueError('Expected all ten published test files')
    model = load_vocoder(directory/'assets/speechbrain-diffwave.ckpt',device)
    mel = MelSpectrogram().to(device)
    upsamplers, checkpoint_hashes = {}, {}
    for condition in args.conditions:
        if 'enhanced' in args.methods:
            path = directory/'runs'/condition/'final.pt'
            checkpoint = torch.load(path,map_location='cpu',weights_only=True)
            if checkpoint['epoch']!=1000:
                raise ValueError('Primary evaluation requires epoch 1000')
            upsamplers[condition] = ConditionerUpsampler().to(device).eval()
            upsamplers[condition].load_state_dict(checkpoint['upsampler'])
            checkpoint_hashes[condition] = sha256(path)
    rows = []
    rows_path = output/'individual.json'
    if rows_path.exists():
        rows = json.loads(rows_path.read_text())
    started = time.perf_counter()
    for condition in args.conditions:
        for sample in samples:
            clean, degraded = load_pair(directory,sample,condition)
            seed = args.seed+int(hashlib.sha256(sample['name'].encode()).hexdigest()[:8],16)
            with torch.inference_mode():
                clean_mel = mel(torch.from_numpy(clean).to(device))
                degraded_mel = mel(torch.from_numpy(degraded).to(device))
            for method in args.methods:
                start = time.perf_counter()
                if method=='diva':
                    waveform = degraded
                else:
                    conditioning = clean_mel if method=='oracle' else degraded_mel
                    upsampler = upsamplers[condition] if method=='enhanced' else None
                    if device.type=='cuda': torch.cuda.synchronize()
                    waveform = synthesize(model,conditioning,upsampler,seed=seed)[0].cpu().numpy()[:len(clean)]
                    if len(waveform)!=len(clean):
                        raise ValueError('Synthesized waveform is shorter than the evaluation reference')
                synthesis_seconds = time.perf_counter()-start
                path = output/'audio'/condition/method/sample['name']
                path.parent.mkdir(parents=True,exist_ok=True)
                sf.write(path,waveform,SR,subtype='FLOAT')
                item = dict(condition=condition,method=method,name=sample['name'],seed=seed,
                    sample_rate=SR,frames=len(clean),duration=len(clean)/SR,
                    synthesis_seconds=synthesis_seconds,peak=float(np.max(np.abs(waveform))),
                    audio_sha256=sha256(path), pretrained_sha256=sha256(directory/'assets/speechbrain-diffwave.ckpt'),
                    upsampler_sha256=checkpoint_hashes.get(condition) if method=='enhanced' else None,
                    evaluation_code_sha256=sha256(__file__), **quality(clean,waveform))
                item['total_seconds'] = time.perf_counter()-start
                rows = [r for r in rows if (r['condition'],r['method'],r['name'])!=(condition,method,sample['name'])]
                rows.append(item)
                rows_path.write_text(json.dumps(rows,indent=2,allow_nan=False)+'\n')
                print(f'{condition}/{method}/{sample["name"]}: PESQ={item["pesq"]:.3f}, '
                      f'raw CSIG={item["csig_raw"]:.3f}, segSNR={item["segsnr"]:.3f}, '
                      f'{item["total_seconds"]:.2f}s',flush=True)
    result = dict(seed=args.seed, diffusion_steps=50, metric_sr=16000,
        inference_precision='fp32', deterministic_algorithms=True,
        boundary='Center crop independently to the shorter signal, without gain or delay matching',
        evaluation_seconds=time.perf_counter()-started, **aggregate(rows))
    (output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    with (output/'individual.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--conditions',nargs='+',choices=['author','corrected','transfer_peaks'],default=['author','corrected'])
    parser.add_argument('--methods',nargs='+',choices=['diva','vanilla','oracle','enhanced'],default=['diva','vanilla','oracle','enhanced'])
    parser.add_argument('--seed',type=int,default=20261023)
    parser.add_argument('--device',default='cuda')
    evaluate(parser.parse_args())
