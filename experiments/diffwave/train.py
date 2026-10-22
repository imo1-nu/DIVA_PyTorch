"""Train on all 482 published training files; evaluate the fixed final epoch."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch.nn import functional as F

from conditions import sha256
from data import load_pair
from model import ConditionerUpsampler, HOP, MelSpectrogram, load_vocoder


def crop_batch(features, indices, rng, frames):
    inputs, targets, lengths = [], [], []
    for index in indices:
        clean, degraded = features[index]
        valid = min(frames, clean.shape[-1])
        extra = max(0, clean.shape[-1] - frames)
        start = int(rng.integers(0, extra+1))
        pair = []
        for spectrum in (degraded, clean):
            cropped = spectrum[:, start:start+frames]
            if cropped.shape[-1] < frames:
                cropped = F.pad(cropped, (0, frames-cropped.shape[-1]), mode='replicate')
            pair.append(cropped)
        inputs.append(pair[0])
        targets.append(pair[1])
        lengths.append(valid*HOP)
    return torch.stack(inputs), torch.stack(targets), torch.tensor(lengths)


def masked_l1(prediction, reference, lengths):
    mask = torch.arange(reference.shape[-1], device=reference.device)[None, None, :] < lengths[:, None, None]
    return ((prediction.float()-reference.float()).abs() * mask).sum() / (mask.sum()*reference.shape[1])


def train(args):
    if not torch.cuda.is_available() and args.device.startswith('cuda'):
        raise RuntimeError('CUDA is required by this run configuration')
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision('high')
    directory, output = args.data_dir.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((directory/'manifest.json').read_text())
    if (not manifest.get('official_metadata_sha256') or
            any(s.get('recording_id')!=s['name'].split('_')[4] or
                s.get('speaker_id')!=s['name'].split('_')[0] for s in manifest['samples'])):
        raise ValueError('Acquire data with official recording/speaker metadata validation first')
    pairs = json.loads((directory/'conditions'/args.condition/'pairs-manifest.json').read_text())
    if pairs['failures']:
        raise RuntimeError('Pair generation has failures; inspect the manifest first')
    available = {sample['name'] for sample in pairs['samples']}
    human_hashes={s['name']:s['human_sha256'] for s in pairs['samples']}
    samples = [row for row in manifest['samples'] if row['split']=='train']
    if any(sample['name'] not in available for sample in samples):
        raise RuntimeError('Generate all published training pairs first')
    if any(human_hashes[s['name']]!=s['wav_sha256'] for s in samples):
        raise ValueError('Generated pairs refer to different human recordings')
    print(f'{args.condition}: train {len(samples)}, published test 10 excluded.', flush=True)
    vocoder = load_vocoder(directory/'assets/speechbrain-diffwave.ckpt', device)
    teacher = vocoder.spectrogram_upsampler
    mel = MelSpectrogram()
    features = []
    input_hash=hashlib.sha256()
    preprocessing = time.perf_counter()
    with torch.no_grad():
        for index, sample in enumerate(samples):
            clean, degraded = load_pair(directory, sample, args.condition)
            input_hash.update(sample['name'].encode()+b'\0')
            input_hash.update(len(clean).to_bytes(8,'little'))
            input_hash.update(clean.astype('<f4',copy=False).tobytes())
            input_hash.update(degraded.astype('<f4',copy=False).tobytes())
            features.append((mel(torch.from_numpy(clean)), mel(torch.from_numpy(degraded))))
            if (index+1) % 100 == 0:
                print(f'Mel preprocessing {index+1}/{len(samples)}.', flush=True)
    model = ConditionerUpsampler().to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)
    amp = device.type == 'cuda' and not args.fp32
    configuration = dict(condition=args.condition, seed=args.seed, epochs=args.epochs,
        learning_rate=args.learning_rate, batch_size=args.batch_size, crop_frames=args.crop_frames,
        precision='bf16 autocast' if amp else 'fp32', gradient_clip=10,
        memory_format='channels_last',
        deterministic_algorithms=True, cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
        train_files=[s['name'] for s in samples], validation_files=[],
        test_files=[s['name'] for s in manifest['samples'] if s['split']=='test'],
        pretrained_sha256=sha256(directory/'assets/speechbrain-diffwave.ckpt'),
        pairs_manifest_sha256=sha256(directory/'conditions'/args.condition/'pairs-manifest.json'),
        training_input_sha256=input_hash.hexdigest(),
        source_fingerprint=pairs['provenance']['source_fingerprint'],
        code_sha256={name:sha256(Path(__file__).parent/name) for name in ('train.py','model.py','data.py')},
        architecture='8 encoder + 8 decoder learned convolutions, 256x upsampling',
        parameters=sum(p.numel() for p in model.parameters()), torch=str(torch.__version__),
        device=str(device), gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        preprocessing_seconds=time.perf_counter()-preprocessing)
    history, first_epoch, steps, previous_seconds = [], 1, 0, 0.0
    if args.resume:
        checkpoint = torch.load(output/'last.pt', map_location=device, weights_only=False)
        for key in ('condition','seed','learning_rate','batch_size','crop_frames','precision','code_sha256',
                    'source_fingerprint','training_input_sha256','pretrained_sha256','train_files','test_files'):
            if checkpoint['configuration'][key] != configuration[key]:
                raise ValueError(f'Resume configuration differs: {key}')
        model.load_state_dict(checkpoint['upsampler'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        rng.bit_generator.state = checkpoint['numpy_rng']
        torch.set_rng_state(checkpoint['torch_rng'].cpu())
        if device.type=='cuda':
            torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint['cuda_rng']])
        first_epoch, steps = checkpoint['epoch']+1, checkpoint['steps']
        old = json.loads((output/'training.json').read_text())
        history, previous_seconds = old['history'], old['seconds']
        if checkpoint['epoch']>args.epochs:
            raise ValueError('Requested final epoch precedes the saved checkpoint')
        if checkpoint['epoch']==args.epochs:
            if not (output/'final.pt').exists():
                torch.save(dict(upsampler=model.state_dict(),epoch=args.epochs,
                                configuration=configuration),output/'final.pt')
            print(f'{args.condition}: epoch {args.epochs} already completed.',flush=True)
            return old
    elif (output/'last.pt').exists():
        raise FileExistsError('Use --resume or a new output directory')
    (output/'config.json').write_text(json.dumps(configuration, indent=2)+'\n')
    started = time.perf_counter()
    for epoch in range(first_epoch, args.epochs+1):
        model.train()
        permutation = rng.permutation(len(features))
        total_loss, count = 0.0, 0
        for start in range(0, len(features), args.batch_size):
            indices = permutation[start:start+args.batch_size]
            degraded, clean, lengths = crop_batch(features, indices, rng, args.crop_frames)
            degraded, clean, lengths = degraded.to(device), clean.to(device), lengths.to(device)
            with torch.no_grad():
                reference = teacher(clean)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
                prediction = model(degraded)
                loss = masked_l1(prediction, reference, lengths)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Nonfinite training loss at epoch {epoch}, step {steps}')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.item()*len(indices)
            count += len(indices)
            steps += 1
        seconds = previous_seconds+time.perf_counter()-started
        item = dict(epoch=epoch, steps=steps, training_l1=total_loss/count,
                    gradient_norm=float(norm), seconds=seconds)
        history.append(item)
        if epoch==1 or epoch % args.log_every==0 or epoch==args.epochs:
            print(f'{args.condition} epoch {epoch}/{args.epochs}: '
                  f'L1={item["training_l1"]:.6f}, elapsed={seconds:.1f}s', flush=True)
            checkpoint = dict(upsampler=model.state_dict(), optimizer=optimizer.state_dict(),
                epoch=epoch, steps=steps, configuration=configuration,
                numpy_rng=rng.bit_generator.state, torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else [])
            torch.save(checkpoint, output/'last.tmp')
            (output/'last.tmp').replace(output/'last.pt')
            result = dict(configuration=configuration, last_epoch=epoch, steps=steps,
                          seconds=seconds, history=history,
                          peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=='cuda' else None,
                          peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=='cuda' else None)
            (output/'training.json').write_text(json.dumps(result, indent=2)+'\n')
    torch.save(dict(upsampler=model.state_dict(), epoch=args.epochs,
                    configuration=configuration), output/'final.pt')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--condition', choices=['author','corrected'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=1000)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--crop-frames', type=int, default=62)
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--seed', type=int, default=20261023)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--fp32', action='store_true')
    parser.add_argument('--resume', action='store_true')
    train(parser.parse_args())
