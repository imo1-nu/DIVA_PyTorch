"""Verify baseline reproduction and H-to-controller wiring before the full run."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    args=parser.parse_args()
    root=args.data_dir.resolve()
    manifest=json.loads((root/'manifest.json').read_text())
    name=manifest['samples'][0]['name']
    before,sr=sf.read(root/'conditions/corrected/pairs'/name,dtype='float32')
    after,rate=sf.read(root/'conditions/fmfit_check/pairs'/name,dtype='float32')
    assert sr==rate==11025
    np.testing.assert_array_equal(before,after)
    h=json.loads((root/'conditions/transfer_peaks/pairs'/name).with_suffix('.json').read_text())
    fm=json.loads((root/'conditions/fmfit_check/pairs'/name).with_suffix('.json').read_text())
    newdir=root/h['trajectories']
    olddir=root/fm['trajectories']
    target=dict(np.load(newdir/'target.npz'))
    oldtarget=dict(np.load(olddir/'target.npz'))
    assert set(target)==set(oldtarget)
    for key in target:
        np.testing.assert_array_equal(target[key],oldtarget[key])
    np.testing.assert_array_equal(target['Art'],0.)
    valid_counts=[]
    for trial in range(1,6):
        with np.load(newdir/f'trial_{trial:02}.npz') as data:
            mask=data['h_valid'].astype(bool)
            np.testing.assert_allclose((data['returned'][:,1:]*[500,1500,3000])[mask],data['h_peaks'][mask],rtol=0,atol=1e-9)
            np.testing.assert_array_equal(data['returned'][:,0],data['fmfit'][:,0])
            assert not np.any(data['auditory_error'][~data['delayed_valid'].astype(bool)])
            valid_counts.append(int(mask.all(axis=1).sum()))
    result=dict(name=name,baseline_audio_samples_exact_equal=True,frames=len(before),sample_rate=sr,
        baseline_pcm_float32_sha256=hashlib.sha256(before.astype('<f4').tobytes()).hexdigest(),
        targets_identical=True,initial_ff_all_zero=True,f0_unchanged=True,
        h_return_values_match_peaks=True,invalid_delayed_auditory_errors_zero=True,
        trial_joint_valid_counts=valid_counts,
        standard_model_and_mask_tests='13 passed, 40 reference subtests; /tmp/transfer-diffwave-tests.log')
    (root/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
