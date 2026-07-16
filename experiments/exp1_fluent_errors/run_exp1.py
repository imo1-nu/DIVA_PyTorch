"""
Experiment 1: Sensorimotor errors in fluent speech.

Reproduces Fig. 2 of Civier et al. (2010).
Compares auditory F2 error time-series for /bid/ (rapid F2 transition) vs
/bad/ (slow F2 transition) under normal and stuttering DIVA gains.

Run from this directory:
    UV_PROJECT_ENVIRONMENT=/tmp/diva_venv python run_exp1.py
"""

import os
import sys
import json

_here = os.path.dirname(os.path.abspath(__file__))
_src = os.path.join(_here, '..', '..', 'src')
_exp = os.path.join(_here, '..')
sys.path.insert(0, os.path.abspath(_src))
sys.path.insert(0, os.path.abspath(_exp))
os.chdir(os.path.abspath(_src))

import numpy as np
import yaml
import torch

from diva_setup import make_diva
from analysis.formant_analysis import aud_err_to_hz, peak_f2_error_hz
from analysis.plot_figures import plot_f2_error_timeseries

RESULTS_DIR = os.path.join(_here, 'results')
FIGURES_DIR = os.path.join(_here, 'figures')


def load_config(name: str) -> dict:
    with open(os.path.join(_here, '..', 'configs', name)) as f:
        return yaml.safe_load(f)


def collect_errors(diva, n_steps_extra: int = 10) -> np.ndarray:
    """Run one production and return per-timestep normalised aud errors (T, 4)."""
    import copy
    import diva_ssm
    import diva_asm as aud_sm_mod
    import diva_som_state_map as som_sm_mod
    import diva_aem as aud_em_mod
    import diva_sem as som_em_mod

    # Reset transient state (vel2pos, delay buffers) while preserving learned weights.
    diva.reset()
    if hasattr(diva, '_initial_cerebellum_buffer'):
        diva.cerebellum.dataBuffer = copy.deepcopy(diva._initial_cerebellum_buffer)
    if hasattr(diva, '_initial_lmc_weight'):
        diva.motor_cortex.LMC.get_set_weights(3, diva._initial_lmc_weight.clone())

    length_idx = diva.production_info[0].index('length\n')
    total_len = float(diva.production_info[1][length_idx])
    num_steps = int(np.ceil(total_len / diva.sample_rate)) + 1

    ssm = diva_ssm.SpeechSoundMap(num_steps)
    aud_sm = aud_sm_mod.AuditoryStateMap();  aud_sm.setinputdims(4)
    som_sm = som_sm_mod.SomatosensoryStateMap();  som_sm.setinputdims(8)
    aud_em = aud_em_mod.AuditoryErrorMap()
    som_em = som_em_mod.SomatosensoryErrorMap()

    cerebellum  = diva.cerebellum
    vocal_tract = diva.vocal_tract
    mc          = diva.motor_cortex

    ssm.InputPorts[0] = torch.tensor([1.0], dtype=torch.float64)
    ssm.InputPorts[1] = 1
    ssm.output()
    ssm.output_seq()

    err_log = []
    with torch.no_grad():
        for _ in range(num_steps + n_steps_extra):
            p = ssm.OutputPorts[0]
            aud_em.InputPorts[0] = ssm.OutputPorts[2]
            som_em.InputPorts[0] = ssm.OutputPorts[3]
            ssm.output_seq()

            aud_sm.output();  som_sm.output()
            aud_em.InputPorts[1] = aud_sm.OutputPorts[0]
            som_em.InputPorts[1] = som_sm.OutputPorts[0]
            aud_em.output();  som_em.output()

            cerebellum.input(p);  cerebellum.output()
            mc.InputPorts[0] = p
            mc.InputPorts[1] = cerebellum.OutputPorts[0]
            mc.InputPorts[2] = som_em.OutputPorts[0]
            mc.InputPorts[3] = aud_em.OutputPorts[0]
            mc.output()

            vocal_tract.InputPorts[0] = mc.OutputPorts[0]
            vocal_tract.output()
            aud_sm.input(vocal_tract.OutputPorts[0])
            som_sm.input(vocal_tract.OutputPorts[1])

            err_log.append(aud_em.OutputPorts[0].numpy().flatten().copy())

    return np.array(err_log)


def run_condition(cfg: dict, target: str, n_test: int = 5, seed: int = 42):
    diva = make_diva(target,
                     feedforward_gain=cfg['feedforward_gain'],
                     feedback_gain=cfg['feedback_gain'],
                     learning_phase_runs=cfg.get('learning_phase_runs', 4),
                     seed=seed)
    trials = [collect_errors(diva) for _ in range(n_test)]
    return trials


def main():
    ns_cfg = load_config('non_stuttering.yaml')
    st_cfg = load_config('stuttering.yaml')

    results = {}
    for target in ['bid', 'bad']:
        for cond, cfg in [('non_stuttering', ns_cfg), ('stuttering', st_cfg)]:
            print(f'target={target}  condition={cond}')
            trials = run_condition(cfg, target)
            peaks = [float(peak_f2_error_hz(t)) for t in trials]
            key = f'{target}_{cond}'
            results[key] = {'peak_f2_hz': peaks,
                            'trials': [t.tolist() for t in trials]}
            print(f'  peak F2 errors (Hz): {[f"{v:.0f}" for v in peaks]}')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'exp1_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    for target in ['bid', 'bad']:
        err_dict = {}
        for cond in ['non_stuttering', 'stuttering']:
            arr = np.array(results[f'{target}_{cond}']['trials'][0])
            err_dict[cond] = aud_err_to_hz(arr.tolist())
        plot_f2_error_timeseries(
            err_dict, os.path.join(FIGURES_DIR, f'fig2_{target}.png'))

    print('Experiment 1 complete.')


if __name__ == '__main__':
    main()
