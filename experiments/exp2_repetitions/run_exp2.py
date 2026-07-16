"""
Experiment 2: Sound/syllable repetitions in stuttering DIVA.

Reproduces Fig. 3 of Civier et al. (2010).
Stuttering DIVA (high α_fb) produces significantly more self-repair events
on /bid/ than non-stuttering DIVA.

Run from this directory:
    UV_PROJECT_ENVIRONMENT=/tmp/diva_venv python run_exp2.py
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
from monitoring_subsystem import ExcessiveErrorDetector, SelfRepairController
from analysis.stats import repetitions_per_100
from analysis.plot_figures import plot_repetition_counts

RESULTS_DIR = os.path.join(_here, 'results')
FIGURES_DIR = os.path.join(_here, 'figures')
N_TRIALS = 100
TARGET = 'bid'


def load_config(name: str) -> dict:
    with open(os.path.join(_here, '..', 'configs', name)) as f:
        return yaml.safe_load(f)


def run_condition(cfg: dict, n_trials: int = N_TRIALS, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    diva = make_diva(TARGET,
                     feedforward_gain=cfg['feedforward_gain'],
                     feedback_gain=cfg['feedback_gain'],
                     learning_phase_runs=cfg.get('learning_phase_runs', 4),
                     seed=seed)

    detector = ExcessiveErrorDetector()
    controller = SelfRepairController(detector)

    repairs = []
    for trial in range(n_trials):
        if trial % 25 == 0:
            print(f'  trial {trial}/{n_trials}')
        n_rep, _ = controller.run(diva, I_masking=0.0, rng=rng)
        repairs.append(n_rep)

    return np.array(repairs)


def main():
    ns_cfg = load_config('non_stuttering.yaml')
    st_cfg = load_config('stuttering.yaml')

    results = {}
    for cond, cfg in [('non_stuttering', ns_cfg), ('stuttering', st_cfg)]:
        print(f'Running: {cond}')
        repairs = run_condition(cfg)
        reps = repetitions_per_100(repairs)
        results[cond] = {'repairs_per_trial': repairs.tolist(),
                         'repetitions_per_100': reps}
        print(f'  → {reps:.1f} reps / 100 syllables')

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'exp2_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_repetition_counts(
        {k: v['repetitions_per_100'] for k, v in results.items()},
        os.path.join(FIGURES_DIR, 'fig3_repetitions.png'))

    print('Experiment 2 complete.')


if __name__ == '__main__':
    main()
