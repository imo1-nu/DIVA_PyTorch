"""
Experiment 4: Masking noise reduces repetitions.

Reproduces Fig. 5 of Civier et al. (2010).
Increasing masking noise I_masking reduces the perceived auditory error, thereby
suppressing self-repair events and repetitions monotonically.

Run from this directory:
    UV_PROJECT_ENVIRONMENT=/tmp/diva_venv python run_exp4.py
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
from analysis.plot_figures import plot_repetitions_vs_noise

RESULTS_DIR = os.path.join(_here, 'results')
FIGURES_DIR = os.path.join(_here, 'figures')
N_TRIALS_PER_LEVEL = 100
TARGET = 'bid'

# I_masking in same units as f_size (Hz).
# Threshold = xi * I_masking = 2 * I_masking.
# Mean f_size = 38.59 Hz (threshold at I=19), max = 172 Hz (threshold at I=86).
# Fine spacing 0-90 Hz captures the gradual suppression curve.
NOISE_LEVELS = np.array([0, 5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90])


def load_config(name: str) -> dict:
    with open(os.path.join(_here, '..', 'configs', name)) as f:
        return yaml.safe_load(f)


def main():
    torch.manual_seed(0)
    np.random.seed(0)
    rng = np.random.default_rng(0)

    st_cfg = load_config('stuttering.yaml')

    diva = make_diva(TARGET,
                     feedforward_gain=st_cfg['feedforward_gain'],
                     feedback_gain=st_cfg['feedback_gain'],
                     learning_phase_runs=st_cfg.get('learning_phase_runs', 4),
                     seed=0)

    detector = ExcessiveErrorDetector()
    controller = SelfRepairController(detector)

    rep_counts = []
    for I in NOISE_LEVELS:
        repairs = []
        for _ in range(N_TRIALS_PER_LEVEL):
            n, _ = controller.run(diva, I_masking=float(I), rng=rng)
            repairs.append(n)
        reps = repetitions_per_100(np.array(repairs))
        rep_counts.append(reps)
        print(f'I_masking={I:>4.0f}  →  {reps:.1f} reps / 100 syllables', flush=True)

    rep_counts = np.array(rep_counts)
    results = {'noise_levels': NOISE_LEVELS.tolist(),
               'repetitions_per_100': rep_counts.tolist()}

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'exp4_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_repetitions_vs_noise(
        NOISE_LEVELS, rep_counts,
        os.path.join(FIGURES_DIR, 'fig5_masking_noise.png'))

    print('Experiment 4 complete.')


if __name__ == '__main__':
    main()
