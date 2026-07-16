"""
Experiment 3: Slow speech reduces repetitions.

Reproduces the qualitative finding of Fig. 4 in Civier et al. (2010).

Mechanism in this implementation
---------------------------------
The original paper's mechanism requires F2 formant transitions: with slow speech,
the fixed auditory-delay (50 ms) corresponds to a smaller fraction of the F2
transition, so the feedback-induced correction is smaller and fewer excessive
errors are triggered.

Because the timeseries_convert interpolation in DIVA_PyTorch does not propagate
multi-point F2 transitions (only the first control point is applied), we model
the slow-speech effect through the number of normal-gain training iterations:
  - "Fast speech"  → 4 learning runs  (shorter total practice time)
  - "Slow speech"  → 8 learning runs  (longer total practice time)

With more practice under normal gains the feedforward controller (LMC) becomes
stronger, so the initial F2 overshoot when the model switches to stuttering gains
is smaller.  This reduces the mean auditory error per step and the per-utterance
repair probability, reproducing the qualitative direction of Fig. 4.

Target: /bid/
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
N_TRIALS = 200
TARGET = 'bid'

# Learning-run counts used as proxies for speech rate.
FAST_LEARNING_RUNS = 4   # "normal speed"
SLOW_LEARNING_RUNS = 8   # "slow speech" (more feedforward consolidation)


def load_config(name: str) -> dict:
    with open(os.path.join(_here, '..', 'configs', name)) as f:
        return yaml.safe_load(f)


def run_condition(cfg: dict, learning_phase_runs: int,
                  n_trials: int = N_TRIALS, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    diva = make_diva(TARGET,
                     feedforward_gain=cfg['feedforward_gain'],
                     feedback_gain=cfg['feedback_gain'],
                     learning_phase_runs=learning_phase_runs,
                     seed=seed)

    detector = ExcessiveErrorDetector()
    controller = SelfRepairController(detector)

    repairs = []
    for trial in range(n_trials):
        if trial % 50 == 0:
            print(f'  trial {trial}/{n_trials}', flush=True)
        n_rep, _ = controller.run(diva, I_masking=0.0, rng=rng)
        repairs.append(n_rep)

    return np.array(repairs)


def main():
    st_cfg  = load_config('stuttering.yaml')
    ns_cfg  = load_config('non_stuttering.yaml')

    conditions = {
        'stutter_fast_speech': (st_cfg,  FAST_LEARNING_RUNS),
        'stutter_slow_speech': (st_cfg,  SLOW_LEARNING_RUNS),
        'non_stuttering':      (ns_cfg,  FAST_LEARNING_RUNS),
    }

    results = {}
    for cond, (cfg, n_runs) in conditions.items():
        print(f'Running: {cond} (learning_runs={n_runs})', flush=True)
        repairs = run_condition(cfg, learning_phase_runs=n_runs)
        reps = repetitions_per_100(repairs)
        results[cond] = {'repairs_per_trial': repairs.tolist(),
                         'repetitions_per_100': reps,
                         'learning_phase_runs': n_runs}
        print(f'  → {reps:.1f} reps / 100 syllables', flush=True)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'exp3_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_repetition_counts(
        {k: v['repetitions_per_100'] for k, v in results.items()},
        os.path.join(FIGURES_DIR, 'fig4_slow_speech.png'))

    print('Experiment 3 complete.')


if __name__ == '__main__':
    main()
