"""
Calibrate ExcessiveErrorDetector epsilon.

Creates diva instances once, then reuses them across all eps levels.
Run from repo root:
  UV_PROJECT_ENVIRONMENT=/tmp/diva_venv UV_LINK_MODE=copy uv run --project experiments python experiments/calibrate_eps.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import numpy as np
from diva_setup import make_diva
from monitoring_subsystem.excessive_error_detector import ExcessiveErrorDetector
from monitoring_subsystem.self_repair_controller import SelfRepairController

NORMAL_FF, NORMAL_FB   = 0.85, 0.30
STUTTER_FF, STUTTER_FB = 0.25, 1.50
N_TRIALS = 200
SEED = 42

print("Building DIVA instances...", flush=True)
diva_st = make_diva('bid', STUTTER_FF, STUTTER_FB, learning_phase_runs=4, seed=SEED)
diva_ns = make_diva('bid', NORMAL_FF,  NORMAL_FB,  learning_phase_runs=4, seed=SEED)
print("Done building.", flush=True)

# ---------- step 1: trial consistency check ----------
print("\n=== Trial consistency check (stutter, eps=1e-5) ===", flush=True)
det0 = ExcessiveErrorDetector(epsilon=1e-5)
ctrl0 = SelfRepairController(det0)
rng0 = np.random.default_rng(0)
for t in range(5):
    _, log = ctrl0.run(diva_st, I_masking=0.0, rng=rng0)
    sizes = np.array([det0.f_size(e) for e in log])
    print(f"  Trial {t}: max={sizes.max():.2f}  mean={sizes.mean():.2f}", flush=True)

# ---------- step 2: calibration sweep ----------
print("\n=== Calibration sweep ===", flush=True)
eps_vals = [5e-6, 1e-5, 2e-5, 3e-5, 5e-5, 8e-5, 1e-4]

for eps in eps_vals:
    det = ExcessiveErrorDetector(epsilon=eps)
    ctrl_st = SelfRepairController(det)
    ctrl_ns = SelfRepairController(det)
    rng_st = np.random.default_rng(SEED)
    rng_ns = np.random.default_rng(SEED)

    st_reps = sum(ctrl_st.run(diva_st, rng=rng_st)[0] for _ in range(N_TRIALS))
    ns_reps = sum(ctrl_ns.run(diva_ns, rng=rng_ns)[0] for _ in range(N_TRIALS))

    st_per100 = st_reps / N_TRIALS * 100
    ns_per100 = ns_reps / N_TRIALS * 100
    print(f"  eps={eps:.0e}  stutter={st_per100:6.1f}  normal={ns_per100:6.1f}  reps/100", flush=True)

print("\nDone.", flush=True)
