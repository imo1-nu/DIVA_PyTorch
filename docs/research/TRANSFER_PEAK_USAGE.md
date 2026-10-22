# Static transfer-peak feedback experiment

The normal application remains `DIVA()` / `VocalTract()` (`fmfit`). Start a new
simulation with `DIVA(observation_mode="transfer_peaks")` to opt in. Configure
this before creating sensory/motor blocks; changing mode mid-run is unsupported.

```python
from main import DIVA
model = DIVA(observation_mode="transfer_peaks")
# Select/prepare a target with the existing application API, then:
model.StartSim(False, None, False)
```

`VocalTract.observe(raw)` returns `(AuditoryObservation, fitted_somatic)` and
`VocalTract.output()` puts its values on the usual two output ports.
`last_observation` contains normalized values, validity, reasons, local 5 ms
step index, original fmfit values, measured Hz, peak count and elapsed seconds.
For custom block wiring, pass the observation with the values:

```python
observation, somatic = vt.observe(raw, step=step)
auditory_state_map.input(observation.values, observation)
# At the next scheduled output of the existing delayed state map:
auditory_state_map.output()
auditory_error_map.InputPorts[1] = auditory_state_map.OutputPorts[0]
auditory_error_map.validity = auditory_state_map.validity
auditory_error_map.output()
```

Use `AuditoryStateMap(masked=True)` for transfer observations. Its initial and
reset F1–F3 masks are false; F0 retains the existing zero-buffer behavior. The
same ten-slot FIFO delays values, masks and observation metadata. A component
with an invalid measurement has a finite transport placeholder 0 and a false
mask; its auditory error is zero before the existing motor/error delays and
learning. Partial observations keep only available, frequency-ordered peaks.
Do not interpret those placeholders as measured 0 Hz or as goal attainment.
Raw nonfinite commands raise an explicit error before model matrix operations;
nonfinite area/H and other geometry measurement failures produce masks/reasons.

`diva_vocaltract`, `vocal_tract_compute`, `ExplicitWeightBlock`, and
`NullSpaceGain` remain the fitted model prediction path. Neither inverse nor
postural Jacobians call `observe`. F0 and all eight fitted somatic components
come from this original path. The geometry call requesting four outputs is used
only to obtain area and section length; its somatic output is discarded.

The observer owns a separate VocalTract geometry/cache object. It does not call
waveform synthesis or mutate source/voicing/pressure/closure/random state. It
applies Input.Scale and tanh exactly once. Native `a2h` evaluates a static H at
11025 Hz/65536 points; these points are a frequency grid, not a measurement
window or an added sensory delay. Negative areas become zero; positive areas
below .025 cm² become .025. Any remaining zero is a closure and invalidates all
three peaks. Extrema use positive-frequency 20log10(abs(H)), prominence1 dB,
spacing≥50 Hz, 50–4950 Hz, lowest three in frequency order. These are H maxima,
not waveform formants, pole frequencies or human ground truth.

CPU operation uses the same native formula and precision as the existing
CUDA-backed shape computation. When CUDA is available it remains selected.
The 401-point initialization compatibility change is separately committed and
matches PR #1; FIR coefficients, delays and learning EPS remain unchanged.

## Reproduce

The existing Torch/NumPy/SciPy environment must also satisfy the upstream imports
(sounddevice, librosa, IPython, matplotlib). Validation used Python3.12.3.

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/diva-mpl-cache \
  /tmp/stutter_diva_audit_venv/bin/python -m unittest discover -s test -p 'test_*.py' -v

# Use a fresh output directory for each mode; never write into source assets.
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/diva-mpl-cache \
  /tmp/stutter_diva_audit_venv/bin/python experiments/run_transfer_peak_feedback.py \
  --target /path/to/existing/diva_control01/inputs/u \
  --mode transfer_peaks --output /tmp/transfer-peak-repeat/transfer_peaks
# Repeat with --mode fmfit and a different output directory.
```

All 20 online trajectories, all four frozen trajectories, errors, instructions,
weights, masks/reasons, timestamps and processing durations are saved to NPZ.
The frozen evaluation bypasses `AdaptiveWeightBlock.output` and asserts exact
weight equality. Independently transcribed outline→area→H equations recompute
all frozen H peaks; the observation function is never used for primary metrics.
Full input copies and source/asset/config/environment hashes accompany results.
Do not compare an online learning trajectory with a frozen generation.

[4回・20回後の固定比較結果](results/transfer_peak01/RESULT.md)と全指標・検証来歴を保存している。
