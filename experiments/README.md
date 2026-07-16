# Civier et al. (2010) Reproduction Experiments

Reproduction of the four modelling experiments in:

> Civier, O., Tasko, S. M., & Guenther, F. H. (2010). Overreliance on auditory
> feedback may lead to sound/syllable repetitions: Simulations of stuttering and
> fluency-inducing conditions with a neural model of speech production.
> *Journal of Fluency Disorders*, 35(3), 246–279.

## Setup

```bash
cd experiments/
UV_PROJECT_ENVIRONMENT=/tmp/diva_venv UV_LINK_MODE=copy uv sync
```

## Gain parameter mapping

The paper uses α_ff / α_fb; the code uses `feedforward_gain` / `feedback_gain`:

| Condition       | α_ff | α_fb | feedforward_gain | feedback_gain |
|-----------------|------|------|-----------------|---------------|
| Non-stuttering  | 0.85 | 0.15 | 0.85            | 0.30          |
| Stuttering      | 0.25 | 0.75 | 0.25            | 1.50          |

`feedback_gain = α_fb / fb_gain_fixed (0.5) = 2 × α_fb`

## Experiments

### Exp 1 — Sensorimotor errors in fluent speech (Fig. 2)

Compares auditory F2 error time-series for `/bid/` (rapid F2 transition) vs
`/bad/` (slow F2 transition) under both gain conditions.

```bash
cd exp1_fluent_errors/
UV_PROJECT_ENVIRONMENT=/tmp/diva_venv python run_exp1.py
```

**Expected**: stuttering DIVA shows larger peak F2 errors for `/bid/` than for
`/bad/`; non-stuttering DIVA shows small errors for both.

### Exp 2 — Sound/syllable repetitions (Fig. 3)

Shows that stuttering DIVA generates significantly more self-repair events
(repetitions) than non-stuttering DIVA on `/bid/`.

```bash
cd exp2_repetitions/
UV_PROJECT_ENVIRONMENT=/tmp/diva_venv python run_exp2.py
```

### Exp 3 — Slow speech reduces repetitions (Fig. 4)

Stuttering DIVA at half speed (`speed_scale=0.5`, target length doubled)
produces fewer repetitions than at normal speed.

```bash
cd exp3_slow_speech/
UV_PROJECT_ENVIRONMENT=/tmp/diva_venv python run_exp3.py
```

### Exp 4 — Masking noise reduces repetitions (Fig. 5)

Increasing masking noise `I_masking` suppresses auditory error detection and
reduces repetitions monotonically.

```bash
cd exp4_masking_noise/
UV_PROJECT_ENVIRONMENT=/tmp/diva_venv python run_exp4.py
```

## Directory structure

```
experiments/
├── configs/
│   ├── stuttering.yaml       # feedforward=0.25, feedback=1.50
│   ├── non_stuttering.yaml   # feedforward=0.85, feedback=0.30
│   └── slow_speech.yaml      # stuttering gains + speed_scale=0.5
├── monitoring_subsystem/
│   ├── excessive_error_detector.py   # P(R_error) formula
│   └── self_repair_controller.py     # step-by-step sim with restart
├── analysis/
│   ├── formant_analysis.py   # normalised → Hz conversion
│   ├── plot_figures.py       # matplotlib figure helpers
│   └── stats.py              # mean/SEM, bootstrap CI
├── exp1_fluent_errors/
├── exp2_repetitions/
├── exp3_slow_speech/
└── exp4_masking_noise/
```

## Custom targets (src/)

| ID  | Label    | Description                               |
|-----|----------|-------------------------------------------|
| 508 | bid      | /bɪd/ — rapid F2 rise (300 ms)           |
| 509 | bad      | /bæd/ — slow F2 rise (300 ms)            |
| 510 | good_dog | /ɡʊd dɒɡ/ — two-syllable target (500 ms) |

## Notes

- All experiment scripts must be run from their own directory, or via the
  `UV_PROJECT_ENVIRONMENT=/tmp/diva_venv` prefix so that the correct venv is used.
- Scripts change `os.chdir` to `src/` internally so DIVA's `.mat` files are found.
- CIFS filesystem (`/home/futagami26/mrnas06home/`) does not support symlinks;
  the uv virtual environment lives in `/tmp/diva_venv`.
