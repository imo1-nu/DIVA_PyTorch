"""
Synthesize stuttering/normal /bid/ audio using Praat KlattGrid via parselmouth.

DIVA formant trajectory (F1/F2/F3) → Praat KlattGrid → WAV

Outputs:
  experiments/stuttering_praat.wav  — /bid/ with 1 repair repetition
  experiments/normal_praat.wav      — /bid/ fluent

Run from repo root:
  UV_PROJECT_ENVIRONMENT=/tmp/diva_venv UV_LINK_MODE=copy \
    uv run --project experiments python experiments/synthesize_praat.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src'))

import copy
import numpy as np
import torch
import parselmouth
from parselmouth.praat import call

from diva_setup import make_diva
from monitoring_subsystem.excessive_error_detector import ExcessiveErrorDetector

FRAME_DUR     = 0.005   # s
SILENCE_STEPS = 13
AUD_SCALE_HZ  = np.array([100.0, 500.0, 1500.0, 3000.0])


# ── DIVA simulation ───────────────────────────────────────────────────────────

def run_logged(diva, detector, seed: int, I_masking: float = 0.0,
               max_restarts: int = 5):
    """Return formant trajectory (Hz), f_size (Hz), events, n_repairs."""
    import diva_ssm
    import diva_asm          as aud_sm_mod
    import diva_som_state_map as som_sm_mod
    import diva_aem          as aud_em_mod
    import diva_sem          as som_em_mod

    length_idx = diva.production_info[0].index('length\n')
    total_len  = float(diva.production_info[1][length_idx])
    num_steps  = int(np.ceil(total_len / diva.sample_rate)) + 1

    aud_sm = aud_sm_mod.AuditoryStateMap();      aud_sm.setinputdims(4)
    som_sm = som_sm_mod.SomatosensoryStateMap(); som_sm.setinputdims(8)
    aud_em = aud_em_mod.AuditoryErrorMap()
    som_em = som_em_mod.SomatosensoryErrorMap()

    cerebellum  = diva.cerebellum
    vocal_tract = diva.vocal_tract
    mc          = diva.motor_cortex

    diva.reset()
    if hasattr(diva, '_initial_cerebellum_buffer'):
        cerebellum.dataBuffer = copy.deepcopy(diva._initial_cerebellum_buffer)
    if hasattr(diva, '_initial_lmc_weight'):
        mc.LMC.get_set_weights(3, diva._initial_lmc_weight.clone())

    rng        = np.random.default_rng(seed)
    formant_hz = []
    events     = []
    n_repairs  = 0
    ssm        = diva_ssm.SpeechSoundMap(num_steps)

    for attempt in range(max_restarts + 1):
        ssm.LDA.reset(); ssm.LDS.reset(); ssm.dataBuffer = []
        ssm.InputPorts[0] = torch.tensor([1.0], dtype=torch.float64)
        ssm.InputPorts[1] = 1
        ssm.output(); ssm.output_seq()

        repair_triggered = False

        for _step in range(num_steps + 10):
            p          = ssm.OutputPorts[0]
            aud_target = ssm.OutputPorts[2]
            som_target = ssm.OutputPorts[3]
            ssm.output_seq()

            aud_em.InputPorts[0] = aud_target
            som_em.InputPorts[0] = som_target
            aud_sm.output();  som_sm.output()
            aud_em.InputPorts[1] = aud_sm.OutputPorts[0]
            som_em.InputPorts[1] = som_sm.OutputPorts[0]
            aud_em.output();  som_em.output()

            cerebellum.input(p); cerebellum.output()
            mc.InputPorts[0] = p
            mc.InputPorts[1] = cerebellum.OutputPorts[0]
            mc.InputPorts[2] = som_em.OutputPorts[0]
            mc.InputPorts[3] = aud_em.OutputPorts[0]
            mc.output()

            vocal_tract.InputPorts[0] = mc.OutputPorts[0]
            vocal_tract.output()
            aud_sm.input(vocal_tract.OutputPorts[0])
            som_sm.input(vocal_tract.OutputPorts[1])

            vt_aud = vocal_tract.OutputPorts[0].numpy().flatten()
            err_np = aud_em.OutputPorts[0].numpy().flatten()
            formant_hz.append(vt_aud * AUD_SCALE_HZ)

            if rng.random() < detector.repair_probability(err_np, I_masking):
                events.append((len(formant_hz) - 1, 'repair_trigger'))
                repair_triggered = True
                break

        if not repair_triggered:
            break

        n_repairs += 1
        for _ in range(SILENCE_STEPS):
            formant_hz.append(np.zeros(4))
        events.append((len(formant_hz), 'restart'))

        if attempt >= max_restarts:
            break

    return np.array(formant_hz), events, n_repairs


# ── Praat KlattGrid synthesis ─────────────────────────────────────────────────

# Approximate formant bandwidths (Hz) for a male voice
_BW = {1: 80.0, 2: 110.0, 3: 160.0}

# F2 threshold to distinguish voiced phonation from closure/silence
_F2_VOICED_THRESHOLD = 100.0   # Hz


def _f0_at(t: float, duration: float,
           f0_start: float = 130.0, f0_end: float = 100.0) -> float:
    """Linear F0 declination (natural intonation)."""
    return f0_start + (f0_end - f0_start) * (t / max(duration, 1e-6))


def formants_to_klatt(formant_hz: np.ndarray,
                      frame_dur: float = FRAME_DUR) -> parselmouth.Sound:
    """
    Synthesize from a formant trajectory using Praat KlattGrid.

    formant_hz : (T, 4) array — F0/F1/F2/F3 in Hz; silence frames = 0
    """
    T        = len(formant_hz)
    duration = T * frame_dur

    # voiced[t] = True when vocal tract is producing sound
    voiced = formant_hz[:, 2] > _F2_VOICED_THRESHOLD

    # ── Create KlattGrid ──────────────────────────────────────────────────────
    # (name, start, end, nOral, nNasal, nNasalAnti, nTracheal,
    #  nTrachealAnti, nDelta, nFrication)
    kg = call("Create KlattGrid...", "bid", 0.0, duration,
              5, 1, 1, 1, 1, 1, 3)

    # Set fixed bandwidths once at t=0 (Praat interpolates constant)
    for fi, bw in _BW.items():
        call(kg, "Add oral formant bandwidth point...", fi, 0.0, bw)

    prev_voiced = False

    for i in range(T):
        t = (i + 0.5) * frame_dur           # mid-frame time
        f1 = formant_hz[i, 1]
        f2 = formant_hz[i, 2]
        f3 = formant_hz[i, 3]
        is_voiced = bool(voiced[i])
        f0 = _f0_at(t, duration)

        # ── Voicing amplitude ─────────────────────────────────────────────────
        if is_voiced:
            call(kg, "Add voicing amplitude point...", t, 80.0)
            call(kg, "Add pitch point...", t, f0)
        else:
            call(kg, "Add voicing amplitude point...", t, 0.0)

        # ── Bilabial burst at voiced onset (simulates /b/ release) ────────────
        if is_voiced and not prev_voiced:
            t_burst_on  = max(0.001, t - frame_dur * 0.8)
            t_burst_off = t - frame_dur * 0.1
            call(kg, "Add aspiration amplitude point...", t_burst_on,  55.0)
            call(kg, "Add aspiration amplitude point...", t_burst_off,  0.0)

        # ── Formant frequencies (voiced frames only) ──────────────────────────
        if is_voiced:
            if f1 > 50.0:
                call(kg, "Add oral formant frequency point...", 1, t, f1)
            if f2 > 50.0:
                call(kg, "Add oral formant frequency point...", 2, t, f2)
            if f3 > 50.0:
                call(kg, "Add oral formant frequency point...", 3, t, f3)

        prev_voiced = is_voiced

    return call(kg, "To Sound")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("Building DIVA instances...", flush=True)
    diva_st = make_diva('bid', 0.25, 1.50, learning_phase_runs=4, seed=42)
    diva_ns = make_diva('bid', 0.85, 0.30, learning_phase_runs=4, seed=42)
    detector = ExcessiveErrorDetector(epsilon=2e-5)
    print("Done.", flush=True)

    # ── Stuttering (seed=0 → 1 repair) ───────────────────────────────────────
    print("Running stuttering simulation (seed=0)...", flush=True)
    fhz_st, events_st, n_rep = run_logged(diva_st, detector, seed=0)
    print(f"  {len(fhz_st)} frames = {len(fhz_st)*FRAME_DUR*1000:.0f} ms, "
          f"repairs={n_rep}", flush=True)
    for fi, label in events_st:
        print(f"  [{label}] frame={fi}, t={fi*FRAME_DUR*1000:.0f} ms")

    print("Synthesizing stutter (Praat KlattGrid)...", flush=True)
    snd_st = formants_to_klatt(fhz_st)
    out_st = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'stuttering_praat.wav')
    snd_st.save(out_st, parselmouth.SoundFileFormat.WAV)
    print(f"  Saved: {out_st}", flush=True)

    # ── Normal (seed=0, no repair) ────────────────────────────────────────────
    print("Running normal simulation (seed=0)...", flush=True)
    fhz_ns, events_ns, _ = run_logged(diva_ns, detector, seed=0)
    print(f"  {len(fhz_ns)} frames = {len(fhz_ns)*FRAME_DUR*1000:.0f} ms",
          flush=True)

    print("Synthesizing normal (Praat KlattGrid)...", flush=True)
    snd_ns = formants_to_klatt(fhz_ns)
    out_ns = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'normal_praat.wav')
    snd_ns.save(out_ns, parselmouth.SoundFileFormat.WAV)
    print(f"  Saved: {out_ns}", flush=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n--- Summary ---")
    print(f"  stuttering_praat.wav : {len(fhz_st)*FRAME_DUR*1000:.0f} ms  "
          f"(attempt→silence→restart pattern)")
    print(f"  normal_praat.wav     : {len(fhz_ns)*FRAME_DUR*1000:.0f} ms  "
          f"(fluent single production)")
    print(f"\n  Stuttering breakdown:")
    prev_fi = 0
    for fi, label in events_st:
        print(f"    [{prev_fi*5:.0f}–{fi*5:.0f} ms] → {label}")
        prev_fi = fi
    print(f"    [{prev_fi*5:.0f}–{len(fhz_st)*5:.0f} ms] → complete /bid/")


if __name__ == '__main__':
    main()
